import os
import queue
import time
from pathlib import Path
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import openai
import pydub
from moviepy.editor import VideoFileClip
from yt_dlp import YoutubeDL
from plyer import notification
import pandas as pd
from datetime import datetime, timedelta
import pytz
import threading
import subprocess

# Configuração da chave da API OpenAI diretamente no código
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configuração do caminho para o ffmpeg
ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'
pydub.AudioSegment.converter = ffmpeg_path

# Configurações gerais
PASTA_TEMP = Path(__file__).parent / 'temp'
PASTA_TEMP.mkdir(exist_ok=True)
ARQUIVO_AUDIO_TEMP = PASTA_TEMP / 'audio.mp3'
ARQUIVO_VIDEO_TEMP = PASTA_TEMP / 'video.mp4'
ARQUIVO_MIC_TEMP = PASTA_TEMP / 'mic.mp3'

# Inicialização segura de session state
if 'grava_process' not in st.session_state:
    st.session_state['grava_process'] = None

# =============================================
# Funções de Transcrição
# =============================================

def transcreve_audio(caminho_audio, prompt):
    with open(caminho_audio, 'rb') as arquivo_audio:
        transcricao = openai.Audio.transcribe(
            model='whisper-1',
            language='pt',
            file=arquivo_audio,
            prompt=prompt,
        )
        return transcricao['text']

if 'transcricao_mic' not in st.session_state:
    st.session_state['transcricao_mic'] = ''

@st.cache_data
def get_ice_servers():
    return [{'urls': ['stun:stun.l.google.com:19302']}]

def adiciona_chunck_de_audio(frames_de_audio, chunck_audio):
    for frame in frames_de_audio:
        sound = pydub.AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels)
        )
        chunck_audio += sound
    return chunck_audio

def transcreve_tab_mic():
    prompt_mic = st.text_input('(opcional) Digite o seu prompt', key='input_mic')
    webrtx_ctx = webrtc_streamer(
        key='recebe_audio',
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={'video': False, 'audio': True}
    )

    if not webrtx_ctx.state.playing:
        st.write(st.session_state['transcricao_mic'])
        return
    
    container = st.empty()
    container.markdown('Comece a falar...')
    chunck_audio = pydub.AudioSegment.empty()
    tempo_ultima_transcricao = time.time()
    st.session_state['transcricao_mic'] = ''
    while True:
        if webrtx_ctx.audio_receiver:
            try:
                frames_de_audio = webrtx_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                continue
            chunck_audio = adiciona_chunck_de_audio(frames_de_audio, chunck_audio)

            agora = time.time()
            if len(chunck_audio) > 0 and agora - tempo_ultima_transcricao > 10:
                tempo_ultima_transcricao = agora
                chunck_audio.export(ARQUIVO_MIC_TEMP)
                transcricao = transcreve_audio(ARQUIVO_MIC_TEMP, prompt_mic)
                st.session_state['transcricao_mic'] += transcricao
                container.write(st.session_state['transcricao_mic'])
                chunck_audio = pydub.AudioSegment.empty()
        else:
            break

def _salva_audio_do_video(video_bytes):
    with open(ARQUIVO_VIDEO_TEMP, mode='wb') as video_f:
        video_f.write(video_bytes.read())
    moviepy_video = VideoFileClip(str(ARQUIVO_VIDEO_TEMP))
    moviepy_video.audio.write_audiofile(str(ARQUIVO_AUDIO_TEMP))

def transcreve_tab_video():
    prompt_input = st.text_input('(opcional) Digite o seu prompt', key='input_video')
    arquivo_video = st.file_uploader('Adicione um arquivo de vídeo .mp4', type=['mp4'])
    if arquivo_video is not None:
        _salva_audio_do_video(arquivo_video)
        transcricao = transcreve_audio(ARQUIVO_AUDIO_TEMP, prompt_input)
        st.write(transcricao)

def transcreve_tab_audio():
    prompt_input = st.text_input('(opcional) Digite o seu prompt', key='input_audio')
    formato_output = st.selectbox('Escolha o formato de saída:', ['Texto', 'Legenda (SRT)'])
    arquivo_audio = st.file_uploader('Adicione um arquivo de áudio .mp3', type=['mp3'])
    
    if arquivo_audio is not None:
        if formato_output == 'Texto':
            transcricao = openai.Audio.transcribe(
                model='whisper-1',
                language='pt',
                file=arquivo_audio,
                prompt=prompt_input
            )
            st.write(transcricao['text'])
        elif formato_output == 'Legenda (SRT)':
            transcricao = openai.Audio.transcribe(
                model='whisper-1',
                language='pt',
                file=arquivo_audio,
                prompt=prompt_input,
                response_format='srt'
            )
            st.text_area("Legendas (SRT):", transcricao, height=400)
            st.download_button("Baixar Legenda", transcricao, file_name="transcricao.srt")

# =============================================
# Funções para Download, Agendamento e Corte de Áudio
# =============================================

# Função para manipular o progresso do download
def progress_hook(d):
    if d['status'] == 'downloading':
        percent_str = d.get('_percent_str', '0.0%').strip()
        percent = float(percent_str.replace('%', ''))
        st.progress(int(percent))

# Função para download de vídeo/áudio com qualidade e formato
def download_video(link, output_format, quality, output_path):
    # Verifica se o caminho é acessível
    if not os.access(output_path, os.W_OK):
        st.error(f"Erro: Permissões insuficientes para escrever no diretório {output_path}.")
        return

    ydl_opts = {
        'format': f'bestaudio/best' if output_format == 'mp3' else quality,
        'outtmpl': os.path.join(output_path, f'%(title)s.%(ext)s'),
        'progress_hooks': [progress_hook],
        'noprogress': True,  # Desativa a barra de progresso padrão do yt-dlp
        'quiet': True,  # Desativa mensagens de log de saída
        'no_color': True,  # Remove caracteres ANSI para evitar erros
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }] if output_format == 'mp3' else []
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

# Função para gravar stream
def grava_stream(link, output_format, output_path, duration):
    # Verifica se o caminho é acessível
    if not os.access(output_path, os.W_OK):
        st.error(f"Erro: Permissões insuficientes para escrever no diretório {output_path}.")
        return

    ydl_opts = {
        'format': 'bestaudio/best' if output_format == 'mp3' else 'best',
        'outtmpl': os.path.join(output_path, f'%(title)s.%(ext)s'),
        'quiet': True,
        'no_color': True,
        'noprogress': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }] if output_format == 'mp3' else []
    }

    st.session_state['grava_process'] = subprocess.Popen(
        ['yt-dlp', link], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    st.write("Gravação iniciada...")  # Confirmação de que o processo de gravação começou.

    # Espera a duração especificada e depois encerra a gravação
    timer = threading.Timer(duration * 60, stop_recording)
    timer.start()

def stop_recording():
    if st.session_state['grava_process'] and st.session_state['grava_process'].poll() is None:
        st.session_state['grava_process'].terminate()
        send_notification("Gravação Encerrada", "A gravação foi encerrada manualmente.")
        st.write("Gravação encerrada manualmente.")
    st.session_state['grava_process'] = None

def save_log(link, formato, output_path):
    log_entry = {
        'Link': link,
        'Formato': formato,
        'Caminho': output_path,
        'Status': 'Concluído'
    }
    df = pd.DataFrame([log_entry])
    if os.path.exists('download_log.csv'):
        df.to_csv('download_log.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('download_log.csv', mode='w', header=True, index=False)

def send_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        timeout=5
    )

# Função para agendar a gravação
def agendar_gravacao(link, output_format, output_path, start_time, duration):
    def job():
        st.write("Executando a função job() para iniciar a gravação.")
        grava_stream(link, output_format, output_path, duration)
    
    # Converter horário atual para o mesmo fuso horário do horário de início
    now = datetime.now(pytz.timezone("America/Sao_Paulo"))
    delay = (start_time - now).total_seconds()

    # Adicionando logs para depuração
    st.write(f"Horário atual (São Paulo): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    st.write(f"Horário de início (São Paulo): {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    st.write(f"Delay calculado: {delay} segundos")

    if delay > 0:
        st.write(f"Gravação agendada para {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        timer = threading.Timer(delay, job)
        timer.start()
        st.write("Timer iniciado.")
    else:
        st.error("O horário de início já passou.")

# Função para carregar e tocar o áudio
def play_audio(audio_file):
    audio = pydub.AudioSegment.from_file(audio_file)
    return audio

# Função para cortar o áudio
def cut_audio(audio, start_time, duration, output_dir, base_filename):
    # Verifica se o caminho é acessível
    if not os.access(output_dir, os.W_OK):
        st.error(f"Erro: Permissões insuficientes para escrever no diretório {output_dir}.")
        return

    start_time_ms = start_time * 1000  # Convertendo para milissegundos
    duration_ms = duration * 1000
    
    end_time_ms = start_time_ms + duration_ms
    clip = audio[start_time_ms:end_time_ms]
    
    output_filename = os.path.join(output_dir, f"{base_filename}_cut_{start_time}-{start_time+duration}.mp3")
    clip.export(output_filename, format="mp3")
    return output_filename

# =============================================
# Interface Principal
# =============================================

def main():
    st.title('Inteligência Eleitoral - Plataforma de Mídia')

    # Menu de navegação
    menu = st.sidebar.selectbox(
        'Escolha a Aplicação:',
        ['IE - Download', 'IE - Agendamento', 'IE - Cut Áudio', 'IE - Transcrição']
    )

    if menu == 'IE - Download':
        st.header('Download de Áudio e Vídeo')
        links = st.text_area('Insira os links dos vídeos/áudios (um por linha):')
        formato = st.selectbox('Escolha o formato:', ['mp3', 'mp4'])
        qualidade = st.selectbox('Escolha a qualidade do vídeo:', ['best', '1080p', '720p', '480p', '360p'])
        output_path = st.text_input('Escolha o diretório para salvar os arquivos:', os.getcwd())
        if st.button('Baixar'):
            if links:
                links_list = links.splitlines()
                for link in links_list:
                    try:
                        download_video(link, formato, qualidade, output_path)
                        save_log(link, formato, output_path)
                        send_notification("Download Concluído", f"Download de {link} foi concluído.")
                        st.success(f'Download de {link} concluído!')
                    except Exception as e:
                        st.error(f'Erro ao baixar {link}: {str(e)}')
                        save_log(link, formato, output_path)
            else:
                st.warning('Por favor, insira um ou mais links válidos.')
        if st.checkbox('Mostrar histórico de downloads'):
            if os.path.exists('download_log.csv'):
                log_df = pd.read_csv('download_log.csv')
                st.dataframe(log_df)
            else:
                st.info('Nenhum download realizado ainda.')

    elif menu == 'IE - Agendamento':
        st.header('Agendamento de Gravação de Stream ao Vivo')
        link = st.text_input('Insira o link do stream ao vivo:')
        formato = st.selectbox('Escolha o formato:', ['mp3', 'mp4'])
        start_date = st.date_input('Escolha a data de início:')
        start_time_str = st.text_input('Escolha o horário de início (HH:MM):')
        duration = st.number_input('Duração da gravação em minutos:', min_value=1, max_value=1440, value=60)
        output_path = st.text_input('Escolha o diretório para salvar os arquivos:', os.getcwd())

        # Verificação do horário de início
        try:
            start_time = datetime.strptime(start_time_str, '%H:%M').time()
        except ValueError:
            st.error('Formato de hora inválido. Use HH:MM.')
            return

        # Ajuste para o fuso horário
        local_tz = pytz.timezone("America/Sao_Paulo")  # Altere para o seu fuso horário se necessário
        start_datetime = local_tz.localize(datetime.combine(start_date, start_time))

        col1, col2 = st.columns(2)
        with col1:
            if st.button('Agendar Gravação'):
                if link:
                    if start_datetime > datetime.now(local_tz):
                        agendar_gravacao(link, formato, output_path, start_datetime, duration)
                    else:
                        st.warning('O horário de início deve ser no futuro.')
                else:
                    st.warning('Por favor, insira um link válido.')

        with col2:
            if st.session_state['grava_process'] and st.session_state['grava_process'].poll() is None:
                if st.button('Parar Gravação'):
                    stop_recording()

        if st.checkbox('Mostrar histórico de gravações'):
            if os.path.exists('download_log.csv'):
                log_df = pd.read_csv('download_log.csv')
                st.dataframe(log_df)
            else:
                st.info('Nenhuma gravação realizada ainda.')

    elif menu == 'IE - Cut Áudio':
        st.header("Cortar Áudio")
        uploaded_file = st.file_uploader("Escolha o arquivo de áudio", type=["mp3", "wav", "ogg"])
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/mp3')
            audio = play_audio(uploaded_file)
            start_time = st.number_input("Defina o tempo de início do corte (em segundos):", min_value=0)
            duration = st.number_input("Defina a duração do corte (em segundos):", min_value=1)
            output_dir = st.text_input("Escolha o diretório para salvar os arquivos cortados:", os.getcwd())
            base_filename = os.path.splitext(uploaded_file.name)[0]
            if st.button("Cortar Áudio"):
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_file = cut_audio(audio, start_time, duration, output_dir, base_filename)
                if output_file:
                    st.success(f"Áudio cortado e salvo em: {output_file}")
                    st.audio(output_file, format='audio/mp3')

    elif menu == 'IE - Transcrição':
        st.header('Transcrição de Áudio e Vídeo')
        tab_mic, tab_video, tab_audio = st.tabs(['Microfone', 'Vídeo', 'Áudio'])
        with tab_mic:
            transcreve_tab_mic()
        with tab_video:
            transcreve_tab_video()
        with tab_audio:
            transcreve_tab_audio()

if __name__ == '__main__':
    main()