import subprocess
import sys

# Tentar instalar o distutils se ele estiver faltando
try:
    import distutils
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "distutils"])
import yt_dlp
import streamlit as st
import whisper
import openai
import numpy as np  # Import necessário para o funcionamento do Whisper
import torch  # Import necessário para o funcionamento do Whisper

def download_facebook_video(url):
    ydl_opts = {
        'format': 'best',
        'outtmpl': 'downloads/%(title)s.%(ext)s',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', None)
            video_ext = info_dict.get('ext', None)
            return f"{video_title}.{video_ext}"
        except yt_dlp.utils.DownloadError as e:
            st.error(f"Erro ao baixar o vídeo: {e}")
            return None

def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']

def summarize_text(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"Resuma o seguinte texto:\n\n{text}",
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Interface Streamlit
st.title('Download e Transcrição de Vídeos do Facebook')

url = st.text_input('Cole o link do vídeo do Facebook')
if st.button('Download e Transcrever'):
    file_name = download_facebook_video(url)
    if file_name:
        st.write(f"Arquivo baixado: {file_name}")
        transcription = transcribe_audio(f'downloads/{file_name}')
        st.text_area('Transcrição', transcription)
        summary = summarize_text(transcription)
        st.text_area('Resumo', summary)
