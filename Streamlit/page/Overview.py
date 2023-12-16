import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from PIL import Image

img = Image.open('icon.png')

st.image(img)

st.title("Overview")

uploaded_file = st.file_uploader("음성 변환 파일을 선택하세요.", type='mp3')