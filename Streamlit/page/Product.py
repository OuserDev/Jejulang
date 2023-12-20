import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from PIL import Image
import base64

font_path = "SKYBORI.ttf"

# 글꼴 적용
st.markdown(
    f"""
    <style>
        @font-face {{
            font-family: 'SKYBORI';
            src: url('data:font/truetype;charset=utf-8;base64,{base64.b64encode(open(font_path, "rb").read()).decode("utf-8")}') format('truetype');
        }}
        h1, h2, h3, h4, h5, h6, .stTitle, .markdown-text {{
            font-family: 'SKYBORI', sans-serif !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.imgur.com/neR8NEG.jpg");
             background-attachment: fixed;
             background-size: cover
             
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

img = Image.open('icon.png')

st.image(img)

st.title("PRODUCT")

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">제주도 방언을 입력하세요.</h2>', unsafe_allow_html=True)

input_text = st.text_area("", "뭐랭하멘?")

if st.button("번역"):
    translated_result = translate_jeju_dialect(input_text)
    st.success(f"번역 결과: {translated_result}")



