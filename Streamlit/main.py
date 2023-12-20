import streamlit as st
from st_pages import Page, show_pages, add_page_title
from streamlit_lottie import st_lottie 
from PIL import Image
import base64

# st.set_page_config(layout="wide")

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

show_pages(
    [
        Page("main.py", "🏠HOME", ""),
        Page("./page/Overview.py", "💿OVERVIEW", ""),
        Page("./page/Data.py", "💿DATA", ""),
        Page("./page/Product.py", "💿PRODUCT", ""),
    ]
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

st.image(img, width=700)

st.markdown('<h1 style="font-family: \'SKYBORI\', sans-serif; font-size:3em; text-align:center;">JEJULANG.COM</h1>', unsafe_allow_html=True)

st.markdown(
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.5em; text-align:center;">'
    '사용자가 제주도 방언으로 텍스트를 입력하면 해당 데이터를 표준어 텍스트로 번역하여 제공하는 서비스'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.5em; text-align:center;">'
    '사용자가 입력한 방언 문구는 처리 후, 표준어로 번역된 결과가 웹 페이지에 정형화되어 출력'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.5em; text-align:center;">'
    '번역된 내용은 사용자가 제공한 제주도 방언의 의미를 명확하게 전달하도록 고려'
    '</p>',
    unsafe_allow_html=True
)

st.markdown('<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.5em; text-align:center;">DEV. 김선혁, 안상우, 석찬비</p>', unsafe_allow_html=True)