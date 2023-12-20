import streamlit as st
from st_pages import Page, show_pages, add_page_title
from streamlit_lottie import st_lottie 
from PIL import Image
import base64

# 이미지를 Base64 문자열로 변환
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
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

# 이미지 로드
img_path = 'icon.png'
img_base64 = get_image_as_base64(img_path)
# HTML을 사용하여 이미지 스타일 적용
img_html = f"""
    <div style="text-align: center; ">
        <img src='data:image/png;base64,{img_base64}' style='width: 400px; display: block; margin-left: auto; margin-right: auto;'>
    </div>
"""
st.markdown(img_html, unsafe_allow_html=True)

st.markdown('<h1 style="font-family: \'SKYBORI\', sans-serif; font-size:5em; text-align:center;">JEJULANG.COM</h1>', unsafe_allow_html=True)

st.markdown(
    '<div>'
    '<p style="margin:70px 0 35px 0; font-family: \'SKYBORI\', sans-serif; font-size:2em; text-align:center;">'
    '사용자가 제주도 방언이 담긴 텍스트를 입력하면,<br>한글 표준어 텍스트로 번역하여 제공하는 서비스'
    '</p>'
    '<p style="margin-bottom:40px; font-family: \'SKYBORI\', sans-serif; font-size:2em; text-align:center;">'
    '사용자가 입력한 방언 문구는 인공지능을 이용하여 변환 처리 후,<br>표준어로 번역된 결과가 웹 페이지에 정형화되어 출력<br>'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:2em; text-align:center;">'
    '번역된 내용은 사용자가 제공한 제주도 방언의 의미를 명확하게 전달하도록 고려'
    '</p>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.8em; text-align:center; font-weight: bold; margin-top:60px; ">DEV. 김선혁, 안상우, 석찬비</p>', unsafe_allow_html=True)
