import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from PIL import Image
import base64

# 이미지를 Base64 문자열로 변환
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

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

# 이미지 로드
img_path = 'icon.png'
img_base64 = get_image_as_base64(img_path)
# HTML을 사용하여 이미지 스타일 적용
img_html = f"""
    <div style="text-align: center; margin-top:100px; ">
        <img src='data:image/png;base64,{img_base64}' style='width: 500px; display: block; margin-left: auto; margin-right: auto;'>
    </div>
"""
st.markdown(img_html, unsafe_allow_html=True)

st.title("PRODUCT")

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">제주도 방언이 포함된 문장을 입력해주세요.</h2>', unsafe_allow_html=True)

input_text = st.text_area("", "뭐랭하멘?")

if st.button("번역"):
    translated_result = translate_jeju_dialect(input_text)
    st.success(f"번역 결과: {translated_result}")



