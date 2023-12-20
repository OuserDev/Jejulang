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
        <img src='data:image/png;base64,{img_base64}' style='width: 350px; display: block; margin-left: auto; margin-right: auto;'>
    </div>
"""
st.markdown(img_html, unsafe_allow_html=True)

st.markdown('<h1 style="font-family: \'SKYBORI\', sans-serif; font-size:3em;">OVERVIEW</h1>', unsafe_allow_html=True)

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.8em;">개발 동기</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '제주 방언은 현재 유네스코에 소멸 위기에 처한 언어로 분류'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '제주 방언의 희소성 때문에 제주도 주민들과의 의사소통에 어려움을 겪는 경우가 많음'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '이에 의사소통 장벽의 해소 및 가치와 중요성 전파를 목표'
    '</p>',
    unsafe_allow_html=True
)

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.8em;">개발 기간</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '2023년 9월1월 ~ 2023년 12월31일'
    '</p>',
    unsafe_allow_html=True
)

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.8em;">개발 인원</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '총 3명 (기획 및 디자인 1명, 개발 2명)'
    '</p>',
    unsafe_allow_html=True
)

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; ont-size:1.8em;">담당 업무</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '김선혁 : PM(프로젝트 매니저), AI 개발(정)'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '안상우 : AI 개발(부), Web 구축'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '석찬비 : 기획 및 UI/UX 디자인'
    '</p>',
    unsafe_allow_html=True
)

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.8em;">프로그램 세부내용</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    'Python3의 TensorFlow를 이용하여 AI 모델 개발'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '확보된 텍스트 데이터셋을 Google Cloud Storage에 저장'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '구축한 서비스를 streamlit를 이용하여 웹으로 구현 후 제공'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    'Git & Github를 활용하여 팀원들과 함께 효과적인 협업 진행'
    '</p>',
    unsafe_allow_html=True
)

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.8em;">개발 환경</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '플랫폼 : Web (Streamlit Framework)'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '사용 언어 : Python3 - Tensorflow (Keras)'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '사용 툴 : Google Colab, VSCode, Git & Github'
    '</p>',
    unsafe_allow_html=True
)

