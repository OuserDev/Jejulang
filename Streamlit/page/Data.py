import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import matplotlib.font_manager as fm
import numpy as np 
import seaborn as sns 
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

st.title("DATA")

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.8em;">AI-Hub 데이터셋</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '중·노년층 한국어 방언 데이터(제주도)'
    '</p>',
    unsafe_allow_html=True
)

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.8em;">소개</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '제주도 지역의 50대 이상 발화자가 발화한 따라말하기(정형), 질문답하기(비정형), 2인대화(비정형) 의 방언 텍스트 데이터'
    '</p>',
    unsafe_allow_html=True
)

img2 = Image.open('./aihub.png') 
st.image(img2, width=700)

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.8em;">데이터 분포</h2>', unsafe_allow_html=True)

# 성별 비율
gender_data = {
    '성별': ['남성', '여성'],
    '백분율': [17, 83]
}

df_gender = pd.DataFrame(gender_data)

# 발화 타입 비율
speech_data = {
    '발화 타입': ['따라말하기', '질문답하기', '2인대화'],
    '발화 시간': [66.1, 100.6, 40.5],
    '발화 타입 비율': [32, 49, 20]
}

df_speech = pd.DataFrame(speech_data)

# 연령대 비율
age_data = {
    '나이': ['50대', '60대', '70대', '80대'],
    '백분율': [65, 28, 6, 1]
}

df_age = pd.DataFrame(age_data)
    
# 그래프 선택
selected_data = st.selectbox("데이터 선택", ["성별 비율", "발화 타입 비율", "연령대 비율"])
selected_graph = st.selectbox("그래프 유형 선택", ["원형 그래프", "막대 그래프"])

@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd()]
    font_files = fm.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)    

def main():
    plt.rc('font', family=fontname)

fontRegistered()
fontNames = [f.name for f in fm.fontManager.ttflist]
filteredFontNames = [font for font in fontNames if "SKYBORI" in font]
fontname = st.selectbox("폰트", filteredFontNames)

show_result = st.button("결과 표시")

if show_result:
    # 성별 그래프 표시
    if selected_data == "성별 비율":
        st.subheader("성별 비율")
        if selected_graph == "원형 그래프":
            fig_gender_pie, ax_gender_pie = plt.subplots()
            ax_gender_pie.pie(df_gender['백분율'], labels=df_gender['성별'], autopct='%1.0f%%', startangle=90)
            ax_gender_pie.axis('equal')
            st.pyplot(fig_gender_pie)
        elif selected_graph == "막대 그래프":
            fig_gender_bar = px.bar(df_gender, x='성별', y='백분율', text='백분율', title='성별 비율')
            st.plotly_chart(fig_gender_bar)

    # 발화 타입 그래프 표시
    elif selected_data == "발화 타입 비율":
        st.subheader("발화 타입 비율")
        if selected_graph == "원형 그래프":
            fig_speech_pie, ax_speech_pie = plt.subplots()
            ax_speech_pie.pie(df_speech['발화 타입 비율'], labels=df_speech['발화 타입'], autopct='%1.0f%%', startangle=90)
            ax_speech_pie.axis('equal')
            st.pyplot(fig_speech_pie)
        elif selected_graph == "막대 그래프":
            fig_speech_bar = px.bar(df_speech, x='발화 타입', y='발화 시간', text='발화 시간', title='발화 타입')
            st.plotly_chart(fig_speech_bar)

    # 연령대 타입 그래프 표시
    elif selected_data == "연령대 비율":
        st.subheader("연령대 비율")
        if selected_graph == "원형 그래프":
            fig_age_pie, ax_age_pie = plt.subplots()
            ax_age_pie.pie(df_age['백분율'], labels=df_age['나이'], autopct='%1.0f%%', startangle=90)
            ax_age_pie.axis('equal')
            st.pyplot(fig_age_pie)
        elif selected_graph == "막대 그래프":
            fig_age_bar = px.bar(df_age, x='나이', y='백분율', text='백분율', title='연령대')
            st.plotly_chart(fig_age_bar)

st.markdown('<h2 style="font-family: \'SKYBORI\', sans-serif; font-size:1.8em;">AI MODEL</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.5em;">'
    'Transformer'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    'Attention 매커니즘에 기반을 둔, 자연어 처리 분야에서 매우 뛰어난 성능을 보이는 심층 학습 모델'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '- 인코더와 디코더 구조'
    '</p>'
    '<p style="font-family: \'SKYBORI\', sans-serif; font-size:1.3em;">'
    '- 문장의 모든 단어를 동시에 병렬 처리'
    '</p>',
    unsafe_allow_html=True
)

img_path = './트랜스포머.png'
img_base64 = get_image_as_base64(img_path)
# HTML을 사용하여 이미지 스타일 적용
img_html = f"""
    <div style="text-align: center; margin-top:100px; ">
        <img src='data:image/png;base64,{img_base64}' style='width: 450px; display: block; margin-left: auto; margin-right: auto;'>
    </div>
"""
st.markdown(img_html, unsafe_allow_html=True)

main()

