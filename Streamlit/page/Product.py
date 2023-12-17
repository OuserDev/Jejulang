import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from PIL import Image

'''def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.imgur.com/EmCsh2V.jpg");
             background-attachment: fixed;
             background-size: cover
             
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()'''

img = Image.open('icon.png')

st.image(img)

st.title("PRODUCT")

st.markdown('<h2 style="font-size:1.3em;">제주도 방언을 입력하세요.</h2>', unsafe_allow_html=True)

input_text = st.text_area("", "뭐랭하멘?")

if st.button("번역"):
    translated_result = translate_jeju_dialect(input_text)
    st.success(f"번역 결과: {translated_result}")



