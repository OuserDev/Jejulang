import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from PIL import Image

def add_bg_from_url():
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

add_bg_from_url()

img = Image.open('icon.png')

st.image(img)

st.title("💿OVERVIEW")
