import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from PIL import Image

img = Image.open('icon.png')

st.image(img)

st.title("Product")