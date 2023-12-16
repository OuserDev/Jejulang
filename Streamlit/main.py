import streamlit as st
from st_pages import Page, show_pages, add_page_title
from streamlit_lottie import st_lottie 
from PIL import Image

img = Image.open('icon.png')

st.image(img)

#st.subheader("제주 방언 번역")
st.subheader("🏠JEJU DIALECT TRANSLATOR")

show_pages(
    [
        Page("main.py", "🏠Home", ""),
        Page("./page/Overview.py", "💿Overview", ""),
        Page("./page/Data.py", "💿Data", ""),
        Page("./page/Product.py", "💿Product", ""),
    ]
)   

def load():     
        st_lottie('https://lottie.host/83c07a2f-dd0e-4227-8fd4-7847ca262063/Xz51rSJpFu.json')

def main():
    load()

if __name__ == '__main__':
    main()