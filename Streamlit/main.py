import streamlit as st
from st_pages import Page, show_pages, add_page_title
from streamlit_lottie import st_lottie 
from PIL import Image

# st.set_page_config(layout="wide")

show_pages(
    [
        Page("main.py", "🏠HOME", ""),
        Page("./page/Overview.py", "💿OVERVIEW", ""),
        Page("./page/Data.py", "💿DATA", ""),
        Page("./page/Product.py", "💿PRODUCT", ""),
    ]
) 

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

st.image(img, width=700)

#폰트는 고쳐야함
font_style = """
        <style>
            @font-face {
              font-family: 'Dongle';
              src: local('Dongle'), url('./Dongle-Regular.ttf') format('truetype');
            }
            body {
                font-family: 'Dongle', sans-serif;
            }
        </style>
    """

session_state = st.session_state
if not hasattr(session_state, 'font_loaded'):
    st.markdown(font_style, unsafe_allow_html=True)
    session_state.font_loaded = True

st.markdown('<h1 style="font-size:3em; text-align:center;">JEJULANG.COM</h1>', unsafe_allow_html=True)

st.markdown(
    '<p style="font-size:1.5em; text-align:center;">'
    '사용자가 제주도 방언으로 텍스트를 입력하면 해당 데이터를 표준어 텍스트로 번역하여 제공하는 서비스'
    '</p>'
    '<p style="font-size:1.5em; text-align:center;">'
    '사용자가 입력한 방언 문구는 처리 후, 표준어로 번역된 결과가 웹 페이지에 정형화되어 출력'
    '</p>'
    '<p style="font-size:1.5em; text-align:center;">'
    '번역된 내용은 사용자가 제공한 제주도 방언의 의미를 명확하게 전달하도록 고려'
    '</p>',
    unsafe_allow_html=True
)

st.markdown('<p style="font-size:1.5em; text-align:center;">DEV. 김선혁, 안상우, 석찬비</p>', unsafe_allow_html=True)