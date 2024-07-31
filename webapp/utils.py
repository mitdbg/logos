import streamlit as st
from streamlit_extras.app_logo import add_logo

def logo():
    add_logo('images/dsg-logo.png')

def hide_default():
    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)

def refresh_page(page: str):
    if 'current_page' not in st.session_state or st.session_state['current_page'] != page:
        for key in st.session_state.keys():
            del st.session_state[key]
        
        st.session_state['current_page'] = page

def reset_session_button():
    def on_click():
        st.session_state['current_page'] = 'reset'
    
    st.sidebar.button('Reset', on_click=on_click)


def display_current_info(display_info: tuple[str, str]):
    for header_text, state in display_info:
        st.sidebar.header(header_text)

        if state not in st.session_state:
            st.sidebar.text('None')
        else:
            st.sidebar.text(st.session_state[state])