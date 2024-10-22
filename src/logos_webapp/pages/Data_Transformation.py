import streamlit as st
from streamlit import components
from streamlit_extras.app_logo import add_logo
import sys
import os

sys.path.append("../")
from src.logos.logos import LOGos
from src.logos_webapp.logos_ui import LOGosUI
from random import sample
from src.logos.graph_renderer import GraphRenderer
from src.definitions import LOGOS_ROOT_DIR

st.set_page_config(page_title="LOGos Demo", page_icon="🪵", layout="wide")

st.title("Data transformation")


def reset_experiment():
    for key in st.session_state.keys():
        del st.session_state[key]


st.sidebar.header("Reset Experiment:")
st.sidebar.button("Reset", on_click=reset_experiment)


def display_current_info(header_text: str, variable: str):
    st.sidebar.header(header_text)

    if variable not in st.session_state:
        st.sidebar.text("None")
    else:
        st.sidebar.text(st.session_state[variable])


display_info = [
    ("File choice:", "file_choice"),
    ("ATE Treatment:", "ate_treatment"),
    ("ATE Outcome:", "ate_outcome"),
    ("ATE Value:", "ate"),
    ("Exploration Score:", "exploration_score"),
    ("Suggested Next Exploration:", "next_exploration"),
]
for header_text, variable in display_info:
    display_current_info(header_text, variable)


if "logos_ui" not in st.session_state:
    st.session_state["logos_ui"] = LOGosUI()
logos_ui = st.session_state["logos_ui"]


###### Content #####
st.subheader("Choosing and previewing the log file")
file_choice_left, file_choice_right = st.columns(2)


with file_choice_left:
    logos_ui.prompt_select_file()

with file_choice_right:
    if "is_file_chosen" in st.session_state and st.session_state["is_file_chosen"]:
        logos_ui.show_log_file()

if "is_file_chosen" in st.session_state and st.session_state["is_file_chosen"]:
    st.markdown('<hr style="border:1px solid lightgray">', unsafe_allow_html=True)
    st.subheader("Deriving the Parsed Table")

parsing_left, parsing_right = st.columns(2)

with parsing_left:
    if "is_file_chosen" in st.session_state and st.session_state["is_file_chosen"]:
        logos_ui.parse()

with parsing_right:
    if "is_parsed" in st.session_state and st.session_state["is_parsed"]:
        logos_ui.show_parsed()

if "is_parsed" in st.session_state and st.session_state["is_parsed"]:
    logos_ui.separate()

    st.markdown('<hr style="border:1px solid lightgray">', unsafe_allow_html=True)
    st.subheader("Deriving the Prepared Table")
    logos_ui.set_causal_unit()


prepare_left, prepare_right = st.columns(2)

with prepare_left:
    if (
        "is_set_causal_unit" in st.session_state
        and st.session_state["is_set_causal_unit"]
    ):
        logos_ui.prepare()

with prepare_right:
    if "is_prepared" in st.session_state and st.session_state["is_prepared"]:
        logos_ui.show_prepared_table()

if "is_prepared" in st.session_state and st.session_state["is_prepared"]:
    st.markdown(
        "### You are now ready to proceed to the `Exploration-based Causal Discovery` Tab!"
    )
