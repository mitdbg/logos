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

st.title("Exploration-based Causal Discovery")


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


if "is_prepared" in st.session_state and st.session_state["is_prepared"]:
    st.markdown(
        "You have a collection of actions at your disposal, each of which is given a horizontal section below:"
    )
    st.markdown('<hr style="border:1px solid lightgray">', unsafe_allow_html=True)

    # Row about exploration
    st.subheader("Rank candidate causes")
    col_1, col_2 = st.columns([0.3, 0.7])
    with col_1:
        logos_ui.prompt_explore()
    with col_2:
        with st.expander("Candidate Cause(s)", expanded=True):
            if "causes_dataframe" in st.session_state:
                st.dataframe(st.session_state["causes_dataframe"])

    st.markdown('<hr style="border:1px solid lightgray">', unsafe_allow_html=True)

    # Row about inspection
    st.subheader("Inspect a prepared variable")
    col_1, col_2 = st.columns([0.3, 0.7])
    with col_1:
        logos_ui.prompt_inspect()

    with col_2:
        with st.expander("Inspection results", expanded=True):
            if "base_var_info_df" in st.session_state:
                st.markdown(
                    "Information about this variable's base variable from the log:"
                )
                st.dataframe(st.session_state["base_var_info_df"])
            if "template_info_df" in st.session_state:
                st.markdown("Information about this variable's template from the log:")
                st.dataframe(st.session_state["template_info_df"])
            if "prepared_log_info_df" in st.session_state:
                st.markdown(
                    "Information about this variable's values in the prepaerd table:"
                )
                st.dataframe(st.session_state["prepared_log_info_df"])

    st.markdown('<hr style="border:1px solid lightgray">', unsafe_allow_html=True)

    # Row about deciding on edges
    st.subheader("Decide whether to include an edge to the causal graph")
    col_1, col_2 = st.columns([0.3, 0.7])
    with col_1:
        logos_ui.prompt_decide()
    with col_2:
        with st.expander("Causal graph", expanded=True):
            if "graph" in st.session_state:
                components.v1.html(
                    GraphRenderer.graph_string_to_html(
                        st.session_state["graph"]
                    )._repr_html_(),
                    height=500,
                )

    st.markdown('<hr style="border:1px solid lightgray">', unsafe_allow_html=True)

    # Row about calculating ATE
    st.subheader("Specify the ATE of interest")
    col_1, col_2 = st.columns([0.3, 0.7])
    with col_1:
        logos_ui.prompt_ate()

else:
    st.markdown("### Please complete the `Data Transformation` tab first!")
