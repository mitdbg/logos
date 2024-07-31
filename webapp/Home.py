import streamlit as st
import sys
sys.path.append('../')
from utils import logo, hide_default, refresh_page, reset_session_button, display_current_info
from streamlit_extras.app_logo import add_logo
from src.sawmill.sawmill import Sawmill
from webapp.sawmillUI import SawmillUI
import os
from io import StringIO
import pandas as pd

st.set_page_config(
    page_title='Home',
    page_icon='🏠',
    layout='wide'
)

st.title('Sawmill:  From Logs to Causal Diagnosis of Large Systems')

add_logo("images/dsg-logo.png")


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


if "sawmill_ui" not in st.session_state:
    st.session_state["sawmill_ui"] = SawmillUI()
sawmill_ui = st.session_state["sawmill_ui"]

st.markdown('Welcome to our demo of Sawmill, the first system that can take users from a log to a principled causal understanding of their system!')



st.subheader("Failures are common in large systems, but the theory of causality can help us respond to them effectively")

st.markdown('- Diagnosing failures in production is notoriously challenging: **not enough time** for testing, formal verification or simulation.')
st.markdown('- Operators have to **work backward** from **observational data** collected from the system at the time of failure.')
st.markdown('- Operations teams want to go **beyond a diagnosis - they want to repair** the system by alerting the appropriate engineering team.')
st.markdown('- Whenever there are multiple ways to fix a problem, they would like to identify the **most efficient** way to utilize engineering time and effort.')
st.markdown('- The theory of causality can help **quantify and compare the impact** of different interventions into the system, helping operators.')

st.subheader('However, there is an important mismatch at play!')
st.markdown('Operators often have access to logs...')

with open('../datasets_raw/tpc-ds/parameter_sweep_1_filtered.log', "r") as log_file:
    log_lines = log_file.readlines(15000)
    st.dataframe([l.strip() for l in log_lines[:60]], use_container_width=True)



st.markdown('...but causal inference needs other inputs!')

left_column, right_column = st.columns(2)

with left_column:
    st.markdown('##### Tabular data per Causal Unit')
    df = pd.read_pickle('../datasets/tpc-ds/parameter_sweep_1_filtered.log_prepared_log_sessionID_None.pkl')
    df_var = pd.read_pickle('../datasets/tpc-ds/parameter_sweep_1_filtered.log_prepared_variables_sessionID_None.pkl')
    df.columns = df_var['Tag'].values
    st.write(df.head(15))

with right_column:
    st.markdown('##### Causal Model')
    st.image('../evaluation/tpc-ds/tpc-ds-graph.png')


st.subheader('Sawmill bridges the gap and makes causal inference based on logs possible')
st.markdown('Switch to the `Data transformation` tab to get started.')