from __future__ import annotations
import streamlit as st
from streamlit_extras.no_default_selectbox import selectbox
from src.sawmill.ate import ATEChallengerMethod
from src.sawmill.sawmill import Sawmill
from src.sawmill.tag_utils import TagUtils
from src.sawmill.variable_name.prepared_variable_name import PreparedVariableName
import pandas as pd
from functools import reduce
import re


DATASET_INFO = {
    "PostgreSQL": {
        "path": "../datasets_raw/tpc-ds/parameter_sweep_1_filtered.log",
        "workdir": "../datasets/tpc-ds",
        "message_prefix": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3}",
        "regex_dict": {
            "Date": r"\d{4}-\d{2}-\d{2}",
            "Time": r"\d{2}:\d{2}:\d{2}\.\d{3}(?= EST \[ )",
            "sessionID": r"(?<=EST \[ )\S+\.\S+",
            "tID": r"3/\d+(?= ] )",
        },
        "custom_imp": {},
    },
    "Proprietary": {
        "path": "../datasets_raw/proprietary_logs/proprietary_1000users_500faulty_100pctfailfaulty_10pctfailnormal.log",
        "workdir": "../datasets/proprietary_logs/proprietary_eval",
        "message_prefix": r".*",
        "regex_dict": {
            "Timestamp": r"\d{4}\-\d{2}\-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
            "UnixTimestamp": r"16\d{11}(?=\sINFO|\sWARN|\sERROR)",
            "User": r"user_\d+",
        },
        "custom_imp": {},
    },
    "XYZ": {
        "path": "../datasets_raw/xyz_extended/log_2024-01-08_20:52:58.log",
        "workdir": "../datasets/xyz_extended",
        "message_prefix": r".*",
        "regex_dict": {
            "timestamp": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
            "machine": r"machine_\d+",
        },
        "custom_imp": {},
    },
    "PostgreSQL-old": {
        "path": "../datasets_raw/tpc-ds/work_mem_2_256kB_2_128kB_parallel_1_2.log",
        "workdir": "../datasets/tpc-ds",
        "message_prefix": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3}",
        "regex_dict": {
            "DateTime": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}(?= EDT \[ )",
            "sessionID": r"(?<=EDT \[ )\S+\.\S+",
            "tID": r"3/\d+(?= ] )",
        },
        "custom_imp": {},
    },
    "XYZ-old": {
        "path": "../datasets_raw/xyzw_logs/log_2023-03-14_20:55:49.log",
        "workdir": "../datasets/xyzw_logs/log_2023-03-14_20:55:49",
        "message_prefix": r".*",
        "regex_dict": {
            "DateTime": r"\d{4}\-\d{2}\-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
        },
        "custom_imp": {"z": "ffill_imp"},
    },
}


class VarTag:
    """
    Holds variable literal name and variable tag
    """

    def __init__(self, name: str, tag: str):
        self._name = name
        self._tag = tag

    @property
    def name(self):
        return self._name

    @property
    def tag(self):
        return self._tag

    def __str__(self) -> str:
        return self.tag

    def __eq__(self, other: VarTag) -> bool:
        if isinstance(other, VarTag):
            return self.name == other.name and self.tag == other.tag

        return False


class SawmillUI:
    def __init__(self):
        pd_options = [
            ("display.max_rows", None),
            ("display.max_columns", None),
            ("expand_frame_repr", False),
            ("display.max_colwidth", None),
        ]
        for option, config in pd_options:
            pd.set_option(option, config)

    def prompt_select_file(self):
        def on_click_select_file():
            pass

        with st.form("select_file_form"):
            st.subheader("Choose a log file to analyze:")

            file_selection = st.selectbox(
                "Select a log file:", list(DATASET_INFO.keys()), key="file_choice"
            )

            submitted = st.form_submit_button(
                "Select file", on_click=on_click_select_file
            )
            if submitted:
                with st.spinner("Selecting file..."):
                    self.sawmill = Sawmill(
                        filename=DATASET_INFO[file_selection]["path"],
                        workdir=DATASET_INFO[file_selection]["workdir"],
                    )
                    st.session_state["is_file_chosen"] = True

    @st.cache_data
    def find_logs(_self, variable, regex):
        var_name = TagUtils.name_of(
            _self.sawmill.prepared_variables, variable, "prepared"
        )
        template_id = PreparedVariableName(var_name).template_id()

        ## FIXME
        df = _self.sawmill.parsed_log
        tp = _self.sawmill.parsed_templates
        df_filtered = df[df[var_name].notnull()]

        logs = []
        for _, row in df_filtered.iterrows():
            template = row["TemplateId"]

            custom_values = {}
            for regex_name in regex:
                custom_values[regex_name] = str(row[regex_name])
            var_value = str(row[var_name])

            log = tp.loc[tp["TemplateId"] == template, "TemplateText"].iloc[0]

            for regex_name in custom_values:
                log = log.replace(regex[regex_name], custom_values[regex_name])

            logs.append(log.replace("<*>", var_value))

        return logs

    def clear_next(self, variables):
        for var in variables:
            if var in st.session_state:
                del st.session_state[var]

    def show_log_file(self):
        """
        Display the first few lines of the chosen log file.
        """

        def on_click():
            pass

        with st.form("log_file_form"):
            st.subheader("Log File Preview")

            background_text = """
                Let's take a quick peek at the log file we are interested in.
            """
            st.markdown(background_text)

            submitted = st.form_submit_button("Show Log File", on_click=on_click)
            if submitted:
                with st.spinner("Showing log file..."):
                    # Open the chosen log file and display the first 15 lines
                    with open(
                        DATASET_INFO[st.session_state["file_choice"]]["path"], "r"
                    ) as log_file:
                        log_lines = log_file.readlines(3000)
                        st.write([l.strip() for l in log_lines[:15]])

    def parse(self):
        """
        Allow users to view and change arguments before parsing the log file.
        """
        with st.form("parse_form"):
            FIELD_REGEX = r"[A-Za-z ?]+"
            DEFAULT_REGEX_DICT = DATASET_INFO[st.session_state["file_choice"]][
                "regex_dict"
            ]

            is_parsed = (
                st.session_state["is_parsed"]
                if "is_parsed" in st.session_state
                else False
            )

            def is_valid_submission(regex_dict: dict) -> bool:
                def is_valid_regex_dict(regex_dict: dict) -> bool:
                    if not regex_dict:
                        return False

                    for regex_field, regex_pattern in regex_dict.items():
                        if (
                            not re.fullmatch(FIELD_REGEX, regex_field)
                            or not regex_pattern
                        ):
                            return False

                    return True

                return is_valid_regex_dict(regex_dict)

            def on_click():
                def construct_regex_dict() -> dict[str, str]:
                    data_editor = st.session_state["regex_dict"]
                    edited_rows, added_rows, deleted_rows = (
                        data_editor["edited_rows"],
                        data_editor["added_rows"],
                        data_editor["deleted_rows"],
                    )
                    regex_list = [
                        {"_index": key, "value": value}
                        for key, value in DEFAULT_REGEX_DICT.items()
                    ]

                    if len(deleted_rows) != 0:
                        for idx in deleted_rows:
                            del regex_list[int(idx)]

                    if len(edited_rows) != 0:
                        for idx, new_mapping in edited_rows.items():
                            regex_list[int(idx)]["_index"] = new_mapping["_index"]

                            if "value" in new_mapping:
                                regex_list[int(idx)]["value"] = new_mapping["value"]
                            else:
                                regex_list[int(idx)]["value"] = None

                    if len(added_rows) != 0:
                        for new_mapping in added_rows:
                            if "value" not in new_mapping:
                                new_mapping["value"] = None
                            regex_list.append(new_mapping)

                    regex_dict = {
                        entry["_index"]: entry["value"] for entry in regex_list
                    }
                    return regex_dict

                regex_dict = construct_regex_dict()
                st.markdown(f"""regex_dict: {regex_dict}""")

            st.subheader("Parsing and Tagging")

            background_text = """
                We can parse the log into the *parsed table* using the [Drain](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf) algorithm.
                Sawmill then assigns a human-understandable *tag* to each parsed variable.
            """
            st.markdown(background_text)

            force_parse = st.checkbox(label="Force re-calculation")

            prefix_left_col, prefix_right_col = st.columns(2)
            with prefix_left_col:
                message_prefix = st.text_input(
                    label="Message Prefix",
                    value=DATASET_INFO[st.session_state["file_choice"]][
                        "message_prefix"
                    ],
                    key="message_prefix",
                    help="""
                    Log messages can often span multiple log lines. In order to treat such multi-line messages correctly,
                    we can specify a regular expression to capture the prefix of each log message. Lines that do not
                    match the regular expression will be treated as a continuation of the previous log message.
                """,
                )
                sim_thresh = st.slider(
                    label="Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=(
                        0.65
                        if not st.session_state["file_choice"] == "Proprietary"
                        else 0.9
                    ),
                    help="The similarity threshold used by Drain. The default similarity threshold is `0.65`",
                )
                depth = st.number_input(
                    label="Depth",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="The fixed depth of the parse tree used by Drain. The default depth is `5`.",
                )
            with prefix_right_col:
                st.text("Custom Regex")
                regex_tool_tip = """
                    You can first extract named variables from each log line using regular expressions,
                    provided here:
                """
                st.markdown(regex_tool_tip)
                regex_dict = st.data_editor(
                    data=DEFAULT_REGEX_DICT,
                    num_rows="dynamic",
                    key="regex_dict",
                )

            submitted = st.form_submit_button(
                label="Parse", on_click=on_click, disabled=is_parsed
            )
            if submitted:
                if not is_valid_submission(regex_dict):
                    st.error(
                        """
                            Regex dictionary must be non-empty and each regex pattern must be non-empty
                        """
                    )
                else:
                    with st.spinner("Parsing log file..."):
                        self.sawmill.parse(
                            regex_dict=regex_dict,
                            sim_thresh=sim_thresh,
                            depth=depth,
                            force=force_parse,
                        )
                        st.success(
                            f"""
                                Successfully parsed log file with  
                                Message Prefix: {message_prefix}         
                                Custom Regex: ```{regex_dict}```    
                                Simulation Threshold: {sim_thresh}   
                                Depth: {depth}
                            """
                        )
                        st.session_state["is_parsed"] = True

    def show_parsed(self):
        """
        Display the parsed table, variables and templates.
        """

        def on_click():
            pass

        with st.form("show_parsed_table_form"):
            st.subheader("Inspect Results of Parsing")
            background_text = "Sawmill parsed the original log into a preliminary tabular representation, the *parsed table*, by separating the *parsed templates* from the *parsed variables* they contain. You can inspect each of these products here:"
            st.markdown(background_text)

            submitted_2 = st.form_submit_button(
                label="Show Parsed Templates", on_click=on_click
            )
            if submitted_2:
                df2 = self.sawmill.parsed_templates.copy().astype(str)
                st.dataframe(df2)

            submitted_3 = st.form_submit_button(
                label="Show Parsed Variables", on_click=on_click
            )
            if submitted_3:
                df3 = self.sawmill.parsed_variables.copy().astype(str)
                st.dataframe(df3)

            submitted_1 = st.form_submit_button(
                label="Show Parsed Table", on_click=on_click
            )
            if submitted_1:
                df1 = self.sawmill.parsed_log.copy().astype(str)
                st.dataframe(df1)

    def separate(self):
        """
        Separate log variables that have erroneously been mapped to the same parsed variable.
        """

        def on_click():
            pass

        with st.form("separate_form"):
            st.subheader("Correct Parsing")
            background_text = "If a part of the template has been erroneously recognized as a variable, you can correct it here."
            st.markdown(background_text)

            to_separate = selectbox(
                label="Erroneous variable",
                options=self.sawmill.parsed_variables.Tag,
                format_func=lambda x: str(x),
                no_selection_label="",
                key="to_separate",
            )

            submitted = st.form_submit_button(
                label="Include in Template", on_click=on_click
            )
            if submitted:
                with st.spinner("Modifying Parsing..."):
                    self.sawmill.include_in_template(to_separate)

                    st.success(
                        f"""
                            Successfully included variable {to_separate} in template.  

                        """
                    )

    def set_causal_unit(self):
        """
        Allow users to view and change arguments before setting the causal unit.
        """
        with st.form("set_causal_unit_form"):
            is_set_causal_unit = (
                st.session_state["is_set_causal_unit"]
                if "is_set_causal_unit" in st.session_state
                else False
            )

            def on_click():
                st.session_state["is_set_causal_unit"] = True

            st.subheader("Define Causal Unit")
            background_text = """
                To continue our analysis, we need to structure and complete the log information around *causal units*.
                If we think of causality in general, the causal units could be patients in a medical context,
                or individuals in an economic/social study.   
                Causal units are defined by one of the available attributes. 
                Pick one of the variables below to define causal units.   
            """
            st.markdown(background_text)

            causal_unit_options = self.sawmill.parsed_variables.Tag
            causal_unit_option = selectbox(
                label="Select the parsed variable that defines each causal unit",
                options=causal_unit_options,
                format_func=lambda x: str(x),
                no_selection_label="",
            )

            left_col, right_col = st.columns([0.7, 0.3])
            with left_col:
                specify_num_units = st.checkbox(
                    label="The chosen variable is numerical and I want to bin the values into a fixed number of causal units.",
                    value=False,
                )
            with right_col:
                num_units = st.number_input(
                    label="Number of Causal Units",
                    min_value=1,
                    key="num_units",
                )

            submitted = st.form_submit_button(
                label="Set Causal Unit", on_click=on_click, disabled=is_set_causal_unit
            )
            if submitted:
                with st.spinner("Setting causal unit..."):
                    causal_unit_text = str(causal_unit_option)

                    self.sawmill.set_causal_unit(
                        causal_unit_option,
                        num_units=num_units if specify_num_units else None,
                    )
                    st.success(
                        f"""
                            Successfully set causal unit with   
                            Causal unit: {causal_unit_text} """
                        + (
                            f""" 
                            Number of causal units: {num_units}
                        """
                            if specify_num_units
                            else ""
                        )
                    )

    def prepare(self):
        """
        Allow users to view and change arguments before preparing the prepared table.
        """
        with st.form("prepare_form"):
            is_prepared = (
                st.session_state["is_prepared"]
                if "is_prepared" in st.session_state
                else False
            )

            def is_valid_submission() -> bool:
                pass

            def on_click():
                st.session_state["is_prepared"] = True

            st.subheader("Prepare for Analysis")
            background_text = """  
                Given the causal unit, we can then ask `Sawmill` to prepare the log for analysis by using `prepare()`.   
                Preparing the log involves two distinct task: `Aggregation` and `Imputation`.   
                After aggregating and imputing, we drop any causal units that still have missing values.
            """
            st.markdown(background_text)

            # st.text('Aggregation Functions')
            # agg_left_col, agg_right_col = st.columns(2)
            # with agg_left_col:
            #    agg_dict = st.data_editor(
            #        data=empty_agg_dict,
            #        num_rows='dynamic',
            #        key='agg_dict',
            #        column_config={
            #             "variable": st.column_config.Column(
            #                 "Prepared Variable",
            #                 help="Name or tag of the prepared variable",
            #                 width="medium",
            #                 required=True,
            #             ),
            #             "agg_func_list": st.column_config.Column(
            #                 "Aggregation Functions",
            #                 help="List of aggregation fucntions to use",
            #                 width="medium",
            #                 required=True,
            #             ),
            #         },
            #    )
            # with agg_right_col:
            #     agg_tool_tip = """
            #         Based on our choice of causal unit, there might be variables that take a multitude of values on different log lines
            #         associated with the same causal unit.
            #         For example, had we chosen a `10 ms` window as our causal unit, there would have been 10 lines reporting values of some variable `x`.
            #         From this multitude of values, a fixed set of values must be derived (e.g. we could always keep the mean, or the last value seen).
            #         By default, `Sawmill` will generate
            #             - the `min`, `max`, and `mean` for numerical variable
            #             - the `mode`, `first` and `last` for categorical variables
            #         We can instead specify a different comma-separated list of functions, per variable.
            #     """
            #     st.markdown(agg_tool_tip)

            imp_left_col, imp_right_col = st.columns(2)
            imp_dict = DATASET_INFO[st.session_state["file_choice"]]["custom_imp"]
            with imp_left_col:
                force_prepare = st.checkbox(label="Force re-calculation")

                count_occurences = st.checkbox(
                    label="Count Occurences of each Log Template", value=False
                )
            with imp_right_col:
                st.text("Custom Imputation")
                imp_tool_tip = """
                    There might be variables that are never observed within some causal unit. In such cases, `Sawmill` can impute missing values based on different strategies.
                """
                st.markdown(imp_tool_tip)
                imp_dict = st.data_editor(
                    data=imp_dict,
                    num_rows="dynamic",
                    key="agg_dict",
                    column_config={
                        "variable": st.column_config.Column(
                            "Prepared Variable",
                            help="Name or tag of the prepared variable",
                            width="medium",
                            required=True,
                        ),
                        "imp_func_list": st.column_config.Column(
                            "Imputation Function",
                            help="Imputation function to use",
                            width="medium",
                            required=True,
                        ),
                    },
                )

            submitted = st.form_submit_button(
                label="Prepare", on_click=on_click, disabled=is_prepared
            )
            if submitted:
                with st.spinner("Preparing log for analysis..."):
                    self.sawmill.prepare(
                        count_occurences=count_occurences,
                        force=force_prepare,
                        custom_imp=imp_dict,
                        reject_prunable_edges=False,
                    )
                    st.success(
                        f"""
                            Successfully prepare log for analysis with   """
                        # Aggregation: {agg_option}
                        # Imputation: {imp_option}
                        f""" Count Occurences: {count_occurences}
                        """
                    )

    def show_prepared_table(self):
        """
        Display the prepared table.
        """

        def on_click():
            pass

        with st.form("show_prepared_table_form"):
            st.subheader("Prepared Table")
            background_text = "Sawmill transformed the parsed table into the *prepared table* based on the selected causal units. Here is a sample:"
            st.markdown(background_text)

            submitted = st.form_submit_button(
                label="Show Prepared Table", on_click=on_click
            )
            if submitted:
                st.write(self.sawmill.prepared_log.head(15))

    def prompt_inspect(self):
        """
        Prompt the user to inspect a particular prepared variable.
        """

        def on_click_inspect():
            pass

        with st.form("inspect_form"):
            st.subheader("Choose a variable to inspect:")
            inspect_var = st.selectbox(
                "Select a variable:",
                self.sawmill.prepared_variable_tags,
                format_func=lambda x: str(x),
                key="inspect_var",
            )

            submitted = st.form_submit_button("Inspect", on_click=on_click_inspect)
            if submitted:
                with st.spinner(f"Inspecting {inspect_var}..."):
                    (
                        st.session_state["base_var_info_df"],
                        st.session_state["template_info_df"],
                        st.session_state["prepared_log_info_df"],
                    ) = self.sawmill.inspect(inspect_var)
                    self.clear_next(["inspect_var"])

    def prompt_explore(self):
        """
        Prompt the user to select a specific outcome variable from a list of all captured.
        """

        def on_click():
            self.clear_next(
                [
                    "current_cause",
                    "source_node",
                    "destination_node",
                    "current_confounder",
                    "causes_plot",
                    "confounders_dataframe",
                    "confounders_plot",
                ]
            )
            st.session_state["current_outcome"] = str(
                st.session_state["selected_exploration_target"]
            )

        with st.form("outcome_form"):
            st.subheader("Choose a variable to explore candidate causes for:")
            self.outcome = st.selectbox(
                "Select a variable:",
                self.sawmill.prepared_variables.Tag,
                format_func=lambda x: str(x),
                key="selected_exploration_target",
            )

            submitted = st.form_submit_button(
                "Explore Candidate Causes", on_click=on_click
            )
            if submitted:
                with st.spinner("Finding candidate cause(s)..."):
                    self.candidate_causes, time = self.sawmill.explore_candidate_causes(
                        self.outcome
                    )
                    st.session_state["causes_dataframe"] = self.candidate_causes

                    if not self.candidate_causes.empty:
                        st.success("Candidate cause(s) found!")
                    else:
                        st.info("No candidate causes(s) found.")

    def prompt_ate(self):
        """
        Prompt the user to select a specific ATE to calculate
        """

        def on_click():
            st.session_state["ate_treatment"] = str(
                st.session_state["selected_treatment"]
            )
            st.session_state["ate_outcome"] = str(st.session_state["selected_outcome"])
            st.session_state["ate"] = self.sawmill.get_adjusted_ate(
                st.session_state["ate_treatment"], st.session_state["ate_outcome"]
            )

        with st.form("ate_form"):
            st.subheader("Choose the ATE you would like to calculate:")
            source_col, destination_col = st.columns(2)

            with source_col:
                selected_source = st.selectbox(
                    "Source node:",
                    self.sawmill.prepared_variable_tags,
                    format_func=lambda x: str(x),
                    key="selected_treatment",
                )
                st.form_submit_button("Calculate", on_click=on_click)

            with destination_col:
                selected_destination = st.selectbox(
                    "Destination node:",
                    self.sawmill.prepared_variable_tags,
                    format_func=lambda x: str(x),
                    key="selected_outcome",
                )

    def prompt_decide(self):
        """
        Prompts the user with options for candidate causes
        """

        def on_click_accept():
            (
                expl_score,
                next_exploration,
                st.session_state["graph"],
            ) = self.sawmill.accept(
                src=st.session_state["source_node"],
                dst=st.session_state["destination_node"],
                interactive=False,
            )

            st.session_state["exploration_score"] = expl_score
            st.session_state["next_exploration"] = TagUtils.tag_of(
                self.sawmill.prepared_variables, next_exploration, "prepared"
            )

            self.clear_next(["source_node", "destination_node"])

            if (
                "ate_treatment" in st.session_state
                and "ate_outcome" in st.session_state
            ):
                st.session_state["ate"] = self.sawmill.get_adjusted_ate(
                    st.session_state["ate_treatment"], st.session_state["ate_outcome"]
                )

        def on_click_reject():
            (
                expl_score,
                next_exploration,
                st.session_state["graph"],
            ) = self.sawmill.reject(
                src=st.session_state["source_node"],
                dst=st.session_state["destination_node"],
                interactive=False,
            )

            st.session_state["exploration_score"] = expl_score
            st.session_state["next_exploration"] = TagUtils.tag_of(
                self.sawmill.prepared_variables, next_exploration, "prepared"
            )

            self.clear_next(["source_node", "destination_node"])

            if (
                "ate_treatment" in st.session_state
                and "ate_outcome" in st.session_state
            ):
                st.session_state["ate"] = self.sawmill.get_adjusted_ate(
                    st.session_state["ate_treatment"], st.session_state["ate_outcome"]
                )

        def on_click_reject_undecided_outgoing():
            (
                expl_score,
                next_exploration,
                st.session_state["graph"],
            ) = self.sawmill.reject_undecided_outgoing(
                src=st.session_state["source_node"], interactive=False
            )

            st.session_state["exploration_score"] = expl_score
            st.session_state["next_exploration"] = TagUtils.tag_of(
                self.sawmill.prepared_variables, next_exploration, "prepared"
            )

            self.clear_next(["source_node", "destination_node"])

            if (
                "ate_treatment" in st.session_state
                and "ate_outcome" in st.session_state
            ):
                st.session_state["ate"] = self.sawmill.get_adjusted_ate(
                    st.session_state["ate_treatment"], st.session_state["ate_outcome"]
                )

        def on_click_reject_undecided_incoming():
            (
                expl_score,
                next_exploration,
                st.session_state["graph"],
            ) = self.sawmill.reject_undecided_incoming(
                dst=st.session_state["destination_node"], interactive=False
            )

            st.session_state["exploration_score"] = expl_score
            st.session_state["next_exploration"] = TagUtils.tag_of(
                self.sawmill.prepared_variables, next_exploration, "prepared"
            )

            self.clear_next(["source_node", "destination_node"])

            if (
                "ate_treatment" in st.session_state
                and "ate_outcome" in st.session_state
            ):
                st.session_state["ate"] = self.sawmill.get_adjusted_ate(
                    st.session_state["ate_treatment"], st.session_state["ate_outcome"]
                )

        # Present two fields side by side, for each of which there is a dropdown that the user can select from
        # The user can then click a button to accept the selected values
        # The user can also click a button to reject the selected values

        def on_click_call_eccs():
            # TODO: only enable this after he user has set the treatment and the outcome
            impactful_edits = self.sawmill.challenge_ate(
                st.session_state["ate_treatment"],
                st.session_state["ate_outcome"],
                method=st.session_state["impact_method"],
            )

            st.session_state["impactful_edits"] = impactful_edits

            self.clear_next(["impact_method"])

        with st.form("decide_edge_form"):
            st.subheader(
                "Choose the endpoints of the edge you would like to decide on:"
            )
            source_col, destination_col = st.columns(2)

            with source_col:
                selected_source = st.selectbox(
                    "Source node:",
                    self.sawmill.prepared_variable_tags,
                    format_func=lambda x: str(x),
                    key="source_node",
                )

            with destination_col:
                selected_destination = st.selectbox(
                    "Destination node:",
                    self.sawmill.prepared_variable_tags,
                    format_func=lambda x: str(x),
                    key="destination_node",
                )

            acc_col, rej_col, _ = st.columns([0.3, 0.3, 0.4])
            with acc_col:
                st.form_submit_button("Accept", on_click=on_click_accept)
            with rej_col:
                st.form_submit_button("Reject", on_click=on_click_reject)
            st.form_submit_button(
                "Reject Undecided Outgoing from Source",
                on_click=on_click_reject_undecided_outgoing,
            )
            st.form_submit_button(
                "Reject Undecided Incoming to Destination",
                on_click=on_click_reject_undecided_incoming,
            )
            impact_col_1, impact_col_2 = st.columns([0.6, 0.4])
            with impact_col_1:
                st.form_submit_button(
                    "Find most impactful edge edit(s)",
                    on_click=on_click_call_eccs,
                )
            with impact_col_2:
                selected_destination = st.selectbox(
                    "Method:",
                    ATEChallengerMethod.values(),
                    format_func=lambda x: str(x),
                    key="impact_method",
                )

            if "impactful_edits" in st.session_state:
                st.write(st.session_state["impactful_edits"])
