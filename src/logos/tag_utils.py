from enum import IntEnum
from typing import Optional

import pandas as pd
from openai import OpenAI

from src.logos.printer import Printer
from src.logos.variable_name.parsed_variable_name import ParsedVariableName


class TagOrigin(IntEnum):
    PRECEDING: int = 0
    """Indicates that the tag was derived from the preceding tokens in the corresponding template."""

    GPT_3POINT5_TURBO: int = 1
    """Indicates that the tag was derived using gpt-3.5-turbo."""

    GPT_4: int = 2
    """Indicates that the tag was derived using gpt-4."""

    NAME: int = 3
    """Indicates that the tag was derived from the name of the variable."""

    REGEX_VARIABLE: int = 4
    """Indicates that the tag was derived from the name of the variable because the name was given by the user."""


class TagUtils:
    """
    A class for managing tags of parsed and prepared variables.
    """

    @staticmethod
    def check_columns(df: pd.DataFrame, columns: list) -> None:
        """
        Check that the specified columns exist in the dataframe.

        Parameters:
            df: The dataframe to be checked.
            columns: The columns to be checked.

        Raises:
            ValueError: If any of the columns are not present in the dataframe.
        """
        if not set(columns).issubset(set(df.columns)):
            raise ValueError(f"Columns {columns} are not all present in the dataframe.")

    @staticmethod
    def check_fields(series: pd.Series, fields: list) -> None:
        """
        Check that the specified fields exist in the specified series.

        Parameters:
            series: The series to be checked.
            fields: The fields to be checked.

        Raises:
            ValueError: If any of the fields are not present in the series.
        """
        if not set(fields).issubset(set(series.index)):
            raise ValueError(f"Fields {fields} are not all present in the series.")

    @staticmethod
    def waterfall_tag(
        templates_df: pd.DataFrame,
        variable_row: pd.Series,
        banned_values: Optional[list[str]] = None,
    ) -> tuple[str, TagOrigin]:
        """
        Apply each of the tagging methods in turn, in order of increasing cost, until a tag is found
        that is not included in the banned values. In partidular, apply `preceding_tokens_tag` first,
        then `gpt_tag` with the GPT-3.5 model, and finally `gpt_tag` with the GPT-4 model. If none of
        these methods succeeds, return the name of the variable as the tag.

        Parameters:
            templates_df: The dataframe containing information about the log templates.
            variable_row: The row of the dataframe containing information about the parsed variable.
            banned_values: A list of values that should not be used as tags.

        Returns:
            A tuple containing (i) the tag for the parsed variable, and (ii) the origin of the tag.
        """
        name = variable_row["Name"]
        if variable_row["From regex"]:
            return (name, TagOrigin.REGEX_VARIABLE)

        # Try to derive a tag from the preceding tokens in the corresponding template
        tag, origin = TagUtils.preceding_tokens_tag(variable_row, banned_values)
        if tag != name:
            return (tag, origin)

        # Try to derive a tag using GPT-3.5
        try:
            tag = TagUtils.gpt_tag(
                templates_df, variable_row, "gpt-3.5-turbo", banned_values
            )
            if tag != name:
                return (tag, TagOrigin.GPT_3POINT5_TURBO)
        except Exception as e:
            print(f"Exception {e} came up while tagging {name} with GPT-3.5.")
            pass

        # Try to derive a tag using GPT-4
        try:
            tag = TagUtils.gpt_tag(templates_df, variable_row, "gpt-4", banned_values)
            if tag != name:
                return (tag, TagOrigin.GPT_4)
        except Exception as e:
            print(f"Exception {e} came up while tagging {name} with GPT-4.")
            pass

        return (name, TagOrigin.NAME)

    @staticmethod
    def preceding_tokens_tag(
        variable_row: pd.Series, banned_values: Optional[list[str]] = None
    ) -> tuple[str, TagOrigin]:
        """
        Try to derive a tag for a parsed variable name based on the preceding tokens in the corresponding template.

        Parameters:
            variable_row: The row of the dataframe containing information about the parsed variable.
            banned_values: A list of values that should not be used as tags.

        Returns:
            A tuple containing (i) the tag for the parsed variable, and (ii) the origin of the tag.
        """

        TagUtils.check_fields(
            variable_row, ["Preceding 3 tokens", "Name", "From regex"]
        )
        name = variable_row["Name"]
        if variable_row["From regex"]:
            return name, TagOrigin.REGEX_VARIABLE

        pr = variable_row["Preceding 3 tokens"]
        tag = name
        origin = TagOrigin.NAME
        if len(pr) >= 2 and (pr[-1] in ":=") and (pr[-2][0] != "<"):
            tag = pr[-2]
            origin = TagOrigin.PRECEDING
        elif (
            len(pr) == 3
            and (pr[2] in """"'""")
            and (pr[1] in ":=")
            and (pr[0][0] != "<")
        ):
            tag = pr[0]
            origin = TagOrigin.PRECEDING

        # Double-check that the tag is not in the banned values
        if banned_values is not None and tag in banned_values:
            return name, TagOrigin.NAME

        return tag, origin

    @staticmethod
    def gpt_tag(
        templates_df: pd.DataFrame,
        variable_row: pd.Series,
        model: str = "gpt-3.5-turbo",
        banned_values: Optional[list[str]] = None,
    ) -> str:
        """
        Use GPT to derive a tag the variable described in `variable_row`,
        using information about the corresponding log template, retrieved from `templates_df`.

        Parameters:
            templates_df: The dataframe containing information about the log templates.
            variable_row: The row of the dataframe containing information about the parsed variable.
            model: The GPT model to use.
            banned_values: A list of values that should not be used as tags.

        Returns:
            The GPT-generated tag for the parsed variable name.
        """

        TagUtils.check_fields(variable_row, ["Name", "Examples"])
        TagUtils.check_columns(templates_df, ["TemplateId", "TemplateExample"])

        template_id = ParsedVariableName(variable_row["Name"]).template_id()
        idx = ParsedVariableName(variable_row["Name"]).index()
        assert idx is not None

        line = templates_df[templates_df["TemplateId"] == template_id][
            "TemplateExample"
        ].values[0]
        line_toks = line.split()

        # Define the messages to send to the model
        messages = [
            {
                "role": "system",
                "content": "You are a backend engineer that knows all about the logging infrastructure of a distributed system.",
            },
            {
                "role": "user",
                "content": f"""Generate a tag for the variable that takes the value {line_toks[idx]} """
                f"""in the following log line:\n {line}\n"""
                f"""Here are the 3 tokens that precede the variable: [{', '.join(line_toks[max(idx-3, 0):idx])} ]\n"""
                f"""Here are some more example values for this variable: [{', '.join(variable_row['Examples'])} ]\n"""
                # f"""Make sure the tag matches none of the following values: [{', '.join(banned_values) if banned_values is not None else ''} ]\n"""
                """Return only the tag as a single word, possibly including underscores. DO NOT EVER REPLY WITH MORE THAN ONE WORD.\n""",
            },
        ]

        client = OpenAI()

        tag = (
            client.chat.completions.create(model=model, messages=messages)  # type: ignore
            .choices[0]
            .message.content
        )
        assert type(tag) == str
        tag_length = len(tag.split())
        if tag_length > 1:
            # GPT didn't listen to us and returned a phrase describing the tag.
            # Extract the word between the second-last and last occurrence of double quotes.
            tag = tag.split('"')[-2]

        with open("gpt_log.txt", "a+") as f:
            f.write("----------------------------------\n")
            f.write(f"Variable name: {variable_row['Name']}\n\n")
            f.write(f"Model used: {model}\n\n")
            f.write(f"Messages sent to the model:\n{messages}\n\n")
            f.write(f"Tag generated by the model:\n{tag}\n\n")
            f.flush()

        # Double-check that the tag is not in the banned values
        if banned_values is not None and tag in banned_values:
            with open("gpt_log.txt", "a+") as f:
                f.write("That tag is banned, returning name.\n")
            return variable_row["Name"]

        return tag

    @staticmethod
    def deduplicate_tags(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the tags in df are unique, by making the tag column of any row
        with a seen-before tag equal to the name column of that row.

        Parameters:
            df: The dataframe to be deduplicated.

        Returns:
            The deduplicated dataframe.
        """

        TagUtils.check_columns(df, ["Name", "Tag", "TagOrigin"])
        seen_tags = set()
        for i, row in df.iterrows():
            if row["Tag"] in seen_tags:
                df.loc[i, "Tag"] = row["Name"]
                df.loc[i, "TagOrigin"] = TagOrigin.NAME
            else:
                seen_tags.add(row["Tag"])

    @staticmethod
    def set_tag(df: pd.DataFrame, name: str, tag: str, info: str = "") -> None:
        """
        Tag a parsed or prepared variable for easier access.

        Parameters:
            df: The dataframe containing the parsed or prepared variables.
            name: The name of the parsed or prepared variable.
            tag: The tag to be set.
            info: A string describing the type of variable being tagged (parsed or prepared).

        Raises:
            ValueError: If the name is not the name of a parsed or prepared variable.
        """
        TagUtils.check_columns(df, ["Name", "Tag"])
        if name in df["Name"].values:
            df.loc[df["Name"] == name, "Tag"] = tag
            Printer.printv(f"Variable {name} tagged as {tag}")
        else:
            raise ValueError(f"{name} is not the name of a {info} variable.")

    @staticmethod
    def get_tag(df: pd.DataFrame, name: str, info: str = "") -> str:
        """
        Retrieve the tag of a parsed or prepared variable.

        Parameters:
            df: The dataframe containing the parsed or prepared variables.
            name: The name of the parsed or prepared variable.
            info: A string describing the type of variable being tagged (parsed or prepared).

        Raises:
            ValueError: If the name is not the name of a parsed or prepared variable.
        """

        TagUtils.check_columns(df, ["Name", "Tag"])
        if name in df["Name"].values:
            return df.loc[df["Name"] == name, "Tag"].values[0]
        else:
            raise ValueError(f"{name} is not the name of a {info} variable.")

    @staticmethod
    def name_of(df: pd.DataFrame, name_or_tag: str, info: str = "") -> str:
        """
        Determine the name of a parsed or prepared variable, given either itself or its tag.

        Parameters:
            df: The dataframe containing the parsed or prepared variables.
            name_or_tag: The name or tag of the parsed or prepared variable.
            info: A string describing the type of variable in question (parsed or prepared).

        Returns:
            The name of the parsed or prepared variable.
        """

        TagUtils.check_columns(df, ["Name", "Tag"])
        name_or_tag = name_or_tag.strip()
        if name_or_tag in df["Name"].values:
            return name_or_tag
        elif name_or_tag in df["Tag"].values:
            return df.loc[df["Tag"] == name_or_tag, "Name"].values[0]
        else:
            raise ValueError(
                f"{name_or_tag} is not the name or tag of a {info} variable."
            )

    @staticmethod
    def tag_of(
        df: pd.DataFrame, name_or_tag: Optional[str], info: str = ""
    ) -> Optional[str]:
        """
        Determine the tag of a parsed or prepared variable, given either itself or its name.
        Retuirn None if the variable is None.

        Parameters:
            df: The dataframe containing the parsed or prepared variables.
            name_or_tag: The name or tag of the parsed or prepared variable.
            info: A string describing the type of variable in question (parsed or prepared).

        Returns:
            The tag of the parsed or prepared variable.
        """

        if name_or_tag is None:
            return None

        TagUtils.check_columns(df, ["Name", "Tag"])
        name_or_tag = name_or_tag.strip()
        if name_or_tag in df["Tag"].values:
            return name_or_tag
        elif name_or_tag in df["Name"].values:
            return df.loc[df["Name"] == name_or_tag, "Tag"].values[0]
        else:
            raise ValueError(
                f"{name_or_tag} is not the name or tag of a {info} variable."
            )
