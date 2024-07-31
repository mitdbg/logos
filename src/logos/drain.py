"""
Inspired by the LogPAI implementation of the Drain algorithm for log parsing, 
available under the MIT license here:
[https://github.com/HelenGuohx/logbert/blob/main/logparser/Drain.py](https://github.com/HelenGuohx/logbert/blob/main/logparser/Drain.py)
"""

import re
import os
import pandas as pd
import hashlib
from datetime import datetime
from tqdm.auto import tqdm
from typing import Optional


class Cluster:
    """
    A cluster in the Drain parse tree.
    """

    def __init__(self, template: str = "", message_ids: list[int] = []):
        """
        Parameters:
            template : the template of log messages in this cluster
            message_ids : the list of log message IDs in this cluster
        """

        self.template = template
        self.message_ids = message_ids


class Node:
    """
    A node in the Drain parse tree.
    """

    def __init__(self, children=None, depth=0, id=None):
        """
        Parameters:
            children : the dictionary of children nodes
            depth : the depth of this node in the tree
            id : the digit or token that this node represents
        """
        if children is None:
            children = dict()
        self.children = children
        self.depth = depth
        self.id = id


class Drain:
    """
    A class implementing the Drain log parsing algorithm.
    """

    def __init__(
        self,
        indir: str = ".",
        depth: int = 4,
        st: float = 0.4,
        max_children: int = 100,
        rex: dict = {},
        skip_writeout: bool = False,
        message_prefix: str = r".*",
    ):
        """
        Initialize a Drain-based parser.

        Parameters:
            indir: the input directory stores the input log file name
            depth: depth of all leaf nodes
            st: similarity threshold
            max_children: max number of children of an internal node
            rex: regular expressions used in preprocessing, provided as a dictionary from field name to field regex
            skip_writeout: whether to skip writing out the parsed log file, templates and variables.
            message_prefix: prefix that starts each message of the log file - lines are merged to their preceding line if they do not start with this prefix.
        """
        self.indir = indir
        self.depth = depth - 2
        self.st = st
        self.max_children = max_children
        self.rex = rex
        self.skip_writeout = skip_writeout
        self.message_prefix = message_prefix

    def parse(self, filename: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Parse a log file.

        Parameters:
            filename: The name of the log file to parse (without path).

        Returns:
            A tuple of three dataframes, containing the parsed log file, the parsed log templates,
            and the parsed variables respectively.
        """

        full_path = os.path.join(self.indir, filename)
        print(f"Parsing file: {full_path}")
        self.filename = filename
        self.root = Node()
        self.cluster_list = []
        self.logdf = self._to_df(full_path)

        tqdm.pandas(desc="Determining template for each line...")
        self.logdf.progress_apply(self._parse_message, axis=1)

        return self._postprocess()

    def _to_df(self, log_file: str) -> pd.DataFrame:
        """
        Transform a log file into a dataframe.

        Parameters:
            log_file: The path to the log file.

        Returns:
            A dataframe containing the log file's lines, tokenized and with regexes replaced.
        """

        log_messages = []
        linecount = 0

        with open(log_file, "r") as f:
            log_message = ""

            for line in tqdm(f.readlines(), desc="Reading and tokenizing log lines..."):
                line = line.strip()

                if re.match(self.message_prefix, line):
                    if log_message:
                        try:
                            log_messages.append(self._preprocess(log_message))
                            linecount += 1
                        except Exception as e:
                            raise ValueError
                    log_message = line
                else:
                    log_message += " " + line

            if log_message:
                try:
                    log_messages.append(self._preprocess(log_message))
                    linecount += 1
                except Exception as e:
                    raise ValueError

        logdf = pd.DataFrame(
            log_messages, columns=["Message", "Tokenized", "Replaced by regex"]
        )
        logdf["MsgId"] = range(len(logdf))
        return logdf

    def _preprocess(self, msg: str) -> tuple[str, list[str], list[str]]:
        """
        Preprocess a message of a log file.

        Parameters:
            msg: The message to preprocess.

        Returns:
            A tuple containing the original message, the tokenized message, and a list of the values replaced by regexes.
        """

        msg = msg.strip()

        regex_matches = []
        for i, rex in enumerate(self.rex.values()):
            matches = re.findall(rex, msg)  ##### ASSUMPTION: only 1 match of interest
            regex_matches.append(matches[0] if matches else "")
            msg = re.sub(rex, "<*" + str(i) + ">", msg, count=1)

        pattern = r'([=,\{\}\[\]\(\);"\'])'  # Add spaces around punctuation
        msg = re.sub(pattern, r" \1 ", msg)
        pattern = r"(?<=\D):|:(?=\D)"  # Colons not in timestamps
        msg = re.sub(pattern, " : ", msg)

        return (msg, msg.strip().split(), regex_matches)

    def _parse_message(self, msg: pd.Series) -> None:
        """
        Parse a single log message and add it to the Drain parse tree in the appropriate cluster.

        Parameters:
            msg: The log message to parse.
        """

        line_id = msg["MsgId"]
        tokenized = msg["Tokenized"]
        cluster = self._tree_search(self.root, tokenized)

        if cluster is None:
            new_cluster = Cluster(template=tokenized, message_ids=[line_id])
            self.cluster_list.append(new_cluster)
            self._add_cluster_to_tree(self.root, new_cluster)
        else:
            new_template = self._get_updated_template(tokenized, cluster.template)
            cluster.message_ids.append(line_id)
            if " ".join(new_template) != " ".join(cluster.template):
                cluster.template = new_template

    def _tree_search(self, root: Node, tokenized: list[str]) -> Optional[Cluster]:
        """
        Search the Drain parse tree for a cluster matching `tokenized`.

        Parameters:
            root: The root of the Drain parse tree.
            tokenized: The tokenized log message to search for.

        Returns:
            The cluster in the Drain parse tree that matches `tokenized`,
            or None if no such cluster exists.
        """

        num_toks = len(tokenized)
        if num_toks not in root.children:
            return None

        node = root.children[num_toks]

        depth = 1
        for token in tokenized:
            if depth >= self.depth or depth > num_toks:
                break
            if token in node.children:
                node = node.children[token]
            elif "<*>" in node.children:
                node = node.children["<*>"]
            else:
                return None
            depth += 1

        cluster_list = node.children
        returned_cluster = self._find_cluster(cluster_list, tokenized)

        return returned_cluster

    def _add_cluster_to_tree(self, root: Node, cluster: Cluster) -> None:
        """
        Add a cluster to the Drain parse tree.

        Parameters:
            root: The root of the Drain parse tree.
            cluster: The cluster to add.
        """

        # Add a node to the first layer of the tree representing the length of the log message.
        length = len(cluster.template)
        first_layer_node = None
        if length not in root.children:
            first_layer_node = Node(depth=1, id=length)
            root.children[length] = first_layer_node
        else:
            first_layer_node = root.children[length]

        # Traverse the tree to add the new cluster.
        node = first_layer_node
        depth = 1
        for token in cluster.template:
            # If out of depth, add current log cluster to the leaf node
            if depth >= self.depth or depth > length:
                if len(node.children) == 0:
                    node.children = [cluster]
                else:
                    node.children.append(cluster)
                break

            # If token not matched in this layer of existing tree.
            if token not in node.children:
                if not any(char.isdigit() for char in token):
                    if "<*>" in node.children:
                        if len(node.children) < self.max_children:
                            new_node = Node(depth=depth + 1, id=token)
                            node.children[token] = new_node
                            node = new_node
                        else:
                            node = node.children["<*>"]
                    else:
                        if len(node.children) + 1 < self.max_children:
                            new_node = Node(depth=depth + 1, id=token)
                            node.children[token] = new_node
                            node = new_node
                        elif len(node.children) + 1 == self.max_children:
                            new_node = Node(depth=depth + 1, id="<*>")
                            node.children["<*>"] = new_node
                            node = new_node
                        else:
                            node = node.children["<*>"]
                else:
                    if "<*>" not in node.children:
                        node.children["<*>"] = Node(depth=depth + 1, id="<*>")
                    node = node.children["<*>"]

            # If the token is matched
            else:
                node = node.children[token]

            depth += 1

    def _similarity(self, seq1: list[str], seq2: list[str]) -> tuple[float, int]:
        """
        Determine the fraction of tokens in `seq1` that are identical to the corresponding token in `seq2`.
        Also return the number of parameters in `seq1`.

        Parameters:
            seq1: The first sequence.
            seq2: The second sequence.

        Returns:
            A tuple containing the fraction of identical tokens and the number of parameters in `seq1`.
        """
        assert len(seq1) == len(seq2)
        matches = 0
        num_params = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == "<*>":
                num_params += 1
            if token1 == token2:
                matches += 1

        similarity = float(matches) / len(seq1)

        return similarity, num_params

    def _find_cluster(
        self, cluster_list: list[Cluster], seq: list[str]
    ) -> Optional[Cluster]:
        """
        Find the cluster in `cluster_list` that is most similar to `seq`.

        Parameters:
            cluster_list: The list of clusters to search.
            seq: The sequence of tokens to compare to.

        Returns:
            The cluster in `cluster_list` that is most similar to `seq`,
            or None if no cluster is sufficiently similar.
        """

        max_similarity = -1
        max_num_params = -1
        max_cluster = None

        for cluster in cluster_list:
            similarity, num_params = self._similarity(cluster.template, seq)
            if similarity > max_similarity or (
                similarity == max_similarity and num_params > max_num_params
            ):
                max_similarity = similarity
                max_num_params = num_params
                max_cluster = cluster

        if max_similarity >= self.st:
            return max_cluster
        else:
            return None

    def _get_updated_template(self, template: list[str], msg: list[str]) -> list[str]:
        """
        Get the updated template from matching `msg` to `template`.

        Parameters:
            template: The template to match to.
            msg: The message to match.

        Returns:
            The updated template.
        """

        assert len(template) == len(msg)
        updated_template = []

        for i, word in enumerate(template):
            if word == msg[i]:
                updated_template.append(word)
            else:
                updated_template.append("<*>")

        return updated_template

    @staticmethod
    def _preceding_3(parsed_templates: pd.DataFrame, x: str) -> list[str]:
        """
        Get the 3 tokens preceding the variable `x` in the template.

        Parameters:
            parsed_templates: The dataframe containing information about the parsed templates.
            x: The name of the variable.

        Returns:
            The 3 tokens preceding the variable `x` in the template.
        """

        splitx = x.split("_")
        if len(splitx) != 2:
            return []
        id = splitx[0]
        position = int(splitx[1])
        start_position = max(0, position - 3)
        return (
            parsed_templates[parsed_templates["TemplateId"] == id]["TemplateText"]
            .values[0]
            .split()[start_position:position]
        )

    def _postprocess(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        template_id_per_msg = [0] * self.logdf.shape[0]
        parsed_templates_list = []

        regex_tokens = ["<*" + str(i) + ">" for i in range(len(self.rex))]

        # Process each cluster to determine template information.
        for cluster in self.cluster_list:
            d = {}

            # Determine the template ID
            d["TemplateText"] = " ".join(cluster.template)
            d["TemplateId"] = hashlib.md5(
                d["TemplateText"].encode("utf-8")
            ).hexdigest()[0:8]

            # Determine the indices of the variables and regexes in the template.
            d["VariableIndices"] = [
                i for i, x in enumerate(cluster.template) if x == "<*>"
            ]
            d["RegexIndices"] = []
            for i in regex_tokens:
                try:
                    d["RegexIndices"].append(cluster.template.index(str(i)))
                except:
                    pass

            # Update the template ID for each log message in the cluster.
            for i, line_id in enumerate(cluster.message_ids):
                self.logdf.loc[line_id, "TemplateId"] = d["TemplateId"]

                if i == 0:
                    d["TemplateExample"] = self.logdf.loc[line_id, "Message"]

            parsed_templates_list.append(d.copy())

        # Create a dataframe of the parsed templates.
        self.parsed_templates = pd.DataFrame(parsed_templates_list)
        template_occurences = dict(self.logdf["TemplateId"].value_counts())
        self.parsed_templates["Occurrences"] = self.parsed_templates["TemplateId"].map(
            template_occurences
        )

        # Create columns for each variable (parsed or regex-derived) and extract them from each log message.
        variable_columns = list(self.rex.keys())
        variable_columns.extend(
            [
                str(i) + "_" + str(j)
                for i in self.parsed_templates["TemplateId"].values
                for j in self.parsed_templates.loc[
                    self.parsed_templates["TemplateId"] == i, "VariableIndices"
                ].values[0]
            ]
        )
        par_df = pd.DataFrame(
            columns=variable_columns, index=range(self.logdf.shape[0])
        )
        self.logdf = pd.concat((self.logdf, par_df), axis=1)
        self._extract_variables()

        # Create a dataframe of the parsed variables.
        parsed_variables = pd.DataFrame()
        parsed_variables["Name"] = variable_columns
        parsed_variables["Occurrences"] = parsed_variables["Name"].map(
            lambda x: self.logdf[x].notna().sum()
        )
        parsed_variables["Preceding 3 tokens"] = parsed_variables["Name"].map(
            lambda x: Drain._preceding_3(self.parsed_templates, x)
        )
        parsed_variables["Examples"] = parsed_variables["Name"].map(
            lambda x: self.logdf[x].loc[self.logdf[x].notna()].unique()[:5].tolist()
        )
        parsed_variables["From regex"] = parsed_variables["Name"].map(
            lambda x: True if x in self.rex.keys() else False
        )

        # Drop unnecessary columns from the parsed log.
        to_drop = ["MsgId", "Message", "Tokenized", "Replaced by regex"]
        to_drop.extend(
            parsed_variables[parsed_variables["Occurrences"] == 0]["Name"].tolist()
        )
        parsed_log = self.logdf.drop(columns=to_drop)
        parsed_variables = (
            parsed_variables[~parsed_variables.isin(to_drop)["Name"]]
            .reset_index()
            .drop(columns="index")
        )

        return parsed_log, self.parsed_templates, parsed_variables

    def _extract_variables(self) -> None:
        """
        Extract the variables from the log messages.
        """

        for row in tqdm(
            self.parsed_templates.itertuples(),
            desc="Extracting variables from each log message...",
            total=len(self.parsed_templates),
        ):
            template_id = row.TemplateId
            variable_indices = row.VariableIndices

            mask = self.logdf["TemplateId"] == template_id
            for i in variable_indices:
                col_name = f"{template_id}_{str(i)}"
                self.logdf.loc[mask, col_name] = self.logdf.loc[mask, "Tokenized"].str[
                    i
                ]

            for i, col_name in enumerate(self.rex.keys()):
                self.logdf.loc[mask, col_name] = self.logdf.loc[
                    mask, "Replaced by regex"
                ].str[i]
