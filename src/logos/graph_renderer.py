"""
A module for rendering causal graphs.
"""

import base64
from io import BytesIO

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from IPython.display import HTML, display


class GraphRenderer:
    """
    Render a digraph with appropriate margins and node tags.
    """

    @staticmethod
    def draw_graph(graph: nx.DiGraph, var_info: pd.DataFrame) -> str:
        """
        Draw a graph with appropriate margins and node tags.

        Parameters:
            graph: The graph to be drawn.
            var_info: A dataframe containing the tags of the variables in the
                graph.

        Returns:
            A base64-encoded string representation of the graph.
        """
        if graph.number_of_nodes() == 0:
            return ""

        pos = nx.spring_layout(graph)
        nx.draw(
            graph,
            pos,
            edgelist=graph.edges(),
            with_labels=False,
            width=2.0,
            node_color="#d3d3d3",
            edge_color=[graph[u][v].get("color", "#7f9aba") for u, v in graph.edges()],
        )
        node_labels = {
            n: (
                n
                if len(var_info.loc[var_info["Name"] == n, "Tag"].values) == 0
                else var_info.loc[var_info["Name"] == n, "Tag"].values[0]
            )
            for n in list(graph.nodes)
        }
        text = nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=12)
        for _, t in text.items():
            t.set_rotation(30)

        # Fix margins
        x_values, y_values = zip(*pos.values())
        x_max, x_min = max(x_values), min(x_values)
        y_max, y_min = max(y_values), min(y_values)
        if x_max != x_min:
            x_margin = (x_max - x_min) * 0.3
            plt.xlim(x_min - x_margin, x_max + x_margin)
        if y_max != y_min:
            y_margin = (y_max - y_min) * 0.3
            plt.ylim(y_min - y_margin, y_max + y_margin)

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.clf()
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        return img_str

    @staticmethod
    def save_graph(graph: nx.DiGraph, var_info: pd.DataFrame, filename: str) -> None:
        """
        Save the graph to a file as a png image.

        Parameters:
            graph: The graph to be saved.
            var_info: A dataframe containing the tags of the variables in the
                graph.
            filename: The name of the file to which the graph should be saved.
        """
        img_str = GraphRenderer.draw_graph(graph, var_info)
        with open(filename, "wb") as f:
            f.write(base64.b64decode(img_str))

    @staticmethod
    def graph_string_to_html(graph: str) -> HTML:
        """
        Convert the string representation of the rgaph to an HTML object

        Parameters:
            graph: The graph to be displayed.
        """
        return HTML(
            f'<img src="data:image/png;base64,{graph}" style="max-width: 100%; height: auto;">'
        )

    @staticmethod
    def display_graph(graph: nx.DiGraph, var_info: pd.DataFrame) -> None:
        """
        Display the graph.

        Parameters:
            graph: The graph to be displayed.
            var_info: A dataframe containing the tags of the variables in the
                graph.
        """
        display(
            GraphRenderer.graph_string_to_html(
                GraphRenderer.draw_graph(graph, var_info)
            )
        )
