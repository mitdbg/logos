import pandas as pd
import sys

sys.path.append("../..")
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.definitions import LOGOS_ROOT_DIR
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_repro_results", action="store_true")
args = parser.parse_args()
prefix = "repro" if args.use_repro_results else "paper"

print("Plotting latency...")

LINE_FORMATTING_DATA = {
    "regression": {
        "label": r"\textsc{Regression}",
        "color": "#7FBA82",
        "marker": "o",
    },
    "langmodel": {
        "label": r"\textsc{LangModel}",
        "color": "#ba8a7f",
        "marker": "o",
    },
    "logos": {
        "label": r"\textsc{LOGos}",
        "color": "#7F9FBA",
        "marker": "o",
    },
}

datasets = ["postgresql", "xyz"]
for dataset in datasets:
    # Read in the data
    methods = ["logos", "regression", "langmodel"]
    results = {}
    for method in methods:
        path = os.path.join(
            LOGOS_ROOT_DIR,
            "dataset_files",
            dataset,
            f"{prefix}_evaluation",
            "8.3-interactive-causal-graph-refinement",
            f"{dataset}_bounded_latency_{method}.csv",
        )
        results[method] = pd.read_csv(path, header=0)
        results[method] = results[method].groupby("judgments").mean()[["latency"]]
        results[method]["judgments"] = results[method].index
        results[method].reset_index(drop=True, inplace=True)

    rc_fonts = {
        "font.family": "serif",
        "text.usetex": True,
        "text.latex.preamble": r"""
            \usepackage{libertine}
            \usepackage[libertine]{newtxmath}
            """,
    }
    mpl.rcParams.update(rc_fonts)
    FONTSIZE = 24

    #####
    # Plot the latency of each method.
    _, ax = plt.subplots(figsize=(6, 5))
    for method in methods:
        ax.plot(
            results[method]["judgments"],
            results[method]["latency"],
            color=LINE_FORMATTING_DATA[method]["color"],
            marker=LINE_FORMATTING_DATA[method]["marker"],
            label=LINE_FORMATTING_DATA[method]["label"],
        )
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.set_ylabel("Mean latency (s)", fontsize=FONTSIZE, labelpad=10)
    ax.set_xlabel("\# Judgments", fontsize=FONTSIZE)
    ax.set_xticks(np.arange(1, 6))
    ax.legend(fontsize=FONTSIZE)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    plt.tight_layout()
    fig_path = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_plots",
        f"8.3-interactive-causal-graph-refinement-{dataset}-bounded-latency.png",
    )
    if not os.path.exists(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
    plt.savefig(fig_path, bbox_inches="tight")

    # Print the x,y values of each data point
    res_path = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_results",
        f"8.3-interactive-causal-graph-refinement-{dataset}-bounded-latency.csv",
    )
    if not os.path.exists(os.path.dirname(res_path)):
        os.makedirs(os.path.dirname(res_path))
    with open(res_path, "w+") as f:
        f.write("Method,Latency\n")
        for method in methods:
            for i, row in results[method].iterrows():
                f.write(f"{method},{i+1},{row['latency']}\n")
