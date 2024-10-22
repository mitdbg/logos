import pandas as pd
import sys

sys.path.append("../..")
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.definitions import LOGOS_ROOT_DIR
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_repro_results", action="store_true")
args = parser.parse_args()
prefix = "repro" if args.use_repro_results else "paper"

print("Plotting judgments...")

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
            f"{dataset}_results_{method}.csv",
        )
        results[method] = pd.read_csv(path, header=0)

    # Take care of latex in plot elements
    mapping = {
        "regression": r"\textsf{Regression}",
        "langmodel": r"\textsf{LangModel}",
        "logos": r"\textsf{LOGos}",
    }
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
    # Plot the judgments required by of each method.
    judgments = {}
    group_cols = list(
        set(results[methods[0]].columns) - set(["judgments", "edge", "ATE", "ARE_ATE"])
    )
    for method in methods:
        if len(group_cols) == 0:
            judgments[method] = (
                results[method].drop(columns=["edge"]).max()["judgments"]
            )
        else:
            judgments[method] = (
                results[method]
                .drop(columns=["edge"])
                .groupby(group_cols)
                .max()
                .mean()["judgments"]
            )
    df = pd.DataFrame(
        {"judgments": [judgments[m] for m in mapping.keys()]}, index=mapping.values()
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    for idx, (label, value) in enumerate(zip(df.index, df["judgments"])):
        ax.bar(idx, value, label=label)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.xticks([])
    plt.ylabel("Judgments Until\nGround Truth ATE", fontsize=FONTSIZE, labelpad=10)
    plt.xlabel("")

    # Print the value of each bar above it
    for i in ax.patches:
        ax.text(
            i.get_x() + i.get_width() / 2,
            i.get_height(),
            f"{i.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE,
        )

    # Leave some gap on the y axis so that the labels are inside the plot
    y_max = max(list(judgments.values()))
    plt.ylim(0, y_max * 1.15)

    # Set the color of the bars
    ax.patches[0].set_facecolor("#7FBA82")
    ax.patches[1].set_facecolor("#ba8a7f")
    ax.patches[2].set_facecolor("#7F9FBA")

    plt.tight_layout()
    fig_path = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_plots",
        f"8.3-interactive-causal-graph-refinement-{dataset}-judgments.png",
    )
    if not os.path.exists(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
    plt.savefig(fig_path, bbox_inches="tight")

    # Print the x,y values of each data point
    res_path = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_plots_data",
        f"8.3-interactive-causal-graph-refinement-{dataset}-judgments.csv",
    )
    if not os.path.exists(os.path.dirname(res_path)):
        os.makedirs(os.path.dirname(res_path))
    with open(res_path, "w+") as f:
        f.write("Method,Judgments\n")
        for method, l in judgments.items():
            f.write(f"{method},{l}\n")
