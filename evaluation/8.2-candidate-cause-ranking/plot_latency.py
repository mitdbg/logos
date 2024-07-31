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

print("Plotting latency...")

datasets = ["postgresql", "proprietary", "xyz"]
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
            "8.2-candidate-cause-ranking",
            f"{dataset}_latency_{method}.csv",
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
    # Plot the latency of each method.
    latency = {}
    for method in methods:
        latency[method] = results[method]["latency"].mean()
    df = pd.DataFrame(
        {"latency": [latency[m] for m in mapping.keys()]}, index=mapping.values()
    )
    ax = df.plot(kind="bar", figsize=(4, 5))
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.xticks(rotation=70)
    plt.ylabel("Mean latency (s)", fontsize=FONTSIZE, labelpad=10)
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
    y_max = max(list(latency.values()))
    plt.ylim(0, y_max * 1.15)

    # Set the color of the bars
    ax.patches[0].set_facecolor("#7FBA82")
    ax.patches[1].set_facecolor("#ba8a7f")
    ax.patches[2].set_facecolor("#7F9FBA")
    ax.get_legend().remove()

    plt.tight_layout()
    fig_path = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_plots",
        f"8.2-candidate-cause-ranking-{dataset}-latency.png",
    )
    if not os.path.exists(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
    plt.savefig(fig_path, bbox_inches="tight")

    # Print the x,y values of each data point
    res_path = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_results",
        f"8.2-candidate-cause-ranking-{dataset}-latency.csv",
    )
    if not os.path.exists(os.path.dirname(res_path)):
        os.makedirs(os.path.dirname(res_path))
    with open(res_path, "w+") as f:
        f.write("Method,Latency\n")
        for method, l in latency.items():
            f.write(f"{method},{l}\n")
