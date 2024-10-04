import pandas as pd
import sys

sys.path.append("../..")
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.definitions import LOGOS_ROOT_DIR
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_repro_results", action="store_true")
args = parser.parse_args()
prefix = "repro" if args.use_repro_results else "paper"

print("Plotting average precision...")

datasets = ["postgresql", "proprietary", "xyz"]
for dataset in datasets:
    # Read in the data
    methods = ["logos", "regression", "langmodel"]
    strip_if_string = lambda x: x.split(":")[0].strip() if isinstance(x, str) else x
    results = {}
    for method in methods:
        path = os.path.join(
            LOGOS_ROOT_DIR,
            "dataset_files",
            dataset,
            f"{prefix}_evaluation",
            "8.2-candidate-cause-ranking",
            f"{dataset}_ranking_{method}.csv",
        )
        results[method] = pd.read_csv(path, header=0)
        results[method]["candidate"] = results[method]["candidate"].apply(
            strip_if_string
        )

    with open(f"conf_{dataset}.json", "r") as f:
        conf = json.load(f)
        ground_truth = conf["true_causes"]

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

    # Plot the average precision of each method

    def calc_avg_precision(results, true_values):
        group_cols = set(results.columns) - set(["rank", "candidate"])
        if len(group_cols) == 0:
            ranking = results["candidate"].tolist()
            denominator = len(true_values)
            numerator = 0
            found = 0
            for i in range(len(ranking)):
                if ranking[i] in true_values:
                    found += 1
                    precision = found / (i + 1)
                    numerator += precision
            return numerator / denominator
        else:
            acc = 0
            count = 0
            for _, group in results.groupby(list(group_cols)):
                group = group.drop(columns=list(group_cols))
                acc += calc_avg_precision(group, true_values)
                count += 1
            return acc / count

    avg_precision = {}
    for method in methods:
        avg_precision[method] = calc_avg_precision(results[method], ground_truth)
    df = pd.DataFrame(
        {"avg_precision": [avg_precision[m] for m in mapping.keys()]},
        index=mapping.values(),
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    for idx, (label, value) in enumerate(zip(df.index, df["avg_precision"])):
        ax.bar(idx, value, label=label)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.xticks([])
    plt.ylabel("Average Precision", fontsize=FONTSIZE, labelpad=10)
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
    y_max = max(list(avg_precision.values()))
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylim(0, 1.15)

    # Set the color of the bars and add legend if needed.
    ax.patches[0].set_facecolor("#7FBA82")
    ax.patches[1].set_facecolor("#ba8a7f")
    ax.patches[2].set_facecolor("#7F9FBA")
    if dataset == 'postgresql':
        plt.legend(fontsize=FONTSIZE, loc="upper center", borderpad=0.2)

    plt.tight_layout()
    fig_path = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_plots",
        f"8.2-candidate-cause-ranking-{dataset}-average-precision.png",
    )
    if not os.path.exists(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
    plt.savefig(fig_path, bbox_inches="tight")

    # Print the x,y values of each data point
    res_path = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_results",
        f"8.2-candidate-cause-ranking-{dataset}-average-precision.csv",
    )
    if not os.path.exists(os.path.dirname(res_path)):
        os.makedirs(os.path.dirname(res_path))
    with open(res_path, "w+") as f:
        f.write("Method,Average Precision\n")
        for method, precision in avg_precision.items():
            f.write(f"{method},{precision}\n")
