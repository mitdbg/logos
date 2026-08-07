import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from logos.filesystem.paths import LOGOS_ROOT_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--use_repro_results", action="store_true")
args = parser.parse_args()
prefix = "repro" if args.use_repro_results else "paper"


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

LINE_FORMATTING_DATA = {
    "length": {
        "xlabel": r"Log Length",
        "color": "#7FBA82",
        "parse_fit_start_idx": 2,
        "parse_fit_end_idx": 6,
        "agg_fit_start_idx": 2,
        "agg_fit_end_idx": 6,
        "loglog": True,
        "xaxis_mult": 1,
    },
    "templates": {
        "xlabel": r"\# Templates",
        "color": "#ba8a7f",
        "parse_fit_start_idx": 1,
        "parse_fit_end_idx": 4,
        "agg_fit_start_idx": 1,
        "agg_fit_end_idx": 4,
        "loglog": True,
        "xaxis_mult": 1,
    },
    "variables": {
        "xlabel": r"\# Variables / \# Line Tokens",
        "color": "#7F9FBA",
        "parse_fit_start_idx": 0,
        "parse_fit_end_idx": 10,
        "agg_fit_start_idx": 0,
        "agg_fit_end_idx": 10,
        "loglog": False,
        "xaxis_mult": 0.01,
    },
}


def form_line_equation_string(slope, intercept, rvalue):
    sign = "+" if intercept >= 0 else "-"
    r2 = rvalue**2
    return (
        rf"$y={slope:.2f}x{sign}{abs(intercept):.2f}$" + "\n" rf"$R^2={r2:.3f}$"
    )


def form_power_law_string(slope, intercept, rvalue):
    sign = "+" if intercept >= 0 else "-"
    r2 = rvalue**2
    return (
        rf"$\log_{{10}}y={slope:.2f}\log_{{10}}x{sign}{abs(intercept):.2f}$"
        + "\n"
        rf"$R^2={r2:.3f}$"
    )


for metric, properties in LINE_FORMATTING_DATA.items():

    # Read data from CSV
    path = os.path.join(
        LOGOS_ROOT_DIR,
        "dataset_files",
        "scaling",
        f"{prefix}_evaluation",
        "8.4-log-converter-scaling",
        f"8.4-log-converter-scaling-{metric}.csv",
    )
    data = pd.read_csv(path)
    data.columns = [x.strip() for x in data.columns]

    # Extract data columns
    x = data[list(data.columns)[0]]
    x = x * properties["xaxis_mult"]
    parse_time = data["Parse Time"]
    prep_time = data["Prep Time"]

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    # Plot 1 - Parse Time
    if properties["loglog"]:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    ax1.plot(
        x,
        parse_time,
        marker="o",
        color=properties["color"],
        markersize=15,
    )
    ax1.set_xlabel(properties["xlabel"], fontsize=FONTSIZE)
    ax1.set_ylabel("Time (s)", fontsize=FONTSIZE)
    ax1.tick_params(axis="both", which="major", labelsize=FONTSIZE)

    # Add trendline
    pfsi = properties["parse_fit_start_idx"]
    pfei = properties["parse_fit_end_idx"]
    if properties["loglog"]:
        pfit = stats.linregress(
            np.log10(x[pfsi:pfei]), np.log10(parse_time[pfsi:pfei])
        )
        trendline_parse = (
            10 ** (pfit.slope * np.log10(x[pfsi:pfei]) + pfit.intercept)
        ).to_numpy()
        parse_label = form_power_law_string(
            pfit.slope, pfit.intercept, pfit.rvalue
        )
    else:
        pfit = stats.linregress(x[pfsi:pfei], parse_time[pfsi:pfei])
        trendline_parse = (
            pfit.slope * x[pfsi:pfei] + pfit.intercept
        ).to_numpy()
        parse_label = form_line_equation_string(
            pfit.slope, pfit.intercept, pfit.rvalue
        )
    ax1.plot(
        x[pfsi:pfei], trendline_parse, "--", color="black", label=parse_label
    )

    ax1.legend(
        loc="lower left", bbox_to_anchor=(-0.08, 1.0001), fontsize=FONTSIZE
    )

    plt.tight_layout()
    plt.show
    fig_path_parsing = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_plots",
        f"8.4-log-converter-scaling-{metric}-parsing.png",
    )
    if not os.path.exists(os.path.dirname(fig_path_parsing)):
        os.makedirs(os.path.dirname(fig_path_parsing))
    plt.savefig(fig_path_parsing, bbox_inches="tight")

    res_path_parsing = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_plots_data",
        f"8.4-log-converter-scaling-{metric}-parsing.csv",
    )
    if not os.path.exists(os.path.dirname(res_path_parsing)):
        os.makedirs(os.path.dirname(res_path_parsing))
    parsing_data = pd.DataFrame(
        {
            "x": x,
            "parse_time": parse_time,
            "trendline_parse": [
                trendline_parse[i - pfsi] if (i >= pfsi and i < pfei) else None
                for i in range(len(x))
            ],
        }
    )
    parsing_data.to_csv(res_path_parsing, index=False)

    fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))

    # Plot 2 - Prep Time
    if properties["loglog"]:
        ax2.set_xscale("log")
        ax2.set_yscale("log")
    ax2.plot(
        x,
        prep_time,
        marker="^",
        color=properties["color"],
        markersize=15,
    )
    ax2.set_xlabel(properties["xlabel"], fontsize=FONTSIZE)
    ax2.set_ylabel("Time (s)", fontsize=FONTSIZE)
    ax2.tick_params(axis="both", which="major", labelsize=FONTSIZE)

    # Add linear trendline
    afsi = properties["agg_fit_start_idx"]
    afei = properties["agg_fit_end_idx"]
    if properties["loglog"]:
        afit = stats.linregress(
            np.log10(x[afsi:afei]), np.log10(prep_time[afsi:afei])
        )
        trendline_prep = (
            10 ** (afit.slope * np.log10(x[afsi:afei]) + afit.intercept)
        ).to_numpy()
        agg_label = form_power_law_string(
            afit.slope, afit.intercept, afit.rvalue
        )
    else:
        afit = stats.linregress(x[afsi:afei], prep_time[afsi:afei])
        trendline_prep = (afit.slope * x[afsi:afei] + afit.intercept).to_numpy()
        agg_label = form_line_equation_string(
            afit.slope, afit.intercept, afit.rvalue
        )
    ax2.plot(x[afsi:afei], trendline_prep, "--", color="black", label=agg_label)
    ax2.legend(
        loc="lower left", bbox_to_anchor=(-0.08, 1.0001), fontsize=FONTSIZE
    )

    plt.tight_layout()
    plt.show
    fig_path_agg = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_plots",
        f"8.4-log-converter-scaling-{metric}-aggregation.png",
    )
    if not os.path.exists(os.path.dirname(fig_path_agg)):
        os.makedirs(os.path.dirname(fig_path_agg))
    plt.savefig(fig_path_agg, bbox_inches="tight")

    res_path_agg = os.path.join(
        LOGOS_ROOT_DIR,
        "evaluation",
        f"{prefix}_plots_data",
        f"8.4-log-converter-scaling-{metric}-aggregation.csv",
    )
    if not os.path.exists(os.path.dirname(res_path_agg)):
        os.makedirs(os.path.dirname(res_path_agg))
    aggregation_data = pd.DataFrame(
        {
            "x": x,
            "agg_time": prep_time,
            "trendline_agg": [
                trendline_prep[i - afsi] if (i >= afsi and i < afei) else None
                for i in range(len(x))
            ],
        }
    )
    aggregation_data.to_csv(res_path_agg, index=False)
