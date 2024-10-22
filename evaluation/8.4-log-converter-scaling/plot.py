import os
import sys

sys.path.append("../..")

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.definitions import LOGOS_ROOT_DIR


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
        "parse_polyfit_deg": 1,
        "agg_fit_start_idx": 1,
        "agg_fit_end_idx": 6,
        "agg_polyfit_deg": 1,
        "loglog": True,
        "xaxis_mult": 1,
    },
    "templates": {
        "xlabel": r"\# Templates",
        "color": "#ba8a7f",
        "parse_fit_start_idx": 0,
        "parse_fit_end_idx": 4,
        "parse_polyfit_deg": 2,
        "agg_fit_start_idx": 0,
        "agg_fit_end_idx": 4,
        "agg_polyfit_deg": 2,
        "loglog": True,
        "xaxis_mult": 1,
    },
    "variables": {
        "xlabel": r"\# Variables / \# Line Tokens",
        "color": "#7F9FBA",
        "parse_fit_start_idx": 0,
        "parse_fit_end_idx": 10,
        "parse_polyfit_deg": 1,
        "agg_fit_start_idx": 0,
        "agg_fit_end_idx": 10,
        "agg_polyfit_deg": 1,
        "loglog": False,
        "polyfit_deg": 1,
        "xaxis_mult": 0.01,
    },
}


def form_polynomial_string(p):
    p_str = r"$"
    for i in range(len(p)):
        if i == 0:
            p_str += f"{p[i]:.2e}"
        else:
            p_str += f"{p[i]:+.2e}"
        if i < len(p) - 1:
            p_str += "x"
        if i < len(p) - 2:
            p_str += r"^" + f"{len(p)-i-1}"

    p_str += r"}$"
    p_str = p_str.replace("e+00", r"{")
    p_str = p_str.replace("e-0", r"\cdot10^{-")
    p_str = p_str.replace("e+0", r"\cdot10^{")
    p_str = p_str.replace("x", r"}x")
    return p_str


for metric, properties in LINE_FORMATTING_DATA.items():

    # Read data from CSV
    path = os.path.join(
        LOGOS_ROOT_DIR,
        "dataset_files",
        "scaling",
        f"{prefix}_evaluation",
        "8.4-log-converter-scaling",
        f"{metric}.csv",
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
    ax1.set_xlabel(
        properties["xlabel"] + (" (log scale)" if properties["loglog"] else ""),
        fontsize=FONTSIZE,
    )
    ax1.set_ylabel(
        "Time " + ("(s, log scale)" if properties["loglog"] else "(s)"),
        fontsize=FONTSIZE,
    )
    ax1.tick_params(axis="both", which="major", labelsize=FONTSIZE)

    # Add trendline
    pfsi = properties["parse_fit_start_idx"]
    pfei = properties["parse_fit_end_idx"]
    pfit_coeffs = np.polyfit(
        x[pfsi:pfei],
        parse_time[pfsi:pfei],
        properties["parse_polyfit_deg"],
    )
    trendline_parse = np.polyval(pfit_coeffs, x[pfsi:pfei])
    ax1.plot(
        x[pfsi:pfei],
        trendline_parse,
        "--",
        color="black",
        label=form_polynomial_string(pfit_coeffs),
    )

    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0001), fontsize=FONTSIZE)

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
    ax2.set_xlabel(
        properties["xlabel"] + (" (log scale)" if properties["loglog"] else ""),
        fontsize=FONTSIZE,
    )
    ax2.set_ylabel(
        "Time " + ("(s, log scale)" if properties["loglog"] else "(s)"),
        fontsize=FONTSIZE,
    )
    ax2.tick_params(axis="both", which="major", labelsize=FONTSIZE)

    # Add linear trendline
    afsi = properties["agg_fit_start_idx"]
    afei = properties["agg_fit_end_idx"]
    afit_coeffs = np.polyfit(
        x[afsi:afei],
        prep_time[afsi:afei],
        properties["agg_polyfit_deg"],
    )
    trendline_prep = np.polyval(afit_coeffs, x[afsi:afei])
    ax2.plot(
        x[afsi:afei],
        trendline_prep,
        "--",
        color="black",
        label=form_polynomial_string(afit_coeffs),
    )
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0001), fontsize=FONTSIZE)

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
