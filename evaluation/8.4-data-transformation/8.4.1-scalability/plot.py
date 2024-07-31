import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import matplotlib as mpl

sys.path.append("../../../")
from src.definitions import LOGOS_ROOT_DIR

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
        "agg_fit_start_idx": 0,
        "agg_fit_end_idx": 6,
        "agg_polyfit_deg": 1,
        "loglog": True,
        "xaxis_mult":1
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
        "xaxis_mult":1
    },
    "variables": {
        "xlabel": r"$\frac{Variables}{Line Tokens}$",
        "color": "#7F9FBA",
        "parse_fit_start_idx": 0,
        "parse_fit_end_idx": 10,
        "parse_polyfit_deg": 1,
        "agg_fit_start_idx": 0,
        "agg_fit_end_idx": 10,
        "agg_polyfit_deg": 1,
        "loglog": False,
        "polyfit_deg": 1,
        "xaxis_mult":0.01
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

plots_dir = os.path.join(LOGOS_ROOT_DIR, "evaluation", "plots")


for metric in LINE_FORMATTING_DATA.keys():

    # Read data from CSV
    path = os.path.join(
        LOGOS_ROOT_DIR, "evaluation", "results", f"8.4.1-scalability-{metric}.csv"
    )
    data = pd.read_csv(path)
    data.columns = [x.strip() for x in data.columns]

    # Extract data columns
    x = data[list(data.columns)[0]]
    x = x * LINE_FORMATTING_DATA[metric]["xaxis_mult"]
    parse_time = data["Parse Time"]
    prep_time = data["Prep Time"]

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    # Plot 1 - Parse Time
    if LINE_FORMATTING_DATA[metric]["loglog"]:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    ax1.plot(
        x,
        parse_time,
        marker="o",
        color=LINE_FORMATTING_DATA[metric]["color"],
        markersize=15,
    )
    ax1.set_xlabel(LINE_FORMATTING_DATA[metric]["xlabel"], fontsize=FONTSIZE)
    ax1.set_ylabel("Time (s)", fontsize=FONTSIZE)
    ax1.tick_params(axis="both", which="major", labelsize=FONTSIZE)

    # Add trendline
    pfsi = LINE_FORMATTING_DATA[metric]["parse_fit_start_idx"]
    pfei = LINE_FORMATTING_DATA[metric]["parse_fit_end_idx"]
    pfit_coeffs = np.polyfit(
        x[pfsi:pfei],
        parse_time[pfsi:pfei],
        LINE_FORMATTING_DATA[metric]["parse_polyfit_deg"],
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
    plt.savefig(os.path.join(plots_dir, f"8.4.1-scalability-{metric}-parsing.jpg"), bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))

    # Plot 2 - Prep Time
    if LINE_FORMATTING_DATA[metric]["loglog"]:
        ax2.set_xscale("log")
        ax2.set_yscale("log")
    ax2.plot(
        x,
        prep_time,
        marker="^",
        color=LINE_FORMATTING_DATA[metric]["color"],
        markersize=15,
    )
    ax2.set_xlabel(LINE_FORMATTING_DATA[metric]["xlabel"], fontsize=FONTSIZE)
    ax2.set_ylabel("Time (s)", fontsize=FONTSIZE)
    ax2.tick_params(axis="both", which="major", labelsize=FONTSIZE)

    # Add linear trendline
    afsi = LINE_FORMATTING_DATA[metric]["agg_fit_start_idx"]
    afei = LINE_FORMATTING_DATA[metric]["agg_fit_end_idx"]
    afit_coeffs = np.polyfit(
        x[afsi:afei],
        prep_time[afsi:afei],
        LINE_FORMATTING_DATA[metric]["agg_polyfit_deg"],
    )
    trendline_prep = np.polyval(afit_coeffs, x[afsi:afei])
    ax2.plot(
        x[afsi:],
        trendline_prep,
        "--",
        color="black",
        label=form_polynomial_string(afit_coeffs),
    )
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0001), fontsize=FONTSIZE)

    plt.tight_layout()
    plt.show
    plt.savefig(os.path.join(plots_dir, f"8.4.1-scalability-{metric}-aggregation.jpg"), bbox_inches="tight")
