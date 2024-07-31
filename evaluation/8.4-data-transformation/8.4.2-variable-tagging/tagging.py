import pandas as pd
import sys

sys.path.append("../..")
from src.logos.tag_utils import TagUtils, TagOrigin
from tqdm.auto import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt


datasets = {
    "PostgreSQL": {
        "vars_filename": "~/causal-log/datasets/tpc-ds/parameter_sweep_1.log_parsed_variables_None_None.pkl",
        "templates_filename": "~/causal-log/datasets/tpc-ds/parameter_sweep_1.log_parsed_templates_None_None.pkl",
    },
    "XYZ\n10 vars": {
        "vars_filename": "~/causal-log/datasets/xyz_extended/log_2023-12-22_13:13:01.log_parsed_variables_None_None.pkl",
        "templates_filename": "~/causal-log/datasets/xyz_extended/log_2023-12-22_13:13:01.log_parsed_templates_None_None.pkl",
    },
    "XYZ\n100 vars": {
        "vars_filename": "~/causal-log/datasets/xyz_extended/log_2023-12-22_13:17:29.log_parsed_variables_None_None.pkl",
        "templates_filename": "~/causal-log/datasets/xyz_extended/log_2023-12-22_13:17:29.log_parsed_templates_None_None.pkl",
    },
    "XYZ\n1000 vars": {
        "vars_filename": "~/causal-log/datasets/xyz_extended/log_2023-12-22_13:27:02.log_parsed_variables_None_None.pkl",
        "templates_filename": "~/causal-log/datasets/xyz_extended/log_2023-12-22_13:27:02.log_parsed_templates_None_None.pkl",
    },
    "OpenStack\nCinder": {
        "vars_filename": "~/causal-log/evaluation/datasets/Openstack/Cinder/Cinder_combined_all.log_parsed_variables_None_None.pkl",
        "templates_filename": "~/causal-log/evaluation/datasets/Openstack/Cinder/Cinder_combined_all.log_parsed_templates_None_None.pkl",
    },
    "OpenStack\nNeutron": {
        "vars_filename": "~/causal-log/evaluation/datasets/Openstack/Neutron/Neutron_combined_all.log_parsed_variables_None_None.pkl",
        "templates_filename": "~/causal-log/evaluation/datasets/Openstack/Neutron/Neutron_combined_all.log_parsed_templates_None_None.pkl",
    },
    "OpenStack\nNova": {
        "vars_filename": "~/causal-log/evaluation/datasets/Openstack/Nova/Nova_combined_all.log_parsed_variables_None_None.pkl",
        "templates_filename": "~/causal-log/evaluation/datasets/Openstack/Nova/Nova_combined_all.log_parsed_templates_None_None.pkl",
    },
    "Proprietary": {
        "vars_filename": "~/causal-log/datasets/proprietary_logs/proprietary_eval/proprietary_1000users_10faulty_20pctfailfaulty_10pctfailnormal.log_parsed_variables_None_None.pkl",
        "templates_filename": "~/causal-log/datasets/proprietary_logs/proprietary_eval/proprietary_1000users_10faulty_20pctfailfaulty_10pctfailnormal.log_parsed_templates_None_None.pkl",
    },
}

d = {
    TagOrigin.PRECEDING: np.array([0] * len(datasets)),
    TagOrigin.GPT_3POINT5_TURBO: np.array([0] * len(datasets)),
    TagOrigin.GPT_4: np.array([0] * len(datasets)),
    TagOrigin.NAME: np.array([0] * len(datasets)),
    TagOrigin.REGEX_VARIABLE: np.array([0] * len(datasets)),
}

d_scaled = {
    TagOrigin.PRECEDING: np.array([0.0] * len(datasets)),
    TagOrigin.GPT_3POINT5_TURBO: np.array([0.0] * len(datasets)),
    TagOrigin.GPT_4: np.array([0.0] * len(datasets)),
    TagOrigin.NAME: np.array([0.0] * len(datasets)),
    TagOrigin.REGEX_VARIABLE: np.array([0.0] * len(datasets)),
}

for dataset_num, dataset in enumerate(datasets.keys()):
    print(f"Starting dataset {dataset}...")
    with open("gpt_log.txt", "a+") as f:
        f.write('========================================\n')
        f.write(f"Starting dataset {dataset}...\n")
    # Do the tagging

    filenames = datasets[dataset]
    vars_filename = filenames["vars_filename"]
    templates_filename = filenames["templates_filename"]
    vars_df = pd.read_pickle(vars_filename)
    templates_df = pd.read_pickle(templates_filename)

    # Tag the variables
    tqdm.pandas(desc="Tagging variables...")
    tags = []

    for _, row in tqdm(vars_df.iterrows(), total=len(vars_df)):
        if row["From regex"]:
            tags.append(row["Tag"])
            d[TagOrigin.REGEX_VARIABLE][dataset_num] += 1
        else:
            tag, origin = TagUtils.waterfall_tag(templates_df, row, tags)
            tags.append(tag)
            d[origin][dataset_num] += 1

    # Save to pickle files
    with open(f"tagging_stats_after_{dataset_num+1}_datasets.pkl", "wb") as f:
        pickle.dump(d, f)

    with open(f"{dataset}_tags.pkl", "wb") as f:
        pickle.dump(tags, f)

    # Move on to plotting
    for i in range(dataset_num + 1):
        dataset_total = float(sum(d.values())[i]) / 100.0

        d_scaled[TagOrigin.PRECEDING][i] = d[TagOrigin.PRECEDING][i] / dataset_total
        d_scaled[TagOrigin.GPT_3POINT5_TURBO][i] = (
            d[TagOrigin.GPT_3POINT5_TURBO][i] / dataset_total
        )
        d_scaled[TagOrigin.GPT_4][i] = d[TagOrigin.GPT_4][i] / dataset_total
        d_scaled[TagOrigin.NAME][i] = d[TagOrigin.NAME][i] / dataset_total
        d_scaled[TagOrigin.REGEX_VARIABLE][i] = (
            d[TagOrigin.REGEX_VARIABLE][i] / dataset_total
        )

    with open(f"tagging_stats_scaled_after_{dataset_num+1}_datasets.pkl", "wb") as f:
        pickle.dump(d_scaled, f)

    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(datasets))

    colors = ["#BA9F7F", "#BA7FB7", "#7F9ABA", "#7FBA82", "#D3D3D3"]
    labels = [
        "Already tagged (parsed by regex)",
        "+ Use preceding 3 tokens",
        "+ Use GPT-3.5-Turbo",
        "+ Use GPT-4",
        "Fall back to variable name",
    ]
    order = [4, 0, 1, 2, 3]

    for i, k in enumerate(order):
        v = d_scaled[k]
        p = ax.bar(
            datasets.keys(),
            v,
            width,
            label=labels[i],
            bottom=bottom,
            color=colors[i],
        )

        # Add text labels to the center of each bar
        # Add a white background to make the text more visible
        for bar in []:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2 + bottom,
                f"{(bar.get_height()):.2f} %",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    pad=0.2,
                    alpha=0.85,
                    boxstyle="round",
                ),
            )

        # Make the first bar dotted and second and third slanted in different directions
        for bar in p:
            if k == TagOrigin.REGEX_VARIABLE:
                bar.set_hatch("O")
            elif k == TagOrigin.PRECEDING:
                bar.set_hatch("o")
            elif k == TagOrigin.GPT_3POINT5_TURBO:
                bar.set_hatch("\\\\")
            elif k == TagOrigin.GPT_4:
                bar.set_hatch("//")

        bottom += v

    ax.legend(loc="upper right")
    plt.savefig(f"tagging_stats_after_{dataset_num+1}_datasets.png", dpi=300)
