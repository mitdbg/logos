import sys
import pandas as pd

sys.path.append("../..")
from src.logos.logos import LOGos
from src.definitions import LOGOS_ROOT_DIR
from src.logos.candidate_cause_ranker import CandidateCauseRankerMethod
import datetime
import os
import argparse
import json
from datetime import datetime

ALL_METHODS = ["logos", "regression", "langmodel"]
ALL_DATASETS = ["postgresql", "proprietary", "xyz"]


def setup_logos_for_postgresql(full_filename, workdir):
    # Parse and prepare the log file
    s = LOGos(full_filename, workdir=workdir)
    s.parse(
        regex_dict={
            "Date": r"\d{4}-\d{2}-\d{2}",
            "Time": r"\d{2}:\d{2}:\d{2}\.\d{3}(?= EST \[ )",
            "sessionID": r"(?<=EST \[ )\S+\.\S+",
            "tID": r"3/\d+(?= ] )",
        },
        message_prefix=r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3}",
    )
    # Print out work_mem
    # s.include_in_template('work_mem')
    s.set_causal_unit("sessionID")
    s.prepare(
        count_occurences=True,
        custom_agg={"sessionID": ["mode"]},
    )

    return s, {}


def setup_logos_for_proprietary(full_filename, workdir):
    s = LOGos(full_filename, workdir=workdir)
    s.parse(
        regex_dict=(
            s.DEFAULT_REGEX_DICT
            | {
                "UnixTimestamp": r"16\d{11}(?=\sINFO|\sWARN|\sERROR)",
                "User": r"user_\d+",
            }
        ),
        sim_thresh=0.9,
    )
    s.set_causal_unit("User")
    s.prepare(
        custom_agg={"User": ["mode"], "73b16c0a_196": ["mean"]},
    )

    full_info_filename = full_filename[:-4] + ".json"

    with open(full_info_filename, "r") as fjson:
        info = json.load(fjson)

    return s, {"F": info["faulty_users"], "p_f": info["fail_prob_pct"][0]}


def setup_logos_for_xyz(full_filename, workdir):
    s = LOGos(filename=full_filename, workdir=workdir)
    s.parse(
        regex_dict={
            "timestamp": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
            "machine": r"machine_\d+",
        },
    )
    s.set_causal_unit("machine")
    s.prepare(count_occurences=True)

    full_info_filename = full_filename[:-4] + ".json"

    with open(full_info_filename, "r") as fjson:
        info = json.load(fjson)

    return s, {"V": info["num_total_variables"], "R": info["noise_radius"]}


def candidate_cause_ranking(dataset: str, methods: list[str]):
    # Initialize paths
    dataset_files_dir = os.path.join(LOGOS_ROOT_DIR, "dataset_files", dataset)
    indir = os.path.join(dataset_files_dir, "datasets_raw")
    workdir = os.path.join(dataset_files_dir, "datasets")
    outdir = os.path.join(
        dataset_files_dir, "repro_evaluation", "8.2-candidate-cause-ranking"
    )
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    runlog_path = os.path.join(outdir, f"{dataset}_runlog.txt")

    # Load configuration
    with open(f"./conf_{dataset}.json", "r") as f:
        conf = json.load(f)

    # Initialize output files
    f = open(runlog_path, "w+")
    sys.stdout = f
    df_ranks = {}
    df_lats = {}
    for method in methods:
        df_ranks[method] = pd.DataFrame(columns=conf["ranking_columns"])
        df_lats[method] = pd.DataFrame(columns=conf["latency_columns"])

    # Run this for each file with suffix .log in the given directory
    for filename in os.listdir(indir):
        if filename.endswith(conf["suffix"]):
            full_filename = os.path.join(indir, filename)

            print(f"{datetime.now()} Processing {filename}")
            sys.stderr.write(f"{datetime.now()} Processing {filename}...\n")

            s, info = globals()[f"setup_logos_for_{dataset}"](full_filename, workdir)
            gpt_log_path = os.path.join(outdir, f"{filename[:-4]}_gpt_log.txt")

            # Print statistics as a sanity check
            print("----------------------------------------")
            print("number of parsed templates:", len(s.parsed_templates))
            print("number of parsed variables:", len(s.parsed_variables))
            print("number of prepared variables:", len(s.prepared_variables))

            for method in methods:
                method_obj = CandidateCauseRankerMethod.from_str(method)
                cands, latency = s.rank_candidate_causes(
                    conf["outcome"], method=method_obj, gpt_log_path=gpt_log_path
                )
                print(cands)

                for i, tag in enumerate(list(cands["Candidate Tag"])):
                    d = info.copy()
                    d['rank'] = i + 1
                    d['candidate'] = tag
                    df_ranks[method].loc[len(df_ranks[method])] = d
                d = info.copy()
                d['latency'] = latency
                df_lats[method].loc[len(df_lats[method])] = d

    # Close all files
    f.close()
    for method in methods:
        ranking_path = os.path.join(outdir, f"{dataset}_ranking_{method}.csv")
        latency_path = os.path.join(outdir, f"{dataset}_latency_{method}.csv")
        df_ranks[method].to_csv(ranking_path, index=False)
        df_lats[method].to_csv(latency_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logos", action="store_true")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--langmodel", action="store_true")
    parser.add_argument("--all_methods", action="store_true")
    parser.add_argument("--postgresql", action="store_true")
    parser.add_argument("--proprietary", action="store_true")
    parser.add_argument("--xyz", action="store_true")
    parser.add_argument("--all_datasets", action="store_true")
    args = parser.parse_args()
    methods = [
        method for method in ALL_METHODS if (getattr(args, method) or args.all_methods)
    ]
    datasets = [
        dataset
        for dataset in ALL_DATASETS
        if (getattr(args, dataset) or args.all_datasets)
    ]

    print(f"Methods: {methods}")
    print(f"Datasets: {datasets}")

    for dataset in datasets:
        candidate_cause_ranking(dataset, methods)
