import sys
import pandas as pd

sys.path.append("../..")
from src.logos.logos import LOGos
from src.definitions import LOGOS_ROOT_DIR
from src.logos.interactive_causal_graph_refiner import (
    InteractiveCausalGraphRefinerMethod,
)
import datetime
import os
import argparse
import json
from datetime import datetime

ALL_METHODS = ["logos", "regression", "langmodel"]
ALL_DATASETS = ["postgresql", "xyz"]


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


def get_are_ate(current, true):
    res = abs((current - true) / true) if true != 0 else 0
    print("Calculating ARE_ATE. Current:", current, "True:", true, "Result:", res)
    return res


def interactive_causal_graph_refinement(dataset: str, methods: list[str]):
    # Initialize paths
    dataset_files_dir = os.path.join(LOGOS_ROOT_DIR, "dataset_files", dataset)
    indir = os.path.join(dataset_files_dir, "datasets_raw")
    workdir = os.path.join(dataset_files_dir, "datasets")
    outdir = os.path.join(
        dataset_files_dir, "repro_evaluation", "8.3-interactive-causal-graph-refinement"
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
        df_ranks[method] = pd.DataFrame(columns=conf["results_columns"])
        df_lats[method] = pd.DataFrame(columns=conf["latency_columns"])

    # Run this for each file with suffix .log in the given directory
    for filename in os.listdir(indir):
        if filename.endswith(conf["suffix"]):
            full_filename = os.path.join(indir, filename)

            print(f"{datetime.now()} Processing {filename}")
            sys.stderr.write(f"{datetime.now()} Processing {filename}...\n")
            gpt_log_path = os.path.join(outdir, f"{filename[:-4]}_gpt_log.txt")

            # Do it once for the ground truth
            s, info = globals()[f"setup_logos_for_{dataset}"](full_filename, workdir)

            print("----------------------------------------")
            print("number of parsed templates:", len(s.parsed_templates))
            print("number of parsed variables:", len(s.parsed_variables))
            print("number of prepared variables:", len(s.prepared_variables))

            for edge in conf["true_graph_edges"]:
                s.accept(edge[0], edge[1], also_fix=False)
            ground_truth_ate = s.get_adjusted_ate(conf["treatment"], conf["outcome"])

            for method in methods:
                # Now do it once for real
                s, info = globals()[f"setup_logos_for_{dataset}"](
                    full_filename, workdir
                )

                print("----------------------------------------")
                print("number of parsed templates:", len(s.parsed_templates))
                print("number of parsed variables:", len(s.parsed_variables))
                print("number of prepared variables:", len(s.prepared_variables))

                for edge in conf["starting_graph_edges"]:
                    s.accept(edge[0], edge[1], also_fix=False, interactive=False)

                ate = s.get_adjusted_ate(conf["treatment"], conf["outcome"])
                d = info.copy()
                d["judgments"] = 0
                d["edge"] = ""
                d["ATE"] = ate
                d["ARE_ATE"] = get_are_ate(ate, ground_truth_ate)
                df_ranks[method].loc[len(df_ranks[method])] = d

                method_obj = InteractiveCausalGraphRefinerMethod.from_str(method)

                j = 0

                while (
                    df_ranks[method].loc[len(df_ranks[method]) - 1, "ARE_ATE"] > 10e-5
                ):
                    edge, latency = s.get_causal_graph_refinement_suggestion(
                        method_obj,
                        conf["treatment"],
                        conf["outcome"],
                        gpt_log_path=gpt_log_path,
                    )
                    print(f"Edge: {edge}")

                    if edge is not None:
                        if list(edge) in conf["true_graph_edges"]:
                            print("This edge is in the ground truth graph")
                            s.accept(edge[0], edge[1], also_fix=True, interactive=False)
                            s.reject(edge[1], edge[0], also_ban=True, interactive=False)
                        elif list(edge)[::-1] in conf["true_graph_edges"]:
                            print(
                                "The inverse of this edge is in the ground truth graph"
                            )
                            s.reject(edge[0], edge[1], also_ban=True, interactive=False)
                            s.accept(edge[1], edge[0], also_fix=True, interactive=False)
                        else:
                            print("This edge is not in the ground truth graph")
                            s.reject(edge[0], edge[1], also_ban=True, interactive=False)
                            s.reject(edge[1], edge[0], also_ban=True, interactive=False)

                    print("After updating the graph, it now has edges:")
                    print(s._graph.edges)

                    ate = s.get_adjusted_ate(conf["treatment"], conf["outcome"])

                    d = info.copy()
                    d["judgments"] = j + 1
                    d["edge"] = edge
                    d["ATE"] = ate
                    d["ARE_ATE"] = get_are_ate(ate, ground_truth_ate)
                    df_ranks[method].loc[len(df_ranks[method])] = d

                    d = info.copy()
                    d["judgments"] = j + 1
                    d["latency"] = latency
                    df_lats[method].loc[len(df_lats[method])] = d

                    j += 1

    # Close all files
    f.close()
    for method in methods:
        results_path = os.path.join(outdir, f"{dataset}_results_{method}.csv")
        latency_path = os.path.join(outdir, f"{dataset}_latency_{method}.csv")
        df_ranks[method].to_csv(results_path, index=False)
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
        interactive_causal_graph_refinement(dataset, methods)
