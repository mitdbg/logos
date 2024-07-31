import sys
import os

sys.path.append("../../../")
from src.logos.logos import LOGos
from src.definitions import LOGOS_ROOT_DIR
from gen_scaling_logs import gen_log


def main():
    variable_min = 1
    variable_max = 10
    variables = [10 * i for i in range(variable_min, variable_max + 1)]

    resultsdir = os.path.join(LOGOS_ROOT_DIR, "evaluation", "results")
    workdir = os.path.join(LOGOS_ROOT_DIR, "dataset_files", "scaling", "variables")
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    f = open("runlog-variables.txt", "w+")
    fr1 = open(os.path.join(resultsdir, "8.4.1-scalability-variables.csv"), "w+")
    fr1.write("variables, Parse Time, Prep Time\n")
    sys.stdout = f

    for v in variables:
        # Generate log
        L = 10000
        S = 10
        V = v
        C = 100 - v
        filename = os.path.join(workdir, f"log_{v}.log")
        gen_log(L, S, V, C, filename)
        print(f"Generated log with {v} variables")

        # Analyze log
        s = LOGos(filename, workdir=workdir, skip_writeout=True)
        parse_time = s.parse(
            regex_dict={"LineID": r"line_\d+"}, sim_thresh=((C + 2) / 102), force=True
        )
        print(f"Shape of parsed log: {s.parsed_log.shape}")
        s.set_causal_unit("LineID")
        d = {k: "zero_imp" for k in s.parsed_log.columns[2:]}
        prep_time = s.prepare(
            custom_agg={"LineID": ["mode"]},
            custom_imp=d,
            ignore_uninteresting=False,
            force=True,
            drop_bad_aggs=False,
            reject_prunable_edges=False,
        )
        print(f"Shape of prepared log: {s.prepared_log.shape}")
        s.prepared_log.head(10)

        fr1.write(f"{v},{parse_time},{prep_time}\n")
        fr1.flush()
        f.flush()

    f.close()
    fr1.close()


if __name__ == "__main__":
    main()
