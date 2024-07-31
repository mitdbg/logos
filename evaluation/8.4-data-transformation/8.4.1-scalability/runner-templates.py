import sys
import os

sys.path.append("../../../")
from src.logos.logos import LOGos
from src.definitions import LOGOS_ROOT_DIR
from gen_scaling_logs import gen_log


def main():
    template_exp_min = 1
    template_exp_max = 4
    templates = [10**i for i in range(template_exp_min, template_exp_max + 1)]

    resultsdir = os.path.join(LOGOS_ROOT_DIR, "evaluation", "results")
    workdir = os.path.join(LOGOS_ROOT_DIR, "dataset_files", "scaling", "templates")
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    f = open("runlog-templates.txt", "w+")
    fr1 = open(os.path.join(resultsdir, "8.4.1-scalability-templates.csv"), "w+")
    fr1.write("Templates, Parse Time, Prep Time\n")
    sys.stdout = f

    for t in templates:
        # Generate log
        L = 10000
        S = t
        V = 1
        C = 10
        filename = os.path.join(workdir, f"log_{t}.log")
        gen_log(L, S, V, C, filename)
        print(f"Generated log with {t} templates")

        # Analyze log
        s = LOGos(filename, workdir=workdir, skip_writeout=True)
        parse_time = s.parse(
            regex_dict={"LineID": r"line_\d+"}, sim_thresh=12 / 13, force=True
        )
        print(f"Number of templates: {len(s.parsed_templates)}")
        print(f"Number of parsed variables: {len(s.parsed_variables)}")
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

        fr1.write(f"{t},{parse_time},{prep_time}\n")
        fr1.flush()
        f.flush()

    f.close()
    fr1.close()


if __name__ == "__main__":
    main()
