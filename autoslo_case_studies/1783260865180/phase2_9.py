"""
Phase 2.9 — What causes prediction_error?

Treats prediction_error as the *outcome* and asks which pre-execution and
runtime context features causally drive routing-model inaccuracy.

Three groups of candidate causes:
  A. Query plan structural features (operator counts + cardinalities from ICONQ)
  B. n_concurrent_on_cluster_at_start — queries on the same cluster that were
     already executing when this query began execution
  C. n_routed_to_cluster_during_exec — new queries routed to the same cluster
     while this query was executing (mid-run resource contention)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/markakis/chunkbench/src")

import autoslo.filesystem.path_utils as pu
from autoslo.filesystem.logos_export import logos_df
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.slo.slo_resolver import SloResolver
from autoslo.config.component_configs import SloResolverConfig
from autoslo.filesystem.yaml_helpers import load_yaml
from logos import Logos

# ── Config ────────────────────────────────────────────────────────────────────
RUN_ID = "1783260865180"
TAUTOLOGICAL = [
    "actual_execution_latency_s", "final_latency_s",
    "slo_s", "slo_overshoot_s", "relative_violation",
]


def compute_concurrent_load(run_id: str, df_logos: pd.DataFrame) -> pd.DataFrame:
    """
    Add two cluster-load features derived from execution-event timestamps.

    n_concurrent_on_cluster_at_start:
        Number of other queries on the same cluster that had already started
        but not yet finished when this query's execution began.

    n_routed_to_cluster_during_exec:
        Number of other queries routed to the same cluster while this query
        was executing (i.e., arrival of new competition mid-run).
    """
    raw = StructuredLog.load(run_id).flat_df(drop_fwd_queries=True)

    routed = (raw[raw["event_type"] == "query_routed"]
              .set_index("query_id")[["cluster_name", "rel_time_s"]]
              .rename(columns={"rel_time_s": "route_time"}))
    starts = (raw[raw["event_type"] == "query_execution_start"]
              .set_index("query_id")["rel_time_s"].rename("start_time"))
    finishes = (raw[raw["event_type"] == "query_execution_finish"]
                .set_index("query_id")["rel_time_s"].rename("finish_time"))

    qinfo = pd.concat([routed, starts, finishes], axis=1).dropna()

    qids    = qinfo.index.to_numpy()
    cluster = qinfo["cluster_name"].to_numpy()
    rt      = qinfo["route_time"].to_numpy()
    st      = qinfo["start_time"].to_numpy()
    ft      = qinfo["finish_time"].to_numpy()
    n       = len(qids)

    # (i, j): is query j on the same cluster as query i?
    same = cluster[:, None] == cluster[None, :]
    not_self = ~np.eye(n, dtype=bool)

    # Feature B: queries j already running when i's execution starts
    #   j.start <= i.start  AND  j.finish > i.start
    concurrent = (
        same & not_self &
        (st[None, :] <= st[:, None]) &
        (ft[None, :] >  st[:, None])
    ).sum(axis=1)

    # Feature C: queries j routed to same cluster during i's execution
    #   i.start <= j.route_time < i.finish
    routed_during = (
        same & not_self &
        (rt[None, :] >= st[:, None]) &
        (rt[None, :] <  ft[:, None])
    ).sum(axis=1)

    load = pd.DataFrame({
        "n_concurrent_on_cluster_at_start": concurrent,
        "n_routed_to_cluster_during_exec":  routed_during,
    }, index=qids)

    df_logos["n_concurrent_on_cluster_at_start"] = (
        df_logos["query_id"].map(load["n_concurrent_on_cluster_at_start"]))
    df_logos["n_routed_to_cluster_during_exec"] = (
        df_logos["query_id"].map(load["n_routed_to_cluster_during_exec"]))

    return df_logos


# ── Load and augment data ─────────────────────────────────────────────────────
exec_cfg = load_yaml(pu.get_runs_dir() / RUN_ID / "execution_config.yml")
resolver = SloResolver(SloResolverConfig(
    slo_s=exec_cfg["slo_resolver_config"]["slo_s"],
    slo_dict_filename=exec_cfg["slo_resolver_config"]["slo_dict_filename"],
))
df = logos_df(run_id=RUN_ID, slo_resolver=resolver, include_named_query_features=True)
df = df.drop(columns=[c for c in TAUTOLOGICAL if c in df.columns])

print("Computing concurrent-load features...")
df = compute_concurrent_load(RUN_ID, df)

# Spot-check
load_cols = ["n_concurrent_on_cluster_at_start", "n_routed_to_cluster_during_exec"]
per_query = (df[df["event_type"] == "query_routed"]
             .set_index("query_id")[load_cols])
print(f"\n  n_concurrent_on_cluster_at_start:")
print(f"    {per_query['n_concurrent_on_cluster_at_start'].value_counts().sort_index().to_dict()}")
print(f"  n_routed_to_cluster_during_exec:")
print(f"    {per_query['n_routed_to_cluster_during_exec'].value_counts().sort_index().to_dict()}")

# ── Build LOGos ───────────────────────────────────────────────────────────────
plan_cols = [c for c in df.columns if "#" in c]
per_unit  = ["slo_violated", "selected_rpu", "prediction_error"] + plan_cols + load_cols

workdir = str(Path(__file__).parent)

lg = Logos.from_parsed_table(
    data=df,
    workdir=workdir,
    source_id="structured_log_p29",
    template_col="event_type",
    passthrough_cols=["query_id", "query_text_id"],
    per_unit_cols=per_unit,
)
lg.set_causal_unit("query_id")
lg.prepare(default_imp="zero_imp", force=True)

plog = lg.prepared_log
pv   = lg.prepared_variables

print(f"\nPrepared variables : {lg.num_prepared_variables}")
print(f"Causal units       : {len(plog)}\n")


def col(tag: str) -> str:
    rows = pv[pv["Tag"] == tag]
    if rows.empty:
        raise KeyError(f"Tag not found: {tag!r}")
    return rows["Name"].values[0]


pred_tag = "prediction_error mean"
slo_tag  = "slo_violated mean"

pred_col = col(pred_tag)
conc_col = col("n_concurrent_on_cluster_at_start mean")
rdur_col = col("n_routed_to_cluster_during_exec mean")

print("─" * 65)
print("Baseline statistics")
print("─" * 65)
print(f"  prediction_error  mean={plog[pred_col].mean():+.3f}  "
      f"std={plog[pred_col].std():.3f}  "
      f"p90={np.percentile(plog[pred_col], 90):+.3f}")
print(f"  n_concurrent_at_start  mean={plog[conc_col].mean():.2f}  "
      f"max={plog[conc_col].max():.0f}")
print(f"  n_routed_during_exec   mean={plog[rdur_col].mean():.2f}  "
      f"max={plog[rdur_col].max():.0f}")

print("\n  Pearson r with prediction_error:")
print(f"    n_concurrent_at_start : "
      f"{plog[[pred_col, conc_col]].corr().iloc[0,1]:+.4f}")
print(f"    n_routed_during_exec  : "
      f"{plog[[pred_col, rdur_col]].corr().iloc[0,1]:+.4f}")

# ── Candidate-cause ranking for prediction_error ───────────────────────────────
print("\n" + "─" * 65)
print("Candidate causes of prediction_error (LOGos ranking, LASSO-pruned)")
print("─" * 65)
ranked = lg.rank_candidate_causes(pred_tag, prune_candidates=True)
print(ranked.to_string())

# ── Where do the load features rank vs plan features? ─────────────────────────
print("\n" + "─" * 65)
print("Load features vs plan features: raw |r| with prediction_error")
print("─" * 65)
load_prep_names = [conc_col, rdur_col]
plan_prep_names = [
    r["Name"] for _, r in pv.iterrows()
    if any(f in r["Tag"] for f in plan_cols) and r["Name"] in plog.columns
]
all_cands = load_prep_names + plan_prep_names
corr_pred = plog[all_cands].corrwith(plog[pred_col]).abs().sort_values(ascending=False)
for name, r in corr_pred.head(15).items():
    tag_val = pv[pv["Name"] == name]["Tag"].values[0]
    marker  = " ◀ load feature" if name in load_prep_names else ""
    print(f"  {tag_val:<55}  |r|={r:.4f}{marker}")

# ── ATE: load features → prediction_error ─────────────────────────────────────
from logos.exploration.ate_calculator import ATECalculator

print("\n" + "─" * 65)
print("Unadjusted ATE: load features → prediction_error")
print("─" * 65)
for feat_tag in ["n_concurrent_on_cluster_at_start mean",
                 "n_routed_to_cluster_during_exec mean"]:
    try:
        res = ATECalculator.get_ate_and_confidence(
            plog, pv, treatment=feat_tag, outcome=pred_tag,
            calculate_p_value=True, calculate_std_error=True,
        )
        p = res.get("p_value", res.get("P-value", float("nan")))
        se = res.get("std_error", res.get("SE", float("nan")))
        print(f"  {feat_tag:<50}  ATE={res['ATE']:+.5f}  SE={se:.5f}  p={p:.2e}")
    except Exception as e:
        print(f"  {feat_tag}: {e}")

# ── Interpretation ─────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("Interpretation")
print("─" * 65)
load_r = corr_pred.get(conc_col, float("nan"))
routed_r = corr_pred.get(rdur_col, float("nan"))
print(
    f"\n  If |r| for the load features is small (< 0.05), concurrency does not"
    f"\n  explain why the routing model errs.  The dominant causes remain query"
    f"\n  plan structural complexity (operator counts and cardinalities)."
    f"\n"
    f"\n  n_concurrent_at_start |r|={load_r:.4f}"
    f"\n  n_routed_during_exec  |r|={routed_r:.4f}"
    f"\n"
    f"\n  Interpretation: {'concurrency is a meaningful confounder' if max(load_r, routed_r) > 0.05 else 'concurrency has negligible effect; plan complexity dominates'}."
)
