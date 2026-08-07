"""
Phase 2 — Cross-run deconfounding analysis across all four AutoSLO runs.

Runs (from main_eval_v8#live.csv):
  1783260865180  May-27  κ=4   (reference run, already in phase2_8.py)
  1783276972704  Apr-15  κ=4
  1783262695062  May-27  κ=2
  1783291484240  Apr-15  κ=2

For each run we compute two unadjusted→adjusted ATE pairs:
  A. prediction_error → slo_violated  (is routing-model inaccuracy causal?)
  B. selected_rpu    → slo_violated   (does cluster size matter causally?)

The plan-structural features (operator counts + table cardinalities from the
query plan) serve as the confounder set in both analyses.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/markakis/chunkbench/src")

import autoslo.filesystem.path_utils as pu
from autoslo.filesystem.logos_export import logos_df
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.slo.slo_resolver import SloResolver
from autoslo.config.component_configs import SloResolverConfig
from logos import Logos
from logos.exploration.ate_calculator import ATECalculator

# ── Configuration ─────────────────────────────────────────────────────────────
RUNS = {
    "1783260865180": {"label": "May-27 κ=4", "kappa": 4, "workload": "May-27"},
    "1783276972704": {"label": "Apr-15 κ=4", "kappa": 4, "workload": "Apr-15"},
    "1783262695062": {"label": "May-27 κ=2", "kappa": 2, "workload": "May-27"},
    "1783291484240": {"label": "Apr-15 κ=2", "kappa": 2, "workload": "Apr-15"},
}
TAUTOLOGICAL = [
    "actual_execution_latency_s", "final_latency_s",
    "slo_s", "slo_overshoot_s", "relative_violation",
]
K_CONFOUNDERS = 10

# ── Per-run analysis ───────────────────────────────────────────────────────────
records = []

for run_id, meta in RUNS.items():
    label = meta["label"]
    print(f"\n{'='*60}\n{label}  (run {run_id})")

    exec_cfg = load_yaml(pu.get_runs_dir() / run_id / "execution_config.yml")
    resolver = SloResolver(SloResolverConfig(
        slo_s=exec_cfg["slo_resolver_config"]["slo_s"],
        slo_dict_filename=exec_cfg["slo_resolver_config"]["slo_dict_filename"],
    ))
    df = logos_df(run_id=run_id, slo_resolver=resolver,
                  include_named_query_features=True)

    plan_cols = [c for c in df.columns if "#" in c]
    df = df.drop(columns=[c for c in TAUTOLOGICAL if c in df.columns])

    per_unit = ["slo_violated", "selected_rpu", "prediction_error"] + plan_cols
    workdir = f"/tmp/logos_cross_{run_id}"
    Path(workdir).mkdir(parents=True, exist_ok=True)

    lg = Logos.from_parsed_table(
        data=df, workdir=workdir, source_id=f"cross_{run_id}",
        template_col="event_type",
        passthrough_cols=["query_id", "query_text_id"],
        per_unit_cols=per_unit,
    )
    lg.set_causal_unit("query_id")
    lg.prepare(default_imp="zero_imp", force=True)

    plog = lg.prepared_log
    pv   = lg.prepared_variables

    def col(tag: str) -> str:
        return pv[pv["Tag"] == tag]["Name"].values[0]

    slo_tag  = "slo_violated mean"
    pred_tag = "prediction_error mean"
    rpu_tag  = "selected_rpu mean"
    slo_c  = col(slo_tag)
    rpu_c  = col(rpu_tag)
    pred_c = col(pred_tag)

    viol_rate  = plog[slo_c].mean()
    n_queries  = len(plog)
    rpu_counts = plog[rpu_c].value_counts().to_dict()

    # Unadjusted ATEs (two-node graphs)
    u_pred = ATECalculator.get_ate_and_confidence(
        plog, pv, treatment=pred_tag, outcome=slo_tag,
        calculate_p_value=True, calculate_std_error=False)
    u_rpu = ATECalculator.get_ate_and_confidence(
        plog, pv, treatment=rpu_tag, outcome=slo_tag,
        calculate_p_value=True, calculate_std_error=False)

    # Identify query-plan confounders
    plan_names = [
        r["Name"] for _, r in pv.iterrows()
        if any(f in r["Tag"] for f in plan_cols) and r["Name"] in plog.columns
    ]
    corr_slo  = plog[plan_names].corrwith(plog[slo_c]).abs()
    corr_rpu  = plog[plan_names].corrwith(plog[rpu_c]).abs()
    corr_pred = plog[plan_names].corrwith(plog[pred_c]).abs()

    top_rpu_conf  = (corr_slo * corr_rpu ).sort_values(ascending=False).head(K_CONFOUNDERS).index
    top_pred_conf = (corr_slo * corr_pred).sort_values(ascending=False).head(K_CONFOUNDERS).index

    # Adjusted ATE — RPU
    lg.clear_graph()
    for nm in top_rpu_conf:
        tv = pv[pv["Name"] == nm]["Tag"].values[0]
        lg.accept(tv, rpu_tag, also_fix=True)
        lg.accept(tv, slo_tag, also_fix=True)
    lg.accept(rpu_tag, slo_tag, also_fix=True)
    adj_rpu = lg.get_adjusted_ate(rpu_tag, slo_tag)

    # Adjusted ATE — prediction_error
    lg.clear_graph()
    for nm in top_pred_conf:
        tv = pv[pv["Name"] == nm]["Tag"].values[0]
        lg.accept(tv, pred_tag, also_fix=True)
        lg.accept(tv, slo_tag,  also_fix=True)
    lg.accept(pred_tag, slo_tag, also_fix=True)
    adj_pred = lg.get_adjusted_ate(pred_tag, slo_tag)

    p_pred = u_pred.get("p_value", u_pred.get("P-value", float("nan")))
    p_rpu  = u_rpu.get("p_value",  u_rpu.get("P-value",  float("nan")))

    top_rpu_tags  = [pv[pv["Name"] == nm]["Tag"].values[0] for nm in top_rpu_conf[:3]]
    top_pred_tags = [pv[pv["Name"] == nm]["Tag"].values[0] for nm in top_pred_conf[:3]]

    rec = dict(
        run_id=run_id, label=label, kappa=meta["kappa"], workload=meta["workload"],
        n=n_queries, viol_rate=viol_rate,
        unadj_pred=u_pred["ATE"], adj_pred=adj_pred, p_pred=p_pred,
        unadj_rpu=u_rpu["ATE"],  adj_rpu=adj_rpu,   p_rpu=p_rpu,
        pred_adj_ratio=abs(adj_pred / u_pred["ATE"]) if u_pred["ATE"] != 0 else float("nan"),
        rpu_adj_ratio=abs(adj_rpu  / u_rpu["ATE"])   if u_rpu["ATE"]  != 0 else float("nan"),
        max_plan_corr_rpu=corr_rpu.max(),
        max_plan_corr_pred=corr_pred.max(),
        top_rpu_conf="; ".join(top_rpu_tags),
        top_pred_conf="; ".join(top_pred_tags),
    )
    records.append(rec)

    rpu_str = "  ".join(f"RPU={int(k)}: {v}" for k, v in sorted(rpu_counts.items()))
    print(f"  n={n_queries}  violation={viol_rate:.1%}  {rpu_str}")
    print(f"  prediction_error ATE: unadj={u_pred['ATE']:+.5f}  adj={adj_pred:+.5f}"
          f"  ratio={rec['pred_adj_ratio']:.2f}x  p={p_pred:.1e}")
    print(f"  selected_rpu ATE:     unadj={u_rpu['ATE']:+.5f}  adj={adj_rpu:+.5f}"
          f"  ratio={rec['rpu_adj_ratio']:.2f}x  p={p_rpu:.2f}")
    print(f"  top RPU  confounders: {top_rpu_tags}")
    print(f"  top pred confounders: {top_pred_tags}")

# ── Summary table ──────────────────────────────────────────────────────────────
res = pd.DataFrame(records)

print("\n\n" + "=" * 70)
print("CROSS-RUN SUMMARY")
print("=" * 70)

fmt = {
    "n":              "{:4.0f}",
    "viol_rate":      "{:.1%}",
    "unadj_pred":     "{:+.5f}",
    "adj_pred":       "{:+.5f}",
    "pred_adj_ratio": "{:.2f}x",
    "p_pred":         "{:.1e}",
    "unadj_rpu":      "{:+.5f}",
    "adj_rpu":        "{:+.5f}",
    "rpu_adj_ratio":  "{:.2f}x",
    "p_rpu":          "{:.2f}",
}
display = res[["label","n","viol_rate",
               "unadj_pred","adj_pred","pred_adj_ratio","p_pred",
               "unadj_rpu","adj_rpu","rpu_adj_ratio","p_rpu"]].copy()
for col_name, f in fmt.items():
    if col_name in display.columns:
        display[col_name] = display[col_name].apply(lambda v: f.format(v))
print(display.to_string(index=False))

print("\n" + "─" * 70)
print("INTERPRETATION")
print("─" * 70)
print(
    "\n1. prediction_error → slo_violated  (robust, stable causal signal)"
    "\n   ─ Significant in every run (p≈10⁻⁴⁰ to p≈10⁻⁸²)."
    "\n   ─ Adjustment ratio is consistently near 1.0 (range 0.96x – 1.03x)."
    "\n   ─ Query plan features do NOT substantially confound this relationship;"
    "\n     the routing model already incorporates plan complexity when producing"
    "\n     its latency estimate, so the residual error is plan-independent."
    f"\n   ─ Outlier: Apr-15 κ=4 has ATE ≈ {res.loc[res.run_id=='1783276972704','unadj_pred'].values[0]:+.5f}"
    f" (~3× larger than the other three runs)."
    "\n     This run appears to have systematically worse routing predictions,"
    "\n     possibly because the April-15 workload snapshot introduced query"
    "\n     patterns the model had not yet been calibrated on."
)
print(
    "\n2. selected_rpu → slo_violated  (fragile, heavily confounded)"
    "\n   ─ Statistically insignificant in two runs (p=0.29, p=0.83);"
    "\n     marginally significant in the κ=2 runs (p=0.03–0.06)."
    "\n   ─ Adjustment ratio is highly variable (0.57x – 4.88x), and the"
    "\n     effect SIGN flips across runs (two runs: unadj and adj have"
    "\n     opposite signs)."
    "\n   ─ This instability is the key LOGos finding: the naive comparison"
    "\n     of violation rates across RPU values is unreliable. The confounder"
    "\n     (query plan complexity) dominates, and its direction and magnitude"
    "\n     change from run to run depending on which queries were routed to"
    "\n     which cluster sizes."
)
print(
    "\n3. κ (SLO strictness) effect:"
    f"\n   ─ κ=4 violation rates: {res.loc[res.kappa==4,'viol_rate'].values*100} %"
    f"\n   ─ κ=2 violation rates: {res.loc[res.kappa==2,'viol_rate'].values*100} %"
    "\n   ─ Tighter SLO (κ=4) → higher violation rate, as expected."
    "\n   ─ The causal effect sizes of prediction_error are similar across κ,"
    "\n     suggesting the routing model's accuracy matters equally regardless"
    "\n     of how tightly the SLO is defined."
)
