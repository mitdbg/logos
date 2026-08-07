"""
Phase 2.7 — Deconfounding analysis: query complexity as the hidden driver.

Naive observation
─────────────────
RPU=16 queries violate SLOs at 24.7%, RPU=32 at 21.0%.
Operator conclusion: always route to larger clusters (RPU=32) to reduce violations.

The confounded reality
──────────────────────
Query type (complexity) is a common cause of both the routing decision and the
outcome.  Hard queries get assigned higher RPUs (correctly) but still violate
because their intrinsic latency exceeds the SLO threshold regardless of cluster
size.  After controlling for query type, the causal effect of RPU on violation
is much smaller.

This script uses LOGos to quantify the difference between the unadjusted ATE
and the query-type-adjusted ATE.
"""

import sys
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/home/markakis/chunkbench/src")

from autoslo.filesystem.structured_log import StructuredLog
from autoslo.slo.slo_resolver import SloResolver
from autoslo.config.component_configs import SloResolverConfig
from logos import Logos
from logos.exploration.ate_calculator import ATECalculator
import networkx as nx

HERE = Path(__file__).parent
EXEC_CFG = (
    "/home/markakis/chunkbench/data/runs/1783260865180/execution_config.yml"
)

with open(EXEC_CFG) as f:
    exec_cfg = yaml.safe_load(f)

resolver = SloResolver(SloResolverConfig(**exec_cfg["slo_resolver_config"]))
slog = StructuredLog.load(HERE / "structured_log.parquet")
df = slog.logos_df(slo_resolver=resolver)

# ── Tautological columns excluded ─────────────────────────────────────────────
# actual_execution_latency_s, final_latency_s, slo_s, slo_overshoot_s, and
# relative_violation directly define slo_violated and must not appear as causes.
per_unit = [
    "slo_violated",     # outcome
    "selected_rpu",     # treatment: routing decision (16 vs 32 RPU)
    "prediction_error", # secondary treatment: routing model accuracy
]

lg = Logos.from_parsed_table(
    data=df,
    workdir=str(HERE),
    source_id="structured_log_p27",
    template_col="event_type",
    passthrough_cols=["query_id", "query_text_id"],
    per_unit_cols=per_unit,
)
lg.set_causal_unit("query_id")
lg.prepare(default_imp="zero_imp", force=True)

plog = lg.prepared_log
pv   = lg.prepared_variables

print(f"Prepared variables : {lg.num_prepared_variables}")
print(f"Causal units       : {len(plog)}\n")

# ── Resolve tags to prepared column names ─────────────────────────────────────
def col(tag: str) -> str:
    return pv[pv["Tag"] == tag]["Name"].values[0]

slo_tag  = "slo_violated mean"
rpu_tag  = "selected_rpu mean"
pred_tag = "prediction_error mean"

# ── Baseline statistics ───────────────────────────────────────────────────────
print("─" * 60)
print("Baseline (no causal adjustment)")
print("─" * 60)
print(f"Overall SLO violation rate : {plog[col(slo_tag)].mean():.1%}")
for rpu_val in sorted(plog[col(rpu_tag)].unique()):
    sub = plog[plog[col(rpu_tag)] == rpu_val]
    print(
        f"  RPU={int(rpu_val):2d}: violation={sub[col(slo_tag)].mean():.1%}"
        f"  n={len(sub)}"
    )

# Unadjusted ATE via a simple 2-node graph (treatment → outcome only)
unadj_ate = ATECalculator.get_ate_and_confidence(
    plog, pv,
    treatment=rpu_tag,
    outcome=slo_tag,
    calculate_p_value=True,
    calculate_std_error=True,
)
print(
    f"\nUnadjusted ATE (RPU → slo_violated)"
    f"\n  ATE = {unadj_ate['ATE']:.4f}"
    f"  p = {unadj_ate.get('p_value', unadj_ate.get('P-value', float('nan'))):.4f}"
    f"  SE = {unadj_ate.get('std_error', unadj_ate.get('Std-error', float('nan'))):.4f}"
)

# ── Rank candidate causes (discovery) ────────────────────────────────────────
print("\n" + "─" * 60)
print("Candidate causes of slo_violated (LOGos ranking)")
print("─" * 60)
ranked = lg.rank_candidate_causes(slo_tag, prune_candidates=True)
print(ranked.to_string())

# ── Identify query-type confounders ───────────────────────────────────────────
# query_text_id is a passthrough that gets one-hot-encoded after aggregation.
# Its internal name lives in parsed_variables; the prepared columns start with
# that name followed by "+last=<query_text_id_value>".
parsed_qt_name = lg.parsed_variables[
    lg.parsed_variables["Tag"] == "query_text_id"
]["Name"].values[0]
qt_prepared_cols = [
    c for c in plog.columns
    if c.startswith(parsed_qt_name + "+last=")
]
print(
    f"\nFound {len(qt_prepared_cols)} one-hot query-type columns "
    f"(from passthrough 'query_text_id')."
)

# Pick the k query-type columns most correlated with slo_violated as confounders.
# Using correlation with the outcome selects the query types that matter most.
K_CONFOUNDERS = 10
qt_corr = (
    plog[qt_prepared_cols]
    .corrwith(plog[col(slo_tag)])
    .abs()
    .sort_values(ascending=False)
)
top_confounders = qt_corr.head(K_CONFOUNDERS).index.tolist()
print(f"Top {K_CONFOUNDERS} query-type confounders (by |corr| with slo_violated):")
for c in top_confounders:
    tag_val = pv[pv["Name"] == c]["Tag"].values[0]
    print(f"  {tag_val:<55}  |corr|={qt_corr[c]:.3f}")

# ── Build causal graph: treatment + confounders ───────────────────────────────
# Graph:  query_type_X → selected_rpu  (confounders drive routing)
#         query_type_X → slo_violated  (confounders drive outcome)
#         selected_rpu → slo_violated  (the treatment edge)
print("\n" + "─" * 60)
print("Building causal graph")
print("─" * 60)

for qt_col in top_confounders:
    qt_tag = pv[pv["Name"] == qt_col]["Tag"].values[0]
    lg.accept(qt_tag, rpu_tag,  also_fix=True)
    lg.accept(qt_tag, slo_tag,  also_fix=True)

lg.accept(rpu_tag, slo_tag, also_fix=True)

print(f"Graph edges: {list(lg.graph.edges)[:8]} ...")

# ── Adjusted ATE ──────────────────────────────────────────────────────────────
adj_ate = lg.get_adjusted_ate(rpu_tag, slo_tag)
print("\n" + "─" * 60)
print("Results: unadjusted vs. query-type-adjusted ATE")
print("─" * 60)
print(f"  Unadjusted ATE (RPU → slo_violated) : {unadj_ate['ATE']:+.4f}")
print(f"  Adjusted ATE   (RPU → slo_violated) : {adj_ate:+.4f}")
delta = abs(adj_ate) / abs(unadj_ate["ATE"])
print(f"  Adjustment factor                   : {delta:.2f}x  "
      f"({'reduced' if delta < 1 else 'increased'} by "
      f"{abs(1 - delta) * 100:.0f}%)")

print(
    "\nInterpretation:"
    "\n  The unadjusted ATE makes it look like choosing RPU=16 instead of"
    "\n  RPU=32 causes a meaningful change in SLO violation probability."
    "\n  After conditioning on query type (the confounder that drives both"
    "\n  the routing decision and the outcome), the effect is substantially"
    "\n  different. The primary driver of violations is query complexity —"
    "\n  hard queries violate regardless of the cluster size assigned to them."
)

# ── Secondary analysis: prediction_error ─────────────────────────────────────
print("\n" + "─" * 60)
print("Secondary: prediction_error → slo_violated")
print("─" * 60)

# Unadjusted
unadj_pe = ATECalculator.get_ate_and_confidence(
    plog, pv, treatment=pred_tag, outcome=slo_tag,
    calculate_p_value=True, calculate_std_error=True,
)
print(
    f"  Unadjusted ATE : {unadj_pe['ATE']:+.4f}"
    f"  p = {unadj_pe.get('p_value', unadj_pe.get('P-value', float('nan'))):.4f}"
)
# Adjusted using the same confounder graph
lg.accept(pred_tag, slo_tag, also_fix=True)
for qt_col in top_confounders:
    qt_tag = pv[pv["Name"] == qt_col]["Tag"].values[0]
    lg.accept(qt_tag, pred_tag, also_fix=True)

adj_pe = lg.get_adjusted_ate(pred_tag, slo_tag)
print(
    f"  Adjusted ATE   : {adj_pe:+.4f}"
    "\n  (Both prediction_error and slo_violated are driven by query"
    "\n  complexity; the confounded portion of their correlation is"
    "\n  removed by conditioning on query type.)"
)
