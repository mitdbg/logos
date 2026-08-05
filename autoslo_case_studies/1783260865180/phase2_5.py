from pathlib import Path

import yaml
import pandas as pd
from autoslo.config.component_configs import SloResolverConfig
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.slo.slo_resolver import SloResolver

from logos import Logos

HERE = Path(__file__).parent
EXEC_CFG = "/home/markakis/chunkbench/data/runs/1783260865180/execution_config.yml"

with open(EXEC_CFG) as f:
    exec_cfg = yaml.safe_load(f)

resolver = SloResolver(SloResolverConfig(**exec_cfg["slo_resolver_config"]))
slog = StructuredLog.load(HERE / "structured_log.parquet")
df = slog.logos_df(slo_resolver=resolver)

# ---------------------------------------------------------------------------
# 1. Raw split — unadjusted violation rate by RPU
# ---------------------------------------------------------------------------
assignments = slog.query_cluster_assignments()
outcomes = slog.query_slo_outcomes(resolver)
merged = outcomes.merge(assignments[["query_id", "rpu"]], on="query_id")

print("=== Unadjusted violation rate by RPU ===")
split = merged.groupby("rpu")["slo_violated"].agg(["mean", "count"]).rename(
    columns={"mean": "violation_rate", "count": "n_queries"}
)
split["violation_rate"] = split["violation_rate"].map("{:.1%}".format)
print(split.to_string())

# ---------------------------------------------------------------------------
# 2. Per-template breakdown — does RPU hurt some templates more than others?
# ---------------------------------------------------------------------------
merged["template"] = merged["query_text_id"].str.extract(r"#(\d+)#")
by_template = (
    merged.groupby(["template", "rpu"])["slo_violated"]
    .mean()
    .unstack(level="rpu")
    .dropna()
    .sort_values(by=merged["rpu"].max(), ascending=False)
)
by_template.columns = [f"viol_rate_rpu{int(c)}" for c in by_template.columns]
by_template["delta_32_minus_16"] = (
    by_template.get("viol_rate_rpu32", 0) - by_template.get("viol_rate_rpu16", 0)
)
print("\n=== Violation rate by template × RPU (templates served by both RPU sizes) ===")
print(by_template.to_string(float_format="{:.2%}".format))

# ---------------------------------------------------------------------------
# 3. Adjusted ATE via Logos — controlling for arrival time
#
# 16-RPU clusters were added later in the run when congestion was already
# high, so raw RPU-violation correlation is confounded by wall_clock_s.
# get_adjusted_ate(RPU, violation | wall_clock_s) isolates the direct RPU
# effect after removing that temporal confound.
# ---------------------------------------------------------------------------
lg = Logos.from_parsed_table(
    data=df, workdir=str(HERE), source_id="structured_log"
)
lg.set_causal_unit("query_id")
custom_imp = {tag: "zero_imp" for tag in lg.parsed_variables["Tag"]}
lg.prepare(custom_imp=custom_imp, force=True)

# Locate selected_rpu in the slo_violated ranking to get unadjusted p-value.
ranking = lg.rank_candidate_causes("slo_violated mean", prune_candidates=False)
rpu_row = ranking[ranking["Candidate Tag"] == "selected_rpu mean"]
print("\n=== selected_rpu in slo_violated ranking (unadjusted) ===")
if rpu_row.empty:
    # fall back to inspecting whatever tag was assigned
    rpu_candidates = ranking[ranking["Candidate Tag"].str.contains("selected_rpu", na=False)]
    print(rpu_candidates[["Candidate Tag", "Slope", "P-value"]].to_string() or "  (not found)")
else:
    print(rpu_row[["Candidate Tag", "Slope", "P-value"]].to_string())

# Adjusted ATE: per-RPU-unit change in violation probability, holding
# wall_clock_s constant.  Multiply by (32-16)=16 for the total 16→32 effect.
ate = lg.get_adjusted_ate(
    "selected_rpu mean",
    "slo_violated mean",
    confounder="wall_clock_s mean",
)
total_effect = ate * 16
print(f"\n=== Adjusted ATE of RPU on slo_violated (confounder: wall_clock_s) ===")
print(f"  Slope (per 1 RPU unit):        {ate:+.6f}")
print(f"  Total effect (16-RPU → 32-RPU): {total_effect:+.4f}  "
      f"({'32-RPU reduces violations' if total_effect < 0 else '32-RPU increases violations'})")
