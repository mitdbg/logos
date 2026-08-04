from pathlib import Path

import yaml
from autoslo.config.component_configs import SloResolverConfig
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.slo.slo_resolver import SloResolver

from logos import Logos

HERE = Path(__file__).parent
EXEC_CFG = (
    "/home/markakis/chunkbench/data/runs/1783260865180/execution_config.yml"
)

with open(EXEC_CFG) as f:
    exec_cfg = yaml.safe_load(f)

resolver = SloResolver(SloResolverConfig(**exec_cfg["slo_resolver_config"]))
slog = StructuredLog.load(HERE / "structured_log.parquet")
df = slog.logos_df(slo_resolver=resolver)

outcomes = slog.query_slo_outcomes(resolver)
print(
    f"Queries: {len(outcomes)}, violation rate: {outcomes['slo_violated'].mean():.1%}, "
    f"max overshoot: {outcomes['slo_overshoot_s'].max():.1f}s"
)

lg = Logos.from_parsed_table(
    data=df, workdir=str(HERE), source_id="structured_log"
)
lg.set_causal_unit("query_id")
custom_imp = {tag: "zero_imp" for tag in lg.parsed_variables["Tag"]}
lg.prepare(custom_imp=custom_imp)

print(f"\nPrepared variables: {lg.num_prepared_variables}")

print("\n--- Candidate causes of slo_violated ---")
print(
    lg.rank_candidate_causes(
        "slo_violated mean", prune_candidates=True
    ).to_string()
)

print("\n--- Candidate causes of slo_overshoot_s ---")
print(
    lg.rank_candidate_causes(
        "slo_overshoot_s mean", prune_candidates=True
    ).to_string()
)
