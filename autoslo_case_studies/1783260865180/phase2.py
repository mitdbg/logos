import sys
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/home/markakis/chunkbench/src')

from autoslo.filesystem.structured_log import StructuredLog
from autoslo.slo.slo_resolver import SloResolver
from autoslo.config.component_configs import SloResolverConfig
from logos import Logos

HERE = Path(__file__).parent
EXEC_CFG = '/home/markakis/chunkbench/data/runs/1783260865180/execution_config.yml'

with open(EXEC_CFG) as f:
    exec_cfg = yaml.safe_load(f)

resolver = SloResolver(SloResolverConfig(**exec_cfg['slo_resolver_config']))
slog = StructuredLog.load(HERE / 'structured_log.parquet')
df = slog.logos_df(slo_resolver=resolver)

per_unit = [
    "slo_violated", "slo_s", "slo_overshoot_s", "relative_violation",
    "final_latency_s", "actual_execution_latency_s",
    "selected_cluster_name", "selected_rpu", "prediction_error",
]

lg = Logos.from_parsed_table(
    data=df,
    workdir=str(HERE),
    source_id="structured_log_p2",
    template_col="event_type",
    passthrough_cols=["query_id", "query_text_id"],
    per_unit_cols=per_unit,
)
lg.set_causal_unit("query_id")
custom_imp = {tag: "zero_imp" for tag in lg.parsed_variables["Tag"]}
lg.prepare(custom_imp=custom_imp)

print(f"\nPrepared variables: {lg.num_prepared_variables}")

print("\n--- Candidate causes of slo_violated ---")
print(lg.rank_candidate_causes('slo_violated mean', prune_candidates=True).to_string())

print("\n--- Candidate causes of prediction_error ---")
print(lg.rank_candidate_causes('prediction_error mean', prune_candidates=True).to_string())
