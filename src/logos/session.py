"""
Session persistence: save and load a full LOGos pipeline state to/from disk.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from logos import Logos
from logos.cache import (
    dump_dataframe,
    dump_metadata,
    load_dataframe,
    load_metadata,
)
from logos.parser import Parser

_logger = logging.getLogger(__name__)


class Session:

    _DEFAULT_SESSION_BASE = Path.home() / ".logos" / "sessions"

    @staticmethod
    def save_session(logos: "Logos", path: Optional[str] = None) -> str:
        """
        Serialize the full LOGos pipeline state to a directory.

        Parameters:
            logos: The Logos instance to save.
            path: Destination directory. Defaults to
                ~/.logos/sessions/<unix_timestamp>/.

        Returns:
            The absolute path of the directory where the session was saved.
        """
        if path is None:
            path = str(Session._DEFAULT_SESSION_BASE / str(int(time.time())))

        os.makedirs(path, exist_ok=True)
        manifest: dict = {}

        # Determine workdir and entry-point type
        if logos._parser is None:
            manifest["entry_point"] = "EP-3"
            workdir = (
                logos._explorer._workdir
                if logos._explorer is not None
                else path
            )
        elif isinstance(logos._parser, Parser):
            manifest["entry_point"] = "EP-1"
            manifest["source_path"] = logos._parser.filename
            workdir = logos._parser.workdir
        else:  # ParsedDataFrameSource
            manifest["entry_point"] = "EP-2"
            manifest["source_path"] = logos._parser.filename  # source_id
            workdir = logos._parser.workdir

        manifest["workdir"] = workdir

        # Save parsed artefacts when available and non-empty
        if logos._parser is not None and not logos._parser.parsed_log.empty:
            dump_dataframe(
                logos._parser.parsed_log,
                os.path.join(path, "parsed_log.parquet"),
            )
            dump_metadata(
                logos._parser.parsed_variables,
                os.path.join(path, "parsed_variables.json"),
            )
            dump_metadata(
                logos._parser.parsed_templates,
                os.path.join(path, "parsed_templates.json"),
            )
            manifest["has_parsed"] = True
        else:
            manifest["has_parsed"] = False

        # Save prepared artefacts when available and non-empty
        if (
            logos._preparer is not None
            and not logos._preparer.prepared_log.empty
        ):
            dump_dataframe(
                logos._preparer.prepared_log,
                os.path.join(path, "prepared_log.parquet"),
            )
            dump_metadata(
                logos._preparer.prepared_variables,
                os.path.join(path, "prepared_variables.json"),
            )
            causal_unit_var, num_causal_units = (
                logos._preparer.get_causal_unit_info()
            )
            manifest["causal_unit_var"] = causal_unit_var
            manifest["num_causal_units"] = num_causal_units
            manifest["has_prepared"] = True
        else:
            manifest["has_prepared"] = False

        # Save explorer state (edge-state matrix + graph)
        if logos._explorer is not None:
            np.save(
                os.path.join(path, "edge_states.npy"),
                logos._explorer._edge_states.m,
            )
            nx.write_graphml(
                logos._explorer.graph,
                os.path.join(path, "graph.graphml"),
            )
            manifest["has_explorer"] = True
        else:
            manifest["has_explorer"] = False

        with open(
            os.path.join(path, "manifest.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(manifest, f, indent=2)

        _logger.debug("Session saved to %s", path)
        return path

    @staticmethod
    def load_session(path: str) -> "Logos":
        """
        Reconstruct a Logos instance from a saved session directory.

        Parameters:
            path: Directory previously created by save_session().

        Returns:
            A fully-restored Logos instance ready for exploration.

        Raises:
            FileNotFoundError: If no manifest is found at the given path.
            ValueError: If the session contains no parseable or prepared data.
        """
        manifest_path = os.path.join(path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"No session manifest found at {manifest_path!r}"
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        workdir = manifest.get("workdir", path)

        # Reconstruct at the most advanced available stage
        if manifest.get("has_prepared"):
            prepared_log = load_dataframe(
                os.path.join(path, "prepared_log.parquet")
            )
            prepared_variables = load_metadata(
                os.path.join(path, "prepared_variables.json")
            )
            variable_tags = dict(
                zip(
                    prepared_variables["Name"].tolist(),
                    prepared_variables["Tag"].tolist(),
                )
            )
            logos = Logos.from_prepared_table(
                prepared_log, workdir, variable_tags
            )

        elif manifest.get("has_parsed"):
            parsed_log = load_dataframe(
                os.path.join(path, "parsed_log.parquet")
            )
            parsed_variables = load_metadata(
                os.path.join(path, "parsed_variables.json")
            )
            variable_tags = dict(
                zip(
                    parsed_variables["Name"].tolist(),
                    parsed_variables["Tag"].tolist(),
                )
            )
            source_id = manifest.get("source_path", "restored_session")
            logos = Logos.from_parsed_table(
                parsed_log,
                workdir,
                source_id=source_id,
                variable_tags=variable_tags,
            )
            if manifest.get("causal_unit_var"):
                logos.set_causal_unit(
                    manifest["causal_unit_var"],
                    manifest.get("num_causal_units"),
                )

        else:
            raise ValueError(
                f"Session at {path!r} contains no parseable or prepared data."
            )

        # Restore explorer state (edge states + graph)
        if manifest.get("has_explorer") and logos._explorer is not None:
            edge_states_path = os.path.join(path, "edge_states.npy")
            graph_path = os.path.join(path, "graph.graphml")
            if os.path.exists(edge_states_path) and os.path.exists(graph_path):
                m = np.load(edge_states_path)
                graph = nx.read_graphml(graph_path)
                logos._explorer._restore_session_state(graph, m)

        _logger.debug("Session loaded from %s", path)
        return logos
