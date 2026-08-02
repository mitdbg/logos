"""Transform screen: set causal unit, configure agg/imp, run prepare()."""
from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Label,
    LoadingIndicator,
    Select,
    Static,
    TextArea,
)

from logos.logos import LOGos
from logos_tui.explore import ExploreScreen


class TransformScreen(Screen):
    """Set causal unit, aggregation/imputation, and run prepare()."""

    CSS_PATH = "transform.tcss"

    def compose(self) -> ComposeResult:
        yield Static("[bold]Data Transformation[/bold]", id="title")

        with Vertical(id="causal-unit-section"):
            yield Static("[bold]Causal Unit[/bold]")
            yield Label(
                "LOGos will suggest candidates based on information utilization "
                "(IUS). Click a row to fill the variable field, or type directly."
            )
            yield DataTable(id="suggestions-table", cursor_type="row")
            with Horizontal(id="cu-inputs"):
                with Vertical(id="cu-var-pane"):
                    yield Label("Variable name or tag:")
                    yield Input(
                        placeholder="e.g. sessionID",
                        id="causal_unit_input",
                    )
                with Vertical(id="cu-bins-pane"):
                    yield Label("Number of bins (numerical variables only):")
                    yield Input(
                        placeholder="e.g. 1000  (leave blank for categorical)",
                        id="num_units_input",
                        type="integer",
                    )
            yield Button("Set Causal Unit", id="btn_set_cu", variant="primary")

        with Horizontal(id="aggimp-row"):
            with Vertical(id="agg-pane"):
                yield Static("[bold]Custom Aggregation (JSON)[/bold]")
                yield Label(
                    '{"var_name": ["mean", "max"]}  — leave empty for auto-selection'
                )
                yield TextArea(
                    text="{}",
                    id="custom_agg_area",
                    language="json",
                )
            with Vertical(id="imp-pane"):
                yield Static("[bold]Custom Imputation (JSON)[/bold]")
                yield Label(
                    '{"var_name": "zero_imp"}  — allowed: zero_imp, ffill_imp, no_imp'
                )
                yield TextArea(
                    text="{}",
                    id="custom_imp_area",
                    language="json",
                )

        yield Label("", id="error-label")
        yield Label("", id="summary-label")
        yield LoadingIndicator(id="loading")
        yield Checkbox(
            "Force re-prepare (required when changing custom agg/imp on an existing workdir)",
            value=True,
            id="force_cb",
        )

        with Horizontal():
            yield Button("Prepare", variant="primary", id="btn_prepare")
            yield Button("Next →", variant="success", id="btn_next", disabled=True)
            yield Button("← Back", variant="default", id="btn_back")
            yield Button("💾 Save Session", id="btn_save_session")

    def on_mount(self) -> None:
        st = self.query_one("#suggestions-table", DataTable)
        st.add_columns("Variable", "Type", "IUS Score", "# Units")
        self._load_suggestions()

    def _load_suggestions(self) -> None:
        try:
            suggestions = self.app.logos.suggest_causal_unit_defs()
        except Exception:
            return
        if suggestions is None or suggestions.empty:
            return
        st = self.query_one("#suggestions-table", DataTable)
        st.clear()
        for _, row in suggestions.iterrows():
            st.add_row(
                str(row.get("Variable", "")),
                str(row.get("Type", "")),
                f"{row.get('IUS', 0.0):.4f}" if "IUS" in row else "—",
                str(row.get("Num Units", "")),
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "suggestions-table":
            # Fill the causal unit input from the clicked row
            row_data = event.data_table.get_row(event.row_key)
            if row_data:
                self.query_one("#causal_unit_input", Input).value = str(row_data[0])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn_set_cu":
            self._set_causal_unit()
        elif btn_id == "btn_prepare":
            self._run_prepare()
        elif btn_id == "btn_next":
            self.app.push_screen(ExploreScreen())
        elif btn_id == "btn_back":
            self.app.pop_screen()
        elif btn_id == "btn_save_session":
            self.app.action_save_session()

    def _set_causal_unit(self) -> None:
        var = self.query_one("#causal_unit_input", Input).value.strip()
        num_str = self.query_one("#num_units_input", Input).value.strip()
        num_units = int(num_str) if num_str else None
        error_lbl = self.query_one("#error-label", Label)
        try:
            self.app.logos.set_causal_unit(var if var else None, num_units)
        except Exception as exc:
            error_lbl.update(f"[red]{exc}[/red]")
            return
        error_lbl.update("")
        self.notify(f"Causal unit set to '{var}'.", severity="information")

    def _run_prepare(self) -> None:
        error_lbl = self.query_one("#error-label", Label)
        error_lbl.update("")

        try:
            custom_agg = json.loads(self.query_one("#custom_agg_area", TextArea).text)
            custom_imp = json.loads(self.query_one("#custom_imp_area", TextArea).text)
            force = self.query_one("#force_cb", Checkbox).value
        except json.JSONDecodeError as exc:
            error_lbl.update(f"[red]Invalid JSON: {exc}[/red]")
            return

        self.query_one("#loading").display = True
        for btn_id in ("btn_prepare", "btn_next", "btn_back"):
            self.query_one(f"#{btn_id}", Button).disabled = True

        def _prepare() -> None:
            try:
                self.app.logos.prepare(
                    custom_agg=custom_agg,
                    custom_imp=custom_imp,
                    force=force,
                )
            except Exception as exc:
                self.app.call_from_thread(self._on_prepare_error, str(exc))
                return
            self.app.call_from_thread(self._on_prepare_done)

        self.run_worker(_prepare, thread=True)

    def _on_prepare_error(self, msg: str) -> None:
        self.query_one("#loading").display = False
        for btn_id in ("btn_prepare", "btn_back"):
            self.query_one(f"#{btn_id}", Button).disabled = False
        self.query_one("#error-label", Label).update(f"[red]Prepare failed: {msg}[/red]")

    def _on_prepare_done(self) -> None:
        self.query_one("#loading").display = False
        for btn_id in ("btn_prepare", "btn_next", "btn_back"):
            self.query_one(f"#{btn_id}", Button).disabled = False

        logos: LOGos = self.app.logos
        n_rows = len(logos.prepared_log)
        n_vars = logos.num_prepared_variables
        self.query_one("#summary-label", Label).update(
            f"[green]Prepared: {n_rows} causal units × {n_vars} variables[/green]"
        )
        self.notify("Preparation complete!", severity="information")
