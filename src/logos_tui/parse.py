"""Parse screen (EP-1): configure Drain parameters and run parsing."""

from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    LoadingIndicator,
    Select,
    Static,
    TextArea,
)

from logos import Logos
from logos_tui.transform import TransformScreen


class ParseScreen(Screen):
    """Configure Drain parameters, run parse(), and inspect results."""

    CSS_PATH = "parse.tcss"

    def compose(self) -> ComposeResult:
        logos: Logos = self.app.logos
        default_regex = json.dumps(logos.DEFAULT_REGEX_DICT, indent=2)
        default_prefix = logos.DEFAULT_MESSAGE_PREFIX

        yield Static("[bold]Parse Log File[/bold]", id="title")

        with Horizontal(id="params-row"):
            with Vertical(id="params-left"):
                yield Label("Similarity threshold (0.0–1.0):")
                yield Input(value="0.65", id="sim_thresh", type="number")
                yield Label("Tree depth (1–10):")
                yield Input(value="5", id="depth", type="integer")
                yield Label("Message prefix regex:")
                yield Input(value=default_prefix, id="message_prefix")

            with Vertical(id="params-right"):
                yield Label("Regex dict (JSON — name → pattern):")
                yield TextArea(
                    text=default_regex,
                    id="regex_dict_area",
                    language="json",
                )

        yield Label("", id="error-label")
        with Horizontal():
            yield Button("Parse", variant="primary", id="btn_parse")
            yield Button("Force re-parse", variant="default", id="btn_force")
            yield Button(
                "Next →", variant="success", id="btn_next", disabled=True
            )
            yield Button("← Back", variant="default", id="btn_back")
            yield Button("💾 Save Session", id="btn_save_session")
        yield LoadingIndicator(id="loading")

        yield Static("[bold]Results[/bold]", id="results-title")
        with Horizontal(id="tables-row"):
            with Vertical(id="table-vars"):
                yield Static("[bold]Parsed Variables[/bold]")
                yield DataTable(id="vars_table", cursor_type="row")
            with Vertical(id="table-templates"):
                yield Static("[bold]Parsed Templates[/bold]")
                yield DataTable(id="templates_table", cursor_type="row")

        with Horizontal(id="include-section"):
            yield Label(
                "Include variable in template (corrects over-segmentation):"
            )
            yield Select(
                options=[],
                prompt="Select parsed variable",
                id="include_select",
            )
            yield Button("Apply", id="btn_include", disabled=True)
        with Horizontal(classes="exit-bar"):
            yield Button(
                "\U0001f4be Save & Exit", id="btn_save_exit", variant="success"
            )
            yield Button(
                "\u2717 Exit without saving",
                id="btn_exit_no_save",
                variant="error",
            )

    def on_mount(self) -> None:
        # Set up DataTable columns
        vt = self.query_one("#vars_table", DataTable)
        vt.add_columns("Name", "Tag", "Type", "Interesting?", "Occurrences")

        tt = self.query_one("#templates_table", DataTable)
        tt.add_columns("TemplateId", "TemplateText", "Occurrences")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id in ("btn_parse", "btn_force"):
            self._run_parse(force=btn_id == "btn_force")
        elif btn_id == "btn_next":
            self.app.push_screen(TransformScreen())
        elif btn_id == "btn_back":
            self.app.pop_screen()
        elif btn_id == "btn_save_session":
            self.app.action_save_session()
        elif btn_id == "btn_include":
            self._include_in_template()

    def _run_parse(self, force: bool = False) -> None:
        error_lbl = self.query_one("#error-label", Label)
        error_lbl.update("")

        try:
            regex_text = self.query_one("#regex_dict_area", TextArea).text
            regex_dict = json.loads(regex_text)
            sim_thresh = float(self.query_one("#sim_thresh", Input).value)
            depth = int(self.query_one("#depth", Input).value)
            message_prefix = self.query_one(
                "#message_prefix", Input
            ).value.strip()
        except Exception as exc:
            error_lbl.update(f"[red]Invalid parameters: {exc}[/red]")
            return

        self.query_one("#loading").display = True
        for btn_id in ("btn_parse", "btn_force", "btn_next", "btn_back"):
            self.query_one(f"#{btn_id}", Button).disabled = True

        def _parse() -> None:
            try:
                self.app.logos.parse(
                    regex_dict=regex_dict,
                    sim_thresh=sim_thresh,
                    depth=depth,
                    force=force,
                    message_prefix=message_prefix,
                )
            except Exception as exc:
                self.app.call_from_thread(self._on_parse_error, str(exc))
                return
            self.app.call_from_thread(self._on_parse_done)

        self.run_worker(_parse, thread=True)

    def _on_parse_error(self, msg: str) -> None:
        self.query_one("#loading").display = False
        for btn_id in ("btn_parse", "btn_force", "btn_back"):
            self.query_one(f"#{btn_id}", Button).disabled = False
        self.query_one("#error-label", Label).update(
            f"[red]Parse failed: {msg}[/red]"
        )

    def _on_parse_done(self) -> None:
        self.query_one("#loading").display = False
        for btn_id in ("btn_parse", "btn_force", "btn_next", "btn_back"):
            self.query_one(f"#{btn_id}", Button).disabled = False
        self._refresh_tables()
        self.notify("Parsing complete!", severity="information")

    def _refresh_tables(self) -> None:
        logos: Logos = self.app.logos
        pv = logos.parsed_variables
        pt = logos.parsed_templates

        vt = self.query_one("#vars_table", DataTable)
        vt.clear()
        for _, row in pv.iterrows():
            vt.add_row(
                str(row.get("Name", "")),
                str(row.get("Tag", "")),
                str(row.get("Type", "")),
                "No" if row.get("IsUninteresting", False) else "Yes",
                str(row.get("Occurrences", "")),
            )

        tt = self.query_one("#templates_table", DataTable)
        tt.clear()
        for _, row in pt.iterrows():
            tt.add_row(
                str(row.get("TemplateId", "")),
                str(row.get("TemplateText", ""))[:80],
                str(row.get("Occurrences", "")),
            )

        # Populate the include-in-template select
        tag_options = [(str(t), str(n)) for n, t in zip(pv["Name"], pv["Tag"])]
        sel = self.query_one("#include_select", Select)
        sel.set_options(tag_options)
        self.query_one("#btn_include", Button).disabled = False

    def _include_in_template(self) -> None:
        sel = self.query_one("#include_select", Select)
        if sel.value is Select.BLANK:
            return
        var = str(sel.value)
        try:
            self.app.logos.include_in_template(var)
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self._refresh_tables()
        self.notify(
            f"Included '{var}' in its template.", severity="information"
        )
