"""
Explore screen: 2×2 grid — rank candidates | causal graph
                            edge decisions  | ATE
"""

from __future__ import annotations

from typing import Optional

import netext
import networkx as nx
from netext import ArrowTip, EdgeProperties, NodeProperties
from netext.edge_rendering.modes import EdgeSegmentDrawingMode
from rich.style import Style
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Label,
    LoadingIndicator,
    Select,
    Static,
)

from logos import Logos

_EMPTY_MSG = "[dim]Graph is empty. Accept some edges to populate it.[/dim]"


def _render_graph_netext(graph: nx.DiGraph, variables, width: int, height: int):
    """Render the causal DAG with netext; fall back to edge list on failure."""
    if len(graph.nodes) == 0:
        return _EMPTY_MSG

    def tag(name: str) -> str:
        m = variables.loc[variables["Name"] == name, "Tag"].values
        return str(m[0]) if len(m) > 0 else name

    try:
        nodes = {
            tag(n): {
                "$properties": NodeProperties(
                    style=Style(color="green", bold=True)
                )
            }
            for n in graph.nodes
        }
        edges = [
            (
                tag(u),
                tag(v),
                {
                    "$properties": EdgeProperties(
                        end_arrow_tip=ArrowTip.ARROW,
                        segment_drawing_mode=EdgeSegmentDrawingMode.BOX_ROUNDED,
                    )
                },
            )
            for u, v in graph.edges
        ]
        return netext.ConsoleGraph(
            nodes=nodes,
            edges=edges,
            max_width=max(width, 20),
            max_height=max(height, 10),
        )
    except Exception:
        lines = [
            f"[green]{tag(u)}[/green] [dim]\u2192[/dim] [blue]{tag(v)}[/blue]"
            for u, v in sorted(
                graph.edges, key=lambda e: (tag(e[0]), tag(e[1]))
            )
        ]
        return "\n".join(lines) if lines else "[dim](no edges yet)[/dim]"


class ExploreScreen(Screen):
    """2×2 exploration grid: rank | graph / edge decisions | ATE."""

    CSS_PATH = "explore.tcss"

    def __init__(self) -> None:
        super().__init__()
        self._suggested_edge: Optional[tuple[str, str]] = None

    def compose(self) -> ComposeResult:
        # ── Header ─────────────────────────────────────────────────────
        with Horizontal(id="explore-header"):
            yield Static(
                "[bold]Causal Graph Exploration[/bold]", id="explore-title"
            )
            yield Button("💾 Save Session", id="btn_save_session")

        # ── Top row ────────────────────────────────────────────────────
        with Horizontal(id="top-row"):
            # TOP-LEFT: candidate ranking
            with Vertical(classes="pane", id="rank-pane"):
                yield Static(
                    "[bold]Rank Candidate Causes[/bold]", classes="pane-title"
                )
                yield Label("Target outcome variable:")
                yield Select(
                    options=[], prompt="Select outcome", id="outcome_select"
                )
                yield Button(
                    "Rank Candidates", id="btn_rank", variant="primary"
                )
                yield LoadingIndicator(id="rank-loading")
                yield DataTable(id="causes-table", cursor_type="row")

            # TOP-RIGHT: causal graph
            with Vertical(classes="pane", id="graph-pane"):
                yield Static("[bold]Causal Graph[/bold]", classes="pane-title")
                yield ScrollableContainer(
                    Static("", id="graph-text"), id="graph-scroll"
                )
                yield Button("Save PNG", id="btn_save_png")

        # ── Bottom row ─────────────────────────────────────────────────
        with Horizontal(id="bottom-row"):
            # BOTTOM-LEFT: edge decisions
            with Vertical(classes="pane", id="edge-pane"):
                yield Static(
                    "[bold]Edge Decisions[/bold]", classes="pane-title"
                )
                yield Label("Source variable (cause):")
                yield Select(
                    options=[], prompt="Select source", id="source_select"
                )
                yield Label("Destination variable (effect):")
                yield Select(
                    options=[], prompt="Select destination", id="dest_select"
                )
                yield Label("", id="suggestion-label")
                with Horizontal():
                    yield Button(
                        "Accept  ✓", id="btn_accept", variant="success"
                    )
                    yield Button("Reject  ✗", id="btn_reject", variant="error")
                yield Button("Reject all undecided → dst", id="btn_rej_inc")
                yield Button(
                    "Reject all undecided from src →", id="btn_rej_out"
                )
                yield LoadingIndicator(id="edge-loading")
                yield Button(
                    "Suggest next edge (ECCS)",
                    id="btn_suggest",
                    variant="primary",
                )
                yield LoadingIndicator(id="refine-loading")

            # BOTTOM-RIGHT: ATE
            with Vertical(classes="pane", id="ate-pane"):
                yield Static(
                    "[bold]Average Treatment Effect[/bold]",
                    classes="pane-title",
                )
                yield Label("Treatment:")
                yield Select(
                    options=[],
                    prompt="Select treatment",
                    id="ate_treatment_select",
                )
                yield Label("Outcome:")
                yield Select(
                    options=[],
                    prompt="Select outcome",
                    id="ate_outcome_select",
                )
                with Horizontal():
                    yield Button(
                        "Calculate adjusted ATE",
                        id="btn_ate_calc",
                        variant="success",
                    )
                    yield LoadingIndicator(id="ate-loading")
                yield Label("", id="ate-result")
                yield Label("", id="ate-error")
        with Horizontal(classes="exit-bar"):
            yield Button(
                "\U0001f4be Save & Exit", id="btn_save_exit", variant="success"
            )
            yield Button(
                "\u2717 Exit without saving",
                id="btn_exit_no_save",
                variant="error",
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._populate_variable_selects()
        self._update_graph_display()

    def _populate_variable_selects(self) -> None:
        logos: Logos = self.app.logos
        try:
            tags = logos.prepared_variable_tags
        except Exception:
            return
        opts = [(t, t) for t in tags]
        for sel_id in (
            "outcome_select",
            "source_select",
            "dest_select",
            "ate_treatment_select",
            "ate_outcome_select",
        ):
            self.query_one(f"#{sel_id}", Select).set_options(opts)

    # ------------------------------------------------------------------
    # Button dispatch
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn_rank":
            self._run_rank()
        elif btn_id == "btn_accept":
            self._run_edge_action("accept")
        elif btn_id == "btn_reject":
            self._run_edge_action("reject")
        elif btn_id == "btn_rej_inc":
            self._run_edge_action("reject_inc")
        elif btn_id == "btn_rej_out":
            self._run_edge_action("reject_out")
        elif btn_id == "btn_suggest":
            self._run_suggest()
        elif btn_id == "btn_save_png":
            self._save_graph_png()
        elif btn_id == "btn_ate_calc":
            self._run_ate()
        elif btn_id == "btn_save_session":
            self.app.action_save_session()

    # ------------------------------------------------------------------
    # Candidate ranking
    # ------------------------------------------------------------------

    def _run_rank(self) -> None:
        outcome_sel = self.query_one("#outcome_select", Select)
        if outcome_sel.value is Select.BLANK:
            self.notify(
                "Please select a target outcome variable.", severity="warning"
            )
            return
        target_tag = str(outcome_sel.value)
        self.query_one("#rank-loading").display = True
        self.query_one("#btn_rank", Button).disabled = True

        def _rank() -> None:
            try:
                df = self.app.logos.rank_candidate_causes(target=target_tag)
            except Exception as exc:
                self.app.call_from_thread(
                    self.notify, str(exc), severity="error"
                )
                self.app.call_from_thread(self._after_rank, None)
                return
            self.app.call_from_thread(self._after_rank, df)

        self.run_worker(_rank, thread=True)

    def _after_rank(self, df) -> None:
        self.query_one("#rank-loading").display = False
        self.query_one("#btn_rank", Button).disabled = False
        if df is None:
            return
        table = self.query_one("#causes-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "Candidate", "Slope", "P-value", "Cand→Tgt", "Tgt→Cand"
        )
        for _, row in df.iterrows():
            table.add_row(
                str(row.get("Candidate Tag", row.get("Candidate", ""))),
                (
                    f"{row.get('Slope', 0.0):.4f}"
                    if row.get("Slope") is not None
                    else "—"
                ),
                (
                    f"{row.get('P-value', 1.0):.4f}"
                    if row.get("P-value") is not None
                    else "—"
                ),
                str(row.get("Candidate->Target Edge Status", "—")),
                str(row.get("Target->Candidate Edge Status", "—")),
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "causes-table":
            row = event.data_table.get_row(event.row_key)
            if row:
                self.query_one("#source_select", Select).value = str(row[0])
            target_sel = self.query_one("#outcome_select", Select)
            if target_sel.value is not Select.BLANK:
                self.query_one("#dest_select", Select).value = str(
                    target_sel.value
                )

    # ------------------------------------------------------------------
    # Edge decisions
    # ------------------------------------------------------------------

    def _run_edge_action(self, action: str) -> None:
        src_sel = self.query_one("#source_select", Select)
        dst_sel = self.query_one("#dest_select", Select)
        if action in ("accept", "reject") and (
            src_sel.value is Select.BLANK or dst_sel.value is Select.BLANK
        ):
            self.notify(
                "Please select both source and destination.", severity="warning"
            )
            return
        if action == "reject_inc" and dst_sel.value is Select.BLANK:
            self.notify(
                "Please select a destination variable.", severity="warning"
            )
            return
        if action == "reject_out" and src_sel.value is Select.BLANK:
            self.notify("Please select a source variable.", severity="warning")
            return

        src = str(src_sel.value) if src_sel.value is not Select.BLANK else ""
        dst = str(dst_sel.value) if dst_sel.value is not Select.BLANK else ""

        self.query_one("#edge-loading").display = True
        for b in ("btn_accept", "btn_reject", "btn_rej_inc", "btn_rej_out"):
            self.query_one(f"#{b}", Button).disabled = True

        def _act() -> None:
            try:
                logos: Logos = self.app.logos
                if action == "accept":
                    score, next_var = logos.accept(src, dst)
                elif action == "reject":
                    score, next_var = logos.reject(src, dst)
                elif action == "reject_inc":
                    score, next_var = logos.reject_undecided_incoming(dst)
                else:
                    score, next_var = logos.reject_undecided_outgoing(src)
            except Exception as exc:
                self.app.call_from_thread(
                    self.notify, str(exc), severity="error"
                )
                self.app.call_from_thread(self._after_edge_action, None, None)
                return
            self.app.call_from_thread(self._after_edge_action, score, next_var)

        self.run_worker(_act, thread=True)

    def _after_edge_action(
        self, score: Optional[float], next_var: Optional[str]
    ) -> None:
        self.query_one("#edge-loading").display = False
        for b in ("btn_accept", "btn_reject", "btn_rej_inc", "btn_rej_out"):
            self.query_one(f"#{b}", Button).disabled = False
        self._update_graph_display()

    # ------------------------------------------------------------------
    # ECCS refinement suggestion
    # ------------------------------------------------------------------

    def _run_suggest(self) -> None:
        outcome_sel = self.query_one("#outcome_select", Select)
        src_sel = self.query_one("#source_select", Select)
        if outcome_sel.value is Select.BLANK or src_sel.value is Select.BLANK:
            self.notify(
                "Select a source (treatment) and outcome before requesting a suggestion.",
                severity="warning",
            )
            return
        treatment_tag = str(src_sel.value)
        outcome_tag = str(outcome_sel.value)
        self.query_one("#refine-loading").display = True
        self.query_one("#btn_suggest", Button).disabled = True

        def _suggest() -> None:
            try:
                edge = self.app.logos.get_causal_graph_refinement_suggestion(
                    treatment_tag, outcome_tag
                )
            except Exception as exc:
                self.app.call_from_thread(
                    self.notify, str(exc), severity="error"
                )
                self.app.call_from_thread(self._after_suggest, None)
                return
            self.app.call_from_thread(self._after_suggest, edge)

        self.run_worker(_suggest, thread=True)

    def _after_suggest(self, edge: Optional[tuple[str, str]]) -> None:
        self.query_one("#refine-loading").display = False
        self.query_one("#btn_suggest", Button).disabled = False
        lbl = self.query_one("#suggestion-label", Label)
        if edge is None:
            lbl.update("[dim]No suggestion (graph may be complete).[/dim]")
            return
        src_tag, dst_tag = edge
        lbl.update(
            f"[yellow]Suggested:[/yellow] [green]{src_tag}[/green]"
            f" [dim]→[/dim] [blue]{dst_tag}[/blue]"
        )
        self.query_one("#source_select", Select).value = src_tag
        self.query_one("#dest_select", Select).value = dst_tag

    # ------------------------------------------------------------------
    # Graph display
    # ------------------------------------------------------------------

    def _update_graph_display(self) -> None:
        try:
            logos: Logos = self.app.logos
            pane = self.query_one("#graph-pane")
            # Subtract borders/padding/title from the usable canvas area
            w = max(20, (pane.content_size.width or 40) - 2)
            h = max(10, (pane.content_size.height or 24) - 4)
            renderable = _render_graph_netext(
                logos.graph, logos.prepared_variables, w, h
            )
        except Exception as exc:
            renderable = f"[red]Could not render graph: {exc}[/red]"
        self.query_one("#graph-text", Static).update(renderable)

    def on_resize(self, _) -> None:
        """Re-render the graph whenever the terminal is resized."""
        self._update_graph_display()

    def _save_graph_png(self) -> None:
        import os

        try:
            logos: Logos = self.app.logos
            workdir = logos._explorer._workdir if logos._explorer else "."
            path = os.path.join(workdir, "causal_graph.png")
            logos.save_graph(path)
            self.notify(f"Graph saved to {path}", severity="information")
        except Exception as exc:
            self.notify(str(exc), severity="error")

    # ------------------------------------------------------------------
    # Inline ATE
    # ------------------------------------------------------------------

    def _run_ate(self) -> None:
        t_sel = self.query_one("#ate_treatment_select", Select)
        o_sel = self.query_one("#ate_outcome_select", Select)
        err = self.query_one("#ate-error", Label)

        if t_sel.value is Select.BLANK or o_sel.value is Select.BLANK:
            err.update("[red]Please select both treatment and outcome.[/red]")
            return
        if t_sel.value == o_sel.value:
            err.update("[red]Treatment and outcome must be different.[/red]")
            return
        err.update("")
        treatment = str(t_sel.value)
        outcome = str(o_sel.value)

        self.query_one("#ate-loading").display = True
        self.query_one("#btn_ate_calc", Button).disabled = True

        def _calc() -> None:
            try:
                ate = self.app.logos.get_adjusted_ate(treatment, outcome)
            except Exception as exc:
                self.app.call_from_thread(
                    self.query_one("#ate-error", Label).update,
                    f"[red]{exc}[/red]",
                )
                self.app.call_from_thread(
                    setattr, self.query_one("#ate-loading"), "display", False
                )
                self.app.call_from_thread(
                    setattr,
                    self.query_one("#btn_ate_calc", Button),
                    "disabled",
                    False,
                )
                return
            self.app.call_from_thread(
                self._on_ate_done, treatment, outcome, ate
            )

        self.run_worker(_calc, thread=True)

    def _on_ate_done(self, treatment: str, outcome: str, ate: float) -> None:
        self.query_one("#ate-loading").display = False
        self.query_one("#btn_ate_calc", Button).disabled = False
        self.query_one("#ate-result", Label).update(
            f"Adjusted ATE of [bold]{treatment}[/bold] on"
            f" [bold]{outcome}[/bold]: [bold green]{ate:.6f}[/bold green]"
        )
