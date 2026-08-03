"""File selection screen: path input + entry-point chooser."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DirectoryTree, Input, Label, Select, Static

from logos import Logos
from logos_tui.explore import ExploreScreen
from logos_tui.parse import ParseScreen
from logos_tui.transform import TransformScreen

_EP_OPTIONS = [
    ("Raw log file (EP-1): run Drain parsing", "EP-1"),
    ("Pre-parsed table (EP-2): skip Drain, provide one row per event", "EP-2"),
    (
        "Pre-prepared table (EP-3): skip parse & prepare, ready for exploration",
        "EP-3",
    ),
]


class FileSelectScreen(Screen):
    """Choose a file and entry-point type to begin analysis."""

    CSS_PATH = "file_select.tcss"

    def compose(self) -> ComposeResult:
        with Horizontal(id="layout"):
            with Vertical(id="tree-pane"):
                yield Static("[bold]Browse[/bold]", id="tree-label")
                yield DirectoryTree(str(Path.home()), id="file-tree")
            with Vertical(id="form-pane"):
                yield Static("[bold]New Session[/bold]", id="title")
                yield Label("Log file path (click a file in the tree to fill):")
                yield Input(placeholder="/path/to/log.txt", id="path_input")
                yield Label("Entry point:")
                yield Select(
                    options=_EP_OPTIONS,
                    prompt="Select entry point",
                    id="ep_select",
                    value="EP-1",
                )
                yield Label("Workdir (click a folder in the tree to fill):")
                yield Input(
                    placeholder="/tmp/logos_workdir", id="workdir_input"
                )
                yield Label("", id="error-label")
                with Horizontal():
                    yield Button("Start", variant="primary", id="btn_start")
                    yield Button(
                        "\u2190 Back", variant="default", id="btn_back"
                    )
        with Horizontal(classes="exit-bar"):
            yield Button(
                "\U0001f4be Save & Exit",
                id="btn_save_exit",
                variant="success",
                disabled=True,
            )
            yield Button("\u2717 Exit", id="btn_exit_no_save", variant="error")
        """Clicking a file fills the path input."""
        self.query_one("#path_input", Input).value = str(event.path)

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        """Clicking a directory fills the workdir input."""
        self.query_one("#workdir_input", Input).value = str(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_back":
            self.app.pop_screen()
            return
        if event.button.id == "btn_start":
            self._start()

    def _start(self) -> None:
        path = self.query_one("#path_input", Input).value.strip()
        workdir = self.query_one("#workdir_input", Input).value.strip()
        ep_select = self.query_one("#ep_select", Select)
        ep = ep_select.value
        error_lbl = self.query_one("#error-label", Label)

        if not path:
            error_lbl.update("[red]Please enter a file path.[/red]")
            return
        if not workdir:
            error_lbl.update("[red]Please enter a workdir path.[/red]")
            return
        if ep is Select.BLANK:
            error_lbl.update("[red]Please select an entry point.[/red]")
            return

        error_lbl.update("")

        # Disable the button during initialization
        btn = self.query_one("#btn_start", Button)
        btn.disabled = True

        def _init() -> None:
            try:
                logos = self._create_logos(path, workdir, str(ep))
            except Exception as exc:
                self.app.call_from_thread(
                    self.query_one("#error-label", Label).update,
                    f"[red]{exc}[/red]",
                )
                self.app.call_from_thread(setattr, btn, "disabled", False)
                return
            self.app.logos = logos
            self.app.call_from_thread(self._navigate, str(ep))

        self.run_worker(_init, thread=True)

    def _navigate(self, ep: str) -> None:
        if ep == "EP-1":
            self.app.push_screen(ParseScreen())
        elif ep == "EP-2":
            self.app.push_screen(TransformScreen())
        else:  # EP-3
            self.app.push_screen(ExploreScreen())

    @staticmethod
    def _create_logos(path: str, workdir: str, ep: str):
        import os

        import pandas as pd

        if ep == "EP-1":
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path!r}")
            return Logos(filename=path, workdir=workdir)

        elif ep == "EP-2":
            # Load CSV/Parquet as parsed table
            df = _load_table(path)
            return Logos.from_parsed_table(df, workdir=workdir, source_id=path)

        else:  # EP-3
            df = _load_table(path)
            return Logos.from_prepared_table(df, workdir=workdir)


def _load_table(path: str):
    import pandas as pd

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format for {path!r}. "
            "Use .csv or .parquet for EP-2/EP-3."
        )
