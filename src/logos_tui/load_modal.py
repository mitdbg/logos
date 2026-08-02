"""Modal dialog for browsing and loading a saved session directory."""
from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Label, Static

_SESSIONS_ROOT = Path.home() / ".logos" / "sessions"


class LoadSessionModal(ModalScreen[str | None]):
    """Browse the sessions directory and pick one to load."""

    CSS = """
    LoadSessionModal > Horizontal {
        width: 90;
        height: 30;
        border: double $primary;
        background: $surface;
    }
    LoadSessionModal #tree-pane {
        width: 2fr;
        border-right: solid $primary-darken-2;
        padding: 0 1;
    }
    LoadSessionModal #tree-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    LoadSessionModal #form-pane {
        width: 1fr;
        padding: 1 2;
    }
    LoadSessionModal #form-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    LoadSessionModal Label {
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        tree_root = _SESSIONS_ROOT if _SESSIONS_ROOT.exists() else Path.home()
        with Horizontal():
            with Vertical(id="tree-pane"):
                yield Static("[bold]Browse sessions[/bold]", id="tree-title")
                yield DirectoryTree(str(tree_root), id="session-tree")
            with Vertical(id="form-pane"):
                yield Static("[bold]Load Session[/bold]", id="form-title")
                yield Label(
                    "Select a session directory in the tree, or type a path:"
                )
                yield Input(
                    placeholder=str(_SESSIONS_ROOT / "<name>"),
                    id="path_input",
                )
                with Horizontal():
                    yield Button("Load", variant="primary", id="btn_load")
                    yield Button("Cancel", variant="default", id="btn_cancel")

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        """Clicking a directory fills the path input."""
        self.query_one("#path_input", Input).value = str(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_load":
            path = self.query_one("#path_input", Input).value.strip()
            self.dismiss(path if path else None)
        else:
            self.dismiss(None)
