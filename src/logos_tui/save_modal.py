"""Modal dialog for naming and saving the current session."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


class SaveSessionModal(ModalScreen[str | None]):
    """Ask the user for a session name; dismisses with the name or None."""

    CSS = """
    SaveSessionModal > Vertical {
        width: 72;
        height: auto;
        border: double $success;
        background: $surface;
        padding: 1 2;
    }
    SaveSessionModal #modal-title {
        text-style: bold;
        color: $success;
        padding-bottom: 1;
    }
    SaveSessionModal Label {
        margin-bottom: 1;
    }
    SaveSessionModal #path-preview {
        color: $text-muted;
        padding-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        default_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path.home() / ".logos" / "sessions"
        with Vertical():
            yield Static("[bold]Save Session[/bold]", id="modal-title")
            yield Label("Session name (becomes the directory name):")
            yield Input(value=default_name, id="name_input")
            yield Label(
                f"[dim]Will be saved to: {base}/{default_name}/[/dim]",
                id="path-preview",
            )
            with Horizontal():
                yield Button("Save", variant="success", id="btn_save")
                yield Button("Cancel", variant="default", id="btn_cancel")

    def on_input_changed(self, event: Input.Changed) -> None:
        base = Path.home() / ".logos" / "sessions"
        name = event.value.strip() or "<name>"
        self.query_one("#path-preview", Label).update(
            f"[dim]Will be saved to: {base}/{name}/[/dim]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_save":
            name = self.query_one("#name_input", Input).value.strip()
            self.dismiss(name if name else None)
        else:
            self.dismiss(None)
