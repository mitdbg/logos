"""Welcome screen: start a new session or load a saved one."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Middle, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, Static

from logos_tui.file_select import FileSelectScreen


class WelcomeScreen(Screen):
    """Landing screen shown on startup."""

    CSS_PATH = "welcome.tcss"

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="welcome-box"):
                    yield Static(
                        "[bold]LOGos[/bold]", id="logo-title"
                    )
                    yield Static(
                        "From Logs to Causal Diagnosis",
                        id="logo-subtitle",
                    )
                    with Center(id="btn-row"):
                        yield Button(
                            "New Session", id="btn_new", variant="primary"
                        )
                        yield Button(
                            "Load Session", id="btn_load", variant="default"
                        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_new":
            self.app.push_screen(FileSelectScreen())
        elif event.button.id == "btn_load":
            self.app.action_load_session()
