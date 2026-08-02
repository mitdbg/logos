"""
LOGos Textual TUI — main application class.

Dev mode with hot autoreload:
    textual run --dev src/logos_tui/app.py

Normal run:
    python -m logos_tui
"""
from __future__ import annotations

from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Footer, Header

from logos.session import Session
from logos_tui.explore import ExploreScreen
from logos_tui.file_select import FileSelectScreen
from logos_tui.load_modal import LoadSessionModal
from logos_tui.parse import ParseScreen
from logos_tui.save_modal import SaveSessionModal
from logos_tui.transform import TransformScreen
from logos_tui.welcome import WelcomeScreen


class LOGosApp(App):
    """The root Textual application; holds the shared LOGos instance."""

    TITLE = "LOGos"
    SUB_TITLE = "From Logs to Causal Diagnosis"
    BINDINGS = [
        Binding("ctrl+s", "save_session", "Save session", show=True),
        Binding("ctrl+o", "load_session", "Load session", show=True),
        Binding("ctrl+r", "reset", "Reset", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]
    CSS_PATH = "app.tcss"

    def __init__(self) -> None:
        super().__init__()
        self.logos = None          # Optional[LOGos]
        self.session_path: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())

    # ------------------------------------------------------------------
    # Global actions
    # ------------------------------------------------------------------

    def action_save_session(self) -> None:
        if self.logos is None:
            self.notify("Nothing to save — no session is active.", severity="warning")
            return
        self.push_screen(SaveSessionModal(), self._on_save_name)

    def action_save_and_exit(self) -> None:
        """Save the current session (prompting for a name) then quit."""
        if self.logos is None:
            self.exit()
            return
        self.push_screen(SaveSessionModal(), self._on_save_and_exit_name)

    def _on_save_and_exit_name(self, name: Optional[str]) -> None:
        if not name:
            return  # user cancelled — stay in the app
        from pathlib import Path
        path = str(Path.home() / ".logos" / "sessions" / name)

        def _save_then_exit() -> None:
            try:
                saved = Session.save_session(self.logos, path)
                self.session_path = saved
            except Exception as exc:
                self.call_from_thread(
                    self.notify, str(exc), title="Save failed", severity="error"
                )
                return
            self.call_from_thread(self.exit)

        self.run_worker(_save_then_exit, thread=True)

    def action_exit_no_save(self) -> None:
        """Quit immediately without saving."""
        self.exit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Catch exit-bar button presses that bubble up from any screen."""
        if event.button.id == "btn_save_exit":
            self.action_save_and_exit()
        elif event.button.id == "btn_exit_no_save":
            self.action_exit_no_save()

    def _on_save_name(self, name: Optional[str]) -> None:
        if not name:
            return
        from pathlib import Path
        path = str(Path.home() / ".logos" / "sessions" / name)

        def _save() -> None:
            saved = Session.save_session(self.logos, path)
            self.session_path = saved
            self.call_from_thread(
                self.notify, f"Session saved to:\n{saved}", title="Saved"
            )

        self.run_worker(_save, thread=True)

    def action_load_session(self) -> None:
        self.push_screen(LoadSessionModal(), self._on_load_result)

    def _on_load_result(self, path: Optional[str]) -> None:
        if not path:
            return

        def _load() -> None:
            try:
                logos = Session.load_session(path)
            except Exception as exc:
                self.call_from_thread(
                    self.notify, str(exc), title="Load failed", severity="error"
                )
                return
            self.logos = logos
            self.session_path = path
            self.call_from_thread(self._navigate_after_load)

        self.run_worker(_load, thread=True)

    def _navigate_after_load(self) -> None:
        # Pop all screens except the root, then push the appropriate one
        while len(self.screen_stack) > 1:
            self.pop_screen()

        if self.logos._explorer is not None:
            self.push_screen(ExploreScreen())
        elif self.logos._preparer is not None and not self.logos._preparer.prepared_log.empty:
            self.push_screen(TransformScreen())
        elif self.logos._parser is not None and not self.logos._parser.parsed_log.empty:
            self.push_screen(ParseScreen())
        else:
            self.push_screen(FileSelectScreen())

        self.notify("Session loaded.", severity="information")

    def action_reset(self) -> None:
        self.logos = None
        self.session_path = None
        while len(self.screen_stack) > 1:
            self.pop_screen()
        self.notify("Session reset.", severity="information")


if __name__ == "__main__":
    import asyncio
    import warnings
    warnings.filterwarnings("ignore", message="Unclosed client session")
    warnings.filterwarnings("ignore", message="Unclosed connector")
    LOGosApp().run()
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.run_until_complete(asyncio.sleep(0))
            loop.close()
    except Exception:
        pass
