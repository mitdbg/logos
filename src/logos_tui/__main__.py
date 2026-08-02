"""Entry point: python -m logos_tui"""
import asyncio
import warnings

from logos_tui.app import LOGosApp

# aiohttp (used by openai/eccs) emits "Unclosed client session" when the asyncio
# event loop is torn down before the HTTP keep-alive pool drains.  The sessions
# are harmless but the warning is noisy; suppress it at the process boundary.
warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")

app = LOGosApp()
app.run()

# Let the event loop drain any remaining async callbacks (closes HTTP sessions).
try:
    loop = asyncio.get_event_loop()
    if not loop.is_closed():
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()
except Exception:
    pass

