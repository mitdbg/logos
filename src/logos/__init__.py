import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        f"LOGos requires Python 3.11 or later (running "
        f"{sys.version_info.major}.{sys.version_info.minor})."
    )

from src.logos.logos import Logos  # noqa: E402

__all__ = ["Logos"]
