"""Consistent, beginner-friendly messages and errors.

In the spirit of the R ``infer``/``cli`` style: a short summary line followed by
``→`` hint bullets that tell a newcomer what to do next. Informational messages
use a dedicated :class:`ModernDiveMessage` warning category so they're easy to
see and easy to silence (``warnings`` filters, or a function's ``quiet=`` flag).
"""

from __future__ import annotations

import warnings


class ModernDiveMessage(UserWarning):
    """An informational (non-error) message emitted by moderndive."""


def _format(summary: str, bullets: tuple[str, ...]) -> str:
    return "\n".join([summary, *[f"  → {b}" for b in bullets]])


def inform(summary: str, *bullets: str) -> None:
    """Emit a one-off, suppressible informational message (a summary + ``→`` hints)."""
    warnings.warn(_format(summary, bullets), ModernDiveMessage, stacklevel=3)


def helpful_error(summary: str, *bullets: str) -> str:
    """Build a beginner-friendly error string (a summary + ``→`` hint bullets).

    Use as ``raise ValueError(helpful_error("...", "do X", "or Y"))``.
    """
    return _format(summary, bullets)
