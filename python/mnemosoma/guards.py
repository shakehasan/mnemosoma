"""Write-time guardrails.

A guard looks at a memory before it is stored and may change it or refuse it.
Guards run in Python, in the order given to :class:`~mnemosoma.MemoryStore`.

The built-in guards are pattern based. They catch common cases cheaply and
will miss others, so they add to the engine's trust tiers and provenance
checks; they do not replace them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

from .types import Trust


@dataclass
class Draft:
    """A memory about to be written. Guards may edit any field."""

    content: str
    trust: Trust
    tags: list[str]
    source: str


class GuardBlocked(ValueError):
    """Raised by a guard to refuse a write."""


Guard = Callable[[Draft], None]

_INJECTION = [
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in (
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions?|rules|prompts?|guidelines)\b",
        r"\b(system|developer)\s+(prompt|message)\b",
        r"\byou\s+are\s+now\b",
        r"\b(do\s+not|don't|never)\s+(tell|inform|alert|notify)\s+the\s+user\b",
        r"\b(always|from\s+now\s+on|in\s+future)\b.{0,40}\b(send|wire|transfer|forward|email|pay)\b",
        r"\bnew\s+instructions?\s*:",
    )
]


def injection_guard(draft: Draft) -> None:
    """Mark text that reads like instructions to a model as untrusted.

    The memory is still stored, tagged ``flagged:injection``. Because it is
    untrusted, default recall leaves it out, and anything derived from it
    inherits that.
    """
    if any(pattern.search(draft.content) for pattern in _INJECTION):
        draft.trust = "untrusted"
        if "flagged:injection" not in draft.tags:
            draft.tags.append("flagged:injection")


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
# Phone numbers written with separators or an area code in brackets. A bare
# run of digits is left alone: it is more often an order or account number.
_PHONE = re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)[\s.-]?|\d{3}[\s.-])\d{3}[\s.-]?\d{4}(?![\w-])")


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    total = 0
    for position, digit in enumerate(reversed(digits)):
        if position % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return len(digits) >= 13 and total % 10 == 0


def pii_guard(mode: Literal["tag", "redact", "block"] = "redact") -> Guard:
    """Detect email addresses, payment card numbers and phone numbers in common formats.

    Args:
        mode: ``"tag"`` stores the text unchanged and adds tags such as
            ``pii:email``. ``"redact"`` also replaces each match with a
            placeholder such as ``[EMAIL]``. ``"block"`` refuses the write.
    """

    def guard(draft: Draft) -> None:
        found: list[str] = []

        def scrub(pattern: re.Pattern[str], label: str, valid: Callable[[str], bool] = lambda _: True) -> None:
            def replace(match: re.Match[str]) -> str:
                if not valid(match.group()):
                    return match.group()
                found.append(label)
                return f"[{label.upper()}]" if mode == "redact" else match.group()

            draft.content = pattern.sub(replace, draft.content)

        scrub(_EMAIL, "email")
        scrub(_CARD, "card", _luhn)
        scrub(_PHONE, "phone")
        if found and mode == "block":
            raise GuardBlocked(f"write refused: contains {', '.join(sorted(set(found)))}")
        for label in sorted(set(found)):
            if f"pii:{label}" not in draft.tags:
                draft.tags.append(f"pii:{label}")

    return guard
