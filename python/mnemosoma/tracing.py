"""Optional LangSmith tracing.

Nothing is sent anywhere unless ``langsmith`` is installed and tracing is
switched on with ``LANGSMITH_TRACING=true`` and an API key. With tracing on,
memory writes and recalls appear as steps in the trace of the agent run that
made them. Traces include the memory text, so turn it on only where that is
acceptable.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from langsmith import traceable as _traceable
except ImportError:  # langsmith is optional
    _traceable = None


def _without_self(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in inputs.items() if k != "self"}


def traced(name: str, run_type: str) -> Callable[[F], F]:
    """Record the call as a LangSmith run when tracing is enabled; otherwise do nothing."""

    def decorate(fn: F) -> F:
        if _traceable is None:
            return fn
        return _traceable(name=name, run_type=run_type, process_inputs=_without_self)(fn)

    return decorate
