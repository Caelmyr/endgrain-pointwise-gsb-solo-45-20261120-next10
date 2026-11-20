from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from .errors import WorkbenchError


def document(value: Any, label: str = "request") -> dict:
    if not isinstance(value, dict):
        raise WorkbenchError(f"{label.capitalize()} must be an object.", label)
    return value


def fields(body: Mapping, allowed: set[str], label: str = "request") -> None:
    unexpected = sorted(set(body) - allowed)
    if unexpected:
        raise WorkbenchError(f"Unexpected {label} field: {unexpected[0]}.", unexpected[0])


def string(
    value: Any,
    label: str,
    limit: int,
    *,
    required: bool = False,
    multiline: bool = False,
) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise WorkbenchError(f"{label.capitalize()} must be text.", label)
    normalized = value.replace("\r\n", "\n").strip()
    if required and not normalized:
        raise WorkbenchError(f"Enter {label}.", label)
    if len(normalized) > limit:
        raise WorkbenchError(f"{label.capitalize()} cannot exceed {limit} characters.", label)
    if any(ord(char) < 32 and (not multiline or char not in "\n\t") for char in normalized):
        raise WorkbenchError(f"{label.capitalize()} contains a control character.", label)
    return normalized


def number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkbenchError(f"Enter a number for {label}.", label)
    converted = float(value)
    if not isfinite(converted) or not minimum <= converted <= maximum:
        raise WorkbenchError(
            f"{label.capitalize()} must be between {minimum:g} and {maximum:g}.",
            label,
        )
    return converted


def integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkbenchError(f"Enter a whole number for {label}.", label)
    number(value, label, minimum, maximum)
    return value


def choice(value: Any, label: str, choices: Sequence[str]) -> str:
    if value not in choices:
        raise WorkbenchError(f"{label.capitalize()} must be one of {', '.join(choices)}.", label)
    return value


def flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise WorkbenchError(f"{label.capitalize()} must be true or false.", label)
    return value


def list_of(value: Any, label: str, minimum: int, maximum: int) -> list:
    if not isinstance(value, list):
        raise WorkbenchError(f"{label.capitalize()} must be an array.", label)
    if not minimum <= len(value) <= maximum:
        raise WorkbenchError(f"{label.capitalize()} needs {minimum} to {maximum} entries.", label)
    return value


def expect_revision(body: Mapping) -> int:
    return integer(body.get("revision"), "revision", 1, 2**31 - 1)
