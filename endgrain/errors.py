from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class WorkbenchError(Exception):
    message: str
    field: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)
    status: int = 400

    def __str__(self) -> str:
        return self.message

    def payload(self) -> dict[str, Any]:
        outcome = {"error": self.message}
        if self.field:
            outcome["field"] = self.field
        if self.details:
            outcome["details"] = self.details
        return outcome


class Missing(WorkbenchError):
    def __init__(self, thing: str):
        super().__init__(f"{thing} was not found.", status=404)


class Conflict(WorkbenchError):
    def __init__(self, message: str, **details: Any):
        super().__init__(message, details=details, status=409)


class Limits(WorkbenchError):
    def __init__(self, message: str, field_name: str | None = None):
        super().__init__(message, field=field_name, status=413)
