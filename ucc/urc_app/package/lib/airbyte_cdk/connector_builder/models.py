# Stub for connector_builder.models (removed from trimmed CDK)
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LogMessage:
    message: str
    level: str = "INFO"
    stack_trace: Optional[str] = None
