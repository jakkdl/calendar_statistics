from .icalparser import Event
from typing import Any, Optional


def events(url: Optional[str] = ..., file: Any | None = ..., string_content: Any | None = ..., start: Any | None = ..., end: Any | None = ..., fix_apple: bool = ..., http: Any | None = ...) -> list[Event]: ...
