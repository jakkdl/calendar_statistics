#from icalendar.prop import vText as vText
from datetime import datetime
from typing import Any

class Event:
    uid: int
    summary: str
    description: str
    start: datetime
    end: datetime
    all_day: bool
    recurring: bool
    location: str
    private: bool
    created: datetime
    last_modified: datetime
    sequence: Any
    attendee: Any
    organizer: str
    def __init__(self) -> None: ...
    def __lt__(self, other: Event) -> bool: ...
