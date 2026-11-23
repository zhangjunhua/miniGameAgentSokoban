"""Append-only JSON Lines for model interactions in one server process."""

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4


class ModelCallLog:
    def __init__(self, directory: Path):
        started = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
        self.path = directory / f"model_calls_{started}_{uuid4().hex[:8]}.jsonl"
        self.lock = Lock()

    def append(self, call_id: str, event: str, fields: dict) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "call_id": call_id,
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(line)
