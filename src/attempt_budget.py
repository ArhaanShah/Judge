import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class AttemptBudget:
    def __init__(self, path: Path, cap: int, model: str, config_sha256: str):
        self.path = Path(path)
        self.cap = cap
        self.model = model
        self.config_sha256 = config_sha256
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
    def current(self) -> int:
        if not self.path.exists():
            return 0
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data.get("api_attempts", 0)
        
    def increment(self) -> int:
        val = self.current()
        if val >= self.cap:
            raise RuntimeError(f"Hard API attempt cap of {self.cap} reached")
        val += 1
        data = {
            "api_attempts": val,
            "model": self.model,
            "config_sha256": self.config_sha256,
            "updated_at": utc_now()
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return val
