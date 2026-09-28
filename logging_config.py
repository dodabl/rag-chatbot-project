import logging
from logging import DEBUG
import sys
import time
import json
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    _RESERVED = set(logging.LogRecord(
        "", 0, "", 0, "", (), None
    ).__dict__.keys()) | {"message", "asctime"}

    def format(self, record=logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
            }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value
        return json.dumps(payload, default=str)
def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True

        
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

class Timer:
    async def __aenter__(self):
        self._start = time.perf_counter()
        return self
    async def __aexit__(self, *exc):
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)