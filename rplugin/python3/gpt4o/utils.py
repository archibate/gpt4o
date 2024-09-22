import json
from typing import Any

def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))

def json_loads(data: str) -> Any:
    return json.loads(data)
