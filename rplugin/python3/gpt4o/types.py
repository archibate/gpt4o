from dataclasses import dataclass
from typing import List

@dataclass
class File:
    path: str
    content: List[str]

@dataclass
class Cursor:
    path: str
    line: int
    col: int

@dataclass
class Prompt:
    instruction: str
    question: str
    force_json: bool = False
