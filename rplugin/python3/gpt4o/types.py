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
    code: str

@dataclass
class Diagnostic:
    type: str
    message: str
    file: str
    line: int
    col: int
    code: str

@dataclass
class Prompt:
    instruction: str
    question: str
    force_json: bool = False

@dataclass
class RecentChange:
    file: str
    line: int
    col: int
    change: str
