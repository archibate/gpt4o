from dataclasses import dataclass

@dataclass
class File:
    path: str
    content: list[str]

@dataclass
class Cursor:
    path: str
    line: int
    col: int

@dataclass
class Prompt:
    instruction: str
    question: str
