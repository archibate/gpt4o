from dataclasses import dataclass

@dataclass
class File:
    path: str
    content: list[str]

@dataclass
class CursorPos:
    path: str
    line: int
    column: int

@dataclass
class Prompt:
    instruction: str
    question: str
