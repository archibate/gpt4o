from dataclasses import dataclass

@dataclass
class RefFile:
    path: str
    content: str
    type: str = ''
    score: int = 1
    special_name: str = ''
