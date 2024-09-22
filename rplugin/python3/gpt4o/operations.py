from dataclasses import dataclass
from typing import Type

class Operation:
    pass

@dataclass
class OperationReplace:
    line_start: int
    line_end: int
    content: list[str]

@dataclass
class OperationDelete:
    line_start: int
    line_end: int

@dataclass
class OperationInsert:
    line_start: int
    content: list[str]

@dataclass
class OperationNop:
    pass

OPERATIONS: dict[str, Type] = dict(
    replace=OperationReplace,
    delete=OperationDelete,
    insert=OperationInsert,
    nop=OperationNop,
)
