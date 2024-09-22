from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

class Operation(ABC):
    @abstractmethod
    def accept(self, visitor: 'OperationVisitor'):
        pass

@dataclass
class OperationReplace(Operation):
    file: str
    line_start: int
    line_end: int
    content: list[str]

    def accept(self, visitor):
        visitor.visit_replace(self)

@dataclass
class OperationDelete(Operation):
    file: str
    line_start: int
    line_end: int

    def accept(self, visitor):
        visitor.visit_delete(self)

@dataclass
class OperationAppend(Operation):
    file: str
    line: int
    content: list[str]

    def accept(self, visitor):
        visitor.visit_append(self)

@dataclass
class OperationNop(Operation):
    def accept(self, visitor):
        visitor.visit_nop(self)

OPERATIONS: dict[str, Type] = dict(
    replace=OperationReplace,
    delete=OperationDelete,
    append=OperationAppend,
    nop=OperationNop,
)

class OperationVisitor(ABC):
    @abstractmethod
    def visit_replace(self, op: OperationReplace):
        pass

    @abstractmethod
    def visit_delete(self, op: OperationDelete):
        pass

    @abstractmethod
    def visit_append(self, op: OperationAppend):
        pass

    @abstractmethod
    def visit_nop(self, op: OperationNop):
        pass
