from dataclasses import dataclass
import neovim.api
from typing import Any

from gpt4o.editing_context import EditingContext
from gpt4o.chat_provider import ChatProviderOpenAI
from gpt4o.response_parser import ResponseParser
from gpt4o.types import File, Cursor, Prompt
from gpt4o.operations import OperationVisitor
from gpt4o.resources import NVIM_BUF_TYPE_MAPS

@neovim.plugin
class NvimPlugin:
    def __init__(self, nvim: neovim.Nvim):
        self.nvim = nvim
        self.provider = ChatProviderOpenAI()

    def alert(self, message: str | Any, level: str = 'INFO'):
        # TODO: use NVIM_PYTHON_LOG_FILE instead?
        if not isinstance(message, str):
            message = repr(message)
        self.nvim.command(f'lua vim.notify and vim.notify({repr(message)}, vim.log.levels.{level.upper()})')

    def get_buffer_path(self, buffer: neovim.api.Buffer) -> str:
        buftype = buffer.options['buftype']
        if buftype:
            return f'[{NVIM_BUF_TYPE_MAPS.get(buftype, buftype).capitalize()}]'
        if not buffer.name:
            return '[No name]'
        return buffer.name

    def find_buffer_by_path(self, path: str) -> neovim.api.Buffer:
        for buffer in self.nvim.buffers:
            if self.get_buffer_path(buffer) == path:
                return buffer
        raise ValueError(f'Could not find buffer with path {repr(path)}')

    def get_files(self) -> list[File]:
        files: list[File] = []
        for buffer in self.nvim.buffers:
            file = File(path=self.get_buffer_path(buffer), content=buffer[:])
            files.append(file)
        return files

    def get_cursor(self) -> Cursor:
        line, col = self.nvim.current.window.cursor
        path = self.get_buffer_path(self.nvim.current.buffer)
        cursor = Cursor(path=path, line=line, col=col + 1)
        return cursor

    def compose_prompt(self, question: str) -> Prompt:
        files = self.get_files()
        cursor = self.get_cursor()
        context = EditingContext(files=files, cursor=cursor)

        prompt = context.compose_prompt(question)
        return prompt

    @neovim.command('GPTEdit', nargs='*', range=True)
    def on_GPTEdit(self, args: list[str], range: tuple[int, int]):
        question = ' '.join(args)
        _ = range

        prompt = self.compose_prompt(question)
        response = self.provider.query_prompt(prompt, force_json=True, seed=42)

        parser = ResponseParser()
        operation = parser.parse_response(response)
        visitor = NvimOperationVisitor(self)
        operation.accept(visitor)

@dataclass
class NvimOperationVisitor(OperationVisitor):
    parent: NvimPlugin

    def visit_replace(self, op):
        buffer = self.parent.find_buffer_by_path(op.path)
        buffer[op.line_start:op.line_end + 1] = op.content

    def visit_delete(self, op):
        buffer = self.parent.find_buffer_by_path(op.path)
        buffer[op.line_start:op.line_end + 1] = []

    def visit_append(self, op):
        buffer = self.parent.find_buffer_by_path(op.path)
        buffer[op.line:op.line] = op.content

    def visit_nop(self, op):
        _ = op
        self.parent.alert('Not a valid change request', 'WARN')
