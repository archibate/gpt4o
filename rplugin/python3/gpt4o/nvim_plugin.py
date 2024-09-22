from dataclasses import dataclass
import neovim.api
from typing import Any, List

from gpt4o.editing_context import EditingContext
from gpt4o.context_simplifier import ContextSimplifier
from gpt4o.chat_provider import ChatProviderOpenAI
from gpt4o.embed_provider import EmbedProviderFastEmbed
from gpt4o.response_parser import ResponseParser
from gpt4o.types import Diagnostic, File, Cursor, Prompt
from gpt4o.operations import OperationVisitor
from gpt4o.utils import json_dumps, json_loads

@neovim.plugin
class NvimPlugin:
    def __init__(self, nvim: neovim.Nvim):
        self.nvim = nvim
        self.chat_provider = ChatProviderOpenAI()
        self.embed_provider = EmbedProviderFastEmbed()
        self.context_simplifier = ContextSimplifier(self.embed_provider)
        self.response_parser = ResponseParser()

    def alert(self, message: str | Any, level: str = 'Normal'):
        if not isinstance(message, str):
            message = repr(message)
        self.log(f'{level}: {message}')
        self.nvim.api.echo([[message, level]], True, [])

    def log(self, message: str | Any):
        if not isinstance(message, str):
            message = repr(message)
        if '\n' in message:
            message = f'{">"*12}\n{message}\n{"<"*12}\n'
        else:
            message = f'{message}\n'
        with open('/tmp/gpt4o.log', 'a') as f:
            f.write(message)

    def get_buffer_path(self, buffer: neovim.api.Buffer) -> str:
        buftype = buffer.options['buftype']
        if buftype:
            return f'[{buftype.capitalize()}]'
        if not buffer.name:
            return '[No name]'
        path = buffer.name
        cwd = self.nvim.call('getcwd') + '/'
        path = path.removeprefix(cwd)
        return path

    def find_buffer_by_path(self, path: str) -> neovim.api.Buffer:
        for buffer in self.nvim.buffers:
            if self.get_buffer_path(buffer) == path:
                return buffer
        raise ValueError(f'Could not find buffer with path {repr(path)}')

    def polish_buffer_content(self, content: List[str]) -> List[str]:
        while len(content) and content[-1].strip() == '':
            content.pop(-1)
        return content

    def get_diagnostics(self) -> List[Diagnostic]:
        diagnostics = json_loads(self.nvim.call('luaeval', 'vim.fn.json_encode(vim.lsp.diagnostic.get_line_diagnostics())'))
        result = []
        for diag in diagnostics:
            line = diag['range']['start']['line'] + 1
            col = diag['range']['start']['character'] + 1
            code = self.nvim.current.buffer[line - 1:line]
            code = '\n'.join(code)
            message = diag['message']
            type = ['error', 'warning', 'info', 'hint'][diag['severity'] - 1]
            result.append(Diagnostic(
                type=type,
                message=message,
                line=line,
                col=col,
                code=code,
            ))
        return result

    def get_files(self) -> List[File]:
        files: List[File] = []
        a = 1
        for buffer in self.nvim.buffers:
            buftype = buffer.options['buftype']
            if buftype == 'nofile':
                continue
                # if len(buffer) == 0 or buffer[0] == '':
                #     continue
                # else:
                #     buftype = buffer.options['filetype']
                #     buftype = NVIM_FILE_TYPE_MAPS.get(buftype, buftype)
            content = self.polish_buffer_content(buffer[:])
            path = self.get_buffer_path(buffer)
            file = File(path=path, content=content)
            files.append(file)
        return files

    def get_cursor(self) -> Cursor:
        line, col = self.nvim.current.window.cursor
        path = self.get_buffer_path(self.nvim.current.buffer)
        code = self.nvim.current.buffer[line - 1]
        cursor = Cursor(path=path, line=line, col=col + 1, code=code)
        return cursor

    def compose_prompt(self, question: str) -> Prompt:
        files = self.get_files()
        cursor = self.get_cursor()
        diagnostics = self.get_diagnostics()
        context = EditingContext(files=files, cursor=cursor, diagnostics=diagnostics)
        self.log(f'all files: {[file.path for file in files]}')
        self.context_simplifier.simplify(context)
        self.log(f'files after simplify: {[file.path for file in files]}')
        prompt = context.compose_prompt(question)
        return prompt

    @neovim.command('GPTInfo', nargs='*', range=True)
    def on_GPTInfo(self, args: List[str], range: tuple[int, int]):
        question = ' '.join(args)
        _ = range

        prompt = self.compose_prompt(question)
        self.alert(prompt.question)

    @neovim.command('GPTEdit', nargs='*', range=True)
    def on_GPTEdit(self, args: List[str], range: tuple[int, int]):
        question = ' '.join(args)
        _ = range

        prompt = self.compose_prompt(question)
        self.log(prompt.question)
        response = self.chat_provider.query_prompt(prompt, force_json=True, seed=42)
        response = ''.join(response)
        self.log(response)

        operation = self.response_parser.parse_response(response)

        visitor = NvimOperationVisitor(self)
        operation.accept(visitor)

@dataclass
class NvimOperationVisitor(OperationVisitor):
    parent: NvimPlugin

    def visit_replace(self, op):
        buffer = self.parent.find_buffer_by_path(op.file)
        buffer[op.line_start - 1:op.line_end] = op.content

    def visit_delete(self, op):
        buffer = self.parent.find_buffer_by_path(op.file)
        if op.line_end > len(buffer):
            op.line_end = len(buffer)
        buffer[op.line_start - 1:op.line_end] = []

    def visit_append(self, op):
        buffer = self.parent.find_buffer_by_path(op.file)
        buffer[op.line:op.line] = op.content

    def visit_prepend(self, op):
        buffer = self.parent.find_buffer_by_path(op.file)
        buffer[op.line - 1:op.line - 1] = op.content

    def visit_nop(self, op):
        _ = op
        self.parent.alert('Nothing we can change', 'WarningMsg')
