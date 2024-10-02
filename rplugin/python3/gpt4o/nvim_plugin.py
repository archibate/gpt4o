from dataclasses import dataclass
import neovim.api
from typing import Any, List

from gpt4o.editing_context import EditingContext
from gpt4o.context_simplifier import ContextSimplifier
from gpt4o.chat_provider import ChatProviderOpenAI
from gpt4o.token_counter import TokenCounterOpenAI
from gpt4o.embed_provider import EmbedProviderOpenAI
from gpt4o.response_parser import ResponseParser
from gpt4o.types import Diagnostic, File, Cursor, Prompt, RecentChange
from gpt4o.operations import OperationVisitor

@neovim.plugin
class NvimPlugin:
    def __init__(self, nvim: neovim.Nvim):
        self.nvim = nvim
        self.chat_provider = ChatProviderOpenAI()
        self.embed_provider = EmbedProviderOpenAI()
        self.context_simplifier = ContextSimplifier(self.embed_provider)
        self.response_parser = ResponseParser()
        self.token_counter = TokenCounterOpenAI()

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

    def get_recent_changed_buffers(self) -> List[neovim.api.Buffer]:
        modified_buffers = [buffer for buffer in self.nvim.buffers if buffer.options['modified']]
        return modified_buffers

    # def get_recent_changes(self) -> List[RecentChange]:
    #     recent_changes = []
    #     for buffer in self.get_recent_changed_buffers():
    #         path = self.get_buffer_path(buffer)
    #         for line_num, line in enumerate(buffer, start=1):
    #             if buffer.options['modified'] and line.strip():
    #                 recent_changes.append(RecentChange(file=path, line=line_num, col=1, change=line))
    #     return recent_changes

    def get_diagnostics(self) -> List[Diagnostic]:
        diagnostics = self.nvim.call('luaeval', 'vim.diagnostic.get()')
        result = []
        for diag in diagnostics:
            bufnr = diag['bufnr']
            file = self.get_buffer_path(self.nvim.buffers[bufnr])
            line = diag['lnum'] + 1
            col = diag['col'] + 1
            code = self.nvim.current.buffer[line - 1:line]
            code = '\n'.join(code)
            message = diag['message']
            type = ['error', 'warning', 'info', 'hint'][diag['severity'] - 1]
            result.append(Diagnostic(
                type=type,
                message=message,
                file=file,
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
        tokens = self.token_counter.count_token(f'{prompt.instruction}\n{prompt.question}')
        self.alert(f'{prompt.question}\n\nTokens: {tokens}')

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
