import neovim.api
from typing import Any

from gpt4o.editing_context import EditingContext
from gpt4o.chat_provider import ChatProviderOpenAI
from gpt4o.response_parser import ResponseParser
from gpt4o.types import File, Cursor
from gpt4o.resources import NVIM_BUF_TYPE_MAPS

@neovim.plugin
class NvimPlugin:
    def __init__(self, nvim: neovim.Nvim):
        self.nvim = nvim
        self.provider = ChatProviderOpenAI()

    def alert(self, message: str | Any):
        if not isinstance(message, str):
            message = repr(message)
        self.nvim.command(f'lua vim.notify({repr(message)})')

    def get_buffer_path(self, buffer: neovim.api.Buffer):
        buftype = buffer.options['buftype']
        if buftype:
            return f'[{NVIM_BUF_TYPE_MAPS.get(buftype, buftype).capitalize()}]'
        if not buffer.name:
            return '[No name]'
        return buffer.name

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

    @neovim.command('GPTEdit', nargs='*', range=True)
    def on_GPTEdit(self, args: list[str], range: tuple[int, int]):
        question = ' '.join(args)

        files = self.get_files()
        cursor = self.get_cursor()
        context = EditingContext(files=files, cursor=cursor)

        prompt = context.compose_prompt(question)
        response = self.provider.query_prompt(prompt, force_json=True, seed=42)

        parser = ResponseParser()
        operation = parser.parse_response(response)
        type(operation).__name__
