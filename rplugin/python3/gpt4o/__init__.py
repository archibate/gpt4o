import neovim
from typing import Any

@neovim.plugin
class NvimPlugin:
    def __init__(self, nvim: neovim.Nvim):
        self.nvim = nvim

    def alert(self, message: str | Any):
        if not isinstance(message, str):
            message = repr(message)
        self.nvim.command(f'lua vim.notify({repr(message)})')

    @neovim.command('GPTEdit', nargs='*', range=True)
    def on_GPTEdit(self, args: list[str], range: tuple[int, int]):
        question = ' '.join(args)

        self.alert([question, range])
