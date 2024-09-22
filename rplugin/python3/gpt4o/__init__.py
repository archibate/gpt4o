try:
    __import__('neovim')
except ImportError:
    pass
else:
    from gpt4o.nvim_plugin import NvimPlugin as _
