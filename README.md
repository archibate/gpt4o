# gpt4o.nvim

Blazing fast 🚀 code assistant in NeoVim powered by 🤖 GPT-4o, offering intelligent code completion and editing to elevate developer productivity.

## Features
- Context-aware suggestions 🌍
- Real-time code completion ⚡
- Incremental streaming result ⏳
- Edit existing code bases 🛠️
- Recognizing terminal errors ⚠️
- Fix your code in place 📝
- Multi-language support 🌐
- Customizable settings ⚙️
- Lightweight and efficient 💨

## Installation
To install gpt4o.nvim, first make sure you have python3 support on your NeoVim.

You may do this by running:

```bash
python3 -m pip install neovim openai
```

So far so good, you should see `OK` in the python3 support when running `:checkhealth`.

Now add the following to your `~/.config/nvim/init.vim`, with your plugin manager, [vim-plug](https://github.com/junegunn/vim-plug) for example:

```vim
Plug 'archibate/gpt4o.nvim'
```

Then, run `:PlugInstall` (or whatever your plugin manager name it) in NeoVim.

We are almost done! The only thing remain is to find an LLM provider.

## LLM providers

### OpenAI (gpt4o)
Goto [OpenAI platform](https://platform.openai.com/api-keys), create a new API key. You want to register an OpenAI account if you haven't yet.

There is a free trial of $5 budget if I recall correctly. Each invocation will consume this budget, and if the free trial has been exhausted, you'll need to set up a payment method (e.g. VISA) to continue using the service.

> The API key will look like this: `sk-xxxxxxxxxxxxx`, do not share it with other people.

Then, append a line to your `~/.bashrc` (`~/.zshrc` for Zsh users):
```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxx"
```

Restart your shell, enter NeoVim and try run `:GPT` and have fun!

> The plugin will invoke `https://api.openai.com` in the background for you for code completion 😎

### DeepSeek (deepseek-coder)
DeepSeek is a Chinese company that specializes in AI-driven programming tools and code assistance solutions. Their slogan is:

> Unravel the mystery of AGI with curiosity. Answer the essential question with long-termism 😎

DeepSeek conveniently provides an OpenAI compatible API at `https://api.deepseek.com`. As you may have already guessed, goto [DeepSeek Platform](https://platform.deepseek.com/api_keys) to obtain an API key. Register an DeepSeek account if you haven't one yet.

> They provide ¥20 free trial for every new customer too; but make sure you use it within 1 month! Otherwise, it seems to become expired and no longer usable. ☹️

Anyway, append these lines to your `~/.bashrc` (`~/.zshrc` for Zsh users):
```bash
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_API_KEY="sk-xxxxxxxxxxxxx"
```

> Note that `OPENAI_BASE_URL` is required for overriding the default OpenAI API into an OpenAI-compatible API (`https://api.deepseek.com` in this case).

## Usage
Once installed and configured, enter VISUAL mode by selecting the region of code you want to edit. Then invoke gpt4o by typing `:GPT <question>`, where `<question>` is your question or instruction to ask gpt4o to help, for example, `:GPT refactor this code` or `:GPT extract chunk_size into variable`. The assistant will edit the selected section of code to fulfill your instruction, the context of your code around are considered.

If no `<question>` provided, gpt4o will try to complete and fix possible mistakes in the editing code.

If you invoke `:GPT` with no selection, i.e. not in VISUAL mode, gpt4o will only edit the current line of code where cursor located at, a single line.

May also instead invoke `:GPT4` to allow gpt4o editing ± 4 lines of code around the cursor, which is usually the small fraction of code you'd like to edit.

> 😂 In case you missed my laughing point: GPT4 = GPT4o with ± 4 lines editing ability 🤣🎉 Hope you find this fun...

## Configuration
You can customize the behavior of gpt4o by adding the following to your `init.vim` or `init.lua`:

```vim
:GPTSetup {
  \ "terminal_history_lines": 100,
  \ "look_back_lines": 180,
  \ "look_ahead_lines": 80,
  \ "limit_attach_lines": 400,
  \ "extra_range_lines": 4,
  \ "api_key": v:null,
  \ "base_url": v:null,
  \ "organization": v:null,
  \ "project": v:null,
  \ "model": "auto",
  \ "max_tokens": v:null,
  \ "temperature": v:null,
  \ "frequency_penalty": v:null,
  \ "presence_penalty": v:null,
  \ "include_usage": v:true,
  \ "include_time": v:true,
  \ "timeout": v:null,
\ }
```

## Contributing
Contributions are welcome! If you'd like to help improve gpt4o.nvim, please open a pull request or issue on the repository. Here are some ways you can contribute:
- Report bugs or suggest new features to help us enhance functionality.
- Submit code contributions to fix bugs or add new features, following the project's coding standards.
- Improve documentation by submitting edits or examples to assist other users in using gpt4o.nvim effectively.
- Share your experiences and best practices in the community forums to foster collaboration and knowledge sharing.
- Engage with other developers by participating in discussions or providing feedback on proposed changes.
- Create tutorials or guides that showcase how to leverage gpt4o.nvim for various use cases and programming languages.
- Test updates and new features thoroughly to ensure they perform as expected and report any issues encountered during testing.
- Promote gpt4o.nvim in relevant developer communities to increase awareness and user adoption.

We appreciate your involvement and look forward to your contributions!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. By contributing to this project, you agree that your contributions will also be licensed under the same MIT License.

## Fun Fact
The README above is written with gpt4o.nvim 😄
![awesome-face](https://142857.red/book/img/favicon.ico)

Usage showcase:
```markdown
May also instead invoke `:GPT4` to allow gpt4o editing +- 4 lines of code around the cursor.
```
And I type `:GPT make this +- the real +- symbol` and gpt4o automatically fix that for me 😎
```markdown
May also instead invoke `:GPT4` to allow gpt4o editing ± 4 lines of code around the cursor.
```
So emojis in the 'Features' section are added by gpt4o too... I can finally uninstall my emoji input method 💔 Have fun!
