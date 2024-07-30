import traceback
from typing import Iterable, Optional
import neovim
import openai

client = openai.OpenAI()

def log(line):
    with open('/tmp/gpt4o.log', 'a') as f:
        print(line, file=f)
        print('======', file=f)

def check_price(model, usage):
    pricing = {
        'gpt-4o': (5, 15),
        'gpt-4o-mini': (0.15, 0.6),
        'gpt-4-turbo': (10, 30),
        'gpt-4': (30, 60),
        'gpt-4-32k': (60, 120),
        'gpt-3.5-turbo': (3, 6),
    }
    prompt_price, completion_price = pricing[model]
    price = (usage.prompt_tokens * prompt_price + usage.completion_tokens * completion_price) / 1000000
    log(f'${round(price, 6):.6f}')

def ask(question, model: str = 'gpt-4o-mini', max_tokens: Optional[int] = None, temperature: Optional[float] = None, frequency_penalty: Optional[float] = None, presence_penalty: Optional[float] = None, include_usage: bool = False, timeout: Optional[float] = None) -> Iterable[str]:
    try:
        log(question)
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": question},
            ],
            stream=True,
            stream_options={"include_usage": True} if include_usage else None,
            timeout=timeout,
        )
        for chunk in completion:
            if chunk.usage:
                check_price(model, chunk.usage)
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except openai.OpenAIError:
        error = traceback.format_exc()
        log(error)
        yield error


@neovim.plugin
class MyPlugin:
    def __init__(self, nvim: neovim.Nvim):
        self.nvim = nvim

    def do_ask(self, question: str):
        self.nvim.command('set paste')
        try:
            first = True
            for chunk in ask(question):
                if first:
                    self.nvim.command('normal! a' + chunk)
                    first = False
                else:
                    self.nvim.command('undojoin|normal! a' + chunk)
                self.nvim.command('redraw')
        finally:
            self.nvim.command('set nopaste')

    def buffer_near(self, range_: tuple[int, int], delta: tuple[int, int]) -> str:
        start, end = range_
        start = max(0, start - delta[0])
        end = min(len(self.nvim.current.buffer), end + delta[1])
        range_ = start, end
        return self.buffer_slice(range_)

    def buffer_slice(self, range_: tuple[int, int]) -> str:
        start, end = range_
        lines = self.nvim.current.buffer[start:end]
        return '\n'.join(lines)

    def buffer_empty_slice(self, range_: tuple[int, int]):
        start, end = range_
        self.nvim.current.buffer[start:end] = ['']

    def fetch_terminal(self):
        content = []
        for buffer in self.nvim.buffers:
            if buffer.name.startswith('term://'):
                content += buffer[-100:]
        return '\n'.join(content)

    def form_prompt(self, question: str, range_: tuple[int, int], include_terminal: bool = False) -> str:
        if include_terminal:
            terminal = self.fetch_terminal().strip()
        else:
            terminal = ''
        code = self.buffer_slice(range_)
        if range_[1] - range_[0] > 100:
            context = code
        else:
            context = self.buffer_near(range_, (200, 80))
        if code.strip():
            self.buffer_empty_slice(range_)
        if context.strip() and context != code:
            if code.strip():
                if question:
                    question = f'precisely follow the user request: {question}\n'
                else:
                    question = 'to make the code complete.\n'
                if terminal:
                    terminal = f'And the following terminal output:\n```\n{terminal}\n```\n'
                else:
                    terminal = ''
                return f'Based on the following context:\n```\n{context}\n```\n{terminal}Edit the following section of code:\n```\n{code}\n```\n{question}Output code only. Do not explain. No code blocks.'
            else:
                context = self.buffer_slice((0, range_[0]))
                if question:
                    question = f'Write code precisely follow the user request: {question}\n'
                else:
                    question = 'Append code to make it complete.\n'
                if terminal:
                    terminal = f'And the following terminal output:\n```\n{terminal}\n```\n'
                else:
                    terminal = ''
                return f'Based on the following context:\n```\n{context}\n```\n{terminal}{question}Output code only. Do not explain. No code blocks.'
        elif code.strip():
            if question:
                question = f'precisely follow the user request: {question}\n'
            else:
                question = ''
            if terminal:
                terminal = f'Based on the following terminal output:\n```\n{terminal}\n```\n'
            else:
                terminal = ''
            return f'{terminal}Edit the following code:\n```\n{code}\n```\n{question}Output code only. Do not explain. No code blocks.'
        else:
            if question:
                question = f'Write code precisely follow the user request: {question}\n'
            else:
                question = 'Write some random code to inspire me. '
            if terminal:
                terminal = f'Based on the following terminal output:\n```\n{terminal}\n```\n'
            else:
                terminal = ''
            return f'{terminal}{question}Output code only. Do not explain. No code blocks.'

    @neovim.command('GPT', nargs='*', range=True, bang=True)
    def on_GPT(self, args, range_, bang):
        if args:
            question = ' '.join(args)
        else:
            question = ''
        range_ = (range_[0] - 1, range_[1])
        assert range_[0] >= range_[1], range_
        self.do_ask(self.form_prompt(question, range_, bang))
