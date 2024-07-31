import pathlib
import traceback
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
from annotated_types import T
import neovim
import openai
import time
import re



@neovim.plugin
class GPT4oPlugin:
    @dataclass
    class Config:
        terminal_history_lines: int = 100
        look_back_lines: int = 180
        look_ahead_lines: int = 80
        limit_attach_lines: int = 400
        extra_range_lines: int = 4
        api_key: Optional[str] = None
        base_url: Optional[str] = None
        organization: Optional[str] = None
        project: Optional[str] = None
        model: str = 'auto'
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None
        frequency_penalty: Optional[float] = None
        presence_penalty: Optional[float] = None
        include_usage: bool = True
        include_time: bool = True
        timeout: Optional[float] = None

    def __init__(self, nvim: neovim.Nvim):
        self.nvim = nvim
        self.running = False
        self.cfg = self.Config()
        self._real_client = None

    @property
    def client(self):
        if not self._real_client:
            self._real_client = openai.OpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                organization=self.cfg.organization,
                project=self.cfg.project,
            )
        return self._real_client

    def log(self, line):
        with open('/tmp/gpt4o.log', 'a') as f:
            print(line, file=f)
            print('======', file=f)

    def check_price(self, model, usage):
        pricing = {
            'gpt-4o': (5, 15),
            'gpt-4o-mini': (0.15, 0.6),
            'gpt-4-turbo': (10, 30),
            'gpt-4': (30, 60),
            'gpt-4-32k': (60, 120),
            'gpt-3.5-turbo': (3, 6),
            'deepseek-coder': (0.14, 0.28),
        }
        if model in pricing:
            prompt_price, completion_price = pricing[model]
            price = (usage.prompt_tokens * prompt_price + usage.completion_tokens * completion_price) / 1000000
            self.log(f'${price:.6f}')

    def check_time(self, sequence: Iterable[T]) -> Iterable[T]:
        if not self.cfg.include_time:
            return sequence
        start_t = time.monotonic()
        first_chunk_t = start_t
        for chunk in sequence:
            if first_chunk_t is None:
                first_chunk_t = time.monotonic()
            yield chunk
        end_t = time.monotonic()
        assert start_t and first_chunk_t
        total_time = end_t - start_t
        latency_time = first_chunk_t - start_t
        self.log(f'total {total_time * 1000:.0f} ms, latency {latency_time * 1000:.0f} ms')

    def streaming_response(self, question) -> Iterable[str]:
        try:
            model = self.cfg.model
            if model == 'auto':
                if 'deepseek' in self.client.base_url.host:
                    model = 'deepseek-coder'
                else:
                    model = 'gpt-4o'
            self.log(question)
            completion = self.client.chat.completions.create(
                model=model,
                temperature=self.cfg.temperature,
                frequency_penalty=self.cfg.frequency_penalty,
                presence_penalty=self.cfg.presence_penalty,
                max_tokens=self.cfg.max_tokens,
                messages=[
                    {"role": "user", "content": question},
                ],
                stream=True,
                stream_options={"include_usage": True} if self.cfg.include_usage else None,
                timeout=self.cfg.timeout,
            )
            for chunk in self.check_time(completion):
                if chunk.usage:
                    self.check_price(model, chunk.usage)
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
        except openai.OpenAIError:
            error = traceback.format_exc()
            self.log(error)
            yield error

    def do_ask(self, question: str, delete_callback: Optional[Callable] = None):
        self.nvim.command('set paste')
        try:
            undojoin = False
            for chunk in self.streaming_response(question):
                if not self.running:
                    break
                if delete_callback:
                    delete_callback()
                    delete_callback = None
                    undojoin = True
                if not undojoin:
                    self.nvim.command('normal! a' + chunk)
                    undojoin = True
                else:
                    self.nvim.command('undojoin|normal! a' + chunk)
                self.nvim.command('redraw')
        finally:
            self.nvim.command('set nopaste')

    def buffer_near(self, range_: tuple[int, int]) -> str:
        start, end = range_
        start = max(0, start - self.cfg.look_back_lines)
        end = min(len(self.nvim.current.buffer), end + self.cfg.look_ahead_lines)
        range_ = start, end
        return self.buffer_slice(range_)

    def buffer_slice(self, range_: tuple[int, int]) -> str:
        start, end = range_
        lines = self.nvim.current.buffer[start:end]
        return self.escape_triple_quotes('\n'.join(lines))

    def buffer_empty_slice(self, range_: tuple[int, int]):
        start, end = range_
        self.nvim.current.buffer[start:end] = ['']

    def fetch_terminal(self):
        content = []
        for buffer in self.nvim.buffers:
            if buffer.name.startswith('term://'):
                content += buffer[-self.cfg.terminal_history_lines:]
        return '\n'.join(content)

    def escape_triple_quotes(self, text: str):
        # e.g.: ``` -> \`\`\`, maybe using re
        return text

    ATTACH_FILES = re.compile(r'attach\s*file:\s*(\S+)(?=\s|$)')

    def gather_attach_files(self, current_file: str, context: str) -> str:
        attach_files: list[str] = re.findall(self.ATTACH_FILES, context)
        current_path = pathlib.Path(current_file).parent if current_file else '.'
        if not attach_files:
            return ''
        attach = 'The following attached files might be useful:\n\n'
        current_file = pathlib.Path(current_file).parts[-1]
        for file in attach_files:
            path = pathlib.Path(file)
            if not path.is_absolute():
                abs_path = current_path / path
            else:
                abs_path = path
            if abs_path.exists():
                with open(abs_path) as f:
                    content = f.read()
                lines = content.split('\n')
                if len(lines) > self.cfg.limit_attach_lines:
                    if file == current_file:
                        lines = context.splitlines()
                    else:
                        lines = lines[:self.cfg.limit_attach_lines]
                content = self.escape_triple_quotes('\n'.join(lines))
                attach += f'### File: `{file}`\n### Content:\n```\n{content}\n```\n\n'
            else:
                self.nvim.command(f'echoerr "attach file not found: {file}"')
        attach += f'### Instruction:\n'
        return attach

    def do_prompt(self, question: str, range_: tuple[int, int], include_terminal: bool = False):
        if include_terminal:
            terminal = self.fetch_terminal().strip()
        else:
            terminal = ''
        code = self.buffer_slice(range_)
        if range_[1] - range_[0] >= self.cfg.look_back_lines + self.cfg.look_ahead_lines:
            context = code
        else:
            context = self.buffer_near(range_)
        delete_callback = None
        if code.strip():
            delete_callback = lambda: self.buffer_empty_slice(range_)
        attach = self.gather_attach_files(self.nvim.current.buffer.name, context)
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
                prompt = f'{attach}Based on the following context:\n```\n{context}\n```\n{terminal}Edit the following section of code:\n```\n{code}\n```\n{question}Output code only. Do not explain. No code blocks.'
            elif range_[1] == len(self.nvim.current.buffer):
                if question:
                    question = f'Append code precisely follow the user request: {question}\n'
                else:
                    question = 'Append code to make it complete.\n'
                if terminal:
                    terminal = f'And the following terminal output:\n```\n{terminal}\n```\n'
                else:
                    terminal = ''
                prompt = f'{attach}Based on the following code:\n```\n{context}\n```\n{terminal}{question}Output code only. Do not explain. No code blocks.'
            else:
                lines = context.splitlines()
                lines[range_[0]:range_[1]] = ['<INSERT>']
                context = '\n'.join(lines)
                if question:
                    question = f'Insert code at the `<INSERT>` mark to precisely follow the user request: {question}\n'
                else:
                    question = 'Insert code at the `<INSERT>` mark to make it complete.\n'
                if terminal:
                    terminal = f'And the following terminal output:\n```\n{terminal}\n```\n'
                else:
                    terminal = ''
                prompt = f'{attach}Based on the following code:\n```\n{context}\n```\n{terminal}{question}Output code only. Do not explain. No code blocks.'
        elif code.strip():
            if question:
                question = f'precisely follow the user request: {question}\n'
            else:
                question = 'to make it complete.'
            if terminal:
                terminal = f'Based on the following terminal output:\n```\n{terminal}\n```\n'
            else:
                terminal = ''
            prompt = f'{attach}{terminal}Edit the following code:\n```\n{code}\n```\n{question}Output code only. Do not explain. No code blocks.'
        else:
            if question:
                question = f'Write code precisely follow the user request: {question}\n'
            elif attach:
                question = 'Write some code based on the attached files above. '
            elif terminal:
                question = 'Write some code based on the terminal output above. '
            else:
                question = 'Write some interesting code to inspire me. '
            if terminal:
                terminal = f'Based on the following terminal output:\n```\n{terminal}\n```\n'
            else:
                terminal = ''
            prompt = f'{attach}{terminal}{question}Output code only. Do not explain. No code blocks.'
        self.do_ask(question=prompt, delete_callback=delete_callback)

    def do_gpt_range(self, args, range_, bang):
        if self.running:
            self.running = False
            return
        self.running = True
        try:
            if args:
                question = ' '.join(args)
            else:
                question = ''
            range_ = (range_[0] - 1, range_[1])
            assert range_[0] <= range_[1], range_
            self.do_prompt(question=question, range_=range_, include_terminal=bang)
        finally:
            self.running = False

    @neovim.command('GPT', nargs='*', range=True, bang=True)
    def on_GPTLine(self, args, range_, bang):
        self.do_gpt_range(args, range_, bang)

    @neovim.command('GPT4', nargs='*', range=True, bang=True)
    def on_GPT4(self, args, range_, bang):
        start, end = range_
        start -= self.cfg.extra_range_lines
        end += self.cfg.extra_range_lines
        if start < 1:
            start = 1
        num_lines = len(self.nvim.current.buffer)
        if end > num_lines:
            end = num_lines
        range_ = start, end
        self.do_gpt_range(args, range_, bang)

    @neovim.function('GPTSetup')
    def on_GPTSetup(self, args):
        assert isinstance(args, dict)
        for k, v in args:
            setattr(self.cfg, k, v)
