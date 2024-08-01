import pathlib
import traceback
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
from annotated_types import T
import neovim
import openai
import math
import time
import re


@neovim.plugin
class GPT4oPlugin:
    DEFAULT_OPENAI_MODEL = 'gpt-4o'
    INSTRUCTION_EDIT = '''You are a code modification assistant. Your task is to modify the provided code based on the user's instructions.

Rules:
1. Return only the modified code, with no additional text or explanations.
2. The first character of your response must be the first character of the code.
3. The last character of your response must be the last character of the code.
4. NEVER use triple backticks (```) or any other markdown formatting in your response.
5. Do not use any code block indicators, syntax highlighting markers, or any other formatting characters.
6. Present the code exactly as it would appear in a plain text editor, preserving all whitespace, indentation, and line breaks.
7. Maintain the original code structure and only make changes as specified by the user's instructions.
8. Ensure that the modified code is syntactically and semantically correct for the given programming language.
9. Use consistent indentation and follow language-specific style guidelines.
10. If the user's request cannot be translated into code changes, respond only with the word NULL (without quotes or any formatting).
11. Do not include any comments or explanations within the code unless specifically requested.
12. Assume that any necessary dependencies or libraries are already imported or available.

IMPORTANT: Your response must NEVER begin or end with triple backticks, single backticks, or any other formatting characters.'''
    INSTRUCTION_CHAT = '''You are an AI programming assistant.
Follow the user's requirements carefully & to the letter.
Your responses should be informative and logical.
You should always adhere to technical information.
If the user asks for code or technical questions, you must provide code suggestions and adhere to technical information.
If the question is related to a developer, you must respond with content related to a developer.
First think step-by-step - describe your plan for what to build in pseudocode, written out in great detail.
Then output the code in a single code block.
Minimize any other prose.
Keep your answers short and impersonal.
Use Markdown formatting in your answers.
Make sure to include the programming language name at the start of the Markdown code blocks.
Avoid wrapping the whole response in triple backticks.
The user works in an IDE of NeoVim which has a concept for editors with open files, integrated language support, and output pane that shows the output of running the code as well as an integrated terminal.
You can only give one reply for each conversation turn.'''

    @dataclass
    class File:
        path: str
        content: str
        type: str = ''

    @dataclass
    class Config:
        extra_range_lines: int = 4
        api_key: Optional[str] = None
        base_url: Optional[str] = None
        organization: Optional[str] = None
        project: Optional[str] = None
        model: str = 'auto'
        embedding_model: str = 'auto'
        limit_context_tokens: int = 10000
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
    def embedding_model(self) -> str:
        if self.cfg.embedding_model == 'auto':
            if 'openai' in self.client.base_url.host:
                return 'embedding-3-small'
            else:
                return ''
        return self.cfg.embedding_model

    @property
    def client(self):
        if not self._real_client:
            self._real_client = openai.OpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                organization=self.cfg.organization,
                project=self.cfg.project,
                timeout=self.cfg.timeout,
            )
        return self._real_client

    def log(self, line):
        with open('/tmp/gpt4o.log', 'a') as f:
            print(line, file=f)
            print('======', file=f)

    def check_price(self, model, usage):
        if not self.cfg.include_usage:
            return
        pricing = {
            'gpt-4o': (5, 15),
            'gpt-4o-mini': (0.15, 0.6),
            'gpt-4-turbo': (10, 30),
            'gpt-4': (30, 60),
            'gpt-4-32k': (60, 120),
            'gpt-3.5-turbo': (3, 6),
            'deepseek-coder': (0.14, 0.28),
            'text-embedding-3-small': (0.02, 0),
            'text-embedding-3-large': (0.13, 0),
        }
        if model in pricing:
            prompt_price, completion_price = pricing[model]
            price = (usage.prompt_tokens * prompt_price + usage.completion_tokens * completion_price) / 1000000
            self.log(f'${price:.6f} @{model}')

    def check_time(self, sequence: Callable[[], Iterable[T]], model: str) -> Iterable[T]:
        if not self.cfg.include_time:
            return sequence
        start_t = time.monotonic()
        first_chunk_t = None
        for chunk in sequence():
            if chunk and first_chunk_t is None:
                first_chunk_t = time.monotonic()
            yield chunk
        end_t = time.monotonic()
        assert start_t
        total_time = end_t - start_t
        latency_time = (first_chunk_t or start_t) - start_t
        self.log(f'total {total_time * 1000:.0f} ms, latency {latency_time * 1000:.0f} ms @{model}')

    def streaming_response(self, question: str, instruction: str) -> Iterable[str]:
        try:
            model = self.cfg.model
            if model == 'auto':
                if 'deepseek' in self.client.base_url.host:
                    model = 'deepseek-coder'
                else:
                    model = self.DEFAULT_OPENAI_MODEL
            self.log(question)
            completion = lambda: self.client.chat.completions.create(
                model=model,
                temperature=self.cfg.temperature,
                frequency_penalty=self.cfg.frequency_penalty,
                presence_penalty=self.cfg.presence_penalty,
                max_tokens=self.cfg.max_tokens,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": question},
                ],
                stream=True,
                stream_options={"include_usage": True} if self.cfg.include_usage else None,
            )
            for chunk in self.check_time(completion, model):
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

    def submit_question(self, question: str, instruction: str, range_: tuple[int, int]):
        self.nvim.command('set paste')
        try:
            first = True
            for chunk in self.streaming_response(question, instruction):
                if not self.running:
                    break
                if first:
                    self.nvim.command(f'normal! {range_[0] + 1}G')
                    self.nvim.current.buffer[range_[0]:range_[1]] = ['']
                    first = False
                self.nvim.command(f'undojoin|normal! a{chunk}')
                self.nvim.command('redraw')
        finally:
            self.nvim.command('set nopaste')

    def buffer_slice(self, range_: tuple[int, int]) -> str:
        start, end = range_
        lines = self.nvim.current.buffer[start:end]
        return '\n'.join(lines)

    def buffer_empty_slice(self, range_: tuple[int, int]):
        start, end = range_
        self.nvim.current.buffer[start:end] = ['']

    def wrap(self, code):
        return f'```\n{code}\n```'

    def buf_into_file(self, buffer, content, buf_type=None):
        if buf_type is None:
            buf_type = self.nvim.api.get_option_value('buftype', {'buf': buffer.number})
        if buf_type == 'terminal':
            file_type = 'bash'
            path = '[terminal]'
        if buf_type == 'quickfix':
            file_type = 'qf'
            path = '[quickfix]'
        else:
            file_type = self.nvim.api.get_option_value('filetype', {'buf': buffer.number})
            path = buffer.name
        return self.File(path=path, content=content, type=file_type)

    def referenced_files(self, code_sample: str) -> list[File]:
        current_buffer = self.nvim.current.buffer
        current_name = pathlib.Path(current_buffer.name).parts[-1]
        current_name = current_name.rsplit('.', maxsplit=1)[0]
        current_content = '\n'.join(current_buffer[:])
        files: list[GPT4oPlugin.File] = []
        for buffer in self.nvim.buffers:
            if buffer.name:
                buf_type = self.nvim.api.get_option_value('buftype', {'buf': buffer.number})
                buf_listed = self.nvim.api.get_option_value('buflisted', {'buf': buffer.number})
                if not buf_type or buf_type in ['terminal', 'quickfix']:
                    content = '\n'.join(buffer[:])
                    if not buf_type:
                        buffer_name = pathlib.Path(buffer.name).parts[-1]
                        buffer_name = buffer_name.rsplit('.', maxsplit=1)[0]
                    else:
                        buffer_name = buf_type
                    ok = buf_listed or re.search(rf'\b{re.escape(current_name)}\b', content) or re.search(rf'\b{re.escape(buffer_name)}\b', current_content)
                    if not ok:
                        continue
                    files.append(self.buf_into_file(buffer, content, buf_type))
        files.append(self.buf_into_file(current_buffer, current_content))
        if self.embedding_model:
            inputs: list[str] = []
            sources: list[int] = []
            for fileid, file in enumerate(files):
                for line in file.content.split('\n'):
                    inputs.append(line)
                    sources.append(fileid)
            inputs.append(code_sample)
            if self.cfg.include_time:
                t0 = time.monotonic()
            else:
                t0 = None
            response = self.client.embeddings.create(input=inputs, model=self.embedding_model)
            if self.cfg.include_time:
                assert t0
                dt = time.monotonic() - t0
                self.log(f'total {dt * 1000:.0f} ms @{self.embedding_model}')
            self.check_price(response.usage, self.embedding_model)
            sample = response.data[-1]
            contents: list[list[str]] = [[] for _ in range(len(files))]
            similiarities = [0.0 for _ in range(len(sources))]
            for i, output in enumerate(response.data[:-1]):
                similiarities[i] = self.cosine_similiarity(output.embedding, sample.embedding)
            indices = [i for i in range(len(similiarities))]
            indices.sort(key=lambda i: similiarities[i], reverse=True)
            tokens = self.count_tokens(inputs)
            total_tokens = 0
            for i in indices:
                fileid = sources[i]
                content = inputs[i]
                total_tokens += tokens[i]
                if total_tokens >= self.cfg.limit_context_tokens:
                    break
                contents[fileid].append(inputs[i])
            new_files: list[GPT4oPlugin.File] = []
            for i, content in enumerate(contents):
                file = files[i]
                if content:
                    file.content = '\n\n...\n\n'.join(content)
                    new_files.append(file)
        return files

    def count_tokens(self, inputs: list[str]) -> list[int]:
        return [(len(input.encode('utf-8')) + 3) // 4 for input in inputs]

    def cosine_similiarity(self, a: list[float], b: list[float]) -> float:
        return sum(ai * bi for ai, bi in zip(a, b)) / (math.sqrt(sum(ai * ai for ai in a)) * math.sqrt(sum(bi * bi for bi in b)))

    def form_context(self, question: str, files: list[File]) -> str:
        if not files:
            return question
        result = 'Use the following context to answer question at the end:\n\n'
        for file in files:
            result += f'File path: {file.path}\nFile content:\n```{file.type}\n{file.content}\n```\n\n'
        result += f'Question: {question}'
        return result

    def do_edit(self, question: str, range_: tuple[int, int]):
        file_type = self.nvim.api.get_option_value('filetype', {'buf': self.nvim.current.buffer.number})
        code = '\n'.join(self.nvim.current.buffer[range_[0]:range_[1]])
        question = question.strip()
        if not question:
            question = 'Fix, complete or continue writing.'
        question = f'Edit the following code:\n```{file_type}\n{code}\n```\nPercisely follow the user instruction: {question}'
        question = self.form_context(question=question, files=self.referenced_files(code))
        self.submit_question(question, self.INSTRUCTION_EDIT, range_)

    def open_scratch(self, title):
        existing_buffers = next((buf for buf in self.nvim.buffers if buf.name == title), None)
        if existing_buffers:
            self.nvim.command(f'buffer {existing_buffers.number}')
        else:
            self.nvim.command(f'new | noswapfile hide enew | setlocal buftype=nofile bufhidden=hide noswapfile nobuflisted | file {title} | redraw')

    def do_chat(self, question: str, range_: tuple[int, int]):
        file_type = self.nvim.api.get_option_value('filetype', {'buf': self.nvim.current.buffer.number})
        code = '\n'.join(self.nvim.current.buffer[range_[0]:range_[1]])
        question = question.strip()
        if question:
            if code.strip():
                question = f'Based on the following code:\n```{file_type}\n{code}\n```\nPercisely answer the user question: {question}'
            question = self.form_context(question=question, files=self.referenced_files(code))
        self.open_scratch('[GPTChat]')
        if question:
            self.submit_question(question, self.INSTRUCTION_CHAT, range_)

    def do_gpt_range(self, args: str, range_: tuple[int, int], bang: bool):
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
            if bang:
                self.do_chat(question=question, range_=range_)
            else:
                self.do_edit(question=question, range_=range_)
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
