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

DEFAULT_OPENAI_MODEL = 'gpt-4o-mini'
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
    score: int = 1

@dataclass
class Config:
    extra_range_lines: int = 4
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    model: str = 'auto'
    embedding_model: str = 'auto'
    limit_context_tokens: int = 1000
    context_chunk_tokens: int = 400
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    include_usage: bool = True
    include_time: bool = True
    timeout: Optional[float] = None

@neovim.plugin
class GPT4oPlugin:
    def __init__(self, nvim: neovim.Nvim):
        self.nvim = nvim
        self.running = False
        self.cfg = Config()
        self._real_client = None
        self._real_fastembed_model = None

    @property
    def embedding_model(self) -> str:
        if self.cfg.embedding_model == 'auto':
            try:
                import fastembed as _
                return 'fastembed'
            except ImportError:
                if 'openai' in self.client.base_url.host:
                    return 'text-embedding-3-small'
                else:
                    return ''
        return self.cfg.embedding_model

    def calculate_embeddings(self, inputs: list[str]) -> list[list[float]]:
        if self.embedding_model == 'fastembed':
            return [list(data) for data in self.fastembed_model.embed(inputs)]
        else:
            assert self.embedding_model
            response = self.client.embeddings.create(input=inputs, model=self.embedding_model)
            self.check_price(response.usage, self.embedding_model)
            return [embedding.embedding for embedding in response.data]

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

    @property
    def fastembed_model(self):
        if not self._real_fastembed_model:
            import fastembed
            self._real_fastembed_model = fastembed.TextEmbedding()
        return self._real_fastembed_model

    def log(self, line: str):
        with open('/tmp/gpt4o.log', 'a') as f:
            print(line, file=f)
            print('======', file=f)

    def check_price(self, usage, model: str):
        if not self.cfg.include_usage or not usage:
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
            price = (usage.prompt_tokens * prompt_price + getattr(usage, 'completion_tokens', 0) * completion_price) / 1000000
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
        self.log(f'[FINISHED] total {total_time * 1000:.0f} ms, latency {latency_time * 1000:.0f} ms @{model}')

    def streaming_response(self, question: str, instruction: str) -> Iterable[str]:
        try:
            model = self.cfg.model
            if model == 'auto':
                if 'deepseek' in self.client.base_url.host:
                    model = 'deepseek-coder'
                else:
                    model = DEFAULT_OPENAI_MODEL
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
                    self.check_price(chunk.usage, model)
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
        return File(path=path, content=content, type=file_type)

    def referenced_files(self, code_sample: str, question: str) -> list[File]:
        files = self.get_all_files()
        if self.embedding_model:
            files = self.filter_similiar_chunks(code_sample, question, files)
        return files

    def is_buffer_valid(self, buffer):
        if buffer.valid and buffer.name:
            buf_type = self.nvim.api.get_option_value('buftype', {'buf': buffer.number})
            if not buf_type or buf_type in ['terminal', 'quickfix']:
                return True
        return False

    def buffer_name_or_type(self, buffer) -> str:
        buf_type: str = self.nvim.api.get_option_value('buftype', {'buf': buffer.number})
        if not buf_type:
            buffer_name: str = buffer.name
            if buffer_name:
                buffer_name = pathlib.Path(buffer_name).parts[-1]
                buffer_name = buffer_name.rsplit('.', maxsplit=1)[0]
            else:
                buffer_name = '[No Name]'
        else:
            buffer_name = buf_type
        return buffer_name

    def is_buffer_related(self, buffer, current_name: str, current_content: str):
        if not self.is_buffer_valid(buffer):
            return False
        buf_listed = self.nvim.api.get_option_value('buflisted', {'buf': buffer.number})
        if buf_listed:
            return True
        buffer_name = self.buffer_name_or_type(buffer)
        content = '\n'.join(buffer[:])
        if re.search(rf'\b{re.escape(current_name)}\b', content) or re.search(rf'\b{re.escape(buffer_name)}\b', current_content):
            return True

    def get_all_files(self) -> list[File]:
        current_buffer = self.nvim.current.buffer
        current_name = self.buffer_name_or_type(current_buffer)
        current_content = '\n'.join(current_buffer[:])
        files: list[File] = []
        for buffer in self.nvim.buffers:
            if buffer.number == current_buffer.number:
                continue
            if self.is_buffer_related(buffer, current_name, current_content):
                self.log(f'adding file: {buffer.name}')
                content = '\n'.join(buffer[:])
                buf_type = self.nvim.api.get_option_value('buftype', {'buf': buffer.number})
                files.append(self.buf_into_file(buffer, content, buf_type))
        self.log(f'current file: {current_buffer.name}')
        files.append(self.buf_into_file(current_buffer, current_content))
        files[-1].score = 2
        return files

    def filter_similiar_chunks(self, code_sample: str, question: str, files: list[File]):
        inputs: list[str] = []
        sources: list[int] = []
        for fileid, file in enumerate(files):
            for line in self.split_content_chunks(file.content):
                line = line.rstrip().lstrip('\n')
                if line:
                    inputs.append(line)
                    sources.append(fileid)
        prompt = code_sample + '\n' + question
        inputs.append(prompt.rstrip().lstrip('\n'))
        # self.log(f'all chunks:\n{'\n--------\n'.join(inputs)}')
        self.log(f'splited into {len(inputs)} chunks')
        tokens = self.count_tokens(inputs)
        if sum(tokens) >= self.cfg.limit_context_tokens:
            # self.log(f'tokens {sum(tokens)} >= {self.cfg.limit_context_tokens}, clipping')
            if self.cfg.include_time:
                t0 = time.monotonic()
            else:
                t0 = None
            response = self.calculate_embeddings(inputs)
            if self.cfg.include_time:
                assert t0
                dt = time.monotonic() - t0
                self.log(f'embedding {dt * 1000:.0f} ms @{self.embedding_model}')
            sample = response[-1]
            contents: list[list[str]] = [[] for _ in range(len(files))]
            similiarities = [0.0 for _ in range(len(sources))]
            for i, output in enumerate(response[:-1]):
                fileid = sources[i]
                similiarities[i] = self.cosine_similiarity(output, sample) ** (1 / files[fileid].score)
            indices = [i for i in range(len(similiarities))]
            indices.sort(key=lambda i: similiarities[i], reverse=True)
            total_tokens = 0
            input_oks = [False for _ in range(len(indices))]
            for i in indices:
                content = inputs[i]
                total_tokens += tokens[i]
                if total_tokens >= self.cfg.limit_context_tokens:
                    break
                input_oks[i] = True
            for i, ok in enumerate(input_oks):
                if ok:
                    fileid = sources[i]
                    contents[fileid].append(inputs[i])
            new_files: list[File] = []
            for i, content in enumerate(contents):
                file = files[i]
                if content:
                    if file.content != '\n'.join(content):
                        file.content = '\n\n...\n\n'.join(content)
                    new_files.append(file)
            files = new_files
        return files

    def split_content_chunks(self, content: str) -> list[str]:
        lines = content.split('\n')
        result: list[str] = []
        tokens = self.count_tokens(lines)
        accum_lines: list[str] = []
        accum_tokens = 0
        for line, ntoken in zip(lines, tokens):
            accum_lines.append(line)
            accum_tokens += ntoken
            limit = self.cfg.context_chunk_tokens
            if not line.strip():
                limit *= 2
            if accum_tokens >= limit:
                result.append('\n'.join(accum_lines))
                accum_lines = []
                accum_tokens = 0
        if accum_lines:
            result.append('\n'.join(accum_lines))
        return result

    def count_tokens(self, inputs: list[str]) -> list[int]:
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(DEFAULT_OPENAI_MODEL)
            tokens = encoding.encode_batch(inputs, disallowed_special=())
            return [len(t) for t in tokens]
        except ImportError:
            return [(len(input.encode('utf-8')) + 3) // 4 for input in inputs]

    def cosine_similiarity(self, a: list[float], b: list[float]) -> float:
        denom = math.sqrt(sum(ai * ai for ai in a)) * math.sqrt(sum(bi * bi for bi in b))
        return sum(ai * bi for ai, bi in zip(a, b)) / denom

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
        original_question = question = question.strip()
        if not question:
            question = 'Fix, complete or continue writing.'
        question = f'Edit the following code:\n```{file_type}\n{code}\n```\nPercisely follow the user instruction: {question}'
        question = self.form_context(question=question, files=self.referenced_files(code, original_question))
        self.submit_question(question, INSTRUCTION_EDIT, range_)

    def open_scratch(self, title):
        bufnr = self.nvim.eval(f'bufnr("^{title}$")')
        if bufnr != -1:
            self.nvim.command(f'buffer {bufnr} | redraw')
        else:
            self.nvim.command(f'new | noswapfile hide enew | setlocal buftype=nofile bufhidden=hide noswapfile nobuflisted | file {title} | redraw')

    def do_chat(self, question: str, range_: tuple[int, int]):
        file_type = self.nvim.api.get_option_value('filetype', {'buf': self.nvim.current.buffer.number})
        if range_[0] < range_[1]:
            code = '\n'.join(self.nvim.current.buffer[range_[0]:range_[1]])
        else:
            code = ''
        original_question = question = question.strip()
        if question:
            if code.strip():
                question = f'Based on the following code:\n```{file_type}\n{code}\n```\nPercisely answer the user question: {question}'
            question = self.form_context(question=question, files=self.referenced_files(code, original_question))
        self.open_scratch('[GPTChat]')
        if question:
            self.submit_question(question, INSTRUCTION_CHAT, range_)

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
    def on_GPT(self, args, range_, bang):
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

    @neovim.command('GPTChat', nargs='*', range=True, bang=True)
    def on_GPTChat(self, args, range_, bang):
        if range_[0] == range_[1]:
            range_ = (range_[0] - 1, range_[1])
        self.do_gpt_range(args, range_, bang)

    @neovim.function('GPTSetup')
    def on_GPTSetup(self, args):
        assert isinstance(args, dict)
        for k, v in args:
            setattr(self.cfg, k, v)
