class Timer:
    def __init__(self):
        import time
        self.t0 = time.monotonic()

    def record(self, tag: str):
        import time
        t1 = time.monotonic()
        dt = t1 - self.t0
        self.add(tag, dt)
        return dt

    def add(self, tag: str, dt: float):
        with open('/tmp/gpt4o.log', 'a') as f:
            f.write(f'[timer] {tag}: {dt:.3f}\n')

timer = Timer()

import pathlib
import traceback
import subprocess
import threading
import unittest
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Callable, Generic, Optional, Iterable, TypeVar
import tempfile
import math
import time
import os
import re

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

try:
    import neovim

except ImportError:
    @eval("lambda x: setattr(globals(), 'neovim', x)")
    class Neovim:
        def plugin(self, cls):
            return cls

        class Nvim:
            pass

        def __getattr__(self, _):
            def decorator(*args, **kwargs):
                _ = args, kwargs
                def wrapper(func):
                    return func
                return wrapper
            return decorator

    assert neovim  # type: ignore

timer.record('import finish')


DEFAULT_OPENAI_MODEL = 'gpt-4o-mini'
DEFAULT_OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'
DEFAULT_DEEPSEEK_MODEL = 'deepseek-coder'

SPECIAL_FILE_TYPES = {
    'terminal': ('[terminal]', 'bash', 'Terminal Output'),
    'toggleterm': ('[terminal]', 'bash', 'Terminal Output'),
    'quickfix': ('[quickfix]', '', 'Quickfix Window'),
}

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


class TestTimer(unittest.TestCase):
    def setUp(self):
        self.timer = Timer()

    def test_record(self):
        self.timer.record('test')
        self.assertGreater(self.timer.t0, 0)

    def test_add(self):
        with open('/tmp/gpt4o.log', 'w') as f:
            pass
        self.timer.add('test', 1)
        with open('/tmp/gpt4o.log', 'r') as f:
            self.assertEqual(f.read(), '[timer] test: 1.000\n')

    def test_timer(self):
        self.timer.record('test')
        self.timer.record('test2')
        self.assertGreater(self.timer.t0, 0)


class LruCache(Generic[K, V]):
    def __init__(self):
        self.cache: dict[K, V] = {}
        self.lru: list[K] = []  # TODO

    @contextmanager
    def usage_guard(self):
        yield

    def find_entry(self, key: K) -> Optional[V]:
        return self.cache.get(key)

    def add_entry(self, key: K, val: V):
        self.cache[key] = val

    def delete_entry(self, key: K):
        self.cache.pop(key, None)

    def cached_calc(self, calc: Callable[[K], V], input: K):
        with self.usage_guard():
            output = self.find_entry(input)
            if output is None:
                output = calc(input)
                self.add_entry(input, output)
        return output

    def batched_cached_calc(self, calc: Callable[[list[K]], list[V]], inputs: list[K]) -> list[V]:
        missed_inputs: list[K] = []
        missed_indices: list[int] = []
        outputs: list[Optional[V]] = []

        with self.usage_guard():
            for i, input in enumerate(inputs):
                output = self.find_entry(input)
                if output is None:
                    missed_inputs.append(input)
                    missed_indices.append(i)
                    outputs.append(None)
                else:
                    outputs.append(output)

            if missed_inputs:
                missed_outputs = calc(missed_inputs)
                assert isinstance(missed_outputs, list)
                for i, output in zip(missed_indices, missed_outputs):
                    outputs[i] = output
                    self.add_entry(inputs[i], output)

        assert all(x is not None for x in outputs)
        return outputs  # type: ignore


class TestLruCache(unittest.TestCase):
    def setUp(self):
        self.cache = LruCache()

    def test_add_entry(self):
        self.cache.add_entry(1, 'a')
        self.assertEqual(self.cache.find_entry(1), 'a')

    def test_delete_entry(self):
        self.cache.add_entry(1, 'a')
        self.cache.delete_entry(1)
        self.assertEqual(self.cache.find_entry(1), None)

    def test_cached_calc(self):
        def calc(x: int) -> int:
            return x + 1

        self.assertEqual(self.cache.cached_calc(calc, 1), 2)
        self.assertEqual(self.cache.cached_calc(calc, 2), 3)
        self.assertEqual(self.cache.cached_calc(calc, 3), 4)
        self.assertEqual(self.cache.cached_calc(calc, 1), 2)

    def test_batched_cached_calc(self):
        def calc(x: list[int]) -> list[int]:
            return [i + 1 for i in x]

        self.assertEqual(self.cache.batched_cached_calc(calc, [1, 2, 3]), [2, 3, 4])
        self.assertEqual(self.cache.batched_cached_calc(calc, [0, 2, 3, 6]), [1, 3, 4, 7])


class BufferHistory:
    def __init__(self):
        self.history: list[tuple[int, str]] = []

    def add_history(self, seq: int, content: str):
        self.history.append((seq, content))

    def shrink_to_newer_than(self, min_seq: int):
        self.history = [(seq, content) for seq, content in self.history if seq >= min_seq]

    def shrink_to_size(self, size: int):
        self.history = self.history[-size:]

    def history_newer_than(self, min_seq: int) -> Iterable[tuple[int, str]]:
        for seq, content in self.history:
            if seq > min_seq:
                yield seq, content

    def __iter__(self):
        return iter(self.history)


class TestBufferHistory(unittest.TestCase):
    def setUp(self):
        self.history = BufferHistory()

    def test_add_history(self):
        self.history.add_history(1, "First entry")
        self.history.add_history(2, "Second entry")
        self.assertEqual(len(self.history.history), 2)

    def test_shrink_to_size(self):
        self.history.add_history(1, "First entry")
        self.history.add_history(2, "Second entry")
        self.history.add_history(3, "Third entry")
        self.history.shrink_to_size(2)
        self.assertEqual(len(self.history.history), 2)
        self.assertEqual(self.history.history[0][1], "Second entry")
        self.assertEqual(self.history.history[1][1], "Third entry")

    def test_shrink_to_newer_than(self):
        self.history.add_history(1, "First entry")
        self.history.add_history(2, "Second entry")
        self.history.add_history(3, "Third entry")
        self.history.shrink_to_newer_than(2)
        self.assertEqual(len(self.history.history), 2)
        self.assertEqual(self.history.history[0][1], "Second entry")
        self.assertEqual(self.history.history[1][1], "Third entry")

    def test_history_newer_than(self):
        self.history.add_history(1, "First entry")
        self.history.add_history(2, "Second entry")
        self.history.add_history(3, "Third entry")
        self.history.add_history(4, "Fourth entry")

        newer_history = list(self.history.history_newer_than(1))
        self.assertEqual(len(newer_history), 3, newer_history)
        self.assertEqual(newer_history[0][1], "Second entry")
        self.assertEqual(newer_history[1][1], "Third entry")
        self.assertEqual(newer_history[2][1], "Fourth entry")

        newer_history = list(self.history.history_newer_than(2))
        self.assertEqual(len(newer_history), 2, newer_history)
        self.assertEqual(newer_history[0][1], "Third entry")
        self.assertEqual(newer_history[1][1], "Fourth entry")

        newer_history = list(self.history.history_newer_than(3))
        self.assertEqual(len(newer_history), 1, newer_history)
        self.assertEqual(newer_history[0][1], "Fourth entry")

        newer_history = list(self.history.history_newer_than(4))
        self.assertEqual(len(newer_history), 0, newer_history)

    def test_iter(self):
        self.history.add_history(1, "First entry")
        self.history.add_history(2, "Second entry")
        entries = list(iter(self.history))
        self.assertEqual(len(entries), 2)


@dataclass
class RefFile:
    path: str
    content: str
    type: str = ''
    score: int = 1
    special_name: str = ''


@dataclass
class Config:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.0
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    include_usage: bool = True
    include_time: bool = True
    timeout: Optional[float] = None

    model: str = 'auto'                    # 'auto' | 'gpt-4o' | 'gpt-4o-mini'
    embedding_model: str = 'auto'          # 'auto' | 'text-embedding-3-small' | 'fastembed'
    use_tiktoken_for_counting: bool = True
    use_fastembed_for_embedding: bool = True
    limit_context_tokens: int = 3200
    context_chunk_tokens: int = 160
    non_empty_line_chunk_punish: int = 3
    recent_diff_tokens: int = 640
    max_recent_diff_count: int = 16
    try_treat_terminal_p10k: bool = True
    accept_unlisted_buffers: str = 'yes'  # 'yes' | 'no' | 'if_content_ref'
    current_buffer_score: int = 3
    extra_range_lines_gpt4: int = 4


@neovim.plugin
class GPT4oPlugin:
    def __init__(self, nvim: neovim.Nvim):
        self.nvim = nvim
        self.running: bool = False
        self.cfg = Config()
        self._real_client = None
        self._real_fastembed_model = None
        self.change_lock = threading.Lock()
        self.buffers_history: dict[int, BufferHistory] = {}
        self.seq_lock = threading.Lock()
        self.embedding_cache: LruCache[str, list[float]] = LruCache()
        self.count_tokens_cache: LruCache[str, int] = LruCache()
        self.seq: int = 0

    def next_seq(self) -> int:
        with self.seq_lock:
            self.seq += 1
            return self.seq

    def get_embedding_model(self) -> str:
        if self.cfg.embedding_model == 'auto':
            use_fastembed = self.cfg.use_fastembed_for_embedding
            if use_fastembed:
                try:
                    timer.record('import fastembed begin')
                    import fastembed as _
                    timer.record('import fastembed end')
                    return 'fastembed'
                except ImportError:
                    use_fastembed = False
            if not use_fastembed:
                if 'openai' in self.get_openai_client().base_url.host:
                    return DEFAULT_OPENAI_EMBEDDING_MODEL
                else:
                    return ''
        return self.cfg.embedding_model

    def calculate_embeddings(self, inputs: list[str]) -> list[list[float]]:
        return self.embedding_cache.batched_cached_calc(self.impl_calculate_embeddings, inputs)

    def impl_calculate_embeddings(self, inputs: list[str]) -> list[list[float]]:
        if self.get_embedding_model() == 'fastembed':
            return [list(data) for data in self.get_fastembed_model().embed(inputs)]
        else:
            assert self.get_embedding_model()
            response = self.get_openai_client().embeddings.create(input=inputs, model=self.get_embedding_model())
            self.check_price(response.usage, self.get_embedding_model())
            return [embedding.embedding for embedding in response.data]

    def get_openai_client(self):
        if not self._real_client:
            timer.record('import openai begin')
            import openai
            timer.record('import openai end')
            self._real_client = openai.OpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                organization=self.cfg.organization,
                project=self.cfg.project,
                timeout=self.cfg.timeout,
            )
        return self._real_client

    def get_fastembed_model(self):
        if not self._real_fastembed_model:
            import fastembed
            timer.record('construct fastembed begin')
            self._real_fastembed_model = fastembed.TextEmbedding()
            timer.record('construct fastembed end' + '\n'.join(traceback.format_stack()))
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
        client = self.get_openai_client()
        try:
            model = self.cfg.model
            if model == 'auto':
                if 'openai' in client.base_url.host:
                    model = DEFAULT_OPENAI_MODEL
                elif 'deepseek' in client.base_url.host:
                    model = DEFAULT_DEEPSEEK_MODEL
                else:
                    model = DEFAULT_OPENAI_MODEL
            self.log(question)

            completion = lambda: client.chat.completions.create(
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

            answer = ''
            for chunk in self.check_time(completion, model):
                if chunk.usage:
                    self.check_price(chunk.usage, model)
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        answer += content
                        yield content
            self.log(answer)

        except __import__(client.__module__).OpenAIError:
            error = traceback.format_exc()
            self.log(error)
            yield error

    def submit_question(self, question: str, instruction: str, range_: tuple[int, int], null_token: Optional[str] = None) -> bool:
        self.nvim.command('set paste')
        try:
            first = True
            for chunk in self.streaming_response(question, instruction):
                if not self.running:
                    break
                if first:
                    if null_token and chunk == null_token:
                        return False
                    self.nvim.command(f'normal! {range_[0] + 1}G')
                    self.nvim.current.buffer[range_[0]:range_[1]] = ['']
                    first = False
                self.nvim.command(f'undojoin|normal! a{chunk}')
                self.nvim.command('redraw')
        finally:
            self.nvim.command('set nopaste')
        return True

    def buffer_slice(self, range_: tuple[int, int]) -> str:
        start, end = range_
        lines = self.nvim.current.buffer[start:end]
        return '\n'.join(lines)

    def buffer_empty_slice(self, range_: tuple[int, int]):
        start, end = range_
        self.nvim.current.buffer[start:end] = ['']

    def treat_terminal_p10k(self, content: str) -> str:
        STANDARD_PROMPT = '❯ '
        if '─╮\n╰─ ' in content:
            content = re.sub(r'❯ |╭─ .* ─╮\n╰─ ', STANDARD_PROMPT, content)
            content = re.sub(r'\s+─╯', '', content)
        content = content.strip()
        content = content.removesuffix(STANDARD_PROMPT.rstrip())
        content = content.strip()
        return content

    def buf_into_file(self, buffer, content: str, buf_type: Optional[str] = None) -> RefFile:
        if buf_type is None:
            buf_type = self.nvim.api.get_option_value('buftype', {'buf': buffer.number})
        special_name = ''
        if buf_type in SPECIAL_FILE_TYPES:
            path, file_type, special_name = SPECIAL_FILE_TYPES[buf_type]
            if path == '[terminal]' and self.cfg.try_treat_terminal_p10k:
                content = self.treat_terminal_p10k(content)
        else:
            path = buffer.name
            file_type = self.nvim.api.get_option_value('filetype', {'buf': buffer.number})
        return RefFile(path=path, content=content, type=file_type, special_name=special_name)

    def get_referenced_files(self, code_sample: str, question: str) -> list[RefFile]:
        files = self.get_all_files()
        if self.get_embedding_model():
            files = self.filter_similiar_chunks(code_sample, question, files)

        diffs = self.get_recent_diffs()
        if diffs:
            files.append(RefFile(path='diff', content='\n'.join(diffs), type='diff', special_name='Recent changes'))
        return files

    def get_recent_diffs(self) -> list[str]:
        buffers = {}
        for buffer in self.nvim.buffers:
            buffers[buffer.number] = buffer

        min_seq = self.seq - self.cfg.max_recent_diff_count
        history_list: list[tuple[int, int, str]] = []
        with self.change_lock:
            for bufnr, history in self.buffers_history.items():
                if bufnr not in buffers:
                    continue
                for seq, content in history.history_newer_than(min_seq):
                    history_list.append((seq, bufnr, content))
                    break
        if not history_list:
            return []

        history_list.sort(key=lambda e: e[0])
        changes: list[str] = []
        for _, bufnr, content in history_list:
            buffer = buffers[bufnr]
            new_content = '\n'.join(buffer[:])
            buffer_path = self.buffer_name_or_type(buffer)
            diff = self.compute_diff(new_content, content, buffer_path)
            if diff:
                changes.append(diff)

        tokens = self.count_tokens(changes)
        accum_diffs = []
        accum_tokens = 0
        for ntoken, diff in zip(tokens, changes):
            accum_diffs.append(diff)
            accum_tokens += ntoken
            if accum_tokens >= self.cfg.recent_diff_tokens:
                break
        return accum_diffs

    def is_buffer_valid(self, buffer):
        if buffer.valid and buffer.name:
            buf_type = self.nvim.api.get_option_value('buftype', {'buf': buffer.number})
            if not buf_type or buf_type in SPECIAL_FILE_TYPES:
                return True
        return False

    def buffer_name_or_type(self, buffer) -> str:
        buf_type: str = self.nvim.api.get_option_value('buftype', {'buf': buffer.number})
        if not buf_type:
            buffer_name: str = buffer.name
        else:
            buffer_name = buf_type
        return buffer_name

    def simplify_buffer_name(self, name: str) -> str:
        if name:
            name = pathlib.Path(name).parts[-1]
            name = name.rsplit('.', maxsplit=1)[0]
        return name

    def is_buffer_related(self, buffer, current_name: str, current_content: str):
        if not self.is_buffer_valid(buffer):
            return False
        if self.cfg.accept_unlisted_buffers == 'yes':
            return True

        buf_listed = self.nvim.api.get_option_value('buflisted', {'buf': buffer.number})
        if buf_listed:
            return True
        if self.cfg.accept_unlisted_buffers == 'no':
            return False

        elif self.cfg.accept_unlisted_buffers == 'if_content_ref':
            buffer_name = self.simplify_buffer_name(self.buffer_name_or_type(buffer))
            content = '\n'.join(buffer[:])
            if re.search(rf'\b{re.escape(current_name)}\b', content) or (buffer_name and re.search(rf'\b{re.escape(buffer_name)}\b', current_content)):
                return True

        else:
            assert False, self.cfg.accept_unlisted_buffers

    def get_all_files(self) -> list[RefFile]:
        current_buffer = self.nvim.current.buffer
        current_name = self.simplify_buffer_name(self.buffer_name_or_type(current_buffer)) or '[No Name]'
        current_content = '\n'.join(current_buffer[:])
        files: list[RefFile] = []
        for buffer in self.nvim.buffers:
            if buffer.number == current_buffer.number:
                continue
            if self.is_buffer_related(buffer, current_name, current_content):
                buf_type = self.nvim.api.get_option_value('buftype', {'buf': buffer.number})
                self.log(f'adding file: {buffer.name}{" " + buf_type if buf_type else ""}')
                content = '\n'.join(buffer[:])
                file = self.buf_into_file(buffer, content, buf_type)
                if file.content:
                    files.append(file)
        self.log(f'current file: {current_buffer.name}')
        current_file = self.buf_into_file(current_buffer, current_content)
        current_file.score = self.cfg.current_buffer_score
        files.append(current_file)
        return files

    def omit_merge_content_chunks(self, contents: list[tuple[int, str]], chunk_count: int) -> str:
        if not contents:
            return ''
        expected_chunkid = 0
        result: list[str] = []
        first = True
        for chunkid, chunk in contents:
            if chunkid != expected_chunkid:
                if first:
                    result.append('...\n')
                else:
                    result.append('\n...\n')
                first = False
            result.append(chunk)
            expected_chunkid = chunkid + 1
        if expected_chunkid != chunk_count:
            result.append('\n...')
        return '\n'.join(result)

    def filter_similiar_chunks(self, code_sample: str, question: str, files: list[RefFile]):
        inputs: list[str] = []
        inputs_to_embed: list[str] = []
        input_chunkids: list[int] = []
        file_chunk_counts: list[int] = []
        input_sources: list[int] = []
        for fileid, file in enumerate(files):
            chunk_count = 0
            for chunkid, chunk in enumerate(self.split_content_chunks(file.content)):
                stripped_chunk = chunk.rstrip().lstrip('\n')
                if stripped_chunk:
                    inputs.append(chunk)
                    inputs_to_embed.append(stripped_chunk)
                    input_chunkids.append(chunkid)
                    input_sources.append(fileid)
                    chunk_count += 1
            file_chunk_counts.append(chunk_count)
        prompt = code_sample + '\n' + question
        prompt = prompt.rstrip().lstrip('\n')
        inputs_to_embed.append(prompt)
        # self.log(f'all chunks:\n{'\n--------\n'.join(inputs)}')
        self.log(f'splited into {len(inputs)} chunks')
        tokens = self.count_tokens(inputs_to_embed)
        if sum(tokens) >= self.cfg.limit_context_tokens:
            # self.log(f'tokens {sum(tokens)} >= {self.cfg.limit_context_tokens}, clipping')
            if self.cfg.include_time:
                t0 = time.monotonic()
            else:
                t0 = None
            response = self.calculate_embeddings(inputs_to_embed)
            if self.cfg.include_time:
                assert t0
                dt = time.monotonic() - t0
                self.log(f'embedding {dt * 1000:.0f} ms @{self.get_embedding_model()}')
            sample = response[-1]
            file_contents: list[list[tuple[int, str]]] = [[] for _ in range(len(files))]
            input_sims = [0.0 for _ in range(len(input_sources))]
            for i, output in enumerate(response[:-1]):
                fileid = input_sources[i]
                input_sims[i] = self.cosine_similiarity(output, sample) ** (1 / files[fileid].score)
            input_indices = [i for i in range(len(input_sims))]
            input_indices.sort(key=lambda i: input_sims[i], reverse=True)
            total_tokens = 0
            input_oks = [False for _ in range(len(input_indices))]
            for i in input_indices:
                total_tokens += tokens[i]
                if total_tokens >= self.cfg.limit_context_tokens:
                    break
                input_oks[i] = True
            for i, ok in enumerate(input_oks):
                if ok:
                    fileid = input_sources[i]
                    chunkid = input_chunkids[i]
                    file_contents[fileid].append((chunkid, inputs[i]))
            new_files: list[RefFile] = []
            for i, content in enumerate(file_contents):
                file = files[i]
                if content:
                    chunk_count = file_chunk_counts[i]
                    file.content = self.omit_merge_content_chunks(content, chunk_count)
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
            if line.strip():
                limit *= self.cfg.non_empty_line_chunk_punish
            if accum_tokens >= limit:
                result.append('\n'.join(accum_lines))
                accum_lines = []
                accum_tokens = 0
        if accum_lines:
            result.append('\n'.join(accum_lines))
        return result

    def count_tokens(self, inputs: list[str]) -> list[int]:
        use_tiktoken = self.cfg.use_tiktoken_for_counting
        if use_tiktoken:
            try:
                timer.record('import tiktoken begin')
                import tiktoken as _
                timer.record('import tiktoken end')
            except ImportError:
                use_tiktoken = False

        if use_tiktoken:
            return self.count_tokens_cache.batched_cached_calc(self.impl_count_tokens, inputs)
        else:
            return [(len(input.encode('utf-8')) + 3) // 4 for input in inputs]

    def impl_count_tokens(self, inputs: list[str]) -> list[int]:
        import tiktoken
        encoding = tiktoken.encoding_for_model(DEFAULT_OPENAI_MODEL)
        tokens = encoding.encode_batch(inputs, disallowed_special=())
        return [len(t) for t in tokens]

    def cosine_similiarity(self, a: list[float], b: list[float]) -> float:
        denom = math.sqrt(sum(ai * ai for ai in a)) * math.sqrt(sum(bi * bi for bi in b))
        return sum(ai * bi for ai, bi in zip(a, b)) / denom

    def add_indent(self, content: str, indent: str = '    ') -> str:
        if content.strip():
            return ('\n' + content).replace('\n', indent + '\n')
        else:
            return indent + '[Empty File]'

    def form_context(self, question: str, files: list[RefFile]) -> str:
        if not files:
            return question
        result = 'Use the following context to answer question at the end:\n\n'
        for file in files:
            if file.special_name:
                prefix = file.special_name
            else:
                prefix = f'File path: {file.path}\nFile content'
            result += f'{prefix}:\n```{file.type}\n{file.content}\n```\n\n'
        if question:
            result += f'Question: {question}'
        return result

    def get_range_code(self, range_: tuple[int, int]) -> tuple[str, str]:
        file_type = self.nvim.api.get_option_value('filetype', {'buf': self.nvim.current.buffer.number})
        if range_[0] < range_[1]:
            code = '\n'.join(self.nvim.current.buffer[range_[0]:range_[1]])
        else:
            code = ''
        return code, file_type

    def alert(self, message: str, level: str = 'INFO'):
        self.nvim.command(f'lua vim.notify({repr(message)}, vim.log.levels.{level})')

    def do_prompt(self, question: str, range_: tuple[int, int], force_chat: bool):
        code, file_type = self.get_range_code(range_)
        original_question = question = question.strip()

        while True:
            if not force_chat:
                if not question:
                    question = 'Fix, complete or continue writing.'
                question = f'Edit the following code:\n```{file_type}\n{code}\n```\nPercisely follow the user instruction: {question}'
            else:
                if question:
                    if code.strip():
                        question = f'Based on the following code:\n```{file_type}\n{code}\n```\nPercisely answer the user question: {question}'
                    question = self.form_context(question=question, files=self.get_referenced_files(code, original_question))

            if force_chat:
                self.open_scratch('[GPTChat]')
            if question:
                instruction = INSTRUCTION_EDIT if not force_chat else INSTRUCTION_CHAT
                success = self.submit_question(question, instruction=instruction, range_=range_)
                if not success:
                    force_chat = True
                    continue
            break

    def open_scratch(self, title):
        bufnr = self.nvim.eval(f'bufnr("^{title}$")')
        if bufnr != -1:
            self.nvim.command(f'buffer {bufnr} | redraw')
        else:
            self.nvim.command(f'new | noswapfile hide enew | setlocal buftype=nofile bufhidden=hide noswapfile nobuflisted | file {title} | redraw')

    @neovim.command('GPTInfo', nargs='*', range=True)
    def on_GPTInfo(self, args: list[str], range_: tuple[int, int]):
        question = ' '.join(args)
        original_question = question = question.strip()
        range_ = (range_[0] - 1, range_[1])
        assert range_[0] < range_[1], range_
        code, file_type = self.get_range_code(range_)
        if not question:
            question = 'Fix, complete or continue writing.'
        question = f'Edit the following code:\n```{file_type}\n{code}\n```\nPercisely follow the user instruction: {question}'
        question = self.form_context(question=question, files=self.get_referenced_files(code, original_question))
        self.alert(question)

    def running_guard(self):
        if self.running:
            self.running = False
            return
        self.running = True
        try:
            yield
        finally:
            self.running = False

    def do_gpt_range(self, args: list[str], range_: tuple[int, int], bang: bool):
        for _ in self.running_guard():
            if args:
                question = ' '.join(args)
            else:
                question = ''
            range_ = (range_[0] - 1, range_[1])
            assert range_[0] < range_[1], range_
            self.do_prompt(question=question, range_=range_, force_chat=bang)

    @neovim.command('GPT', nargs='*', range=True, bang=True)
    def on_GPT(self, args: list[str], range_: tuple[int, int], bang: bool):
        self.do_gpt_range(args, range_, bang)

    @neovim.command('GPT4', nargs='*', range=True, bang=True)
    def on_GPT4(self, args: list[str], range_: tuple[int, int], bang: bool):
        start, end = range_
        start -= self.cfg.extra_range_lines_gpt4
        end += self.cfg.extra_range_lines_gpt4
        if start < 1:
            start = 1
        num_lines = len(self.nvim.current.buffer)
        if end > num_lines:
            end = num_lines
        range_ = start, end
        self.do_gpt_range(args, range_, bang)

    @neovim.command('GPTSetup', nargs='*', sync=True, allow_nested=False)
    def on_GPTSetup(self, args: list[str]):
        for _ in self.running_guard():
            if not args:
                args = list(self.cfg.__dict__.keys())
            alert_result = []
            for kv in args:
                kv = kv.split('=', maxsplit=1)
                k = kv[0]
                if k.startswith('_') or not hasattr(self.cfg, k):
                    self.alert(f'no such config key {k}', 'ERROR')
                if len(kv) > 1:
                    v = kv[1]
                    type_ = self.cfg.__annotations__[k]
                    can_be_none = False
                    if hasattr(type_, '__args__') and hasattr(type_.__args__, '__getitem__'):
                        type_ = type_.__args__[0]
                        can_be_none = True
                    if type_ is bool and v.lower() in ('true', 'false'):
                        v = v[0] in 'tT'
                    elif can_be_none and v == '' or v.lower() == 'none':
                        v = None
                    if v is not None:
                        v = type_(v)
                    setattr(self.cfg, k, v)
                else:
                    v = getattr(self.cfg, k)
                    alert_result.append(f'{k}={v}')
            self.alert('\n'.join(alert_result))

    @contextmanager
    def eventignore_guard(self):
        ei_backup = self.nvim.api.eval('&eventignore')
        try:
            assert isinstance(ei_backup, str), ei_backup
            yield
            self.nvim.command('set eventignore=all')
        finally:
            self.nvim.command(f'set eventignore={ei_backup}')

    # @neovim.autocmd('InsertLeave,BufEnter,CursorHold,BufLeave,BufWritePost')
    @neovim.autocmd('InsertLeave,BufEnter,CursorHold,BufLeave,BufWritePost')
    def on_GPTHold(self):
        timer.record('GPTHold enter')

        with self.eventignore_guard():
            if not self.nvim.api.eval('&modifiable'):
                return
            new_content = self.nvim.current.buffer[:]
            new_content = '\n'.join(new_content)
            current_number = self.nvim.current.buffer.number

            with self.change_lock:
                if current_number not in self.buffers_history:
                    history = BufferHistory()
                    self.buffers_history[current_number] = history
                else:
                    history = self.buffers_history[current_number]
                seq = self.next_seq()
                history.add_history(seq, new_content)
                history.shrink_to_newer_than(seq - self.cfg.max_recent_diff_count)

        timer.record('GPTHold exit')

    def compute_diff(self, new_content: str, old_content: str, current_path: str = '') -> Optional[str]:
        old_label = 'a'
        new_label = 'b'
        if current_path:
            current_path = '/' + current_path.lstrip('/')
            old_label += current_path
            new_label += current_path

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            old_file = temp_dir / 'a'
            new_file = temp_dir / 'b'
            with open(old_file, 'w') as f:
                f.write(old_content)
            with open(new_file, 'w') as f:
                f.write(new_content)
            with subprocess.Popen(['diff', '-u', '-d',
                                   '--label', old_label,
                                   '--label', new_label,
                                   str(old_file), str(new_file),
                                   ],
                                  stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=dict(os.environ, LC_ALL='C'),
                                  ) as proc:
                output, error = proc.communicate()
                if error:
                    self.alert(error.decode('utf-8'), 'ERROR')
                ret = proc.wait()
                if ret == 0:
                    return None
                if ret != 1:
                    self.alert(f'diff exited with {ret}', 'ERROR')
                    return None

        diff = output.decode('utf-8')
        return diff


class TestGPT4oPlugin(unittest.TestCase):
    def setUp(self):
        nvim: neovim.Nvim = None  # type: ignore
        self.plugin = GPT4oPlugin(nvim)

    def test_compute_diff(self):
        diff = self.plugin.compute_diff('a\nb', 'a\nb')
        self.assertIsNone(diff)

        diff = self.plugin.compute_diff('a\nb', 'a\nb\nc')
        self.assertEqual(diff, '--- a\n+++ b\n@@ -1,3 +1,2 @@\n a\n-b\n-c\n\\ '
                         'No newline at end of file\n+b\n\\ No newline at end '
                         'of file\n')

    def test_treat_terminal_p10k(self):
        content = '╭─ ~/Codes/gpt4o.nvim │ main ⇡5 *1 +1 !1 ▓▒░················' \
                  '····························································' \
                  '····················░▒▓ 1 ✘ │ 16:46:18 ─╮\n╰─ cd -q ../build' \
                  '                                                          ─╯'
        content = self.plugin.treat_terminal_p10k(content)
        self.assertEqual(content, '❯ cd -q ../build')

if __name__ == '__main__':
    unittest.main()

timer.record('script finish')
