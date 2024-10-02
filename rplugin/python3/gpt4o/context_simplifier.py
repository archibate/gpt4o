import unittest
from typing import List
import math

from gpt4o.types import File, Cursor
from gpt4o.embed_provider import EmbedProvider
from gpt4o.editing_context import EditingContext

class ContextSimplifier:
    def __init__(self, embed: EmbedProvider):
        self.embed = embed

    @staticmethod
    def cosine_similiarity(a: List[float], b: List[float]) -> float:
        denom = math.sqrt(sum(ai * ai for ai in a)) * math.sqrt(sum(bi * bi for bi in b))
        return sum(ai * bi for ai, bi in zip(a, b)) / denom

    @staticmethod
    def get_file_content(file: File) -> str:
        return '\n'.join(file.content)

    def split_file_chunks(self):
        pass

    def simplify(self, context: EditingContext) -> EditingContext:
        current_file = None
        other_files: List[File] = []

        if len(context.files) <= 2:
            return context

        for file in context.files:
            if file.path == context.cursor.path:
                current_file = file
            else:
                other_files.append(file)

        assert current_file is not None
        chunks = [self.get_file_content(current_file)] + [self.get_file_content(file) for file in other_files]
        embeds = self.embed.batched_embed(chunks)
        assert len(embeds) == len(chunks)
        current_embed = embeds.pop(0)

        # similiarities: List[tuple[float, File]] = []
        # for embed, file in zip(embeds, other_files):
        #     similiarity = self.cosine_similiarity(current_embed, embed)
        #     similiarities.append((similiarity, file))
        # similiarities.sort(key=lambda x: x[0], reverse=True) # large similiarities goes first

        relevant_files: List[File] = []
        for embed, file in zip(embeds, other_files):
            similiarity = self.cosine_similiarity(current_embed, embed)
            # print(file.path, similiarity)
            if similiarity >= 0.4:
                relevant_files.append(file)
        relevant_files.append(current_file)

        context.files = relevant_files
        return context

class TestContextSimplifier(unittest.TestCase):
    def setUp(self):
        from gpt4o.embed_provider import EmbedProviderOpenAI
        self.embed = EmbedProviderOpenAI()
        self.simplifier = ContextSimplifier(self.embed)

    def test_simplify(self):
        context = EditingContext(
            cursor=Cursor('test.py', 2, 3, 'This is a test file for CI purpose.'),
            files=[
                File('test.py', ['This is a test file for CI purpose.']),
                File('assignments.py', ['a = 1', 'b = 2', 'c = 3']),
                File('ci_check.py', ['def test():', '    do_ci_checks()']),
                File('random.py', ['def test():', '    randomness_validation()']),
                File('dummy_check.py', ['This is a dummy check file for test. No matter what. Well, you must be worrying about this novel book...']),
                File('novel_book.py', ['Previously on Lost: The tainted soul was frustrated by entangled love.']),
            ],
            diagnostics=[],
        )

        simplified = self.simplifier.simplify(context)
        self.assertEqual(simplified.files, [File(path='ci_check.py', content=['def test():', '    do_ci_checks()']), File(path='dummy_check.py', content=['This is a dummy check file for test. No matter what. Well, you must be worrying about this novel book...']), File(path='test.py', content=['This is a test file for CI purpose.'])])

if __name__ == '__main__':
    unittest.main()
