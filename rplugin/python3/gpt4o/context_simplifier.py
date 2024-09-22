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

    def simplify(self, context: EditingContext) -> EditingContext:
        current_file = None
        other_files: List[File] = []

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

        similiarities: List[float] = []
        for embed in embeds:
            similiarities.append(self.cosine_similiarity(current_embed, embed))

        return context

class TestContextSimplifier(unittest.TestCase):
    def setUp(self):
        from gpt4o.embed_provider import EmbedProviderFastEmbed
        self.embed = EmbedProviderFastEmbed()
        self.simplifier = ContextSimplifier(self.embed)

    def test_simplify(self):
        context = EditingContext(
            cursor=Cursor('test.py', 2, 3),
            files=[
                File('test.py', ['x = 5', 'y = 10']),
                File('other.py', ['a = 1', 'b = 2']),
                File('another.py', ['i = 7', 'j = 8']),
            ])

        simplified = self.simplifier.simplify(context)
        print(simplified)

if __name__ == '__main__':
    unittest.main()
