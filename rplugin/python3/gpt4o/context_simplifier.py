from dataclasses import dataclass
import unittest
from typing import List
import math

from gpt4o.types import File
from gpt4o.embed_provider import EmbedProvider
from gpt4o.editing_context import EditingContext

@dataclass
class Chunk:
    content: str
    line_start: int
    line_end: int
    file: File
    similiarity: float = 0

class ContextSimplifier:
    def __init__(self, embed: EmbedProvider):
        self.embed = embed

    @staticmethod
    def cosine_similiarity(a: List[float], b: List[float]) -> float:
        denom = math.sqrt(sum(ai * ai for ai in a)) * math.sqrt(sum(bi * bi for bi in b))
        return sum(ai * bi for ai, bi in zip(a, b)) / denom

    @staticmethod
    def split_into_chunks(file: File) -> List[Chunk]:
        return [
            Chunk(
                content='\n'.join(file.content),
                line_start=1,
                line_end=len(file.content),
                file=file,
            ),
        ]

    def simplify(self, context: EditingContext) -> EditingContext:
        current_file = None
        other_files: List[File] = []

        for file in context.files:
            if file.path == context.cursor.path:
                current_file = file
            else:
                other_files.append(file)

        assert current_file is not None
        chunks = self.split_into_chunks(current_file) + sum((
            self.split_into_chunks(file) for file in other_files),
            start=[])
        embeds = self.embed.batched_embed([chunk.content for chunk in chunks])
        assert len(embeds) == len(chunks)
        current_embed = embeds.pop(0)

        for chunk, embed in zip(chunks, embeds):
            chunk.similiarity = self.cosine_similiarity(current_embed, embed))

        return context

if __name__ == '__main__':
    unittest.main()
