import unittest
from abc import ABC, abstractmethod
from typing import List

class TestEmbedProvider(unittest.TestCase):
    def test_batched_embed(self):
        provider: EmbedProvider = EmbedProviderFastEmbed()
        inputs = ['hello', 'world']
        answer = ''.join(provider.batched_embed(inputs))
        self.assertEqual(len(answer), 2, answer)

class EmbedProvider(ABC):
    @abstractmethod
    def batched_embed(self, inputs: List[str]) -> List[List[float]]:
        pass

class EmbedProviderFastEmbed(EmbedProvider):
    def __init__(self):
        self.model = None

    def batched_embed(self, inputs):
        if self.model is None:
            import fastembed
            self.model = fastembed.TextEmbedding()

        outputs = []
        for embed in self.model.embed(inputs):
            outputs.append(list(embed))
        return outputs

if __name__ == '__main__':
    unittest.main()
