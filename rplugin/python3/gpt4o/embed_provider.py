import unittest
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

class TestEmbedProvider(unittest.TestCase):
    def test_batched_embed(self):
        provider: EmbedProvider = EmbedProviderOpenAI()
        inputs = ['hello', 'world']
        answer = provider.batched_embed(inputs)
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

@dataclass
class OpenAIEmbedConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    timeout: Optional[float] = None

    model: str = 'text-embedding-3-small'

class EmbedProviderOpenAI(EmbedProvider):
    def __init__(self):
        self.__config = OpenAIEmbedConfig()
        self.__client = None

    def batched_embed(self, inputs):
        if self.__client is None:
            import openai
            self.__client = openai.OpenAI(
                api_key=self.__config.api_key,
                base_url=self.__config.base_url,
                organization=self.__config.organization,
                project=self.__config.project,
                timeout=self.__config.timeout,
            )

        response = self.__client.embeddings.create(
            input=inputs,
            model=self.__config.model,
        )
        outputs = [embed.embedding for embed in response.data]
        return outputs

if __name__ == '__main__':
    unittest.main()
