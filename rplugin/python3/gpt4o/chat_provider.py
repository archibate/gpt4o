import unittest
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from gpt4o.types import Prompt
from gpt4o.utils import json_loads
from gpt4o.resources import ENSURE_JSON_COMPATIBLE

@dataclass
class ChatConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    timeout: Optional[float] = None

    model: str = 'gpt-4o-mini'
    temperature: Optional[float] = 0.0
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None

class TestChatProvider(unittest.TestCase):
    def setUp(self):
        self.provider: ChatProvider = ChatProviderOpenAI()

    def test_query_prompt(self):
        prompt = Prompt(instruction='', question='Hello?')
        answer = ''.join(self.provider.query_prompt(prompt, seed=42))
        self.assertEqual(answer, r'Hello! How can I assist you today?')

    def test_query_prompt_json(self):
        prompt = Prompt(instruction='', question='List three bullet points of Python. Output in JSON format.')
        answer = ''.join(self.provider.query_prompt(prompt, force_json=True, seed=42))
        answer = json_loads(answer)
        self.assertTrue(isinstance(answer, dict) or isinstance(answer, list))

class ChatProvider(ABC):
    @abstractmethod
    def get_config(self) -> ChatConfig:
        pass

    @abstractmethod
    def query_prompt(self, prompt: Prompt,
                     *,
                     force_json: bool = False,
                     seed: int | None = None,
                     ) -> Iterable[str]:
        pass

class ChatProviderOpenAI(ChatProvider):
    def __init__(self):
        import openai

        self.__config = ChatConfig()

    def get_config(self):
        return self.__config

    def query_prompt(self, prompt, *, force_json = False, seed = None):
        import openai

        if force_json:
            prompt.question = f'{prompt.question} {ENSURE_JSON_COMPATIBLE}'

        self.__client = openai.OpenAI(
            api_key=self.__config.api_key,
            base_url=self.__config.base_url,
            organization=self.__config.organization,
            project=self.__config.project,
            timeout=self.__config.timeout,
        )
        completion = self.__client.chat.completions.create(
            model=self.__config.model,
            temperature=self.__config.temperature,
            frequency_penalty=self.__config.frequency_penalty,
            presence_penalty=self.__config.presence_penalty,
            max_tokens=self.__config.max_tokens,
            seed=seed,
            messages=[
                {"role": "system", "content": prompt.instruction},
                {"role": "user", "content": prompt.question},
            ],
            stream=True,
            response_format={"type": "json_object"} if force_json else openai.NotGiven(),
        )

        for chunk in completion:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

if __name__ == '__main__':
    unittest.main()
