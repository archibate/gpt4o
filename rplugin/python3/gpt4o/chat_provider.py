import unittest
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from gpt4o.types import Prompt
from gpt4o.utils import json_loads
from gpt4o.resources import ENSURE_JSON_COMPATIBLE, FORCE_JSON_OK_WHITELIST

class TestChatProvider(unittest.TestCase):
    def test_query_prompt(self):
        provider: ChatProvider = ChatProviderOpenAI()
        prompt = Prompt(instruction='', question='Hello?')
        answer = ''.join(provider.query_prompt(prompt, seed=42))
        self.assertTrue('How can I assist you today?' in answer, answer)

    def test_query_prompt_json(self):
        provider: ChatProvider = ChatProviderOpenAI()
        prompt = Prompt(instruction='', question='List three bullet points of Python. Output in JSON format.')
        answer = ''.join(provider.query_prompt(prompt, force_json=True, seed=42))
        answer = json_loads(answer)
        self.assertTrue(isinstance(answer, dict) or isinstance(answer, list))

    def test_is_force_json_supported(self):
        provider = ChatProviderOpenAI()
        provider.get_config().base_url = None
        self.assertTrue(provider.is_force_json_supported())

        provider = ChatProviderOpenAI()
        provider.get_config().base_url = 'https://api.openai.com/v1'
        self.assertTrue(provider.is_force_json_supported())

        provider = ChatProviderOpenAI()
        provider.get_config().base_url = 'https://api.openai.com/beta'
        self.assertTrue(provider.is_force_json_supported())

        provider = ChatProviderOpenAI()
        provider.get_config().base_url = 'https://api.deepseek.com/v1'
        self.assertTrue(provider.is_force_json_supported())

        provider = ChatProviderOpenAI()
        provider.get_config().base_url = 'https://api.minimax.com/v1'
        self.assertFalse(provider.is_force_json_supported())

@dataclass
class OpenAIChatConfig:
    force_json_supported: Optional[bool] = None

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    timeout: Optional[float] = None

    model: str = 'gpt-4o'
    temperature: Optional[float] = 0.0
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None

class ChatProvider(ABC):
    @abstractmethod
    def query_prompt(self, prompt: Prompt,
                     *,
                     force_json: bool = False,
                     seed: Optional[int] = None,
                     ) -> Iterable[str]:
        pass
    
class ChatProviderOpenAI(ChatProvider):
    def __init__(self):
        self.__config = OpenAIChatConfig()
        self.__client = None

    def get_config(self) -> OpenAIChatConfig:
        return self.__config

    def is_force_json_supported(self) -> bool:
        supported = self.__config.force_json_supported
        if supported is None:
            if self.__config.base_url is None:
                return True
            else:
                for whitelist in FORCE_JSON_OK_WHITELIST:
                    if self.__config.base_url.lstrip('https://').lstrip('http://').startswith(whitelist):
                        return True
                else:
                    return False
        return supported

    def query_prompt(self, prompt, *, force_json = False, seed = None):
        import openai

        if force_json and not self.is_force_json_supported():
            prompt.question = f'{prompt.question} {ENSURE_JSON_COMPATIBLE}'
            force_json = False

        if self.__client is None:
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

class ChatProviderFree(ChatProviderOpenAI):
    def __init__(self):
        super().__init__()
        if self.get_config().base_url is None and self.get_config().api_key is None:
            self.get_config().base_url = 'https://142857.red/openai/v1'
            self.get_config().api_key = 'sk-free-of-charge'

if __name__ == '__main__':
    unittest.main()
