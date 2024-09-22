import unittest
from typing import Iterable

from gpt4o.types import Prompt
from gpt4o.utils import json_loads

class TestPromptQuerier(unittest.TestCase):
    def setUp(self):
        self.querier = PromptQuerier()

    def test_query_prompt(self):
        prompt = Prompt(instruction='', question='Hello?')
        answer = ''.join(self.querier.query_prompt(prompt, seed=42))
        self.assertEqual(answer, r'Hello! How can I assist you today?')

    def test_query_prompt_json(self):
        prompt = Prompt(instruction='', question='List three bullet points of Python. Output in JSON format.')
        answer = ''.join(self.querier.query_prompt(prompt, force_json=True, seed=42))
        answer = json_loads(answer)
        self.assertTrue(isinstance(answer, dict) or isinstance(answer, list))

class PromptQuerier:
    def __init__(self):
        import openai

        self.client = openai.OpenAI()

    def query_prompt(self, prompt: Prompt,
                     *,
                     force_json: bool = False,
                     seed: int | None = None,
                     ) -> Iterable[str]:
        from openai import NotGiven

        completion = self.client.chat.completions.create(
            model='gpt-4o-mini',
            temperature=0,
            seed=seed,
            messages=[
                {"role": "system", "content": prompt.instruction},
                {"role": "user", "content": prompt.question},
            ],
            stream=True,
            response_format={"type": "json_object"} if force_json else NotGiven(),
        )

        for chunk in completion:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

if __name__ == '__main__':
    unittest.main()
