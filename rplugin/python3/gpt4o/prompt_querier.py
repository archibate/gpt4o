import unittest
from gpt4o.types import Prompt

class TestPromptQuerier(unittest.TestCase):
    def setUp(self):
        self.querier = PromptQuerier()

    def test_query_prompt(self):
        prompt = Prompt(instruction='', question='Hello?')
        answer = self.querier.query_prompt(prompt)
        self.assertEqual(answer, r'Hello, how can I assist you today?')

class PromptQuerier:
    def query_prompt(self, prompt: Prompt) -> str:
        files_data = [{"path": file.path, "content": file.content} for file in files]
        return json_dumps(files_data)

if __name__ == '__main__':
    unittest.main()
