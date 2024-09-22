import unittest
import json
from gpt4o.file import File

class TestPromptComposer(unittest.TestCase):
    def setUp(self):
        self.composer = PromptComposer()

    def test_compose_files(self):
        files = [RefFile(path='hello.py', content='def main():\n    pass')]
        composed = self.composer.compose_files(files)
        self.assertEqual(composed, r'{"file":"hello.py","content":"def main():\n    pass"}')


class PromptComposer:
    def compose_files(self, files: list[File]) -> str:
        data = {file.path: file.content for file in files}
        return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
