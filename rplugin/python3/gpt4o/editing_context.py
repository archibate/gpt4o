import unittest
from dataclasses import dataclass

from gpt4o.types import File, CursorPos, Prompt
from gpt4o.utils import json_dumps
from gpt4o.resources import INSTRUCTIONS

class TestEditingContext(unittest.TestCase):
    def test_compose_files(self):
        files = [File(path='hello.py', content=['def main():', '    pass'])]
        cursor = CursorPos(path='hello.py', line=2, column=5)
        context = EditingContext(files=files, cursor=cursor)
        composed = context.compose_prompt('Implement the `main` function.')
        self.assertEqual(composed.instruction, INSTRUCTIONS.FILE_EDIT)
        self.assertEqual(composed.question, r'''
Input JSON:
[{"file":"hello.py","content":{"1":"def main():","2":"    pass"}}]

Current Cursor:
{"file":"hello.py","line":2,"col":5}

Request Changes:
1. Implement the `main` function.

Output the changes in the specified JSON format. Ensure the output JSON is raw and compatible, without any triple quotes or additional formatting. Do not explain.
                         '''.strip())

@dataclass
class EditingContext:
    files: list[File]
    cursor: CursorPos

    def compose_prompt(self, change: str) -> Prompt:
        files = [
            {"file": file.path, "content": {str(line + 1): text for line, text in enumerate(file.content)}}
            for file in self.files
        ]
        cursor = {
            "file": self.cursor.path,
            "line": self.cursor.line,
            "col": self.cursor.column,
        }
        question = rf'''
Input JSON:
{json_dumps(files)}

Current Cursor:
{json_dumps(cursor)}

Request Changes:
1. {change}

Output the changes in the specified JSON format. Ensure the output JSON is raw and compatible, without any triple quotes or additional formatting. Do not explain.
        '''.strip()
        return Prompt(instruction=INSTRUCTIONS.FILE_EDIT,
                      question=question)

if __name__ == '__main__':
    unittest.main()
