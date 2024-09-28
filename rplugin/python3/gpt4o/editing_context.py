import unittest
from dataclasses import dataclass
from typing import Dict, List

from gpt4o.types import File, Cursor, Prompt, Diagnostic
from gpt4o.utils import json_dumps
from gpt4o.resources import INSTRUCTIONS, DEFAULT_CHANGE_REQUEST

class TestEditingContext(unittest.TestCase):
    def test_compose_files(self):
        files = [File(path='hello.py', content=['def main():', '    pass'])]
        cursor = Cursor(path='hello.py', line=2, col=5, code='    pass')
        context = EditingContext(files=files, cursor=cursor, diagnostics=[])
        composed = context.compose_prompt('Implement the `main` function.')
        self.assertEqual(composed.instruction, INSTRUCTIONS.FILE_EDIT)
        self.assertEqual(composed.question, r'''
Current Cursor:
{"file":"hello.py","line":2,"col":5,"code":"    pass"}

Input JSON:
[{"file":"hello.py","content":[{"line":1,"text":"def main():"},{"line":2,"text":"    pass"}]}]

User Instruction:
1. Implement the `main` function.

Output the changes in the specified JSON format.
                         '''.strip())

@dataclass
class EditingContext:
    files: List[File]
    cursor: Cursor
    diagnostics: List[Diagnostic]

    @classmethod
    def __polish_change_request(cls, change: str) -> str:
        change = change.strip() or DEFAULT_CHANGE_REQUEST
        if not change.startswith('1.'):
            change = f'1. {change}'
        return change

    def compose_prompt(self, change: str) -> Prompt:
        change = self.__polish_change_request(change)

        table: Dict[str, str] = {}

        # table['Related context'] = json_dumps([
        #     {
        #         "file": file.path,
        #         "content": '\n'.join(file.content),
        #     }
        #     for file in self.files
        # ])

        table['Current Cursor'] = json_dumps({
            "file": self.cursor.path,
            "line": self.cursor.line,
            "col": self.cursor.col,
            "code": self.cursor.code,
        })

        table['Input JSON'] = json_dumps([
            {
                "file": file.path,
                "content": [
                    {
                        "line": line + 1,
                        "text": text,
                    } for line, text in enumerate(file.content)
                ],
            }
            for file in self.files
        ])

        if self.diagnostics:
            table['Diagnostics'] = json_dumps([
                {
                    "type": diag.type,
                    "message": diag.message,
                    "file": diag.file,
                    "line": diag.line,
                    "col": diag.col,
                    "code": diag.code,
                }
                for diag in self.diagnostics
            ])

        table['User Instruction'] = change

        question = ''.join(f'{key}:\n{value}\n\n' for key, value in table.items())
        question += 'Output the changes in the specified JSON format.'

        return Prompt(instruction=INSTRUCTIONS.FILE_EDIT,
                      question=question)

if __name__ == '__main__':
    unittest.main()
