import unittest
from typing import Iterable

from gpt4o.operations import OPERATIONS, Operation
from gpt4o.utils import json_loads

class TestResponseParser(unittest.TestCase):
    def setUp(self):
        self.parser = ResponseParser()

    def test_query_prompt(self):
        response = r'{"operation":"replace","line_start":3,"line_end":4,"content":["hello","world"]}'
        operation = self.parser.parse_response([response])
        self.assertEqual(operation, OPERATIONS['replace'](
            line_start=3, line_end=4, content=['hello', 'world']))

class ResponseParser:
    def __init__(self):
        pass

    def parse_response(self, response: Iterable[str]) -> Operation:
        data = ''.join(response)
        data = json_loads(data)
        op: str = data['operation']
        cls = OPERATIONS[op]
        members = {key: data[key] for key in cls.__annotations__.keys()}
        operation: Operation = cls(**members)
        return operation
                
if __name__ == '__main__':
    unittest.main()
