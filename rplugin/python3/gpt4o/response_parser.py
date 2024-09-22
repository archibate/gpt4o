import unittest
from typing import Iterable

from gpt4o.operations import OPERATIONS, Operation
from gpt4o.utils import json_loads

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

class TestResponseParser(unittest.TestCase):
    def setUp(self):
        self.parser = ResponseParser()

    def test_parse_replace(self):
        response = r'{"operation":"replace","line_start":3,"line_end":4,"content":["hello","world"]}'
        operation = self.parser.parse_response([response])
        self.assertEqual(operation, OPERATIONS['replace'](
            line_start=3, line_end=4, content=['hello', 'world']))

    def test_parse_delete(self):
        response = r'{"operation":"delete","line_start":3,"line_end":4}'
        operation = self.parser.parse_response([response])
        self.assertEqual(operation, OPERATIONS['delete'](
            line_start=3, line_end=4))

    def test_parse_append(self):
        response = r'{"operation":"append","line":3,"content":"hello"}'
        operation = self.parser.parse_response([response])
        self.assertEqual(operation, OPERATIONS['append'](line=3, content='hello'))

    # def test_parse_move(self):
    #     response = r'{"operation":"move","line_start":3,"line_end":4,"new_line":5}'
    #     operation = self.parser.parse_response([response])
    #     self.assertEqual(operation, OPERATIONS['move'](
    #         line_start=3, line_end=4, new_line=5))
                
if __name__ == '__main__':
    unittest.main()
