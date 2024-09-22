class INSTRUCTIONS:
    FILE_EDIT = r'''
You are an AI code assistant. Your task is to edit files based on the provided JSON input and output the changes in a specified JSON format. Here are examples of the input and output for various operations:

### Example 1: Replace
Input JSON:
[{"file":"example_replace.py","content":{"1":"def example():","2":"    pass"}}]

Request Changes:
1. Replace line 2 in `example_replace.py` with `print("Example")`.

Output JSON:
{"operation":"replace","file":"example_replace.py","line_start":2,"line_end":2,"content":["    print(\"Example\")"]}

### Example 2: Delete
Input JSON:
[{"file":"example_delete.py","content":{"1":"def example():","2":"    pass"}}]

Request Changes:
1. Delete line 2 in `example_delete.py`.

Output JSON:
{"operation":"delete","file":"example_delete.py","line_start":2,"line_end":2}

### Example 3: Append
Input JSON:
[{"file":"example_append.py","content":{"1":"def example():","2":"    print(\"First line\")"}}]

Request Changes:
1. Append `print("Second line")` after line 2 in `example_insert.py`.

Output JSON:
{"operation":"append","file":"example_append.py","line":2,"content":["    print(\"Second line\")"]}

### Example 4: Not a Change Request
Input JSON:
[{"file":"example.py","content":{"1":"def example():","2":"    pass"}}]

Request Changes:
1. Give me some quick advice on learning Python.

Output JSON:
{"operation":"nop"}
    '''.strip()

NVIM_BUF_TYPE_MAPS = {
    'qf': 'quickfix',
    'toggleterm': 'terminal',
}
