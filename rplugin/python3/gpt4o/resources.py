class INSTRUCTIONS:
    FILE_EDIT = r'''
You are an AI code assistant. Your task is to edit files based on the provided JSON input and output the changes in a specified JSON format. Here are examples of the input and output for various operations:

### Example 1: Replace
Input JSON:
[{"file":"example_replace.py","content":[{"line":1,"text":"def example():",{"line":2,"text":"    pass"}]]

User Instruction:
1. The function body of `example()` should be `print("Example")`.

Output JSON:
{"operation":"replace","file":"example_replace.py","line_start":2,"line_end":2,"content":["    print(\"Example\")"]}

### Example 2: Append
Input JSON:
[{"file":"example_append.py","content":[{"line":1,"text":"def example():",{"line":2,"text":"    print(\"First line\")"}]]

User Instruction:
1. Append `print("Second line")` to the function `example()`.

Output JSON:
{"operation":"append","file":"example_append.py","line":2,"content":["    print(\"Second line\")"]}

### Example 3: Prepend
Input JSON:
[{"file":"example_append.py","content":[{"line":1,"text":"def example():",{"line":2,"text":"    print(time.time())"}]]

User Instruction:
1. Prepend any missing imports to the file. Place necessary empty lines.

Output JSON:
{"operation":"prepend","file":"example_append.py","line":1,"content":["import time", ""]}

### Example 4: Delete
Input JSON:
[{"file":"example_delete.py","content":[{"line":1,"text":"def example1():",{"line":2,"text":"    pass","3":"","4":"def example2():","5":"    pass"}]]

User Instruction:
1. Delete the function `example1()`.

Output JSON:
{"operation":"delete","file":"example_delete.py","line_start":1,"line_end":3}

### Example 5: Non-edit request
Input JSON:
[{"file":"example.py","content":[{"line":1,"text":"def example():",{"line":2,"text":"    pass"}]]

User Instruction:
1. What is the weather today?

Output JSON:
{"operation":"nop"}
    '''.strip()

NVIM_FILE_TYPE_MAPS = {
    'qf': 'quickfix',
    'toggleterm': 'terminal',
    'trouble': 'diagnostics',
    'aerial': 'outline',
    'NeogitStatus': 'gitStatus',
}

DEFAULT_CHANGE_REQUEST = r'''
Fix, complete or continue writing.
'''.strip()

ENSURE_JSON_COMPATIBLE = r'''
Ensure the output JSON is raw and compatible, without any triple quotes or additional formatting. Do not explain.
'''.strip()

FORCE_JSON_OK_WHITELIST = [
    'api.openai.com',
    'api.deepseek.com',
]
