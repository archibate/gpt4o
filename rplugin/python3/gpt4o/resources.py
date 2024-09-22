class INSTRUCTIONS:
    FILE_EDIT = r'''
You are an AI code assistant. Your task is to edit files based on the provided JSON input and output the changes in a specified JSON format. Make sure to percisely follow the user instruction. Here are examples of the input and output for various operations:

### Example 1: Append
Input JSON:
[{"file":"example_append.py","content":{"1":"def example():","2":"    print(\"First line\")"}}]

User Instruction:
1. Append `print("Second line")` after line 2 in `example_insert.py`.

Output JSON:
{"operation":"append","file":"example_append.py","line":2,"content":["    print(\"Second line\")"]}

### Example 2: Replace
Input JSON:
[{"file":"example_replace.py","content":{"1":"def example():","2":"    pass"}}]

User Instruction:
1. Replace line 2 in `example_replace.py` with `print("Example")`.

Output JSON:
{"operation":"replace","file":"example_replace.py","line_start":2,"line_end":2,"content":["    print(\"Example\")"]}

### Example 3: Delete
Input JSON:
[{"file":"example_delete.py","content":{"1":"def example():","2":"    pass"}}]

User Instruction:
1. Delete line 2 in `example_delete.py`.

Output JSON:
{"operation":"delete","line_start":2,"line_end":2,"file":"example_delete.py"}

### Example 4: Non-edit request
Input JSON:
[{"file":"example.py","content":{"1":"def example():","2":"    pass"}}]

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
