from typing import Iterable

def rid_triple_quotes(sequence: Iterable[str]) -> Iterable[str]:
    new_line = True
    escape = False
    ignore_until_nl = False
    count = 0
    for chunk in sequence:
        result = ''
        for c in chunk:
            if ignore_until_nl:
                escape = False
                new_line = False
                if c == '\n':
                    ignore_until_nl = False
                    new_line = True
                continue
            if c == '`' and new_line and not escape:
                count += 1
            else:
                count = 0
                result += c
                new_line = False
                escape = False
                if c == '\n':
                    new_line = True
                elif c == '\\':
                    escape = True
            if count >= 3:
                ignore_until_nl = True
        yield result

print(''.join(rid_triple_quotes('Yet.\n```python\nhello\n```\nGood.')))
