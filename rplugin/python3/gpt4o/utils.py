import json
from typing import Any

def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))

def json_loads(data: str) -> Any:
    return json.loads(data)

def tokenize(content: str):
    import tiktoken
    from colorama import Fore, Style
    enc = tiktoken.encoding_for_model('gpt-4o')
    tokens = enc.decode_tokens_bytes(enc.encode_ordinary(content))
    result = ''
    for i, token in enumerate(tokens):
        try:
            token = token.decode('utf-8')
        except UnicodeDecodeError:
            token = token.decode('latin-1')
        # token = token.replace(' ', '▁')
        result += f'{Style.BRIGHT}{Fore.YELLOW if i % 2 == 0 else Fore.BLUE}{token}{Fore.RESET}{Style.NORMAL}'
    return result

if __name__ == '__main__':
    import sys
    print(tokenize(' '.join(sys.argv[1:])))
