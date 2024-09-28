class TokenCounterOpenAI:
    def __init__(self):
        self.__encoding = None

    def count_token(self, text: str) -> int:
        import tiktoken
        if self.__encoding is None:
            self.__encoding = tiktoken.encoding_for_model('gpt-4o')
        return len(self.__encoding.encode_ordinary(text))
