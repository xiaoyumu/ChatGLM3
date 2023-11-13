from typing import Optional

import tiktoken

# Refer to: https://github.com/openai/tiktoken
# encoding = tiktoken.get_encoding("cl100k_base")
# encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
DEFAULT_TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


def calc_tokens(text: str or list[str], model: Optional[str] = None) -> int:
    if not text:
        return 0

    if model:
        encoding = tiktoken.encoding_for_model(model)
    else:
        encoding = DEFAULT_TOKEN_ENCODING

    if isinstance(text, str):
        return len(encoding.encode(text))

    tokens = 0
    if isinstance(text, list):
        for t in text:
            tokens += len(encoding.encode(t))
    return tokens
