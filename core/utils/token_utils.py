from math import trunc

def chars_to_tokens(chars: int) -> int:
    return trunc(chars / 4.7)

def tokens_to_chars(tokens: int) -> int:
    return trunc(tokens * 4.7)