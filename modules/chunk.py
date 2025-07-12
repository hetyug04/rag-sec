def chunk_tokens(tokens: list[str], size: int = 2000, overlap: int = 200) -> list[str]:
    """
    Splits a list of tokens into overlapping chunks of a specified size.
    """
    if not tokens:
        return []

    step = size - overlap
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i : i + size])
        yield chunk
