from collections.abc import Iterator


def chunk_tokens(
    tokens: list[str], size: int = 500, overlap: int = 50
) -> Iterator[str]:
    """
    Splits a list of tokens into overlapping chunks of a specified size.
    """
    if not tokens:
        return

    step = size - overlap
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i : i + size])
        yield chunk
