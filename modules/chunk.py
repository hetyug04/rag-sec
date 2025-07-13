from collections.abc import Iterator

from transformers import AutoTokenizer

# Load the tokenizer once to be reused
TOKENIZER = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")


def chunk_text(text: str, window_size: int = 500, stride: int = 450) -> Iterator[str]:
    """
    Splits a long text into overlapping chunks based on the model's actual tokenizer.
    """
    if not text:
        return

    # Tokenize the entire text once
    token_ids = TOKENIZER.encode(text, add_special_tokens=False)

    # Use a sliding window over the token IDs
    for i in range(0, len(token_ids), stride):
        chunk_ids = token_ids[i : i + window_size]

        # Skip chunks that are too short to be meaningful
        if len(chunk_ids) < 10:
            continue

        # Decode the token IDs back into a string
        yield TOKENIZER.decode(chunk_ids)
