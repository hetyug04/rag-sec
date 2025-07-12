import argparse
import os
import sys

# Import local modules
from chunk import chunk_tokens

import boto3
from clean import html_to_text, tokenize
from colbert import Indexer
from colbert.infra import ColBERTConfig
from tqdm import tqdm


def main():
    # --- 1. Argument Parsing ---
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prefix", default="raw/", help="Input prefix in R2 for raw HTML files."
    )
    p.add_argument(
        "--out", default="embeddings/", help="Output prefix in R2 for index shards."
    )
    p.add_argument(
        "--bucket", default=os.environ.get("R2_BUCKET"), help="R2 bucket name."
    )
    p.add_argument(
        "--index-name",
        default="sec_filings_index",
        help="A name for the ColBERT index.",
    )
    args = p.parse_args()

    if not all(
        [
            os.environ.get("R2_ENDPOINT"),
            args.bucket,
            os.environ.get("R2_KEY"),
            os.environ.get("R2_SECRET"),
        ]
    ):
        print(
            "Error: R2 environment variables (ENDPOINT, BUCKET, KEY, SECRET) must be set."
        )
        sys.exit(1)

    # --- 2. S3/R2 Client Setup ---
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET"],
    )

    # --- 3. Data Preparation ---
    print("Gathering and chunking passages from R2...")
    passages = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=args.bucket, Prefix=args.prefix)
    for page in tqdm(pages, desc="Listing pages"):
        if "Contents" not in page:
            continue
        for obj in tqdm(page["Contents"], desc="Processing filings", leave=False):
            html_content = (
                s3.get_object(Bucket=args.bucket, Key=obj["Key"])["Body"]
                .read()
                .decode("utf-8", "ignore")
            )
            clean_text = html_to_text(html_content)
            tokens = tokenize(clean_text)
            for chunk in chunk_tokens(tokens):
                passages.append(chunk)

    if not passages:
        print("No passages found to embed. Exiting.")
        return

    # --- 4. ColBERT-v2 Embedding and Indexing ---
    print(f"\nFound {len(passages)} passages. Configuring ColBERT indexer...")
    colbert_config = ColBERTConfig(
        nbits=2,  # Number of bits for compression. 2 bits is standard for ColBERTv2.
        root="/content/colbert_indices/",  # Local path in Colab to store index parts
    )

    indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=colbert_config)

    print("Indexing passages... (This will take a while on the GPU)")
    indexer.index(name=args.index_name, collection=passages, overwrite=True)

    print("Index created locally.")

    # --- 5. Upload Index Shards to R2 ---
    print(f"Uploading index shards to r2://{args.bucket}/{args.out}{args.index_name}/")
    local_index_path = os.path.join(colbert_config.root, args.index_name)

    for root, _, files in os.walk(local_index_path):
        for file_name in tqdm(files, desc="Uploading shards"):
            local_file_path = os.path.join(root, file_name)
            # Create a relative path to maintain directory structure in R2
            relative_path = os.path.relpath(local_file_path, colbert_config.root)
            r2_key = os.path.join(args.out, relative_path).replace(
                "\\", "/"
            )  # Use forward slashes

            s3.upload_file(local_file_path, args.bucket, r2_key)

    print("\nDone. Index uploaded successfully.")


if __name__ == "__main__":
    main()
