import argparse
import os
import sys

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from colbert import Indexer
from colbert.infra import ColBERTConfig
from tqdm import tqdm

# Import local modules
from .chunk import chunk_tokens
from .clean import html_to_text, tokenize


def main():
    # --- 1. Argument Parsing ---
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prefix", default="sec/raw/", help="Input prefix in R2 for raw HTML files."
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
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of filings to process for testing.",
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
        config=Config(signature_version="s3v4"),
    )

    # --- 3. Data Preparation ---
    print(f"Listing all objects from r2://{args.bucket}/{args.prefix}...")
    all_objects = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=args.bucket, Prefix=args.prefix)
    for page in pages:
        if "Contents" in page:
            all_objects.extend(
                [obj for obj in page["Contents"] if not obj["Key"].endswith("/")]
            )

    if args.limit:
        print(
            f"Applying limit: processing the first {args.limit} of {len(all_objects)} filings."
        )
        all_objects = all_objects[: args.limit]

    print(f"Gathering and chunking passages from {len(all_objects)} filings...")
    passages = []
    for obj in tqdm(all_objects, desc="Processing filings"):
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
        print(
            "Error: No passages found to embed. Check if the --prefix is correct and if files exist in the bucket."
        )
        return

    # --- 4. ColBERT-v2 Embedding and Indexing ---
    print(f"\nFound {len(passages)} passages. Configuring ColBERT indexer...")
    # Let ColBERT use its default root directory for creating the index.
    colbert_config = ColBERTConfig(
        doc_maxlen=2048,
        nbits=2,
    )

    indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=colbert_config)

    print("Indexing passages... (This will take a while on the GPU)")
    indexer.index(name=args.index_name, collection=passages, overwrite=True)

    print("Index created locally.")

    # --- 5. Upload Index Shards to R2 ---
    # FIX: Get the correct index path directly from the indexer object.
    local_index_path = indexer.get_index()
    print(
        f"Uploading index shards from '{local_index_path}' to r2://{args.bucket}/{args.out}..."
    )

    # ADDED: Verification to ensure the local index path exists before uploading.
    if not os.path.exists(local_index_path) or not os.listdir(local_index_path):
        print(
            f"FATAL ERROR: Index directory '{local_index_path}' not found or is empty."
        )
        print("The ColBERT indexer may have failed to write the files.")
        sys.exit(1)

    for root, _, files in os.walk(local_index_path):
        for file_name in tqdm(files, desc="Uploading shards"):
            local_file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_file_path, local_index_path)
            r2_key = os.path.join(args.out, args.index_name, relative_path).replace(
                "\\", "/"
            )

            try:
                s3.upload_file(local_file_path, args.bucket, r2_key)
            except ClientError as e:
                print(f"\nERROR uploading {file_name}: {e}")
                sys.exit(1)

    print("\nDone. Index uploaded successfully.")


if __name__ == "__main__":
    main()
