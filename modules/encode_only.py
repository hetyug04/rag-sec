#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import tempfile
from chunk import chunk_text

import boto3
from colbert import Indexer
from colbert.infra import ColBERTConfig
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--in-prefix", default="sec/processed/")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    # --- Load & (re-)chunk passages ---
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET"],
    )

    passages = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=args.bucket, Prefix=args.in_prefix
    ):
        for o in page.get("Contents", []):
            if o["Key"].endswith("/") or (args.limit and len(passages) >= args.limit):
                continue
            data = json.loads(
                s3.get_object(Bucket=args.bucket, Key=o["Key"])["Body"].read()
            )
            for c in data["content"]:
                ids = TOKENIZER(c, add_special_tokens=False)["input_ids"]
                if len(ids) > 512:
                    passages.extend(chunk_text(c))
                else:
                    passages.append(c)

    if not passages:
        print("No passages to encode; exiting.")
        return

    # --- Encode-only via Indexer(nbits=0) on GPU ---
    config = ColBERTConfig(doc_maxlen=512, nbits=0)  # nbits=0 → skip k-means/PQ
    indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)

    print(f"Encoding {len(passages)} passages on GPU…")
    indexer.index(
        name="flat_only",
        collection=passages,
        overwrite=True,
    )  # no `out=` argument :contentReference[oaicite:4]{index=4}

    # --- Locate where files were written ---
    local_index_path = indexer.get_index()
    print(
        f"Index written to {local_index_path}"
    )  # e.g. “…/experiments/default/indexes/flat_only”

    # --- Zip up raw embeddings and upload ---
    raw_dir = os.path.join(local_index_path, "raw")
    if not os.path.isdir(raw_dir):
        raise RuntimeError(f"Could not find raw embeddings in {raw_dir}")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_base = os.path.join(tmpdir, "flat_embeddings")
        archive_path = shutil.make_archive(
            base_name=archive_base, format="zip", root_dir=raw_dir
        )
        print(
            f"Uploading {archive_path} → r2://{args.bucket}/embeddings/flat_embeddings.zip"
        )
        s3.upload_file(archive_path, args.bucket, "embeddings/flat_embeddings.zip")

    print("✅ GPU encode-only run complete.")


if __name__ == "__main__":
    main()
