import argparse
import json
import os

import boto3
import numpy as np
from colbert import Checkpoint
from colbert.infra import ColBERTConfig
from transformers import AutoTokenizer

from modules.chunk import chunk_text

TOKENIZER = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--in-prefix", default="sec/processed/")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET"],
    )

    # 1) load all JSON chunks
    all_chunks = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=args.bucket, Prefix=args.in_prefix
    ):
        for o in page.get("Contents", []):
            if o["Key"].endswith("/") or (args.limit and len(all_chunks) >= args.limit):
                continue
            data = json.loads(
                s3.get_object(Bucket=args.bucket, Key=o["Key"])["Body"].read()
            )
            all_chunks.extend(data["content"])

    # 2) optionally re-chunk oversized (shouldn't happen)
    safe_chunks = []
    for c in all_chunks:
        if len(TOKENIZER(c, add_special_tokens=False)["input_ids"]) > 512:
            safe_chunks.extend(chunk_text(c))
        else:
            safe_chunks.append(c)

    # 3) encode on GPU only
    ckpt = Checkpoint(
        "colbert-ir/colbertv2.0", colbert_config=ColBERTConfig(doc_maxlen=512, nbits=0)
    )
    indexer = ckpt.indexer
    vectors = indexer.encode(safe_chunks)  # returns numpy (Nx128)

    # 4) save & upload to R2
    npy_path = "/content/embeddings.npz"
    np.savez(npy_path, embeddings=vectors)
    s3.upload_file(npy_path, args.bucket, "embeddings/flat_embeddings.npz")


if __name__ == "__main__":
    main()
