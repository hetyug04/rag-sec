"""
process.py - clean raw SEC filings, chunk to ≤500-token windows,
             and upload the results as JSON to Cloudflare R2
"""

from __future__ import annotations

import argparse
import json
import os
from chunk import TOKENIZER, chunk_text  # ← same tokenizer as ColBERT
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from clean import html_to_markdown
from tqdm import tqdm

# ---------------- S3 / R2 client ----------------
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_KEY"],
    aws_secret_access_key=os.environ["R2_SECRET"],
)


# -------------- main entry point ---------------
def main() -> None:
    p = argparse.ArgumentParser(description="Clean + chunk raw SEC filings")
    p.add_argument("--bucket", required=True, help="R2 bucket name")
    p.add_argument("--in-prefix", default="sec/raw/", help="Raw HTML prefix")
    p.add_argument("--out-prefix", default="sec/processed/", help="Output JSON prefix")
    p.add_argument("--limit", type=int, default=None, help="Limit filings processed")
    args = p.parse_args()

    # ---------- list raw HTML objects ----------
    objects: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.in_prefix):
        objects.extend(
            [o for o in page.get("Contents", []) if not o["Key"].endswith("/")]
        )

    if args.limit:
        objects = objects[: args.limit]
    print(f"Found {len(objects)} raw filings to process.")

    # ---------- process each file --------------
    for obj in tqdm(objects, desc="Cleaning & chunking"):
        try:
            raw_html = (
                s3.get_object(Bucket=args.bucket, Key=obj["Key"])["Body"]
                .read()
                .decode("utf-8", errors="ignore")
            )

            md = html_to_markdown(raw_html)  # clean → Markdown
            passages = list(chunk_text(md))  # ≤500 WP tokens each

            # length safety assertion
            assert all(
                len(TOKENIZER(p, add_special_tokens=True)["input_ids"]) <= 512
                for p in passages
            ), f"Oversize chunk in {obj['Key']}"

            if not passages:
                print(f"WARN: {obj['Key']} produced 0 passages; skipping.")
                continue

            out_key = os.path.join(
                args.out_prefix, Path(obj["Key"]).name + ".json"
            ).replace("\\", "/")

            payload = json.dumps({"source": obj["Key"], "content": passages})
            s3.put_object(
                Bucket=args.bucket,
                Key=out_key,
                Body=payload,
                ContentType="application/json",
            )

        except ClientError as ce:
            print(f"ERROR S3 on {obj['Key']}: {ce}")
        except AssertionError as ae:
            print(f"ASSERT {ae}")
        except Exception as e:
            print(f"ERROR processing {obj['Key']}: {e}")

    print("✅ Processing complete.")


if __name__ == "__main__":
    main()
