import argparse
import json
import os
from chunk import chunk_text

import boto3

# Import your custom cleaning and chunking logic
from clean import html_to_markdown
from tqdm import tqdm

# --- S3 Client Setup ---
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("R2_ENDPOINT"),
    aws_access_key_id=os.environ.get("R2_KEY"),
    aws_secret_access_key=os.environ.get("R2_SECRET"),
)


def main():
    p = argparse.ArgumentParser(
        description="Clean, chunk, and process raw SEC filings."
    )
    p.add_argument("--bucket", required=True, help="R2 bucket name.")
    p.add_argument("--in-prefix", default="sec/raw/", help="Prefix for raw HTML files.")
    p.add_argument(
        "--out-prefix",
        default="sec/processed/",
        help="Prefix for processed JSON output.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of filings to process.",
    )
    args = p.parse_args()

    # List all raw HTML objects
    all_objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=os.environ.get("R2_BUCKET"), Prefix=args.in_prefix
    ):
        if "Contents" in page:
            all_objects.extend(
                [obj for obj in page["Contents"] if not obj["Key"].endswith("/")]
            )

    if args.limit:
        all_objects = all_objects[: args.limit]

    print(f"Found {len(all_objects)} raw filings to process.")

    # Process each file
    for obj in tqdm(all_objects, desc="Cleaning and Chunking Files"):
        try:
            # 1. Get Raw HTML
            raw_html = (
                s3.get_object(Bucket="sec", Key=obj["Key"])["Body"]
                .read()
                .decode("utf-8")
            )

            # 2. Clean HTML to Markdown using your logic
            markdown_text = html_to_markdown(raw_html)

            # 3. Chunk the clean text using your logic
            passages = list(chunk_text(markdown_text))

            if not passages:
                print(f"WARN: No passages generated for {obj['Key']}. Skipping.")
                continue

            # 4. Prepare JSON output
            output_data = {
                "source": obj["Key"],
                "content": passages,
            }

            # 5. Save the processed JSON to the new prefix
            output_key = os.path.join(
                args.out_prefix, os.path.basename(obj["Key"]) + ".json"
            )
            s3.put_object(
                Bucket=os.environ.get("R2_BUCKET"),
                Key=output_key,
                Body=json.dumps(output_data, indent=2),
                ContentType="application/json",
            )
        except Exception as e:
            print(f"ERROR processing {obj['Key']}: {e}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
