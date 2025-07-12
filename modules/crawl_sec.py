import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

import aiofiles
import boto3
import dotenv
import httpx
from bs4 import BeautifulSoup

# --- Configuration ---
SEC_RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-K%7C10-Q&owner=include&count=100&output=atom"
TARGET_PREFIX = "sec/raw/"
SIZE_LIMIT_GB = 2.0

# --- Boto3 and User-Agent Setup ---
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("R2_ENDPOINT"),
    aws_access_key_id=os.environ.get("R2_KEY"),
    aws_secret_access_key=os.environ.get("R2_SECRET"),
)
UA = {"User-Agent": "rag-sec bot <hetyug04@gmail.com>"}


def prune_bucket_if_needed(bucket: str, prefix: str, size_limit_gb: float):
    """
    Checks the total size of objects under a prefix and deletes the oldest
    files until the total size is under the specified limit.
    """
    print("\n" + "-" * 60)
    print(f"Checking storage size for prefix: s3://{bucket}/{prefix}")
    size_limit_bytes = size_limit_gb * 1024**3

    all_objects = []
    total_size_bytes = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                all_objects.append(obj)
                total_size_bytes += obj["Size"]

    print(
        f"Found {len(all_objects)} objects with a total size of "
        f"{total_size_bytes / 1024**3:.2f} GB."
    )

    if total_size_bytes <= size_limit_bytes:
        print("Storage size is within the limit. No action needed.")
        return

    print(f"Total size exceeds the {size_limit_gb} GB limit. Pruning oldest files...")
    bytes_to_delete = total_size_bytes - size_limit_bytes
    all_objects.sort(key=lambda x: x["LastModified"])

    keys_to_delete = []
    bytes_deleted = 0
    for obj in all_objects:
        if bytes_deleted < bytes_to_delete:
            keys_to_delete.append({"Key": obj["Key"]})
            bytes_deleted += obj["Size"]
        else:
            break

    print(
        f"Deleting {len(keys_to_delete)} files to free up {bytes_deleted / 1024**2:.2f} MB..."
    )
    for i in range(0, len(keys_to_delete), 1000):
        delete_batch = keys_to_delete[i : i + 1000]
        s3.delete_objects(
            Bucket=bucket, Delete={"Objects": delete_batch, "Quiet": True}
        )

    print("Pruning complete.")


async def process_filing(
    filing_info: dict,
    client: httpx.AsyncClient,
    log,
    bucket: str,
    sem: asyncio.Semaphore,
):
    """
    Fetches, stores, and logs a single filing discovered from the RSS feed.
    """
    cik = filing_info["cik"]
    acc_no = filing_info["accession_number"]
    form_type = filing_info["form_type"]
    filed_date = filing_info["filed_date"]

    async with sem:
        print(f"Processing: {form_type} from CIK {cik} (Acc No: {acc_no})")

        # Use the submissions API to get the primary document name
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            response = await client.get(submissions_url)
            response.raise_for_status()
            js = response.json()
        except httpx.HTTPStatusError as e:
            print(f"  -> HTTP Error for CIK {cik}: {e.response.status_code}")
            return
        except Exception as e:
            print(f"  -> An unexpected error occurred for CIK {cik}: {e}")
            return

        # Find the specific filing by accession number to get the document name
        primary_doc = None
        try:
            filings = js.get("filings", {}).get("recent", {})
            acc_numbers = filings.get("accessionNumber", [])
            doc_names = filings.get("primaryDocument", [])
            for i, acc in enumerate(acc_numbers):
                if acc == acc_no:
                    primary_doc = doc_names[i]
                    break
        except (KeyError, IndexError):
            print(f"  -> Could not find filing {acc_no} in JSON for CIK {cik}.")
            return

        if not primary_doc:
            print(
                f"  -> Filing {acc_no} not found in recent submissions for CIK {cik}."
            )
            return

        key = f"{TARGET_PREFIX}{cik}/{acc_no}_{primary_doc}"

        # Check if the object already exists before downloading
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"  -> Already exists: s3://{bucket}/{key}")
            return
        except s3.exceptions.ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                raise

        # Construct the final document URL and download
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no.replace('-', '')}/{primary_doc}"
        html_content = (await client.get(doc_url)).text

        # Upload to S3
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=html_content,
            ContentType="text/html",
            Metadata={"form": form_type, "filed": filed_date},
        )
        print(f"    -> Uploaded to S3 at s3://{bucket}/{key}")
        log_message = {
            "cik": cik,
            "key": key,
            "form": form_type,
            "filed": filed_date,
            "status": "ok",
        }
        await log.write(json.dumps(log_message) + "\n")
        await asyncio.sleep(0.1)


async def main(max_filings: int, bucket: str):
    """
    Main function to discover the latest filings from the SEC RSS feed,
    process them concurrently, and then prune storage if necessary.
    """
    # 1. Fetch the latest filings from the SEC RSS feed
    print("Fetching latest filings from SEC RSS feed...")
    async with httpx.AsyncClient(headers=UA, timeout=30) as client:
        try:
            response = await client.get(SEC_RSS_URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "xml")
            entries = soup.find_all("entry")
        except httpx.HTTPStatusError as e:
            print(f"Could not fetch SEC RSS feed. Error: {e.response.status_code}")
            return

    # 2. Parse feed entries to get filing metadata
    filings_to_process = []
    for entry in entries:
        try:
            # Extract CIK from title, e.g., "10-K - TESLA, INC. (0001318605) (Filer)"
            cik_match = re.search(r"\((\d{10})\)", entry.title.text)
            if not cik_match:
                continue

            filing_info = {
                "cik": cik_match.group(1),
                "accession_number": entry.id.text.split("=")[-1],
                "form_type": entry.category["term"],
                "filed_date": entry.updated.text,
            }
            filings_to_process.append(filing_info)
        except (AttributeError, KeyError, IndexError):
            print("  -> Skipping a malformed RSS entry.")
            continue

    if not filings_to_process:
        print("No new filings found in the RSS feed.")
        return

    # 3. Process the discovered filings concurrently
    filings_to_process = filings_to_process[:max_filings]
    print(f"Discovered {len(filings_to_process)} new filings to process.")
    print("-" * 60)

    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    log_file_path = log_path / f"crawl_{datetime.now():%Y-%m-%d}.jsonl"

    sem = asyncio.Semaphore(10)
    async with aiofiles.open(log_file_path, mode="a") as log:
        async with httpx.AsyncClient(headers=UA, timeout=30) as client:
            tasks = [
                process_filing(filing, client, log, bucket, sem)
                for filing in filings_to_process
            ]
            await asyncio.gather(*tasks)

    # 4. After all crawling is done, check and prune the bucket
    prune_bucket_if_needed(bucket, TARGET_PREFIX, SIZE_LIMIT_GB)


if __name__ == "__main__":
    dotenv.load_dotenv()
    p = argparse.ArgumentParser(description="Discover and fetch latest SEC filings.")
    p.add_argument(
        "--max-filings",
        type=int,
        default=100,
        help="Max number of filings to process from the RSS feed.",
    )
    p.add_argument(
        "--R2_BUCKET",
        type=str,
        default=os.environ.get("R2_BUCKET"),
        help="Name of the R2 bucket (or other S3-compatible bucket).",
    )
    args = p.parse_args()

    if not args.R2_BUCKET:
        raise ValueError(
            "Bucket name must be provided via --R2_BUCKET or R2_BUCKET env var."
        )

    asyncio.run(main(args.max_filings, args.R2_BUCKET))
