import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import aiofiles
import boto3
import dotenv
import httpx

# --- Configuration ---
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
    Fetches, stores, and logs a single filing discovered from the daily index.
    """
    cik = filing_info["cik"]
    acc_no = filing_info["accession_number"]
    form_type = filing_info["form_type"]
    filed_date = filing_info["filed_date"]

    async with sem:
        print(f"Processing: {form_type} from CIK {cik} (Acc No: {acc_no})")

        # FIX: Pad the CIK with leading zeros to make it 10 digits long.
        padded_cik = cik.zfill(10)
        submissions_url = f"https://data.sec.gov/submissions/CIK{padded_cik}.html"
        print(f"  -> Fetching metadata from: {submissions_url}")

        try:
            response = await client.get(submissions_url)
            response.raise_for_status()
            js = response.json()
        except httpx.HTTPStatusError as e:
            print(
                f"  -> WARN: HTTP {e.response.status_code} for metadata on CIK {cik}. This can be expected due to data lag. Skipping."
            )
            return
        except Exception as e:
            print(f"  -> ERROR: An unexpected error occurred for CIK {cik}: {e}")
            return

        primary_doc = None
        try:
            # The submissions JSON uses the non-padded CIK as a key
            filings = js.get("filings", {}).get("recent", {})
            acc_numbers = filings.get("accessionNumber", [])
            doc_names = filings.get("primaryDocument", [])
            for i, acc in enumerate(acc_numbers):
                if acc == acc_no:
                    primary_doc = doc_names[i]
                    break
        except (KeyError, IndexError):
            print(
                f"  -> WARN: Could not find filing {acc_no} in JSON for CIK {cik}. Skipping."
            )
            return

        if not primary_doc:
            print(
                f"  -> WARN: Filing {acc_no} not found in recent submissions for CIK {cik}. Skipping."
            )
            return

        key = f"{TARGET_PREFIX}{cik}/{acc_no}_{primary_doc}"

        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"  -> INFO: Already exists in R2: s3://{bucket}/{key}")
            return
        except s3.exceptions.ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                raise

        # The document URL path uses the non-padded, integer CIK
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no.replace('-', '')}/{primary_doc}"
        print(f"  -> Fetching document from: {doc_url}")

        try:
            html_content_response = await client.get(doc_url)
            html_content_response.raise_for_status()
            html_content = html_content_response.text
        except httpx.HTTPStatusError as e:
            print(
                f"  -> WARN: HTTP {e.response.status_code} for document URL on CIK {cik}. Skipping."
            )
            return

        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=html_content,
            ContentType="text/html",
            Metadata={"form": form_type, "filed": filed_date},
        )
        print(f"    -> SUCCESS: Uploaded to S3 at s3://{bucket}/{key}")
        log_message = {
            "cik": cik,
            "key": key,
            "form": form_type,
            "filed": filed_date,
            "status": "ok",
        }
        await log.write(json.dumps(log_message) + "\n")


async def main(max_filings: int, bucket: str):
    """
    Main function to discover the latest filings by walking backwards through
    SEC daily master index files, processing them concurrently, and then
    pruning storage if necessary.
    """
    print("Discovering latest filings from SEC daily indexes...")
    filings_to_process = []
    current_date = datetime.now()

    async with httpx.AsyncClient(headers=UA, timeout=30) as client:
        while len(filings_to_process) < max_filings:
            # Safety break to avoid infinite loops
            if (datetime.now() - current_date).days > 365:
                print("Searched back a full year. Stopping discovery.")
                break

            # Master index files are not available for weekends
            if current_date.weekday() >= 5:
                current_date -= timedelta(days=1)
                continue

            date_str = current_date.strftime("%Y%m%d")
            qtr = (current_date.month - 1) // 3 + 1
            year = current_date.year

            master_index_url = f"https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{qtr}/master.{date_str}.idx"

            try:
                print(f"Fetching index: {master_index_url}")
                response = await client.get(master_index_url)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    print(
                        f"  -> INFO: No index found for {date_str} (likely a holiday)."
                    )
                if e.response.status_code == 403:
                    if b"<Code>AccessDenied</Code>" in e.response.content:
                        # file truly not present - treat like 404
                        print(f"  -> INFO: {date_str} is holiday / no index.")
                        current_date -= timedelta(days=1)
                        continue
                else:
                    print(
                        f"Stopping due to non-404 HTTP error: {e.response.status_code}"
                    )
                    break

            # Add a delay *after every request* to respect rate limits
            await asyncio.sleep(0.12)

            # If the request was successful, parse the content
            if response.status_code == 200:
                lines = response.text.splitlines()
                start_line = next(
                    (i for i, line in enumerate(lines) if line.startswith("CIK|")), -1
                )

                if start_line != -1:
                    for line in reversed(
                        lines[start_line + 2 :]
                    ):  # Process newest first
                        parts = line.strip().split("|")
                        if len(parts) != 5:
                            continue

                        cik, _, form_type, date_filed, file_path = parts

                        if form_type in ("10-K", "10-Q"):
                            acc_no_match = re.search(r"(\d{10}-\d{2}-\d{6})", file_path)
                            if not acc_no_match:
                                continue

                            filing_info = {
                                "cik": cik,
                                "accession_number": acc_no_match.group(1),
                                "form_type": form_type,
                                "filed_date": date_filed,
                            }
                            filings_to_process.append(filing_info)

                            if len(filings_to_process) >= max_filings:
                                break

            # Move to the previous day for the next iteration
            current_date -= timedelta(days=1)

    if not filings_to_process:
        print("No new filings found after searching.")
        return

    print(f"\nDiscovered {len(filings_to_process)} total filings. Processing now.")
    print("-" * 60)

    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    log_file_path = log_path / f"crawl_{datetime.now():%Y-%m-%d}.html"

    sem = asyncio.Semaphore(8)  # Use a safe concurrency limit
    async with aiofiles.open(log_file_path, mode="a") as log:
        async with httpx.AsyncClient(headers=UA, timeout=30) as client:
            tasks = [
                process_filing(filing, client, log, bucket, sem)
                for filing in filings_to_process
            ]
            await asyncio.gather(*tasks)

    prune_bucket_if_needed(bucket, TARGET_PREFIX, SIZE_LIMIT_GB)


if __name__ == "__main__":
    dotenv.load_dotenv()
    p = argparse.ArgumentParser(description="Discover and fetch latest SEC filings.")
    p.add_argument(
        "--max-filings",
        type=int,
        default=500,
        help="Max number of recent filings to discover and process.",
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
