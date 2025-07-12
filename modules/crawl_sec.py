import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import aiofiles
import boto3
import dotenv
import httpx

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("R2_ENDPOINT"),
    aws_access_key_id=os.environ.get("R2_KEY"),
    aws_secret_access_key=os.environ.get("R2_SECRET"),
)
UA = {"User-Agent": "rag-sec bot <hetyug04@gmail.com>"}


def load_company_tickers(file_path="company_tickers.json") -> list[dict]:
    """
    Loads company ticker data from the specified JSON file.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return [v for k, v in data.items()]
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON from {file_path}.")
        return []


async def fetch_and_store_filings(
    company: dict,
    client: httpx.AsyncClient,
    max_filings_per_type: int,
    log,
    bucket: str,
    sem: asyncio.Semaphore,
):
    """
    Fetches, stores, and logs recent filings for a company,
    respecting a semaphore to limit concurrency.
    """
    # CIK must be zero-padded to 10 digits
    cik = str(company.get("cik_str", "")).zfill(10)
    company_name = company.get("title", "N/A")

    if not cik.isdigit():
        print(f"Skipping {company_name} due to invalid CIK.")
        return

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    async with sem:  # Acquire semaphore before making a request
        print(f"Processing: {company_name} (CIK: {cik})")
        try:
            response = await client.get(submissions_url)
            response.raise_for_status()
            js = response.json()
        except httpx.HTTPStatusError as e:
            print(f"  -> HTTP Error for {company_name}: {e.response.status_code}")
            return
        except Exception as e:
            print(f"  -> An unexpected error occurred for {company_name}: {e}")
            return

    filings = js.get("filings", {}).get("recent", {})
    forms = zip(
        filings.get("form", []),
        filings.get("accessionNumber", []),
        filings.get("primaryDocument", []),
        filings.get("filingDate", []),
    )

    filing_counts = {"10-K": 0, "10-Q": 0}
    for form, acc, doc, date in forms:
        if form not in ("10-K", "10-Q"):
            continue

        if (
            filing_counts["10-K"] >= max_filings_per_type
            and filing_counts["10-Q"] >= max_filings_per_type
        ):
            break

        if filing_counts.get(form, max_filings_per_type) < max_filings_per_type:
            key = f"sec/raw/{cik}/{acc}_{doc}"

            # Check if object already exists before downloading
            try:
                s3.head_object(Bucket=bucket, Key=key)
                print(f"  -> Already exists: s3://{bucket}/{key}")
                continue
            except s3.exceptions.ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                    raise  # Re-raise unexpected errors

            filing_counts[form] += 1
            print(f"  -> Found {form} filed on {date}. Accession No: {acc}")

            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"

            async with sem:  # Acquire semaphore for each download
                html_content = (await client.get(doc_url)).text

            # S3 upload is blocking, but fast enough for this context
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=html_content,
                ContentType="text/html",
                Metadata={"form": form, "filed": date},
            )
            print(f"    -> Uploaded to S3 at s3://{bucket}/{key}")
            log_message = {
                "cik": cik,
                "key": key,
                "form": form,
                "filed": date,
                "status": "ok",
            }
            await log.write(json.dumps(log_message) + "\n")

            await asyncio.sleep(0.1)  # Be a good citizen


async def main(max_companies: int, max_filings_per_type: int, bucket: str):
    """
    Main function to load companies and process their filings concurrently.
    """
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    log_file_path = log_path / f"crawl_{datetime.now():%Y-%m-%d}.jsonl"

    companies = load_company_tickers()
    if not companies:
        print("No company data found. Exiting.")
        return

    companies_to_process = companies[:max_companies]
    print(f"Processing the first {len(companies_to_process)} companies.")
    print("-" * 60)

    # Create a semaphore to limit concurrent requests to SEC EDGAR
    sem = asyncio.Semaphore(10)

    async with aiofiles.open(log_file_path, mode="a") as log:
        async with httpx.AsyncClient(headers=UA, timeout=30) as client:
            tasks = []
            for company in companies_to_process:
                task = fetch_and_store_filings(
                    company, client, max_filings_per_type, log, bucket, sem
                )
                tasks.append(task)
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    dotenv.load_dotenv()

    p = argparse.ArgumentParser(
        description="Fetch recent SEC filings and upload to an S3-compatible store."
    )
    p.add_argument(
        "--max_companies",
        type=int,
        default=50,
        help="Max number of companies to process.",
    )
    p.add_argument(
        "--max_filings_per_type",
        type=int,
        default=5,
        help="Max number of 10-K/10-Q filings to retrieve per company.",
    )
    # Argument to pull bucket name from env var, making it configurable
    p.add_argument(
        "--R2_BUCKET",
        type=str,
        default=os.environ.get("R2_BUCKET"),
        help="Name of the R2 bucket (or other S3-compatible bucket).",
    )
    args = p.parse_args()

    # Ensure bucket is specified
    if not args.R2_BUCKET:
        raise ValueError(
            "Bucket name must be provided via --R2_BUCKET argument or R2_BUCKET environment variable."
        )

    asyncio.run(main(args.max_companies, args.max_filings_per_type, args.R2_BUCKET))
