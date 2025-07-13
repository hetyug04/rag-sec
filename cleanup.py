import os

import boto3
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# --- Configuration ---
BUCKET_NAME = os.environ.get("R2_BUCKET")  # Your bucket name
PREFIX_TO_DELETE = (
    ""  # Set to "" to delete everything, or e.g., "sec/raw/" to delete a folder
)

# --- Boto3 Setup ---
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("R2_ENDPOINT"),
    aws_access_key_id=os.environ.get("R2_KEY"),
    aws_secret_access_key=os.environ.get("R2_SECRET"),
)


def delete_all_objects(bucket: str, prefix: str):
    """
    Deletes all objects under a specified prefix in an R2 bucket.
    """
    print(f"Fetching all objects under prefix '{prefix}' in bucket '{bucket}'...")

    # 1. Collect all object keys to delete
    keys_to_delete = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                keys_to_delete.append({"Key": obj["Key"]})

    if not keys_to_delete:
        print("No objects found to delete.")
        return

    # 2. Delete objects in batches of 1000
    print(f"Found {len(keys_to_delete)} objects. Preparing to delete...")
    confirm = input("Are you sure you want to proceed? (yes/no): ")
    if confirm.lower() != "yes":
        print("Deletion aborted.")
        return

    for i in range(0, len(keys_to_delete), 1000):
        delete_batch = keys_to_delete[i : i + 1000]
        print(f"Deleting batch {i//1000 + 1}...")
        s3.delete_objects(
            Bucket=bucket, Delete={"Objects": delete_batch, "Quiet": True}
        )

    print(f"Successfully deleted {len(keys_to_delete)} objects.")


if __name__ == "__main__":
    if not BUCKET_NAME:
        raise ValueError("R2_BUCKET environment variable not set.")
    delete_all_objects(BUCKET_NAME, PREFIX_TO_DELETE)
