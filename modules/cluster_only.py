import os

import boto3
import faiss
import numpy as np

# 1) Download embeddings
bucket = os.environ["R2_BUCKET"]
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_KEY"],
    aws_secret_access_key=os.environ["R2_SECRET"],
)
s3.download_file(bucket, "embeddings/flat_embeddings.npz", "embeddings.npz")

# 2) Load vectors
data = np.load("embeddings.npz")["embeddings"].astype("float32")
print("Loaded", data.shape[0], "vectors of dim", data.shape[1])

# 3) Build IVF index
d = data.shape[1]
nlist = 8192  # adjust for speed/recall tradeoff
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

# 4) Train + add
faiss.normalize_L2(data)
index.train(data)  # CPU k-means
index.add(data)

# 5) Save & (optional) upload
faiss.write_index(index, "ivf.index")
s3.upload_file("ivf.index", bucket, "embeddings/ivf.index")

print("Clustering complete.")
