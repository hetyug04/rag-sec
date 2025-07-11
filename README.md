@'
# rag-sec

Self-hosted Retrieval-Augmented Generation over SEC filings, edge-native and zero-cost.
'@ | Set-Content -Encoding UTF8 README.md

git add README.md
git commit -m "docs: add README"
git push -u origin main
