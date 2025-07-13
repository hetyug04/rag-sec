![CI](https://github.com/hetyug04/rag-sec/actions/workflows/ci.yml/badge.svg)


Self-hosted Retrieval-Augmented Generation over SEC filings, edge-native and zero-cost.

7/11/25:
Quick update journal. Today I figured out how to retrieved the publick 10K (Annual) and 10Q (Quartely) reports from the SEC EDGAR
database through the API. Was simpler than I thought once I figured it out. I also set up my R2 bucket on cloudflare which will store the most recent
reports. I set up a workflow on github to rerun the script at 11PM EST, which will retrieve the most recent results up to date. Since the EDGAR system gets updated pretty frequently, once a day is a pretty good reup as new reports, which can be filled from 6am - 10pm, so 11pm is the perfect time. Only
problems I see in the forseeable future is my free R2 bucket storage overflowing, but im not that concerned about it since ~500 reports uploaded today to the bucket only took ~40 MB. Tomorrow Ill start working on turning the text documents into embeddings

7/12/25:
I ran into some problems with yesterdays crawling and uploading method. Instead of getting the most recent 10k and 10Q filings, it was getting the most recent filings for a specific number of companies. I fixed that by changing the crawling algorithm to work backwards from the present day and skip weekends and holidays. I also implemented by cleaning, chunking, and embedding sections. I had some trouble figuring out how I wanted to run the chunking because the reports contained tables as well, so trying to hold meaning with those tables after passing them through an embedder was challenging. The clean.py module strips all the xml and styling, further reducing the size of the total embeddings. I have a basic script to run my embed.py on google colab, but im running into some weird dynamic GPU allocation issues and colab is not giving me anymore access to the T4's. I tested the whole pipline with 1 report initially and it succesfully saved the embeddinds to the cloudflare bucket, which I consider a success for the overall embedding process.