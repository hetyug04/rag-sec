![CI](https://github.com/hetyug04/rag-sec/actions/workflows/ci.yml/badge.svg)


Self-hosted Retrieval-Augmented Generation over SEC filings, edge-native and zero-cost.

7/11/25:
Quick update journal. Today I figured out how to retrieved the publick 10K (Annual) and 10Q (Quartely) reports from the SEC EDGAR
database through the API. Was simpler than I thought once I figured it out. I also set up my R2 bucket on cloudflare which will store the most recent
reports. I set up a workflow on github to rerun the script at 11PM EST, which will retrieve the most recent results up to date. Since the EDGAR system gets updated pretty frequently, once a day is a pretty good reup as new reports, which can be filled from 6am - 10pm, so 11pm is the perfect time. Only
problems I see in the forseeable future is my free R2 bucket storage overflowing, but im not that concerned about it since ~500 reports uploaded today to the bucket only took ~40 MB. 