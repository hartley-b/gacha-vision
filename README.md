# Gacha Vision

Upload screenshots of the Yo-kai Watch Puni Puni gacha rate list and get back the character IDs they contain, e.g. `{1002, 4032}`, with a review table to catch and correct any wrong row before you copy the results.

## Setup

Requires Python 3.11+ and Tesseract with the Japanese language pack:

```bash
brew install tesseract tesseract-lang
```

Then:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open **http://127.0.0.1:5063** and drop one or more screenshots on the page. First run may take a minute to download the CLIP model weights; after that, startup is fast (reference embeddings ship precomputed in `cache/`).

## Directory Uutline

```
app.py                    # the web app (Flask, port 5063)
src/                      # identification pipeline
  row_extraction.py       #   find icon crops in a screenshot
  rank_detection.py       #   template-match rank banners
  rank_grouping.py        #   assign rows to ranks
  matching.py             #   CLIP embedding nearest-neighbor vs final_sorted
  name_ocr.py             #   OCR the on-screen name, fuzzy lookup (2nd signal)
  identification.py       #   ties it together; text can override visual
data/character_names.json # id, rank, jpn/eng name for all characters
ref/banner_templates/     # one cropped rank-banner graphic per rank
final_sorted/             # reference icons, by rank + ALL/ (the match database)
cache/                    # precomputed reference embeddings (rebuilt if stale)
requirements.txt
```

## How it decides

Each row is identified two independent ways: visually (CLIP embedding nearest-neighbor against the reference icons, scoped to the row's rank) and textually (OCR of the on-screen name, fuzzy-matched against the name database). The visual match wins by default; a near-exact text read that disagrees overrules it. Rows where the signals disagree are tinted and tagged in the review table. 

Initial testing was measured on a fully audited 180-row ground-truth data set. The combined pipeline scores 180/180.
