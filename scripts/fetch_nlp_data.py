"""Fetch the SECONDARY NLP dataset: Federal Register EPA air-quality documents.

The primary sensor dataset (EPA AQS) has no legitimate text field, so the
NLP syllabus module uses a clearly-labeled secondary dataset:

    U.S. Federal Register (federalregister.gov API) - documents published
    by the Environmental Protection Agency matching "air quality",
    2021-2025. Public domain (US government work, 17 USC 105).
    API docs: https://www.federalregister.gov/developers/documentation/api/v1

Two document TYPES give a genuine, non-synthetic classification task:
    RULE   (rules & regulations)  vs  NOTICE  (notices)
The classification task is binary text classification with Naive Bayes
(syllabus: Naive Bayes), plus tokenization, stopword handling, word
frequency and WordCloud (syllabus: text mining, WordCloud).

No labels are fabricated: `type` is the document's official metadata.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_EXTERNAL  # noqa: E402

API = "https://www.federalregister.gov/api/v1/documents.json"
FIELDS = ["document_number", "title", "type", "abstract", "publication_date",
          "agencies", "raw_text_url"]
PER_PAGE = 100


def fetch_page(page: int) -> dict:
    params = urllib.parse.urlencode({
        "conditions[term]": "air quality",
        "conditions[agencies][]": "environmental-protection-agency",
        "conditions[publication_date][gte]": "2021-01-01",
        "conditions[publication_date][lte]": "2025-12-31",
        "conditions[type][]": ["RULE", "NOTICE"],
        "fields[]": FIELDS,
        "per_page": PER_PAGE,
        "page": page,
        "order": "newest",
    }, doseq=True)
    url = f"{API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "atmos-project/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main(max_pages: int = 12) -> int:
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    docs = []
    for page in range(1, max_pages + 1):
        try:
            data = fetch_page(page)
        except Exception as e:  # noqa: BLE001
            print(f"page {page} failed: {e}")
            break
        results = data.get("results", [])
        docs.extend(results)
        print(f"page {page}: +{len(results)} (total {len(docs)}); "
              f"count={data.get('count')}")
        if page * PER_PAGE >= data.get("count", 0) or not results:
            break
        time.sleep(1)  # polite rate limiting
    out = DATA_EXTERNAL / "federal_register_epa_airquality.json"
    out.write_text(json.dumps(docs, indent=1), encoding="utf-8")
    types = {}
    for d in docs:
        types[d["type"]] = types.get(d["type"], 0) + 1
    print(f"saved {len(docs)} docs -> {out}")
    print("type distribution:", types)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
