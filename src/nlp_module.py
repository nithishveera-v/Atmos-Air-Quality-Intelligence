"""NLP module for the Atmos project (SECONDARY dataset - clearly labeled).

Corpus: 1,200 U.S. Federal Register documents (EPA, "air quality",
2021-2025), fetched by scripts/fetch_nlp_data.py. Public domain.
THIS IS A SECONDARY DATASET: the primary AQS sensor data contains no
text, so the NLP syllabus module is deliberately separate from the
primary air-quality analysis. Nothing here feeds the sensor models.

Task: classify document TYPE (Rule vs Notice) from title+abstract text.
The label is the official document metadata (no fabricated labels, no
label leakage: words come from the document itself, and the TF-IDF
vectorizer is fitted on the TRAINING SPLIT ONLY inside a Pipeline).

Includes: text cleaning, tokenization, stopword handling, word
frequency, WordCloud, Multinomial Naive Bayes classification, and
per-class precision/recall/F1 + confusion matrix.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_EXTERNAL, RANDOM_SEED  # noqa: E402

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


def load_corpus() -> pd.DataFrame:
    docs = json.loads(
        (DATA_EXTERNAL / "federal_register_epa_airquality.json").read_text(encoding="utf-8"))
    rows = []
    for d in docs:
        abstract = d.get("abstract") or ""
        text = f"{d.get('title', '')}. {abstract}".strip()
        rows.append({"document_number": d.get("document_number"),
                     "publication_date": d.get("publication_date"),
                     "doc_type": d["type"],
                     "text": text})
    return pd.DataFrame(rows)


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"https?://\S+", " ", s)
    tokens = TOKEN_RE.findall(s)
    return " ".join(tokens)


def word_frequencies(texts: pd.Series, stop_words, top_n: int = 30) -> pd.DataFrame:
    """Word frequency over cleaned, stopword-filtered tokens."""
    counts: dict[str, int] = {}
    for t in texts:
        for w in clean_text(t).split():
            if w in stop_words or len(w) < 3:
                continue
            counts[w] = counts.get(w, 0) + 1
    return (pd.DataFrame({"word": list(counts), "count": list(counts.values())})
            .sort_values("count", ascending=False).head(top_n)
            .reset_index(drop=True))


def run_nlp() -> dict:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    corpus = load_corpus()
    corpus = corpus[corpus["text"].str.len() > 40].copy()
    corpus["clean"] = corpus["text"].apply(clean_text)

    X = corpus["clean"]
    y = corpus["doc_type"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("nb", MultinomialNB(alpha=1.0)),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    labels = sorted(y.unique())
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=labels).ravel()

    freq_all = word_frequencies(corpus["text"], ENGLISH_STOP_WORDS)
    freq_rule = word_frequencies(corpus.loc[corpus["doc_type"] == "Rule", "text"],
                                 ENGLISH_STOP_WORDS, 20)
    freq_notice = word_frequencies(corpus.loc[corpus["doc_type"] == "Notice", "text"],
                                   ENGLISH_STOP_WORDS, 20)

    results = {
        "n_docs": int(len(corpus)),
        "class_counts": corpus["doc_type"].value_counts().to_dict(),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "precision_macro": round(float(precision_score(yte, pred, average="macro")), 4),
        "recall_macro": round(float(recall_score(yte, pred, average="macro")), 4),
        "f1_macro": round(float(f1_score(yte, pred, average="macro")), 4),
        "confusion": {"labels": labels,
                      "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "word_freq_all_top10": freq_all.head(10).to_dict("records"),
        "distinctive_note": (
            "Multinomial NB with TF-IDF (fit on train only). Rules and notices "
            "are distinguished largely by procedural vocabulary; this is a "
            "demonstration of the NLP syllabus module on a real, clearly "
            "labeled secondary corpus - not part of the primary sensor analysis."),
    }
    tables = {"freq_all": freq_all, "freq_rule": freq_rule, "freq_notice": freq_notice,
              "corpus": corpus[["document_number", "publication_date", "doc_type", "clean"]]}
    return {"results": results, "tables": tables, "pipeline": pipe,
            "stop_words": ENGLISH_STOP_WORDS}


def make_wordcloud(freq_df: pd.DataFrame):
    """Build a WordCloud from a frequency table (returns None if unavailable)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
    freqs = dict(zip(freq_df["word"], freq_df["count"]))
    wc = WordCloud(width=900, height=500, background_color="white",
                   colormap="viridis").generate_from_frequencies(freqs)
    return wc
