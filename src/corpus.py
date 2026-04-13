"""Corpus loader and BM25 index for the U.S. Treasury Bulletins.

On first use this downloads `treasury_bulletins_transformed.zip` from the
databricks/officeqa repo (pinned commit), extracts the 697 plain-text
bulletins into memory, splits each into overlapping character chunks,
and builds a BM25 index.

Everything is kept in memory — total corpus is ~100 MB uncompressed,
~70k chunks.  No embeddings, no external service at query time.
"""
from __future__ import annotations

import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

CORPUS_URL_DEFAULT = (
    "https://github.com/databricks/officeqa/raw/6aa8c32/"
    "treasury_bulletins_parsed/transformed/treasury_bulletins_transformed.zip"
)

# Chunking parameters — Treasury Bulletins have dense tables, so we keep
# chunks generous so a single table row + its column headers land together.
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 300

# Filename pattern:  treasury_bulletin_1941_01.txt
FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.txt$", re.IGNORECASE)

# Very small English stopword list — we keep numbers, units, and domain
# terms in the token stream.
_STOPWORDS = frozenset(
    """
    a an the of in on at to for from by with and or but if then else when
    as is are was were be been being have has had do does did not no yes
    this that these those it its i we you he she they them our your their
    me my his her what which who whom whose why how
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.%]*")


def tokenize(text: str) -> list[str]:
    """Lower-cased token stream keeping numbers, %, hyphens, decimals."""
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Chunk:
    chunk_id: int
    filename: str          # e.g. treasury_bulletin_1941_01.txt
    year: int
    month: int
    text: str


def _chunk_file(text: str, filename: str, start_id: int) -> list[Chunk]:
    """Split a bulletin file into overlapping character chunks, aligned to newlines."""
    m = FILENAME_RE.search(filename)
    year = int(m.group(1)) if m else 0
    month = int(m.group(2)) if m else 0

    chunks: list[Chunk] = []
    i = 0
    n = len(text)
    cid = start_id
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        # Snap end to a newline if close, to keep table rows intact.
        if end < n:
            nl = text.rfind("\n", i + CHUNK_CHARS // 2, end)
            if nl != -1:
                end = nl
        piece = text[i:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    filename=filename,
                    year=year,
                    month=month,
                    text=piece,
                )
            )
            cid += 1
        if end >= n:
            break
        i = max(end - CHUNK_OVERLAP, i + 1)
    return chunks


class Corpus:
    """Loaded corpus + BM25 index."""

    def __init__(self, chunks: list[Chunk], bm25: BM25Okapi) -> None:
        self.chunks = chunks
        self.bm25 = bm25

    @classmethod
    def load(cls, cache_dir: Path, url: str = CORPUS_URL_DEFAULT) -> "Corpus":
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / "treasury_bulletins_transformed.zip"
        if not zip_path.exists():
            logger.info("Downloading corpus from %s", url)
            t0 = time.monotonic()
            with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            logger.info("Downloaded %d bytes in %.1fs", zip_path.stat().st_size, time.monotonic() - t0)
        else:
            logger.info("Using cached corpus at %s", zip_path)

        logger.info("Extracting and chunking corpus…")
        t0 = time.monotonic()
        all_chunks: list[Chunk] = []
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(n for n in zf.namelist() if n.endswith(".txt"))
            for name in names:
                with zf.open(name) as f:
                    text = f.read().decode("utf-8", errors="replace")
                all_chunks.extend(_chunk_file(text, os.path.basename(name), len(all_chunks)))
        logger.info("Built %d chunks from %d files in %.1fs", len(all_chunks), len(names), time.monotonic() - t0)

        logger.info("Tokenizing + building BM25 index…")
        t0 = time.monotonic()
        tokenized = [tokenize(c.text) for c in all_chunks]
        bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index ready (%.1fs)", time.monotonic() - t0)

        return cls(all_chunks, bm25)

    def search(
        self,
        query: str,
        top_k: int = 20,
        year_filter: Iterable[int] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """BM25 search with optional year filter.

        If a year filter is given we first widen the scores across the whole
        index, then keep the top-k chunks whose filename-year is in the filter.
        Falls back to unfiltered top-k if the filter yields too few hits.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)

        idx_scored = [(i, s) for i, s in enumerate(scores)]
        idx_scored.sort(key=lambda x: x[1], reverse=True)

        results: list[tuple[Chunk, float]] = []
        if year_filter:
            year_set = set(year_filter)
            for i, s in idx_scored:
                c = self.chunks[i]
                if c.year in year_set:
                    results.append((c, s))
                if len(results) >= top_k:
                    break
            if len(results) >= max(5, top_k // 2):
                return results
            # not enough in-filter hits → fall back
        return [(self.chunks[i], s) for i, s in idx_scored[:top_k]]
