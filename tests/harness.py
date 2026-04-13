"""Local sample evaluation harness.

Usage:
    python -m tests.harness --n 25
    python -m tests.harness --n 10 --uids UID0001,UID0042
    python -m tests.harness --n 25 --workers 8

Runs the OfficeQAAgent in-process against a sample of the officeqa.csv
dataset and scores with a copy of the green judge's fuzzy matcher.

Never prints the OPENAI_API_KEY.  Reads it from the environment directly
via the `openai` client.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import logging
import os
import random
import sys
import time
from pathlib import Path

import httpx

# Make the src package importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import OfficeQAAgent  # noqa: E402
from src.corpus import Corpus         # noqa: E402
from tests.score import score_answer  # noqa: E402

logger = logging.getLogger("harness")

CSV_URL = "https://raw.githubusercontent.com/databricks/officeqa/6aa8c32/officeqa.csv"


def load_questions(cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "officeqa.csv"
    if not path.exists():
        logger.info("Downloading officeqa.csv")
        path.write_bytes(httpx.get(CSV_URL, timeout=60, follow_redirects=True).raise_for_status().content)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=25, help="Sample size")
    p.add_argument("--uids", type=str, default="", help="Comma-separated UIDs; overrides --n")
    p.add_argument("--difficulty", choices=["all", "easy", "hard"], default="all")
    p.add_argument("--workers", type=int, default=6, help="Concurrent questions")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cache-dir", type=str, default=str(ROOT / "cache"))
    p.add_argument("--tolerance", type=float, default=0.0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Dampen OpenAI SDK chatter
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in env.", file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir)
    logger.info("Loading corpus (this is ~60s first time, instant on reuse)…")
    corpus = Corpus.load(cache_dir)
    agent = OfficeQAAgent(corpus=corpus)

    all_q = load_questions(cache_dir)
    if args.difficulty != "all":
        all_q = [q for q in all_q if q.get("difficulty") == args.difficulty]

    if args.uids:
        want = set(u.strip() for u in args.uids.split(",") if u.strip())
        sample = [q for q in all_q if q["uid"] in want]
    else:
        rng = random.Random(args.seed)
        sample = rng.sample(all_q, min(args.n, len(all_q)))

    logger.info("Evaluating %d questions with %d workers (tolerance=%g)",
                len(sample), args.workers, args.tolerance)

    def _run_one(q: dict) -> dict:
        t0 = time.monotonic()
        try:
            res = agent.answer_question(q["question"])
            raw = res.raw_response
            final = res.final_answer
            err = ""
        except Exception as e:
            logger.exception("Question %s failed: %s", q["uid"], e)
            raw = f"<REASONING>error</REASONING>\n<FINAL_ANSWER>0</FINAL_ANSWER>"
            final = "0"
            err = str(e)
        ok, rationale = score_answer(q["answer"], raw, args.tolerance)
        return {
            "uid": q["uid"],
            "difficulty": q.get("difficulty"),
            "gt": q["answer"],
            "pred": final,
            "ok": ok,
            "rationale": rationale,
            "elapsed": time.monotonic() - t0,
            "error": err,
            "question": q["question"],
        }

    t_start = time.monotonic()
    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(_run_one, sample):
            results.append(r)
            mark = "✓" if r["ok"] else "✗"
            gt = r["gt"][:60].replace("\n", " ")
            pred = r["pred"][:60].replace("\n", " ")
            print(f"  {mark} {r['uid']:<8} [{r['difficulty']:>4}] {r['elapsed']:5.1f}s  gt={gt!r}  pred={pred!r}  ({r['rationale'][:60]})")

    wall = time.monotonic() - t_start
    correct = sum(1 for r in results if r["ok"])
    easy = [r for r in results if r["difficulty"] == "easy"]
    hard = [r for r in results if r["difficulty"] == "hard"]
    print()
    print(f"Total      : {correct}/{len(results)}  = {100 * correct / max(1, len(results)):.1f}%")
    if easy:
        e_ok = sum(1 for r in easy if r["ok"])
        print(f"Easy       : {e_ok}/{len(easy)}  = {100 * e_ok / len(easy):.1f}%")
    if hard:
        h_ok = sum(1 for r in hard if r["ok"])
        print(f"Hard       : {h_ok}/{len(hard)}  = {100 * h_ok / len(hard):.1f}%")
    print(f"Wall clock : {wall:.1f}s   (avg {wall / max(1, len(results)):.1f}s/q)")

    return 0 if correct > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
