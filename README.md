# purple-agent-officeqa

Purple agent for the **OfficeQA** benchmark (AgentBeats phase 2, Finance
track). Uses BM25 retrieval over the full U.S. Treasury Bulletin corpus
(697 files, 1939-2025) and **gpt-5.4-mini** for reasoning (configurable
via the `OPENAI_MODEL` env var; gpt-5-mini is a cheaper fallback).

## Architecture

On startup the agent downloads `treasury_bulletins_transformed.zip` from
`databricks/officeqa` (pinned to commit `6aa8c32`), extracts the 697
text files into memory, splits each into overlapping character chunks,
and builds an in-memory BM25 index (~70k chunks, ~20 s of indexing).

Per question the agent runs three stages:

1. **Analyze** — gpt-5-mini extracts relevant years, a BM25-friendly
   search query, expected answer unit, and expected answer type.
2. **Retrieve** — top-K chunks from BM25, filtered by year when the
   question is year-specific.
3. **Answer** — gpt-5-mini drafts a `<REASONING>` + `<FINAL_ANSWER>`
   response. If the draft admits missing information, a second
   retrieval pass + refine call produces the final answer.

Before emission, a lightweight normalizer ensures the response has both
tags, de-hedges multi-number final answers (the judge fails those
automatically), and preserves `[...]` list answers.

## Layout

```
purple-agent-officeqa/
├── src/
│   ├── __init__.py
│   ├── corpus.py     # corpus download + chunking + BM25
│   ├── agent.py      # 3-stage reasoning pipeline
│   ├── executor.py   # A2A executor
│   └── server.py     # uvicorn + A2A HTTP entry point
├── tests/
│   ├── harness.py    # local sample evaluator
│   └── score.py      # copy of the green judge's fuzzy matcher
├── amber-manifest.json5      # agentbeats component manifest
├── submission-template.json  # fill in and drop into the leaderboard fork
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Local evaluation

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...          # never commit this
python -m tests.harness --n 20 --seed 3
```

Flags:
- `--n N` — sample size (default 25)
- `--uids UID0001,UID0008` — evaluate specific question IDs
- `--difficulty {all,easy,hard}` — filter
- `--workers K` — concurrent questions (default 6)

The harness downloads the corpus + dataset into `./cache/` on first run
and reuses them afterwards.

## Build the Docker image

```bash
docker build -t ghcr.io/<YOUR_GHCR_USER>/purple-agent-officeqa:v1 .
docker push ghcr.io/<YOUR_GHCR_USER>/purple-agent-officeqa:v1
```

## Submitting to the leaderboard

See `SUBMISSION.md` for the full step-by-step.
