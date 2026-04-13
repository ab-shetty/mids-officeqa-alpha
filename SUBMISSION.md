# Submitting to the OfficeQA leaderboard

AgentBeats has a **Quick Submit** web UI at
<https://agentbeats.dev/agentbeater/officeqa/submit> that handles the
leaderboard PR + secret encryption for you. The steps below take you
from this RunPod workspace to a running evaluation.

Your target GitHub repo is already created and empty:
**`<YOUR_GH_USER>/mids-officeqa-alpha`**. Replace `<YOUR_GH_USER>` with
your actual username in the commands below.

## 0. Ship the code from RunPod to your local machine

**On the RunPod pod**, create the tarball (excluding the big corpus
cache and any pycache directories):

```bash
cd /root
tar --exclude='purple-agent-officeqa/cache' \
    --exclude='purple-agent-officeqa/**/__pycache__' \
    --exclude='purple-agent-officeqa/.*.sw[a-p]' \
    -czf /root/purple-agent-officeqa.tar.gz purple-agent-officeqa
ls -lh /root/purple-agent-officeqa.tar.gz
```

**On your local machine**, pull it down (pick whichever matches your
RunPod access):

- From the RunPod web UI: use the **File Browser → Download** button
  on `/root/purple-agent-officeqa.tar.gz`.
- Or via SSH:
  ```bash
  scp -P <RUNPOD_SSH_PORT> root@<RUNPOD_HOST>:/root/purple-agent-officeqa.tar.gz .
  ```

Extract and enter the directory:

```bash
tar -xzf purple-agent-officeqa.tar.gz
cd purple-agent-officeqa
```

## 1. Push to your empty `mids-officeqa-alpha` repo

```bash
git init -b main
git add .
git commit -m "Initial purple agent for OfficeQA benchmark"

# The repo was created empty on github.com, so no conflicts.
git remote add origin git@github.com:<YOUR_GH_USER>/mids-officeqa-alpha.git
git push -u origin main
```

(SSH URL shown; swap for the HTTPS URL if you prefer.)

The repo needs to be **public** so AgentBeats can fetch
`amber-manifest.json5` from a raw.githubusercontent URL. If you created
it private, flip it to public now:
Settings → General → Danger Zone → **Change repository visibility**.

## 2. Let GitHub Actions build and publish the image

The workflow at `.github/workflows/publish.yml` fires on every push to
`main` and pushes the image to **`ghcr.io/<YOUR_GH_USER>/mids-officeqa-alpha`**.
Your first `git push` from step 1 already triggered it — watch it at:

```
https://github.com/<YOUR_GH_USER>/mids-officeqa-alpha/actions
```

When the run is green, open the job summary — it prints the pinned
digest in a box like:

```
ghcr.io/<you>/mids-officeqa-alpha@sha256:abcd1234...
```

**Copy this line** — you need it for step 4.

### Make the GHCR package public

AgentBeats' runner pulls your image anonymously, so the GHCR package
must be public. After the first successful workflow run:

1. Go to `https://github.com/<YOUR_GH_USER>?tab=packages` and click
   **mids-officeqa-alpha**.
2. On the right side click **Package settings**.
3. Scroll to **Danger Zone** → **Change visibility** → Public.
4. Confirm with the package name.

One-time only. Re-runs of the workflow reuse the same package.

Verify the image is anonymously pullable:

```bash
docker logout ghcr.io
docker pull ghcr.io/<YOUR_GH_USER>/mids-officeqa-alpha:latest
```

## 3. Pin `amber-manifest.json5` to the digest

Open `amber-manifest.json5` and replace the `image:` line:

```diff
- image: "ghcr.io/<YOUR_GHCR_USER>/purple-agent-officeqa:latest",
+ image: "ghcr.io/<YOUR_GH_USER>/mids-officeqa-alpha@sha256:abcd1234...",
```

Commit and push:

```bash
git add amber-manifest.json5
git commit -m "Pin manifest to image digest"
git push
```

Record the resulting commit SHA — AgentBeats pulls the manifest from
that commit. (It will pull from whatever is on `main` when you
register in step 4, so as long as you don't push more changes between
now and then, the latest commit is fine.)

## 4. Register the agent on agentbeats.dev

1. Sign in to <https://agentbeats.dev>.
2. Click **Register Agent** (top right).
3. Select **Purple**.
4. Fill in:
   - **Display name**: `mids-officeqa-alpha` (or any name — this is
     what shows in the Quick Submit dropdown)
   - **Docker image**: the pinned digest,
     `ghcr.io/<YOUR_GH_USER>/mids-officeqa-alpha@sha256:…`
   - **Repository URL**:
     `https://github.com/<YOUR_GH_USER>/mids-officeqa-alpha`
   - **Track / tags**: Finance, OfficeQA, RAG
5. Save.

AgentBeats indexes your repo's `amber-manifest.json5` automatically
and issues an `agentbeats_id`. You don't need to copy it — Quick
Submit's dropdown finds the agent by name.

## 5. Quick Submit

Open <https://agentbeats.dev/agentbeater/officeqa/submit>:

- **Participant 1 → Purple agent**: search for `mids-officeqa-alpha`
  and select it.
- **Participant role name**: `officeqa_agent` (the role the green agent
  expects — leave the default).
- **Participant secrets → Add secret**:
  - name: **`openai_api_key`** (lowercase — it's a config key, not an
    env var; our `amber-manifest.json5` maps `${config.openai_api_key}`
    → env `OPENAI_API_KEY`)
  - value: your OpenAI key. Encrypted client-side in your browser;
    AgentBeats only stores a key + encrypted bundle.
- **Green agent secrets**: leave empty (the OfficeQA green doesn't
  take any).
- **Config**: paste (this matches the leaderboard's default — lower
  `num_questions` while smoke-testing if you want):

  ```json
  {
    "dataset_url": "https://raw.githubusercontent.com/databricks/officeqa/6aa8c32/officeqa.csv",
    "num_questions": 246,
    "difficulty": "all",
    "tolerance": 0,
    "max_concurrent": 10,
    "run_id": 1
  }
  ```
- Click **Submit**.

AgentBeats opens a PR against `officeqa-agentbeats-leaderboard` with
the encrypted secrets bundle attached and kicks off the workflow.
Follow the PR link it gives you.

## 6. Watch the run

The full 246-question evaluation takes 30–60 minutes of wall time.
When it finishes, the PR gets a second commit with your scores at
`submissions/<uuid>.json` plus a provenance JSON, and the leaderboard
ingests them.

## Smoke test first

Before the full 246-question run, submit once with
**`num_questions: 5`** to confirm the image boots, the corpus
downloads, and your OpenAI key decrypts correctly. That's ~$0.30 and
~5 minutes on gpt-5.4-mini. Then re-submit with `246`.

## Troubleshooting

- **"image pull failed"** — the GHCR package is still private (step 2)
  or the `image:` digest in `amber-manifest.json5` is stale.
- **"amber-manifest not found"** — the repo is private, or the
  `amber-manifest.json5` at the registered commit has a syntax error.
- **OPENAI_API_KEY errors in the agent logs** — the Quick Submit
  secret name must be **`openai_api_key`** (lowercase), matching
  `${config.openai_api_key}` in the manifest. `OPENAI_API_KEY` as the
  secret name won't substitute.
- **below-10% score** — bump `reasoning_effort` to `"high"` via the
  participant config in Quick Submit. Rebuild not needed.
- **cheaper run** — change `openai_model` in the participant config
  from `gpt-5.4-mini` (default) to `gpt-5-mini`. On our 20-question
  seed=42 sample the cheaper model scored 10% vs 15% for 5.4-mini,
  but full-benchmark variance is ±3pp, so gpt-5-mini is a reasonable
  budget-constrained choice (~$3 vs ~$9 for 246 questions).
- **GitHub Actions build fails with `denied: permission_denied`** —
  your repo's Actions don't have `packages: write`. Go to Settings →
  Actions → General → Workflow permissions → **Read and write
  permissions**.

## What's unused under the Quick Submit flow

- `submission-template.json` — only needed if AgentBeats' UI is down
  and you need the manual PR fallback documented in
  `RDI-Foundation/officeqa-agentbeats-leaderboard/README.md`. Safe to
  leave in the repo; it doesn't get read by anything automatic.
