# Offline memory evidence controls

## Research question

Can facts written through the real memory manager be recovered by a fresh agent,
and can an independent scorer distinguish missing evidence, current evidence,
obsolete evidence and an explicit correction? Can it distinguish a correct short
answer from a complete, evidence-backed dispatch record?

This experiment verifies storage, retrieval, model/tool wiring and scoring. It
does **not** measure the benefit, reliability or freshness reasoning of a real
language model.

## Experiment design

`fixtures.json` contains six synthetic cases with two facts each. Every fact has
one current value and a different obsolete value. The subject and requested keys
are visible to the responding agent; the values and grading rules are not passed
as arguments or imported into its module.

Each repeat creates four separate temporary SQLite databases:

| Arm | Stored facts | Reader behavior |
| --- | --- | --- |
| `off` | Current facts | Memory skill is unavailable |
| `current` | Current facts | Retrieve through `MemorySkill.recall()` |
| `stale` | Obsolete facts | Retrieve through the same skill and policy |
| `corrected` | Initially obsolete | A separate writer explicitly updates the facts, then closes; a new reader retrieves them |

The importer uses `MemoryManager.remember()` with deduplication disabled so all
fixture records exist. It closes its manager before readers are created. Every
case uses a new `EvidenceReader` and a new memory connection. The correction arm
also uses a separate manager and commits through `MemoryManager.update()`.
`MemorySkill`, the SQLite store, NumPy search and the 256-dimensional hashing
embedder are the production implementations. Automatic event/episode recording,
spontaneous injection and reflection are disabled to keep retrieval explicit.

The same `FakeLLMClient` script runs in all arms. It issues real lookup calls,
copies values from matching retrieved records and cites their actual memory IDs.
It contains no fixture values or expected answers. `scoring.py` independently
checks the returned answer, the entire dispatch record and its citations against
the fixed answer key and observed tool results. A regression test changes only a
stored value: the response changes with it and the unchanged scorer rejects it.

The dispatch is a structured return value; this harness executes no deployment or
external operation. The scripted reader deliberately has no freshness judgment.
The correction arm measures an explicit update and subsequent retrieval, not
automatic contradiction resolution, reflection or model learning.

## Key metrics

Metrics stay separate for all four arms and for every case/repeat:

| Metric | Definition |
| --- | --- |
| Retrieval coverage | Required subject/key facts present in actual tool results, regardless of whether their value is current |
| Current-fact coverage | Required facts retrieved with their fixed current values |
| Obsolete-fact adoption | A returned answer or dispatch field equals its known obsolete value |
| Answer correctness | The primary answer exactly equals the fixed current answer |
| Task completion / `verification_passed` | The primary answer and every dispatch field are correct, with citations to corresponding facts actually retrieved |
| `agent_completed` | The responding agent's completion claim; it cannot override independent verification |
| Calls | Observed `FakeLLMClient` attempts; external provider calls are separately recorded as zero |
| Usage and cost | Reported quantities, known subtotals and completeness flags; missing data remains `null` |

An explicit abstention is an observed incorrect answer. A failed invocation has no
observed answer, so answer correctness and obsolete adoption remain unknown; task
completion is false because no completed artifact was returned. Incomplete
retrieval observations also stay unknown. Aggregate rates report observed and
unknown counts and never treat an unknown as a successful or zero-valued result.

The default fake responses contain no provider usage. Their token and cost totals
therefore remain unknown even though model-interface calls can be counted.
The limit record includes repeats, cases, retrieval size, task/cell deadlines,
CodeAct iteration limits and the prohibition on external provider calls.

`report_complete` means every planned case/arm/repeat has an observation. It does
not mean every task succeeded, every usage field is known, or a real model was
evaluated. `recovery_scope` is `new-agent-existing-memory-store`: these are fresh
agents reopening a durable store, not conversation-snapshot or process recovery
claims. The separate terminal lifecycle tests cover ACP process restoration.

## How to run

From the repository root, after syncing the development dependencies with `uv`:

```bash
uv run --offline --no-sync python -m experiments.memory_evidence.run --output .aster/evidence/memory-offline.json --repeats 2
uv run --offline --no-sync python -m pytest packages/aster-memory/tests/memory/test_memory_evidence_harness.py -q
```

Choose a new output filename for each run; existing evidence is not overwritten.
`--top-k` defaults to 3 per query and `--timeout` to 30 seconds per task. The
runner accepts only an in-process `FakeLLMClient` and hashing embeddings. It
neither loads saved model aliases nor invokes remote models. A live comparison
requires a separately authorized evaluation with resolved model identities,
comparable limits and independently captured outcomes.

## Evidence and provenance

The JSON report includes the fixture hash, runner/policy/scorer and shared-reporting
source hashes, candidate commit/dirty/diff metadata, resolved fake-client identity
and its fingerprint, platform, limits, per-arm results and per-case observations. It
reuses the terminal acceptance reporter's identity and usage rules. Only case
identifiers, verdicts, token/cost metadata and hashed answer/record values are
retained; prompts, raw note content and response transcripts are omitted.
Temporary databases are closed and removed after the run.

The committed summary is synthetic evidence from this harness. A dirty candidate
records that the run occurred before the combined change was committed; exact
source hashes identify the files used. It is not clean-release acceptance or a
reconstruction of unpublished historical results.

## Results summary

Measured on **2026-11-23 at 00:41 +08:00** on native Windows with Python 3.12.13.
The saved report is
[`results/offline-windows-20261123.json`](results/offline-windows-20261123.json).
It records the UTC timestamp, dirty candidate and exact source hashes.

```bash
uv run --offline --no-sync python -m experiments.memory_evidence.run --output experiments/memory_evidence/results/offline-windows-20261123.json --repeats 2 --timeout 60
```

Six cases ran twice in each arm: **12 tasks and 24 required facts per arm,
48 task observations in total**. All 48 returned a response without an invocation
error. The report is complete, and every scoring metric has zero unknown cases.

| Arm | Retrieved facts | Current facts | Correct answers | Independently completed tasks | Obsolete-fact adoption |
| --- | --- | --- | --- | --- | --- |
| `off` | 0/24 | 0/24 | 0/12 | 0/12 | 0/12 |
| `current` | 24/24 | 24/24 | 12/12 | 12/12 | 0/12 |
| `stale` | 24/24 | 0/24 | 0/12 | 0/12 | 12/12 |
| `corrected` | 24/24 | 24/24 | 12/12 | 12/12 | 0/12 |

Each arm made 12 completed `FakeLLMClient` calls, for 48 model-interface calls
and **zero external provider calls**. Token totals and cost remain `null`, with
all token/cost completeness flags false; no provider usage was reported. The
correction arm made 24 explicit fact updates across the two repeats. No manual
intervention occurred.

The final harness and README checks passed: **10 tests** (eight harness checks
and two README checks), plus Ruff lint and format checks. The saved report's
fixture, source, policy and model fingerprints were verified against the files
used, and its text was checked for omitted fixture questions and raw fact values.

These are repeated deterministic controls, not estimates of real model accuracy.
The obsolete arm shows that retrieval coverage alone cannot establish correctness;
the correction arm shows that an explicit durable update changes what a fresh
reader observes. This artifact contains native Windows measurements only and
makes no real model, automatic reflection or process-recovery claim.
