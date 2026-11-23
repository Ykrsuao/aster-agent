# `aster_memory` — long-term memory for agents

An **opt-in, additive, brain-inspired long-term memory subsystem** for aster
Agents. Toggle it on per agent; it changes no core framework code. Its defining
idea: **the agent authors and curates its own memories** through native tools —
rather than a harness extracting them behind the agent's back.

- Runnable demo: [`examples/quickstart/12_memory.py`](../../../../examples/quickstart/12_memory.py).

---

## Table of contents
1. [Why](#why) · 2. [Quick start](#quick-start) · 3. [Architecture](#architecture) ·
4. [Memory records & schema](#memory-records--schema) · 5. [The operations](#the-operations) ·
6. [Agent-facing tools](#agent-facing-tools) · 7. [Retrieval & scoring](#retrieval--scoring) ·
8. [Spontaneous retrieval](#spontaneous-retrieval-automatic-recall) · 9. [Reflection](#reflection-offline-consolidation) ·
10. [Forgetting](#forgetting) · 11. [Storage & embeddings](#storage--embedding-backends) ·
12. [Configuration](#configuration-reference) · 13. [Monitoring](#monitoring--debugging) ·
14. [What it's good for](#when-does-memory-help) · 15. [Module map](#module-map) · 16. [Limitations](#limitations)

---

## Why

A aster agent is normally amnesic between method calls and sessions. This add-on
gives it durable memory across **long-horizon autonomous tasks** and **tasks
accreted over time**, modeled on how the brain uses memory.

| | other systems (Mem0, Zep, RAG) | **this system** |
|---|---|---|
| who creates memories | a harness pipeline extracts/stores them | **the agent** writes them via tools |
| instruction | implicit / external | the agent is **told it owns its memory** and given the schema (`MEMORY_SCHEMA_GUIDE`, injected at install) |
| refinement | system-side | the agent **`update_memory` / `forget` / `associate`** as it goes |

When `enabled=False` (or not installed), the agent is byte-for-byte unchanged — the
*additive guarantee* (regression-tested).

A disabled installation does not construct an embedder or open its database.
Explicitly accessing the manager's store or calling its operations initializes
the resources on demand; agent-facing memory tools remain disabled. Uninstalling
an unused disabled manager does not create resources either.

---

## Quick start

```python
from aster import Agent
from aster_memory import MemoryConfig, MemoryManager, MemoryToolsMixin

class MyAgent(MemoryToolsMixin, Agent, llm=my_llm):
    async def work(self, task: str) -> str:
        # Do {task}; recall with self.recall(...) and save with self.remember(...).
        ...

agent = MyAgent()
MemoryManager.install(agent, config=MemoryConfig(enabled=True))

# The agent now writes/reads its own memory; you can too:
agent.remember("Deploy with `make ship`.", type="skill", importance="HIGH")
hits = agent.recall("how do I deploy?")          # list[Memory]
report = agent.reflect()                          # offline consolidation
print(agent.memory_stats().summary())             # usage counters
```

Default config needs **no external services**: a local SQLite file + a
deterministic offline embedder. Point it at real models when you want (below).
`MemoryManager.install(...)` remains available as lower-level plumbing for hosts
that need manual lifecycle control, but normal users should enable `aster.memory`
as a skill.

---

## Architecture

`MemoryManager.install(agent, config=...)` wires three surfaces onto the agent's
**existing** machinery (mirrors `agents/summarization.py`; **no core edits**):

```
                    ┌─────────────────────── your Agent ───────────────────────┐
   conscious tools  │  remember / recall / search / update_memory / forget /    │
   (MemoryToolsMixin)│  associate         ── visible via doc(self) ──            │
                    │                                                            │
   pre-turn inject  │  BeforeTurn ─▶ recall_for_context() ─▶ ContextManager     │  "spontaneous association"
                    │                  (dynamic block, configurable cadence)     │
                    │                                                            │
   write-on-event   │  EventManager.on("Error"/"Notification"/…) ─▶ remember()  │
                    │                                                            │
   post-task reflect│  intercept("agent_call", top-level) ─▶ reflect()          │  "consolidation"
                    └────────────────────────────┬───────────────────────────-─┘
                                                  │
                       MemoryManager  ──────────  owns:
                         • MemoryStore   (SQLite: records + graph + vectors)
                         • RetrievalEngine (hybrid recall + ACT-R scoring + k-hop)
                         • ReflectionEngine (merge · reconsolidate · abstract · prune)
                         • ForgettingEngine (decay + prune)
                         • Embedder      (hashing | litellm)
                         • VectorIndex   (numpy | sqlite-vec | chroma)
```

`install()` also injects `MEMORY_SCHEMA_GUIDE` (the "you own your memory + here's the
schema" instruction) as a static context block, and `register`s runtime
memory-events. `uninstall()` removes every hook and the instruction block.

---

## Memory records & schema

A memory (`schema.Memory`) is intentionally **loose** — only `id`, `type`,
`content`, `created_at` are required; everything else has sane defaults.

- **types** (`MemoryType`): `info` (semantic fact), `skill` (verified procedure),
  `episode` (a specific experience), `intent` (future reminder/TODO),
  `reflection` (insight distilled from episodes), `scratch` (transient working memory).
- **descriptors:** `importance` (verbal: CRITICAL/HIGH/MEDIUM/LOW/TRIVIAL), `salience`, `confidence`, `mood`, `strength`
  (spaced-repetition counter), `reinforcement_count`.
- **metadata:** `created_at`, `last_accessed_at`, `access_log`, `access_count`,
  `source_task_ref`, `related_files`, `chat_turn_ref`, `valid_from/valid_to`.
- **graph:** typed directed `edges` (`EdgeType`): causal `derived_from` / `created_by`
  / `causes`, plus `refines`, `supports`, `contradicts`, `related`, `precedes`,
  `part_of`, `triggers`.

The directed graph lets retrieval do **multi-hop associative spread** and lets
reflection record provenance.

---

## The operations

| operation | what | how it fires |
|---|---|---|
| **Encode (write)** | create a memory; dedup-on-write reinforces near-duplicates | conscious `remember()` + automatic event-driven |
| **Spontaneous association** | inject relevant memories into context each turn | `BeforeTurn` → dynamic block (cadence: `self_gated`/`per_task`/`every_turn`) |
| **Deliberate recall** | the agent looks something up | `recall()` / `search()` tools |
| **Reflection** | offline consolidation after a task | top-level `intercept("agent_call")` or `manager.reflect()` |
| **Forgetting** | decay + prune | online (lazy) + offline (in reflection) |

---

## Agent-facing tools

Mixed in by `MemoryToolsMixin`, enabled/disabled per `MemoryConfig.tools`:

| tool | signature | purpose |
|---|---|---|
| `remember` | `(content, *, type="info", importance="MEDIUM", tags=None, title=None) -> id` | write a new memory (dedup-on-write) |
| `recall` | `(query, k=5) -> list[Memory]` | associative recall (similarity + graph spread) |
| `search` | `(query, k=5) -> list[Memory]` | term/keyword recall (no graph hop) |
| `update_memory` | `(id, *, content=, importance=, type=, tags=) -> bool` | refine an existing memory (re-embeds if content changed) |
| `forget` | `(id) -> bool` | archive an obsolete/wrong memory |
| `associate` | `(a_id, b_id, relation="related") -> None` | add a directed graph edge |

The same operations are available programmatically on `agent._memory`
(`MemoryManager`), which also exposes `reflect()`, `recall_for_context()`,
`memory_stats()`, `log_summary()`.

---

## Retrieval & scoring

`recall(query)` runs (see `retrieval.py`):

1. **Hybrid candidates** — dense (embedding KNN) ∪ sparse (keyword) pools.
2. **ACT-R-style score** per candidate (each term min-max normalised, then weighted):
   ```
   rel = λ·cos(query, mem) + (1−λ)·context-overlap          # encoding specificity
   rec = σ( ln Σ_k (t_now − access_k)^−d )                  # ACT-R base-level activation
   imp = importance / 10
   score = α_rel·rel̂ + α_rec·reĉ + α_imp·imp̂
   ```
3. **Multi-hop spread** (`hops ≥ 1`) — activation propagates over graph edges with
   per-hop decay, surfacing memories linked to (but not directly similar to) the query.

On retrieval, memories are "touched" (recency + strength bumped) so hot paths
self-strengthen. Weights/decays/hops are all in `RetrievalConfig`/`ScoringWeights`.

---

## Spontaneous retrieval (automatic recall)

The agent does **not** have to call `recall()` to benefit from memory. Before each
turn, a `BeforeTurn` hook **synthesises a query from the conversation** ("the anchor"),
runs the **same full `recall()` pipeline** (dense ANN ∪ keyword → ACT-R score → graph
spread — *not* ANN-only), and injects the top hits into a context block the model reads
that turn (`manager.py` → `ContextManager.set_dynamic`). As the conversation moves, the
anchor changes and different memories surface — "spontaneous association".

**How to use it.** Nothing — it's automatic once memory is installed (`enabled=True`).
The recalled memories appear in the agent's context under the block:

```
## Recalled memories (associative)
- [info] prod DB is pg-east-1
- [skill] deploy with `make ship`
```

Use explicit `self.recall(query)` only when you want to *deliberately* fetch for a
specific query; spontaneous recall covers the passive "what's relevant right now" case.

**The anchor** is built by `SpontaneousConfig.query_strategies` (composable, de-duped):

| strategy | anchor it builds |
|---|---|
| `last_message` *(default)* | the most recent user-text message |
| `recent_events` | the last `recent_events_n` events, concatenated |
| `working_state` | recent `PythonOutput`/`LLMResponse` (the agent's scratch state) |
| `distilled` | LLM-distilled query — falls back to `recent_events` with no LLM |

If no anchor can be derived (e.g. empty turn) **nothing is injected** — it never
"retrieves from nothing". Spontaneous recall uses **`touch=False`**: passively surfaced
memories are *not* reinforced (only memories the agent deliberately `recall()`s get their
recency/strength bumped), so the ACT-R signal stays meaningful.

Injection IDs, per-memory injection counters, and `MemoryInjected.n_memories`
describe records retained after the character budget is applied. A partially
rendered record counts once its complete `[type#id]` marker is visible (including
the status in a todo marker); a cut inside that marker does not count. Omitted
records are not logged as injected. This rule applies to both recalled memories
and the additional open-todo section, without changing passive ACT-R behavior.

**How to configure** (`SpontaneousConfig`):

```python
from aster_memory import MemoryConfig, MemoryManager
from aster_memory.config import SpontaneousConfig

MemoryManager.install(agent, config=MemoryConfig(
    enabled=True,
    spontaneous=SpontaneousConfig(
        enabled=True,                       # set False to disable auto-injection entirely
        query_strategies=("last_message",), # anchor source(s), tried in order
        inject_cadence="self_gated",        # self_gated | per_task | every_turn
        top_k=5,                            # memories injected (0 -> retrieval.top_k)
        context_char_budget=2000,           # hard cap on the injected block
        context_block_key="recalled_memories",
        recent_events_n=5,                  # window for recent_events / working_state
    ),
))
```

- **`inject_cadence`** — `self_gated` (default): re-inject only when the derived anchor
  changes (cheapest); `per_task`: inject once per task; `every_turn`: refresh every turn.
- Turn it **off** with `spontaneous=SpontaneousConfig(enabled=False)` and rely on the
  agent's deliberate `recall`/`search` tools instead.

---

## Reflection (offline consolidation)

`manager.reflect()` (see `reflection.py`) runs an ordered, brain-inspired pipeline:

1. **merge** near-identical memories into one canonical record;
2. **reconsolidate** — cluster related memories and let an optional LLM `reconciler`
   resolve outdated/contradicted values (**keep-latest**: archive stale, store the
   current consolidated one);
3. **edge formation** — link memories whose embeddings are close;
4. **re-score** importance (salience + access-frequency aware);
5. **abstract** — an optional LLM `reasoner` turns clusters of episodes into compact
   `reflection` memories (the generative abstraction step);
6. **prune** — forget decayed memories.

Deterministic steps (1, 3, 4, 6) run with no LLM. The generative steps (2, 5) only
run when you pass `reasoner=` / `reconciler=` to `install()`.

**How to use it.** Two ways to trigger a pass:

- **Automatic (default):** with `ReflectionPolicy(trigger="post_task")`, a top-level
  `agent_call` interception runs `reflect()` after each task (only top-level calls, not
  nested subagent calls, when `only_top_level=True`).
- **Manual:** call `manager.reflect()` (or `self.memory.reflect()` via the skill) whenever
  you want — returns a `ReflectionReport` (`merged`, `reconciled`, `superseded`,
  `edges_added`, `rescored`, `created`, `pruned`).

**How to configure** (`ReflectionPolicy`):

```python
from aster_memory import MemoryConfig, MemoryManager
from aster_memory.config import ReflectionPolicy

MemoryManager.install(
    agent,
    config=MemoryConfig(enabled=True, reflection=ReflectionPolicy(
        enabled=True,
        trigger="post_task",        # post_task (auto) | manual
        only_top_level=True,        # nested subagent calls do NOT each reflect
        background=False,           # False = await inline (deterministic); True = create_task
        merge_threshold=0.95,       # cosine >= this -> merge duplicates
        edge_threshold=0.80,        # this <= cosine < merge_threshold -> add a 'related' edge
        recon_threshold=0.6,        # cluster radius for reconsolidation
        recon_max_cluster=6,
        max_episodes_per_reflection=50,
    )),
    reasoner=my_llm_reasoner,        # OPTIONAL — enables step 5 (abstraction)
    reconciler=my_llm_reconciler,    # OPTIONAL — enables step 2 (reconsolidation)
)
```

**LLM or not.** Without `reasoner`/`reconciler`, `reflect()` makes **zero LLM calls** —
the deterministic steps (merge, edges, re-score, prune) run over the already-stored
embeddings + ACT-R math (the skill installs this way by default). Wire the two callables
to enable the generative steps. They're plain callables you build from your LLM, e.g.:

```python
def my_llm_reasoner(episodes):                  # list[Memory] -> list[Memory] (new reflections)
    text = my_llm.call([{ "role": "user",
        "content": "Distil durable insights from these episodes:\n" + format(episodes)}]).content
    return [Memory(content=line, type=MemoryType.REFLECTION) for line in parse(text)]

def my_llm_reconciler(cluster):                 # list[Memory] (old->new) -> (current|None, [archive_ids])
    # ask the LLM which value is current; return the consolidated memory + ids to retire
    ...
```

The stubs above are the contract. [`examples/quickstart/12_memory.py`](../../../../examples/quickstart/12_memory.py)
shows the deterministic (no-LLM) reflection path.

**When does reflection help?** Empirically (real gpt-5.4):

| task shape | retrieval bottleneck? | reflect effect |
|---|---|---|
| synthesis of many scattered facts | yes (small `top_k` ≪ #facts) | **+50%** |
| aggregation / latest value, memories already fit budget | no | **+0%** (neutral) |
| one pinpoint fact | no | **−20%** (abstraction blurs it) |

Consolidation pays off **only when retrieval is the bottleneck** and the model
can't reconcile in-context. It's off by default for pure retrieval-QA.

---

## Forgetting

`forgetting.py`: every memory has a retention that **decays with time** (Ebbinghaus
`R = e^(−Δt/S)`), slowed by `strength` (retrieval boosts it). The offline prune
archives memories below threshold, with guards: never prune very young memories,
high-importance memories, or `protected_types` (default: `skill`). Archive (soft
tombstone) vs hard delete is configurable.

---

## Storage & embedding backends

**One SQLite file** holds records + the association graph + vector blobs (separate
from the agent's session DB; WAL). The vector index is pluggable behind one
protocol (`vector_backends.py`):

| `vector.backend` | what | extra dep |
|---|---|---|
| `numpy` (default) | exact brute-force cosine, in-process | none |
| `sqlite_vec` | ANN inside the same SQLite file | `sqlite-vec` |
| `chroma_embedded` | Chroma over a local dir | `chromadb` |
| `chroma_http` | a running Chroma server | `chromadb` |

Indexes use cosine similarity; approximate backends can return different neighbors.
An index can be rebuilt from SQLite's vector blobs, but changing the embedding
model requires re-embedding the text. Embeddings (`embeddings.py`):

| `embedding.backend` | what |
|---|---|
| `hashing` (default) | deterministic, offline, dependency-free (great for tests) |
| `litellm` | OpenAI-compatible `/embeddings` (e.g. `text-embedding-3-large` via the NVIDIA gateway) |

```python
MemoryConfig(
    enabled=True,
    vector=VectorConfig(backend="sqlite_vec"),
    embedding=EmbeddingConfig(backend="litellm",
                              model="openai/azure/openai/text-embedding-3-large",
                              endpoint="https://inference-api.nvidia.com/v1",
                              dimensions=1024),
)
```

### Embedding compatibility and rebuilding

Each database binds a stable embedding identity and dimension. `MemoryManager`,
retrieval and reflection check this contract; vector writes and queries validate
their shape and reject NaN/Inf before modifying records. A different model is
incompatible even when its vectors have the same dimension.

The offline hashing identity versions its algorithm. LiteLLM's identity hashes
the normalized model name, endpoint and requested `dimensions`; credentials,
timeouts and retry settings are excluded. The actual dimension is recorded
separately and cannot change after a successful response. Use a pinned model
revision: a provider replacing the model behind an unchanged alias/endpoint
cannot be detected from those settings. Custom embedders must implement the
`Embedder` protocol's non-empty, stable `identity` property and change it whenever
the model or preprocessing changes.

For direct store access with an embedder, declare both parts of the contract:

```python
from aster_memory import HashingEmbedder, MemoryStore

embedder = HashingEmbedder(dim=256)
store = MemoryStore(
    "memory.sqlite",
    embedding_identity=embedder.identity,
    embedding_dim=embedder.dim,
)
```

A new empty `MemoryStore()` without an identity supports low-level, caller-supplied
vectors under a `raw-vectors:v1` contract, binding the dimension on the first
vector write. This does not establish which model produced them. Named embedders
cannot adopt raw vectors, and dimensions alone cannot identify vectors in a
legacy database. Opening either a legacy database or a named database without
declaring its identity allows text/graph inspection and raw vector reads; it does
not authorize semantic queries or new embedding writes. An incompatible named
open raises `EmbeddingMismatchError` without replacing existing data. The viewer's
hashing-only retrieval debugger reports HTTP 422 for an incompatible identity.

Explicitly rebuild when changing models/dimensions or adopting a legacy/raw store:

```python
from aster_memory import HashingEmbedder, MemoryStore

source = MemoryStore("memory.sqlite")
try:
    rebuilt = source.rebuild_embeddings(
        HashingEmbedder(dim=512), "memory-rebuilt.sqlite", batch_size=64
    )
finally:
    source.close()
# After checking the new file, set MemoryConfig.path to str(rebuilt) and use
# EmbeddingConfig(backend="hashing", dim=512).
```

Rebuilding copies a consistent snapshot, including archived memories, IDs, full
record metadata, relationships and maintenance history. It re-embeds every record
and publishes a new NumPy-backed SQLite file only after all batches succeed.
The destination must not exist; failures, interruption and a concurrent creator
never replace it or overwrite the source. Pause writers when planning a cutover:
source writes after the snapshot are retained in the source, not the new file.
`LiteLLMEmbedder` can also be passed explicitly; doing so sends those memory texts
to its configured embedding service.

Derived sqlite-vec tables use a persisted namespace per database. Chroma collection
names also include a digest of the normalized absolute SQLite path, so database
copies at different paths cannot share an index even when their persisted
namespaces match. Reopening the same path reuses its collection; moving the database
creates a new derived collection. Independent copies at the same absolute path on
different hosts still need distinct `VectorConfig.collection` values when sharing
one Chroma HTTP service. Rebuilding allocates a new namespace and leaves old indexes
intact; it does not attempt a transaction spanning SQLite and an external Chroma
service.
After validating the replacement, optional indexes can be created from its blobs
using the same embedding contract. Publication uses a hard link in the destination
directory, requiring a filesystem that supports hard links (such as NTFS/ext4).

Opening a database upgrades its schema stamp to v3 so older library versions
cannot silently bypass the contract. Legacy vector identities are never guessed
during this schema upgrade; preserve an original backup before a library downgrade.

---

## Configuration reference

`MemoryConfig` (frozen Pydantic; `.merge_with(**overrides)`) with nested sub-configs:

| group | key knobs |
|---|---|
| **top-level** | `enabled`, `path`, `tools`, `instruct`, `chunk_size/overlap` |
| **spontaneous** | `enabled`, `query_strategies` (`last_message`/`recent_events`/`working_state`), `inject_cadence`, `context_char_budget`, `top_k` |
| **retrieval** | `top_k`, `hops`, `n_dense`, `n_sparse`, `per_hop_decay`, `min_similarity`, `weights` (relevance/recency/importance/λ/decay/spread) |
| **embedding** | `backend`, `model`, `endpoint`, `api_key`, `dim`, `dimensions`, `batch_size` |
| **vector** | `backend`, `collection`, `host`, `port` |
| **write** | `on_events`, `salience_min`, `dedup_threshold`, `dedup_top_k`, `write_episodic` |
| **reflection** | `enabled`, `trigger`, `only_top_level`, `merge_threshold`, `edge_threshold`, `recon_threshold`, budgets |
| **forget** | `enabled`, `decay_half_life_hours`, `prune_activation_threshold`, `prune_min_age_hours`, `archive_vs_delete`, `protected_types` |

---

## Monitoring & debugging

Reuses the framework's existing observability (no new system):

- **logger** `aster_memory` — `setLevel(logging.DEBUG)` for per-op traces.
- **events** on the agent's `EventManager`: `MemoryWritten`, `MemoryRecalled`,
  `MemoryInjected`, `ReflectionCompleted` (RUNTIME_EVENT role → visible to any telemetry
  subscriber, never injected into the LLM context).
- **counters** `manager.memory_stats()` → `MemoryStats` (writes, reinforced, recalls,
  recalled_items, injections, reflections, merged, edges_added, pruned, store_size);
  `manager.log_summary()` logs a one-liner.

```python
import logging; logging.getLogger("aster_memory").setLevel(logging.DEBUG)
agent.event_manager.on("MemoryWritten", lambda e: print("wrote", e.memory_id, e.op))
```

---

## When does memory help?

This checkout includes a reproducible
[offline memory control experiment](../../../../experiments/memory_evidence/README.md).
It imports fixed synthetic facts through `MemoryManager`, closes the writer and
retrieves them through `MemorySkill` in fresh agents. Separate memory-off,
current, obsolete and explicitly corrected arms use the same scripted reader.
An independent scorer checks retrieval coverage, obsolete-fact adoption, answer
correctness and completion of the entire evidence-backed task. Reports retain
source/fixture hashes and distinguish unknown usage from reported zero.

These controls verify storage, retrieval and evaluation wiring with a
`FakeLLMClient` and local hashing. They do not establish the benefit or reliability
of memory for a real model, remote embedding provider or representative workload.
An explicit update in the correction arm does not test reflection or automatic
freshness judgment. See the experiment README for the actual offline results and
their saved evidence. For a smaller API demo, see
[`examples/quickstart/12_memory.py`](../../../../examples/quickstart/12_memory.py).

Earlier versions of this section quoted internal GPT-5.4 and
text-embedding-3-large measurements named `recall_qa.py`, `locomo.py`,
`reflecting.py`, `longmemeval.py` and `memory_effect.py`. Their benchmark harness,
raw outcomes and complete run configuration were not published in this
repository. Those historical reports cannot be independently reproduced from
this checkout and are not current acceptance evidence. The new synthetic report
does not recreate or validate those missing measurements.

Whether memory improves a real task, whether stale facts are adopted and whether
reflection preserves useful detail require controlled comparisons on the intended
workload, with independent grading and comparable model, prompt and budget settings.

---

## Module map

| file | contents |
|---|---|
| `schema.py` | `Memory`, `MemoryType`, `Edge`, `EdgeType` |
| `config.py` | `MemoryConfig` + sub-configs (frozen Pydantic) |
| `embeddings.py` | `Embedder` protocol, `HashingEmbedder`, `LiteLLMEmbedder`, `get_embedder` |
| `store.py` | `MemoryStore` (SQLite: records + graph + vectors) |
| `vector_backends.py` | `VectorIndex` protocol + numpy / sqlite-vec / chroma impls + `make_vector_index` |
| `retrieval.py` | `RetrievalEngine` (hybrid recall, ACT-R scoring, k-hop), query strategies |
| `forgetting.py` | `ForgettingEngine`, `retention()` |
| `reflection.py` | `ReflectionEngine` (merge · reconsolidate · edges · re-score · abstract · prune), `ReflectionReport` |
| `monitoring.py` | `MemoryStats` + runtime memory events |
| `manager.py` | `MemoryManager` (install/hooks/ops) + `MemoryToolsMixin` + `MEMORY_SCHEMA_GUIDE` |

Tests: [`tests/memory/`](../../tests/memory/).

---

## Limitations

- **Agentic authoring can require LLM calls.** The deterministic offline path
  (hashing embedder, no reasoner/reconciler) requires no remote model calls.
- **Reconsolidation** only fires when contradictory memories cluster *and* the agent
  authored both values; a missing fact can't be reconciled.
- **Reflection changes stored evidence.** Its usefulness and potential loss of
  detail need workload-specific evaluation; the offline correction control does
  not measure reflection.
- `numpy`/`sqlite-vec` backends are exact/brute-force — fine to ~10⁵ vectors; beyond
  that prefer a Chroma backend.

See [`examples/quickstart/12_memory.py`](../../../../examples/quickstart/12_memory.py)
for a runnable demo.
