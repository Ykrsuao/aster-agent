# aster-memory

Long-term memory subsystem for [Aster](../../README.md)
agents. Installs the `aster.memory` skill, which gives an agent persistent recall
across sessions backed by vector search.

```bash
uv sync --extra memory    # from the repository root
```

The SQLite + NumPy + local hashing path supports native Windows as well as
Linux. Offline native Windows tests cover saving, cross-session recall,
restoration, deletion and persistent opt-out through real ACP processes, plus
embedding compatibility and explicit rebuilding in the host integration.
This does not verify remote embedding providers or the optional vector backends
on Windows.

For the opt-in terminal integration and its project scope, settings and commands,
see [project memory](../aster-cli/README.md#project-memory). It reuses this memory
skill and store, with automatic event/episode recording and reflection disabled.

The default vector backend is numpy-only (no extra dependencies). `sqlite-vec`
and `chromadb` backends are imported lazily if you install them yourself.

Stores now validate embedding identities and dimensions. Existing vectors with
an unknown identity require an explicit rebuild to a new database; their text
and relationships remain available for inspection. See
[embedding compatibility and rebuilding](src/aster_memory/README.md#embedding-compatibility-and-rebuilding)
for the API, migration steps and failure guarantees.

See the [main repository](../../README.md) for
documentation, and `examples/arc_agi_3` for a worked usage example.

Apache-2.0 licensed.
