# Frozen terminal coding baseline

This is a small, deterministic coding-workspace baseline for the existing
`CodingAgent`. The ten tasks cover boundary bugs, Unicode, CSV, timezones,
cross-file changes, a shared-helper refactor, failing-test diagnosis, path input
validation, CLI exit behavior and configuration preservation.

Run without a provider:

```powershell
uv run --no-sync python scripts/terminal_acceptance.py --list
uv run --no-sync python scripts/terminal_acceptance.py --self-check
uv run --no-sync pytest tests/test_terminal_acceptance.py
```

Run a saved model alias explicitly on native Windows (this sends real,
potentially billable requests):

```powershell
uv run --no-sync python scripts/terminal_acceptance.py --model ALIAS --output artifacts/terminal-candidate
```

`--task ID` selects a frozen task for diagnosis. The defaults are 180 seconds per
task, eight model calls across the shared client, 4,096 output tokens per call,
and no endpoint retries. These settings do not modify saved model configuration.
Use a fresh output directory for each attempt; failures and their workspaces are
retained. Run the complete ten-task set for comparisons and keep the fixture,
runner and candidate hashes with the result. Do not replace failed tasks after a run.

Live runs currently require native Windows: the runner invokes the real coding
agent in a separate process and uses a Windows Job Object to terminate its process
tree at the deadline. `--list` and `--self-check` also work on other platforms;
POSIX process groups alone cannot clean up shell children that create new sessions. Only seed files and the prompt enter the workspace;
reference solutions and final checks are not copied there. After the agent has
stopped, the parent supplies independent checks over stdin. Completion messages
alone never pass a task; protected public tests must stay unchanged. Self-check
proves that every original seed fails and each reference solution passes.

Reports include the overall success verdict and separate `verification_passed` /
`verification_reason` fields, elapsed time, attempted/completed calls, provider
token usage and cost, and manual intervention count (zero for unattended runs).
Correct files do not count as an overall pass if the agent failed to complete
within the declared budget. Missing usage or an unfinished billed request makes totals unknown; a known
subtotal is retained separately. Raw provider responses and exception messages
are not exported. Generated files remain in each task workspace for inspection.

This measures small programming tasks through `CodingAgent`, not terminal input,
session recovery, installation, upgrade compatibility or real-world project
success rates. Native execution is not an OS sandbox: subprocess deadlines and
keeping verifier files outside the workspace do not contain hostile generated
Python. Use a disposable OS-isolated environment for untrusted models or inputs.
