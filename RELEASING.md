# Releasing Aster

`aster`, `aster-cli`, `aster-acp`, `aster-memory`, and `aster-bench` are built from
one commit. Versions come from Git tags: `vX.Y.Z` produces `X.Y.Z`; commits between
tags produce development versions.

## Build and check locally

Start in the repository root with Python 3.12–3.13, uv, Rust 1.88+, and the native
compiler toolchain. Source installation uses the local workspace packages.

```powershell
uv sync --group dev --extra tui
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_license_headers.py
uv run pytest
cargo test --locked --manifest-path packages/aster-tui/Cargo.toml
cargo build --locked --release --manifest-path packages/aster-tui/Cargo.toml
uv build --no-sources --package aster --out-dir dist
uv build --no-sources --package aster-cli --out-dir dist
uv build --no-sources --package aster-acp --out-dir dist
uv build --no-sources --package aster-memory --out-dir dist
uv build --no-sources --package aster-bench --out-dir dist
```

Rebuild the viewer with `npm ci --ignore-scripts` and `npm run build` in
`src/aster/viewer/frontend-react/` when its sources change; include the resulting
`dist/`. Keep `LICENSE`, SPDX headers, and `THIRD_PARTY_NOTICES.md` in distributions.

Smoke-test all five built wheels together in a clean environment. Check imports,
`aster --help`, `aster --version`, and `aster-acp --help` from outside the checkout.
Ordinary Python wheels do not bundle the native terminal; build the Windows
terminal bundle below to distribute it with the matching Python packages.
Retain the commit, version, test results, and artifact checksums with release evidence.

## Windows terminal bundle

On native Windows x64, build from the candidate checkout:

```powershell
uv run --no-sync python scripts/build_terminal_release.py --out-dir dist/terminal-windows
```

The output directory must be new or empty. The command builds the Rust release
executable and five matching wheels, then writes a ZIP, `MANIFEST.json` and
`SHA256SUMS`. Only the CLI wheel is platform-specific (`py3-none-win_amd64`);
its native executable is at `aster_cli/bin/aster.exe`. Use `--skip-cargo-build`
only when the release executable was just built from this candidate. Default
wheel and source-distribution builds remain pure Python.

The extracted `install.ps1` validates the bundle and installs the isolated
`aster-cli` tool with the matching local wheels and locked dependency versions.
Run a trusted bundle with
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1`; this
sets policy only for that process, avoiding a permanent policy change.
It requires uv and can obtain Python 3.12 and third-party packages from the
configured index/cache. These dependencies are not inside the ZIP. Close Aster
before reinstalling. Use `-ToolsDirectory` and `-BinDirectory` for an isolated
test installation; `-Python` selects an existing supported interpreter and
`-Offline` forbids downloads. Workspace and user configuration are not migrated
or deleted by the installer. Checksums detect corruption, not publisher identity.

The Windows CI job builds and installs this bundle outside the checkout, runs
`scripts/terminal_release_smoke.py seed`, reinstalls, then runs `verify` with the
installed environment's `python -I`. It checks installed-package origins, exact
bundled binary selection, a real ACP conversation with scripted model responses,
replay without repeating file edits, continuation and preserved input/configuration
bytes. This proves same-version reinstall behavior; an upgrade from a different
released version needs its own seed artifact. CI uploads the ZIP and results;
it does not publish a release.

## Frozen coding-task baseline

The ten tasks in `tests/fixtures/terminal_acceptance/tasks.json` cover bug fixes,
parsing, a small feature, a refactor, input validation and CLI/state behavior.
They use small isolated repositories and the real `CodingAgent`; this is a
repeatable baseline, not a substitute for full terminal or real-repository use.

```powershell
uv run --no-sync python scripts/terminal_acceptance.py --self-check
uv run --no-sync python scripts/terminal_acceptance.py --model YOUR_SAVED_ALIAS --output results/terminal-candidate
```

The first command makes no provider requests and checks that every broken seed
fails while its reference solution passes. The second currently supports native
Windows only and sends real model requests; defaults cap each task at 180 seconds,
eight model calls and 4,096 output tokens per call without modifying saved model
settings. A Windows Job stops the worker and its children at the deadline, including
blocking code. Live runs on other platforms are refused until their process-tree
cleanup is supported. Generated code runs natively in disposable workspaces, which
are not an OS security boundary. Use a dedicated test environment for untrusted models.

The runner preserves failures and checks outputs with separate verifiers after
execution. JSON and Markdown reports retain task identity, candidate information,
elapsed time and observed usage; unknown cost is not reported as zero. All ten
tasks must pass for this baseline's exit status to be successful. Keep first-run
results when rerunning after a fix. Real model runs remain manual; offline fixture
and runner tests run with the ordinary test suite.

The [2026-11-20 record](packages/aster-tui/docs/stability.md) retains an eight-call
baseline (6/10), a targeted twelve-call comparison (4/4) and a complete twelve-call
candidate run (10/10). To reproduce the last run's limits, add `--max-calls 12`;
do not compare or merge results with different budgets as one run. This is one
model and one run per condition, not a measured long-term reliability rate.

## Terminal candidate acceptance (manual)

For the next terminal candidate, complete the native Windows user journey from
installation through editing, verification, interruption, recovery and upgrade.
Linux remains supported and must still pass its existing checks. The candidate
gate below requires installation, upgrade, restored-task continuation and frozen
task evidence. The other user-journey checks in this checklist remain manual;
passing the focused regressions below does not complete those checks.

After the development environment is installed, get focused feedback with:

```powershell
uv run --no-sync python -m pytest tests/test_native_windows.py packages/aster-cli/tests/test_tui_command.py
cargo test --locked --manifest-path packages/aster-tui/Cargo.toml --test cli --test first_send --test startup_recovery --test recovery_and_rollback
cargo test --locked --manifest-path packages/aster-tui/Cargo.toml --lib cancel
cargo test --locked --manifest-path packages/aster-tui/Cargo.toml --bin aster draft
```

These tests use local fixtures and scripted model responses. They cover launcher
behavior, first-send ordering, cancellation, saved input and recovery conflicts;
they do not prove fresh installation, real-model task completion or VM isolation.
Keep the full checks above for a release candidate.

Record the following evidence against one candidate commit and its artifacts:

- [ ] **Installation:** a user can install and start the complete terminal outside
  the checkout without compiling Rust or copying a launcher from a developer's
  environment. Retain the exact instructions, package checksums and clean-environment
  results. The bundle and CI smoke above provide a repeatable check; retain a
  successful run for the exact candidate rather than assuming the job passed.
- [ ] **Real tasks:** freeze ten representative coding tasks before the run, with
  repository revisions, exact prompts, checks of the resulting code and per-task
  time/cost budgets. Include a bug fix, a small feature, a refactor, a failing test
  investigation and a task spanning several files. Keep failed tasks in the set.
  Record model/configuration, success or failure, elapsed time, total cost and
  manual interventions for every attempt. Include delegated work in cost; mark
  unavailable usage as unknown. Set the required success threshold before running.
- [ ] **Interruption:** exercise cancellation during tool execution, a model
  timeout and a backend disconnect. Record whether the original task can finish
  after recovery, not just whether the conversation reopens. Confirm that pending
  inputs stay paused and that old file edits are not silently executed again.
- [ ] **Saved work:** exercise exit and restart with unsent text, attachments and
  queued inputs, plus a storage write failure and two terminals editing the same
  saved draft. Check that errors remain visible and newer work is not overwritten.
  Data loss, an unintended resend or an approval bypass blocks acceptance.
- [ ] **Upgrade:** install the candidate over the previous distributed version,
  reopen a saved conversation and draft, and complete another turn. Keep a backup
  of the test workspace and record the old and new artifact versions.
- [ ] **Support description:** the installation guide and current support limits
  agree with observed behavior. Keep dated historical results separate from the
  candidate's results; leave anything not exercised explicitly pending.

The optional `make_release.py` checks described below remain separate evidence.
The publication workflow does not invoke that script. In-process cancellation is
cooperative and cannot promise to interrupt arbitrary blocking Python code.
Any claim about sandbox containment needs the separate real-microVM checks.

## Candidate evidence required for publication

Publication consumes the exact accepted wheel, source-distribution and Windows
bundle bytes; it does not rebuild from a tag. `scripts/check_release_evidence.py`
checks their hashes, five-package versions, clean source SHA, installed native
executable, restored conversation and subsequent turn before making files
available to the existing publishing jobs. A same-version reinstall is useful
ordinary CI evidence but cannot satisfy the upgrade requirement.

The manual real-model producer is
`.github/workflows/terminal-acceptance.yml`; its setup and dispatch are described
below. Strict candidate acceptance still needs a successful live report for the
exact candidate and a previously distributed complete Windows bundle. A manually
uploaded JSON, an offline scripted run, a historical dirty build or `passed: true`
cannot replace those results. Adding the workflow does not establish a
distribution channel or prove a real-version upgrade or real-model candidate.

Ordinary push and pull-request CI still runs without release inputs. Once the
prerequisites are available, a maintainer can use the following sequence:

1. Freeze the final `vX.Y.Z` tag on the chosen clean commit already reachable from
   the repository's default branch. Run the candidate workflow on that tag so
   wheels and the native bundle carry the final version before acceptance. Tag
   creation does not trigger publication.
2. Obtain a `terminal-task-acceptance` artifact from a successful manual run of
   the trusted task workflow on that exact SHA. Its `Terminal task acceptance`
   job must have succeeded. Select the non-secret resolved model configuration
   fingerprint before requesting candidate acceptance.
3. Manually dispatch `ci.yml` on the final tag with `release_candidate=true`,
   `release_tag`, `upgrade_from_tag`, `task_report_artifact_id` and
   `model_fingerprint`. `upgrade_from_tag` must identify an older, published
   final release in this repository with its complete Windows ZIP and GitHub
   asset digest. Missing history blocks this explicit candidate request, not
   normal CI. The job installs the old bundle outside the checkout, saves a
   task and draft, installs the candidate over it, reopens without rerunning the
   old edit, then completes a new scripted turn and checks preserved bytes.
4. After all required Linux, Windows, Python, Rust, build and security checks
   succeed, CI seals one artifact containing the tested distributions and
   evidence. Its name includes tag, full SHA and run attempt. The job summary
   prints `Candidate artifact: ID`; retain that exact artifact ID. Candidate
   artifacts are retained for 90 days, and required upstream task artifacts
   must also remain available through publication.
5. Rehearse via the existing `publish.yml` manual dispatch, supplying
   `release_tag` and `candidate_artifact_id`. It still targets TestPyPI only.
   For production, put the single unmodified `Candidate artifact: ID` line in
   the GitHub release notes and publish that release using the existing
   environment approval and Trusted Publishing setup. The production upload
   remains reachable only through `release: published`.

For each source artifact, the checker queries the GitHub API for repository and
head-repository identity, workflow ID and path, full SHA, successful dispatch
run, current attempt, required successful jobs and artifact digest. It downloads
only through that API, verifies the downloaded archive and its file inventory,
and resolves the final tag again before materializing upload files. Fork runs,
other workflows, skipped jobs, expired artifacts and stale attempts are rejected.
The publisher executes the current default-branch controller; source hashes for
the task runner and fixtures are read at the frozen candidate SHA.

The task producer contract extends `terminal_acceptance.py`'s version-1 report:

- `report.json` retains the clean `candidate.commit`, `fixture_sha256`,
  `runner_sha256`, all ten unique task IDs, exact `limits`, and per-task outcomes.
  The current policy fixes 180 seconds, eight calls, 4,096 output tokens per call
  and zero endpoint retries. Changing that policy requires a reviewed source
  change before the run; twelve-call historical evidence is not interchangeable.
- `measurement_kind` must be `live-model`, `report_complete` must be true and
  `repeat_id` must be a positive integer. Every task must have `agent_completed`,
  `verification_passed` and `passed` true, with `status: completed`.
- The top-level and every task's `model_identity` and SHA256 `model_fingerprint`
  must agree with the actual resolved worker configuration and the requested
  fingerprint. The canonical fingerprint encoding matches
  `terminal_acceptance.fingerprint_model`: sorted JSON keys, compact separators,
  ASCII escaping and no NaN values. The producer must enforce its non-secret
  identity whitelist; an alias alone is insufficient.
- Integer `source_run_id` and `source_run_attempt` must match the API-authenticated
  source. The checker verifies runner and fixture hashes against that candidate's
  source files, not against a report's own claims.

This gate accepts one explicitly identified complete task run. It does not
aggregate repeats or turn one successful run into a reliability claim. Keep
earlier failures and use the separate evaluation reports for comparisons.

The checker performs no model requests, builds, uploads or release creation.
Offline regression tests simulate GitHub responses and exercise rejected
provenance, tampered files, incomplete tasks and insufficient upgrade evidence.
Passing those tests validates the gate, not a distributable candidate.

## Run the trusted task producer

The workflow accepts only `workflow_dispatch`, with `confirm_live` defaulting to
false. Configure the GitHub environment `terminal-acceptance` with required
reviewers and deployment branch/tag rules appropriate to the release policy.
Declaring the environment in YAML does not configure its protections. Keep
model credentials in the environment secret `TERMINAL_ACCEPTANCE_API_KEY` and
one non-secret model entry in its variable `TERMINAL_ACCEPTANCE_MODEL_CONFIG`:

```json
{
  "model_name": "openai/your-exact-model-id",
  "client_type": "completion",
  "api_base": "https://your-approved-gateway.example/v1"
}
```

The entry accepts only `model_name`, `client_type`, `api_base`, and optional
`temperature`, `top_p`, `reasoning_effort`, `context_window`. Completion routes
use an explicit `openai/` or `anthropic/` prefix; Responses routes use `openai/`
and `client_type: responses`. The endpoint must use HTTPS without embedded
credentials, query or fragment. Unknown fields are rejected. The producer fixes
the key environment reference, reply cap, disabled endpoint retries and text-only
media policy. Responses also fixes `store: false` and encrypted reasoning state
replay. Other provider or transport settings require a reviewed producer change.

On the same candidate, supply the identical non-secret JSON through
`TERMINAL_ACCEPTANCE_MODEL_CONFIG` and resolve its fingerprint offline:

```powershell
uv run --no-sync python scripts/terminal_acceptance_ci.py --describe-model
```

This command requires no API key and sends no inference requests. It prints only
the resolved model identity and fingerprint. Review these settings before a live
dispatch; this is not a connection or model-quality test.

Select a branch or final tag whose commit is already reachable from the default
branch, then dispatch `terminal-acceptance.yml` on that ref with its complete
`candidate_sha`, the resolved `model_fingerprint` and `confirm_live=true`. The
workflow checks out `github.sha`; the driver requires that SHA, the dispatch input
and the clean checkout to agree. It creates a temporary saved alias outside the
checkout and isolates user/project registry, settings and secrets discovery.
The worker checks the approved fingerprint again before every shared-client call.

All ten tasks run sequentially, once each. Runs share one workflow concurrency
group, with cancellation disabled; a pending GitHub run may still be replaced by
a newer pending run. Within a task, async calls through the measured client queue
one at a time; overlapping synchronous calls fail before dispatch. Cancelling an
async caller retains its slot until the provider call finishes, or the task's
Windows process-tree deadline terminates it. This keeps
this producer's measured calls below a two-request gateway limit, but does not
coordinate other applications using that gateway. Native generated code remains
trusted test-environment code, not an OS sandbox or a network-wide rate limiter.

The fixed per-task limits are 180 seconds, eight model calls and 4,096 output
tokens per call with zero endpoint retries: at most 80 admitted calls and 327,680
requested output tokens across ten tasks. These are request limits, not a dollar
spending cap; provider-side limits and explicit approval remain necessary when
cost or unreported usage needs a hard bound. Each fixture runs once; any
strategy-level retry still consumes one of its eight admitted model calls.

Only `report.json` is uploaded as `terminal-task-acceptance`, retained for 30 days.
Workspaces, prompts, responses, registry files and credentials are not uploaded.
Failed runs retain an available report for diagnosis, but cannot satisfy release
acceptance. Success requires all ten independently verified tasks, live model
identity, unchanged source and the same report checks used by the release consumer.
Preserve the successful artifact ID through publication, and retain each failed
attempt instead of replacing it with a later result.

Implementation and offline contract tests do not create the GitHub environment,
set its variables/secrets, approve a run or generate live-model evidence.

## Configure a release channel

This checkout does not designate an Aster hosting organization or package-index
owner. Before publication, configure the intended repository, package ownership,
and credentials for that release channel.

The optional `scripts/make_release.py` gate requires
`ASTER_RELEASE_REPOSITORY=owner/repo` before it can operate on GitHub. Supply your
actual Aster repository; the script validates the value. It can create a draft
release after its checks, and cannot publish that draft itself.

The gate includes live model checks and fresh-wheel quickstarts. The quickstarts
still run with `--skip-capability`, `--checks-only`, or `--dry-run`. Configure the
selected model aliases and provider credentials before using it. `--ci` requires
an explicit candidate SHA and a compatible model-alias wheel supplied through
`--internal-wheel`; use `--help` for the full gate controls.

The runner retains private reports, logs, traces, distributions, checksums, and a
manifest. Review sanitized notes and the exact tested commit before approving a
draft. A failed gate creates no new draft; a failure after draft creation may need
reconciliation before retrying.

## Publish and rehearse

`.github/workflows/publish.yml` verifies and downloads the accepted candidate.
Manual workflow runs
target TestPyPI; a published GitHub release targets PyPI. Configure Trusted
Publishing for your actual repository and `publish.yml`, with a distinct
`testpypi-<package>` or `pypi-<package>` environment for each distribution. Configure
environment reviewers to match the project's release policy.

Rehearse on TestPyPI and inspect artifact versions and contents before creating
a production release. A `release: published` event enables the production upload
jobs, so use manual dispatch for publication rehearsals.
