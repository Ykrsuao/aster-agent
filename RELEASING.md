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
Linux remains supported and must still pass its existing checks. This checklist
is a manual acceptance procedure; it is not an automated CI or publication gate.
Passing the focused regressions below does not complete the acceptance process.

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

The optional `make_release.py` checks described below remain useful release
evidence. `publish.yml` currently does not invoke them or enforce this checklist;
review the recorded candidate evidence before publishing. In-process cancellation
is cooperative and cannot promise to interrupt arbitrary blocking Python code.
Any claim about sandbox containment needs the separate real-microVM checks.

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

`.github/workflows/publish.yml` builds all five packages. Manual workflow runs
target TestPyPI; a published GitHub release targets PyPI. Configure Trusted
Publishing for your actual repository and `publish.yml`, with a distinct
`testpypi-<package>` or `pypi-<package>` environment for each distribution. Configure
environment reviewers to match the project's release policy.

Rehearse on TestPyPI and inspect artifact versions and contents before creating
a production release. A `release: published` event enables the production upload
jobs, so use manual dispatch for publication rehearsals.
