# Contributing to Aster

## Developer Certificate of Origin

By contributing to this project, you agree to the Developer Certificate of Origin (DCO).
This is a lightweight way for contributors to certify that they wrote or otherwise have
the right to submit the code they are contributing to the project.

You can read the full DCO text at https://developercertificate.org/

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

To indicate acceptance of the DCO, sign off your commits with `git commit -s`, which
appends a `Signed-off-by` line to your commit message:

```
Signed-off-by: Jane Smith <jane.smith@example.com>
```

## How to contribute

This repository contains the Aster terminal workspace and agent framework.
Contributions use the pull-request flow below.

**1.** **Fork** this repository and create a topic branch for your change.

**2.** **Make your change** with tests where applicable. Keep the change focused.

**3.** **Sign off every commit** with `git commit -s` (see the DCO section above) — PRs
without a `Signed-off-by` line on each commit cannot be merged.

**4.** **Open a pull request** against the default branch. Describe the motivation and
what you changed.

**5.** **Automated checks run on your PR**: DCO sign-off verification, license/notice
checks, linting, and the test suite. Please get these green.

**6.** **A maintainer reviews and lands your change.** Your authorship and sign-off
are preserved in Git history.

For substantial features, open an issue first to discuss the design before investing
in a large PR.

## Platform support

Aster currently supports Linux and native Windows. Host features must provide native implementations; Windows shell commands use PowerShell and process-tree cleanup uses Job Objects. Sandbox execution uses microsandbox, with fine-grained permissions applied inside its Linux guest. macOS is outside the current support scope: existing compatibility code is retained, and its CI checks run only when explicitly requested through `workflow_dispatch` (`macos_native` or `macos_sandbox`).

- Import platform-specific modules inside the implementation that uses them.
- Keep microsandbox initialization and virtualization requirements confined to sandbox execution; in-process execution must work without its runtime.

The default root `uv run pytest` collection includes the CLI, ACP, bench and memory packages. Linux unit CI runs this offline suite; `windows-core-smoke` also runs the memory package explicitly and keeps its JUnit results. Optional vector backends and live provider tests remain separate from that offline baseline.

`linux-terminal` and `windows-core-smoke` run Rust formatting, Clippy, tests and a release build. The Linux job also runs the Python terminal launcher and real ACP subprocess protocol regressions; Rust transport tests cover backend startup and process-tree cleanup using the synced Python environment. Windows additionally checks installed wheels and the complete terminal bundle. Tests in `tests/test_native_windows.py` exercise public interfaces on Windows, and WSL supplies additional Linux coverage.

For terminal development, install Python dependencies before running Rust tests:

```sh
uv sync --extra tui --extra mcp --extra tracing --extra viewer
uv run --no-sync pytest packages/aster-memory/tests packages/aster-cli/tests/test_tui_command.py packages/aster-acp/tests/test_protocol.py
cargo fmt --manifest-path packages/aster-tui/Cargo.toml -- --check
cargo clippy --locked --manifest-path packages/aster-tui/Cargo.toml --all-targets -- -D warnings
cargo test --locked --manifest-path packages/aster-tui/Cargo.toml
cargo build --locked --release --manifest-path packages/aster-tui/Cargo.toml
```

## Automated AI Agents

If you are an automated agent, you must include 🤖🤖🤖 at the end of all commit
messages, issue titles, PR titles, and any comments or replies on issues and pull
requests (for example, `Add feature name 🤖🤖🤖`). Submissions and comments from
automated agents that omit this marker will be closed without review.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
