# Security

## Reporting a vulnerability

Report vulnerabilities privately to the maintainer of the Aster distribution
you installed. When its repository enables private vulnerability reporting,
use **Security → Report a vulnerability**. If that channel is unavailable,
request a private contact without including exploit details in a public issue.

Include the Aster version or commit, operating system, affected component,
steps to reproduce, and expected impact. Remove credentials, private source,
and personal conversation data from logs and examples.

## Scope

This checkout provides Aster's terminal, agent runtime, and editor integration.
Reports about an external dependency should also follow that dependency's
security policy. Dependency attribution and source links are recorded in
[Third-Party Notices](THIRD_PARTY_NOTICES.md).

An Aster release channel and support policy have not yet been published.
Maintainers should configure a private reporting contact before distributing
releases publicly.

## Production operation

- Run code-generating agents with an OS-level isolation boundary and a dedicated,
  least-privilege identity. Restrict mounts, network destinations, credentials,
  runtime, and resource consumption. Python visibility annotations and workspace
  trust prompts do not replace isolation. See [Safety](docs/concepts/safety.md).
- Treat the Viewer as an administrative interface: its token grants access to
  all stored traces and the playground. It does not provide per-user or
  per-tenant authorization. Keep it on a private network or behind an
  authenticated TLS gateway; do not use the Vite development or preview server
  as the production server.
- Set `ASTER_VIEWER_AUTH_TOKEN` to a strong, independently generated secret on
  the Viewer and every exporter. When set, **all protected API requests require
  authentication, including loopback requests**. With no token, only loopback
  clients are accepted; that mode is for local development.
- Keep the default loopback bind when using a reverse proxy. Configure the exact
  public hostname using `ASTER_VIEWER_PUBLIC_HOST` or
  `ASTER_VIEWER_ALLOWED_HOSTS`, and set `ASTER_VIEWER_CORS_ORIGINS` to the intended
  browser origins. Never use `ASTER_VIEWER_ALLOWED_HOSTS=*` in production.
  Exporters using another alias must send an explicit bearer token; browser
  cookies do not exempt an unknown hostname from validation.
- Protect bootstrap links containing `?token=` as credentials. A successful
  bootstrap strips the query token, sets an HttpOnly session cookie, and returns
  `no-store` and `no-referrer` headers. The Viewer redacts query tokens in its
  access logs; configure equivalent redaction at proxies and ingress services.
  Use HTTPS so session cookies receive the Secure flag.
- Restrict access to `.aster/mcp_tokens.json`, trace databases, JSONL exports, and
  backups. POSIX token caches use owner-only mode; on Windows, configure a
  user/service-only directory ACL. Trace scrubbing uses known patterns and
  sensitive field names, not an exhaustive detector for arbitrary secrets.
  Historical exports are not rewritten. A scrubbing error drops the affected
  span rather than exporting its original contents; monitor that warning.

## Security upgrade compatibility

MCP OAuth caches bind credentials to the authorization server, exact token
endpoint, protected resource, and client configuration. Old entries without
that binding, or entries whose binding changes, are rejected before cached
credentials can be sent. After verifying the intended server configuration,
stop clients using the cache, remove only that server URL's entry from
`.aster/mcp_tokens.json`, and reconnect to authorize again. Preserve other
servers' entries. Do not add newly discovered binding fields to old credentials.
When `ASTER_PROJECT_DIR` is set, the cache is relative to that directory.

Viewer installations that configure a token must now configure it on local
exporters too, and open the authenticated URL printed by `aster start-dev`.
Custom public hostnames are explicitly allowed through
`ASTER_VIEWER_PUBLIC_HOST`; additional aliases use `ASTER_VIEWER_ALLOWED_HOSTS`.

Automatic `AGENTS.md` loading is confined to the repository root, or the current
directory outside Git. Internal symbolic links remain supported. External
symbolic links and files with multiple hardlinks are skipped. The opened file
handle is checked before reading; Linux requires `/proc/self/fd`, and unsupported
platforms fail closed. Instruction files remain untrusted input, not an
authorization policy.

## Dependency and release checks

Install from `uv.lock` and `package-lock.json`; install JavaScript build
dependencies with `npm ci --ignore-scripts`. CI checks the committed Viewer
bundle against a clean rebuild, audits its full dependency tree and the TUI's
`Cargo.lock`, scans
Git history for secrets, and runs the security regressions on Linux and Windows.
Configure these as required branch checks in the repository host before
releasing. Real microVM containment tests remain a separate CI job and must pass
for deployments relying on that boundary.

As of 2026-11-19, the Viewer build dependency `picomatch` advisory
[GHSA-fc5q-39xc-vjpf](https://github.com/advisories/GHSA-fc5q-39xc-vjpf)
was withdrawn upstream on 2026-11-18. The
[maintainer's explanation](https://github.com/micromatch/picomatch/pull/235#issuecomment-6315749381)
identifies globs as regular expressions and warns against accepting untrusted
patterns. Full `npm audit` now reports zero vulnerabilities for this lockfile;
this is an advisory withdrawal, not a patched `picomatch` release. Build only
reviewed source in a resource-limited CI worker without production credentials,
and recheck the full dependency tree before each release.

The TUI now uses `ratatui 0.30.2` and `ratatui-textarea 0.9.4`, resolving to
`lru 0.18.5`. That version exceeds the `lru >=0.18.2` fix floor for both
[RUSTSEC-2026-0002](https://rustsec.org/advisories/RUSTSEC-2026-0002.html)
and [RUSTSEC-2026-0253](https://rustsec.org/advisories/RUSTSEC-2026-0253.html).
The dependency upgrade retains the existing `tui_textarea` import alias and
saved draft cursor format. CI audits the full lockfile without advisory
exceptions; do not restore the affected `lru 0.12` dependency range.

The audit still reports maintenance notices for
[`chrono`](https://rustsec.org/advisories/RUSTSEC-2026-0543.html) and
[`bincode`](https://rustsec.org/advisories/RUSTSEC-2025-0141.html). These remain
outstanding maintenance work; the default vulnerability audit does not fail on
unmaintained-package notices.
