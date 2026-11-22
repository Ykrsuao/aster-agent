# aster-acp

**Run the Aster coding agent inside your editor.** `aster-acp` is an
[Agent Client Protocol](https://agentclientprotocol.com) server, so any
ACP-speaking client — Zed today — can drive the same agent the terminal host
uses: CodeAct, repository tools, a persistent shell, installed skills, workspace
slash commands and durable sessions, with file edits and terminal commands
surfaced as structured activity.

It hosts `aster_cli.coding.CodingAgent` directly. Repository instructions
(`AGENTS.md`), coding tools, summarization, installed `aster.skills` entry points
and semantic file and terminal activity therefore have no separate ACP
implementations — fix something here and the terminal host gets it too.

This is new and we would like it exercised. If something breaks, please say so.

## Install

```bash
uv sync --extra acp    # from the repository root
```

> **Windows:** ACP runs natively, including stdio transport, durable sessions and PowerShell tools. No WSL environment or platform override is required. See the [main README](../../README.md#installation).

There is no default model. Set `ASTER_MODEL` or pass `--model`, or the command
exits with a usage error.

Startup uses LiteLLM's bundled pricing and model catalogue to avoid waiting for
a remote download before opening a session. Set
`LITELLM_LOCAL_MODEL_COST_MAP=False` before launch to request the remote catalogue.

On Windows, startup caches the installed LiteLLM and OpenAI Python code in a
single archive to reduce the number of individual file checks before the first
message. The cache is built once after SDK installation and reused across
reboots; Python and SDK wheel changes select a new cache. Its files live under
`~/.cache/aster/provider-code/` (or `XDG_CACHE_HOME/aster/provider-code/`).
SDK resources and native extensions use their installed paths. Editable SDKs
and project sources use ordinary imports. Set `ASTER_PROVIDER_CODE_CACHE=0` to
disable this cache when troubleshooting or editing installed SDK source files.
An unavailable or unwritable cache falls back to ordinary imports.

## Quick start: Zed

Zed launches ACP agents as "external agents". Add Aster to `settings.json`
(`cmd-,`):

```json
{
  "agent_servers": {
    "Aster": {
      "type": "custom",
      "command": "uv",
      "args": ["run", "--project", "/absolute/path/to/aster", "--package", "aster-acp", "--", "aster-acp"],
      "env": {
        "ASTER_MODEL": "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        "NVIDIA_API_KEY": "nvapi-..."
      }
    }
  }
}
```

Replace `/absolute/path/to/aster` with this checkout's path.
Open a repository, then pick **Aster** from the `+` menu in the agent panel. Zed
runs the command with your worktree as its working directory, so repository
instructions, project skills and sessions resolve against the open project.

Before running generated code, the agent sends the standard ACP
`session/request_permission` request, so Zed shows a prompt with **Allow once**,
**Allow for this session** and **Reject**. This applies to every client. A client
that does not answer the request, or answers with anything else, gets no
execution: the code is not run.

Credentials go in `env` here rather than in Zed's own settings: the agent is a
separate process and inherits only what Zed passes it. Use a secret-manager
wrapper as the `command` if you would rather not put a key in `settings.json`.

The equivalent command from the source checkout is:

```bash
uv run --project "$PWD" --package aster-acp -- aster-acp
```

### MCP servers do not carry over from Zed

**Remote MCP servers you authenticated inside Zed are not usable from an ACP
agent.** Zed holds those OAuth tokens itself and does not pass them down, so a
server showing a green indicator in Zed's own UI arrives at the agent either
with no tools at all or with nothing but its `authenticate` /
`__complete_authentication` stubs. Local stdio MCP servers are unaffected.

This is a known Zed limitation, tracked in
[zed-industries/zed#54410](https://github.com/zed-industries/zed/issues/54410)
(open, labelled `area:ai/mcp` + `area:ai/acp`). A maintainer has said the
plumbing largely exists and the work is queued, but as of this writing it is
unresolved.

Configure the MCP server directly for Aster instead — through Aster's own
`.mcp.json` — and it works normally, because the agent then owns the
connection and its credentials rather than borrowing Zed's.

## Launching the server yourself

```bash
aster-acp --model nvidia_nim/nvidia/nemotron-3-super-120b-a12b
```

This is a JSON-RPC server, not an interactive program: it speaks ACP on
stdin/stdout and exits when its input closes, so running it in a terminal
without a client does nothing. Launch it this way to wire up an ACP client
other than Zed, or to watch the diagnostics it writes to stderr while a client
drives it. `--model` accepts any LiteLLM model name or configured Aster alias.

### Startup timing

Set `ASTER_STARTUP_TIMING=1` in the agent's environment to diagnose slow loading.
Lines prefixed with `[aster-startup]` go to stderr (and Zed's agent log), leaving
the ACP protocol on stdout unchanged. They record import, configuration, model,
storage, MCP and agent preparation phases, including elapsed time and process
CPU time. Each phase logs its start, so an unfinished phase remains visible.
Only stage names, process IDs and timings are recorded.

These timings work on any fresh process. Comparing a first launch after boot
with later launches helps distinguish cold startup from recurring costs; a
fresh process alone does not reset the operating system's filesystem cache.
Unset the variable to disable the diagnostic output.

### Waiting for a reply

Each prompt shows a separate activity card while Aster prepares the execution
environment, waits for a model response, or works on the request. Titles follow
runtime events; they do not estimate a percentage or a remaining duration. The
card completes when the first answer is ready to send. If the turn ends earlier,
it shows completion, cancellation, or failure instead. These updates are optional
ACP `other` tool activity, separate from answer text and durable conversation
history. A client rejecting an activity update does not suppress the answer.

Set `ASTER_PROMPT_TIMING=1` to diagnose time from `session/prompt` to the answer.
`ASTER_STARTUP_TIMING=1` also enables these records for an end-to-end diagnosis.
Lines beginning with `[aster-prompt]` go only to stderr. Each accepted prompt has
an opaque `turn_id`, a process ID, an execution backend and elapsed wall time:

| Stage | What it measures |
| --- | --- |
| `received`, `accepted` | Entry to the adapter and acquisition of this session's foreground turn |
| `prefill_start`, `prefill_end` | Input inspection/setup, including lazy sandbox preparation |
| `model_start`, `model_end` | Each logical model call; `call` pairs its boundaries and `outcome` reports its exit |
| `first_model_response` | The first complete, valid response available to the runtime |
| `first_status` | First successful activity write to the ACP transport |
| `first_answer` | First successful nonempty answer write to the ACP transport |
| `dispatch_end` | Foreground execution returns, including its runtime cleanup |
| `complete` | Final updates have reached the transport, or delivery has failed; includes the outcome |

The model client currently collects a streaming response before returning it.
`first_model_response` therefore does **not** measure the provider's first token.
Likewise, transport completion cannot measure when Zed paints the text. Model
call time includes client preparation, middleware, queueing and retries; it is
not pure network latency. If cancellation or failure prevents an answer, there
is no `first_answer` record. The host's stop/error note is not counted as an
answer, and status updates are never counted as answer latency.

Only fixed stage/outcome names, opaque correlation IDs and timings are logged;
these records contain no prompts, responses, Python source, paths, model
configuration or reasoning. Both timing switches are off by default.

### Usage notifications

Each model response sends a `usage_update`. With reported token usage, `used` is
the input-token count and `size` is the known context-window size, or `0` when
the limit is unknown. Clients should show a token count when `size` is zero;
they must not interpret that value as a full context window.

Before the first response, `_meta.aster.usagePending: true` and
`_meta.aster.usageAvailable: false` distinguish pending statistics from a measured
zero. `used` is `0`, while `size` preserves a known model capacity. After a model
switch, the adapter clears the old measurement and publishes the new capacity.

Before creating or loading the live agent, and before adopting a replacement
model, ACP refreshes optional endpoint metadata. Explicit OpenAI-compatible
routes use one bounded `GET /models` and an exact wire-model ID match. The
two-second lookup does not perform inference or save configuration; failure
leaves configured and LiteLLM limits available. Explicit context settings win,
and automatic summarization uses the same resolved window as usage reporting.

If the response has no usage report, the notification has `used: 0`, `size: 0`
and `_meta.aster.usageAvailable: false`. Those zeros are protocol placeholders:
the terminal displays unavailable statistics rather than a measured zero.

`cost` contains the estimated USD total for calls since this session was opened
in the current server process. It is omitted once any response lacks a valid
price or usage, since the accumulated amount would then be incomplete. An
explicitly reported zero is valid. Loading a saved session does not reconstruct
historical costs, and new sessions start a separate accumulator.

The `_aster/session/runtime` extension reports `contextWindow`, `reasoningLevels`
(a list of declared labels, `null` for unknown support, or `[]` for unsupported),
`reasoningSelection` (the explicit selection), `reasoningDefault` (documented
route default) and `reasoningLevel` (the display label). No provider settings or
credentials are included. Each entry in `models` includes the same reasoning
fields so a picker can preview a model and effort together. Other models' choices
come from the local registry without client construction or network checks;
the active model reports its live selection. `_aster/session/configure` accepts a `reasoningLevel`
label to select its declared request settings, or `null` to clear the selection.
Omitting the field preserves the selection. Changes require an idle session with
no pending approval or background jobs; combined model/effort changes validate
against the candidate model before committing. Configuration updates publish
runtime status and refresh usage metadata without making an inference call.

Runtime status also reports `permissionMode`. Set it through
`_aster/session/configure` with one of these values:

| `permissionMode` | Execution behavior |
|---|---|
| `auto` | Allow generated code and file changes for this live session. |
| `manual` | Request approval before generated execution. |
| `accept_edits` | Automatically handle supported file reads and edits; request approval for commands and other code. |
| `plan` | Use supplied context to plan, with generated execution disabled. |

Selecting the first three modes also enters Code mode. The legacy `mode` and
`approval` fields remain available; do not combine them with `permissionMode`
in one configuration request. New or reconnected sessions start in Manual.
The selected native or sandbox backend does not change with permission mode.
A request containing only `sessionId` and `permissionMode` may change permissions
while foreground or background work runs. The new policy applies at the next cell's
authorization check; work already authorized or started continues. Plan disables
subsequent generated execution without cancelling the current task. An open approval
must still be resolved or cancelled first: changing modes never approves that request.
Requests combined with model, reasoning, or other settings retain the idle-session
requirement.

Accept edits recognizes a single direct `await self.shell.read(...)`,
`await self.shell.write_file(...)`, or path-based `await self.shell.replace(...)`
with literal arguments. It executes the parsed file request through private
host file tools, retaining activity events and the runtime's validation,
middleware, and hooks. It does not evaluate the submitted Python or trust
bindings in the native or sandbox REPL. Computed arguments, assignments,
printing, shell commands, and mixed cells retain ordinary execution approval.
The operation result is available in the usual execution output.

Files with multiple hard links and new files whose parent directory does not
exist require ordinary approval. Automatic operations verify the opened file's
identity and location before reading or writing, and use that same handle for
the operation. If a directory changes while a new file is being opened, execution
is refused before content is written; an empty file may remain. It is not removed
by pathname because another writer may already have replaced it.

Protected path components (`.git`, `.aster`, `.agents`, `.cursor`, `.claude`)
always require approval. At each workspace or granted root, these entries and
links inside their trees also protect their real targets, even when addressed
directly. This check runs again after opening the file; unreadable trees or a
scan exceeding 16,384 entries fall back to ordinary approval.

An explicit `model` change saves its configured alias as `tui.default_model`
before confirming the runtime update. Validation or save failures preserve the
previous runtime and default. New and loaded sessions in the same workspace use
the last confirmed model; reasoning effort, mode and execution consent remain
session settings. The save uses the highest-priority existing `settings.yaml`
layer, or creates project settings if none exists. The terminal launcher still
gives `--model` and `ASTER_MODEL` priority over this saved default.
The initial `_aster/models/select` picker remains lazy: it saves only after
`session/new` or `session/load` has prepared the selected model successfully.
Embedding hosts can pass the startup alias with `CodingACPAdapter(..., model=alias)`;
without it, the initial client's wire ID is used for display only and is not saved.

## Sandbox execution

Pass `--sandbox` to run CodeAct's generated Python in a microsandbox microVM.
From the repository root, install both extras with `uv sync --extra acp --extra sandbox`.
On Windows, launch the ACP server with:

```powershell
uv run --project . --extra acp --extra sandbox aster-acp --model my-alias --sandbox
```

Add `--sandbox` to the command arguments in your ACP client, or set
`ASTER_SANDBOX=true` in the server's environment. `--no-sandbox` explicitly
overrides that environment setting; native execution is the default and works
on Windows with PowerShell tools.

Clients can change the current session through
`_aster/session/configure` with `executionBackend: "sandbox"` or `"native"`.
Runtime status advertises these choices in `executionBackends`. Send this change
separately from model, reasoning, mode, and approval settings, while the session
is idle and has no background jobs or pending approval. Enabling runs a test cell
in a temporary microVM before confirming the change; startup failure or
`session/cancel` keeps the previous backend. The next task creates its own worker.
Conversation history and `self.vars` remain available; ordinary REPL locals are
scoped to each task. The backend choice is live-session only; new and reloaded
sessions use the server's startup setting. Approvals remain a separate setting.

The adapter creates a separate strategy for each ACP session and sets
`SandboxConfig.workspace` to that session's repository root. On Windows, the
guest sees this root at `/workspace`. Default guest permissions allow read-write
access to the workspace and read-only access to interpreter files, with internet
sockets blocked. Framework callers can customize image, resources, network and
extra file rules through
`CodingACPAdapter(llm_factory, sandbox=True, sandbox_config=SandboxConfig(...))`;
the workspace is always resolved from the ACP session.

ACP transport, model requests, workspace skill imports and brokered `self.*`
tools run on the host. The microVM isolates generated Python's direct
operations. Cancelling a running cell or closing its session tears down the VM.
The real stdio tests cover guest execution and cancellation using simulated
model responses.

Prepared dependency snapshots are shared across workspaces using the same image
and dependency versions. They live under `MSB_HOME` (default `~/.microsandbox`),
survive process restarts, and use a process lock to prevent duplicate preparation.
Snapshots from older Aster versions are reused when their environment matches.
Changing the image or dependencies creates a separate cached environment.

## Workspace trust

`aster-acp` checks folder trust before using a repository's Python skills,
libraries, model settings, secrets or saved sessions. Untrusted folders use
built-in and user-level configuration; their conversations are stored in the
user directory. The first session message names the repository files being
ignored. Paths explicitly configured through the server's environment remain
available.

Run `aster trust path/to/repo` to allow its files, then reopen the session.
Use `aster trust --list` to inspect the saved choices and
`aster trust --remove path/to/repo` to remove a choice. Trusting a directory also
trusts its descendants. The TUI asks before loading an unfamiliar folder's files;
other clients can direct the user to the same CLI commands. Unattended hosts can
set `ASTER_TRUST_ALL_WORKSPACES=1` to allow every folder.

Hosts that embed `CodingACPAdapter` directly must enable the check before
reading project configuration or creating sessions:

```python
from aster import workspace_trust

workspace_trust.enforce()
```

Library use leaves the check disabled until the host enables it.

Once trusted, workspace skill roots (including additional configured roots) and
`.aster/libs` are imported on the host when the session opens, in both native and
sandbox execution modes. Trust only repositories whose Python you would run.
Python libraries share the server process, so use separate servers when
workspaces need independent dependencies or credentials.

## How it behaves

ACP uses standard input and output for JSON-RPC. Diagnostics are written to
standard error. Without `--sandbox`, generated Python shares the agent's process
environment, including model credentials. With `--sandbox`, generated Python
runs in the microVM under the configured guest permissions. Shell commands and
other brokered tools still execute on the host under the agent's permissions;
launch the server with credentials and host access scoped to the session.
Cancellation stops cooperative local work immediately. An in-flight provider
request may finish in the background when its client does not support
transport-level aborts. Slash commands run on the agent's event loop so they
have the same semantics as the native TUI and can safely start agent jobs. An
async command is cooperatively cancellable; a synchronous command that blocks
that loop cannot be preempted by the current in-process adapter. The planned
one-process-per-agent boundary is the safe kill mechanism for that case.

## Sessions and skills

Each ACP session has an independent live agent and allows one foreground prompt
at a time. Sessions are stored in `<workspace>/.aster/sessions`, where the TUI
and ACP adapter can share list and replay metadata. A repository could supply
session records of its own, so that directory is used only for a trusted folder.
Until the user runs `aster trust`, the folder's sessions are kept in the user
directory, and its settings, secrets, libraries and skills are ignored; the
session opens with a message naming what was left out. See "Trusted folders" in
the `aster-cli` README. The
adapter also advertises session close; closing a live session preserves its
durable history.

### Read-only skill browser

Clients with `_meta.aster.skills` support can call `_aster/session/skills` with
`{"sessionId": "..."}` to list discovered skills, including text and MCP skills.
The result contains `skills`, `total`, and `truncated`; each skill has `name`,
`description`, `active`, `loaded`, and `kind` (`text`, `python`, or `mcp`). The list
is capped at 2,000 entries. Add `name` to read one skill's documentation under
`skill`, with `text` and `truncated` fields; documentation is capped at 65,536
characters and list descriptions at 512 characters.

Browsing works while a task is running and makes no model request. It neither
loads nor activates skills. Loaded skills expose their class documentation only;
text skills include their already-loaded SKILL.md body. Unloaded entry-point
skills show their import target and explicitly report that documentation is not
yet available. Instance fields and arbitrary filesystem contents are not read.

### Session tree extension

`initialize` advertises Aster's message-tree extension through
`_meta.aster.sessionTree`. Clients using this capability can browse saved user
messages and assistant replies, then restore a chosen path while keeping the
other branches. The native terminal exposes it as `/tree`.

Call `_aster/session/tree` with `{"sessionId": "current-runtime-id"}`. The response
contains `nodes` and `currentNodeId` (a node ID or `null`). Each node has:

| Field | Meaning |
| --- | --- |
| `id` | Stable message-node ID |
| `parentId` | Parent message-node ID, or `null` at the root |
| `role` | `user` or `assistant` |
| `preview` | Text for the tree display; use the navigation result to recall the original prompt |
| `navigable` | Whether a usable restore point exists for this node |

When `_meta.aster.sessionTreeSearch` is advertised, the same request accepts a
`query` string of at most 4096 characters. Search matches complete saved message
text, including content beyond the display preview:

```json
{"sessionId": "current-runtime-id", "query": "a phrase from the message"}
```

The response still contains the entire `nodes` graph and its real
`currentNodeId`. It also echoes `query` unchanged and returns `matchedNodeIds`
for direct matches. Show those matches with their ancestors; the current node
need not be a match. An empty or whitespace-only query matches every node.
Discard results for an obsolete query or session before updating the selection.
Requests without `query` keep the original response shape.

To navigate, call `_aster/session/navigate` with the current node returned by the
tree request:

```json
{
  "sessionId": "current-runtime-id",
  "nodeId": "selected-node-id",
  "expectedCurrentNodeId": "current-node-id"
}
```

`expectedCurrentNodeId` is required, including when its value is `null`. A stale
value is rejected; refresh the tree before trying again. Navigation returns:

```json
{
  "sessionId": "target-runtime-id",
  "prompt": "Original user text to edit"
}
```

Load the returned `sessionId` through ordinary `session/load` to install the
target runtime. For a user node, the restored context ends before that message
and `prompt` contains its original text. For an assistant node, context ends
after the reply and `prompt` is `null`. Put recalled text in the editor for review;
do not dispatch it automatically or overwrite an existing draft. Navigation
does not call the model, generate a summary, or roll back workspace files.

The logical tree uses Aster's immutable snapshots and separately owned session
databases. Navigation copies the selected saved boundary into a fresh session,
including when selecting the current reply, so clients must load the returned
handle. A restore point belongs to the specific saved reply, so later background
messages do not replace it. Progress messages and older nodes without a verified
restore point can be listed with `navigable: false`. Hosts must finish active
work, resolve queued input and save drafts before switching. Recheck these
conditions after `session/load` finishes, before committing the new current
session; if the switch is rejected, keep the source and close the loaded target.
See the
[terminal guide](../aster-tui/README.md#session-tree)
for the user-facing behavior.

The current stdio adapter hosts those live agents in its own process. That is
an adapter-private implementation detail rather than part of the durable
session API: the live-session registry is isolated inside `aster-acp` so it can
later be replaced by handles to an agent daemon without changing stored
sessions, the shared coding agent, or the ACP protocol surface.

Python skill packages use the interpreter's normal import machinery. Multiple
sessions may use distinct skill package names, but two workspaces must not load
different checkouts under the same top-level Python package name in one ACP
server process. Launch a separate stdio server for those workspaces. A future
one-process-per-agent daemon will make that isolation an OS process boundary.

Installed `aster.skills` entry points are loaded into the shared skill registry
but remain opt-in. The agent can activate a relevant skill with
`self.skills.activate(["name"])`. Stdio MCP servers supplied by an ACP client
are registered and activated as `mcp.<name>` skills for that session.

Workspace and user skill roots are shared with the terminal host through
layered `settings.yaml`. New configuration should use:

```yaml
coding:
  additional_skills_dirs:
    - ../aster-skills
```

The existing `tui.additional_skills_dirs` key remains supported during the
migration, as does the older project-local `.aster/config.toml` key
`[tui].libs_dirs`. Packaged libraries declared through `aster.skills`, `SKILL.md`
skills, and standalone Python skills are discovered from each configured root.
Loaded `@slash_command` methods are advertised through ACP and matching
`/command arguments` prompts are dispatched through the shared typed command
router. Command discovery is refreshed when loaded skills change.

Text skills are advertised as individual commands. Use `/aster-agent-authoring
Add a classifier`, or mention `$aster-agent-authoring` anywhere in a task.
The command uses a valid frontmatter `name`, falling back to the directory ID;
the directory ID also remains an alias. Duplicate aliases and names occupied by
terminal or Python commands use the full registry name, such as `/cmd.review`.
A qualified missing name, an
ambiguous alias, or a disabled user invocation produces an error before a model
call. Unknown `$variables`, escaped mentions, and Markdown code are kept as
literal text.

Several mentions supply their instructions and source directories to one
cancellable task, with the complete original request preserved. Repeated aliases
for the same skill are included once. Arguments are literal: there is no shell
parsing or `$ARGUMENTS` substitution. Skill selection does not execute scripts or
change execution permissions. Session restoration reloads current skills from disk.

For automatic selection, the agent initially sees only names, descriptions, and
source directories. It chooses relevant skills for the current task and reads
their full instructions through `self.skills.read_text_skill(name)`. Reading does
not activate the skill or add its whole body to the persistent context. This
generated read follows the session's execution policy; Plan mode still disables
generated execution.

SKILL.md may set `argument-hint`, `user-invocable: false` (hide and reject explicit
invocations), or `disable-model-invocation: true` (allow only explicit use).
`policy.allow_implicit_invocation: false` in `agents/openai.yaml` also disables
automatic selection. If either automatic-use setting disables it, explicit use
is required. Malformed boolean settings reject discovery or preserve the previous
skill on a failed reload.

`available_commands_update` includes the command's standard `input.hint` and
`_meta.aster.kind: "skill"` for text-skill commands. Their optional
`_meta.aster.aliases` array supplies valid alternate command names without
duplicating the menu entry. Python commands can also
provide a `_meta.aster.completions` array for first-argument completion. Clients
that ignore this metadata can still send commands and mentions as text.
Text-skill requests preserve file and image attachments in the task context;
Python slash commands retain their existing text-only dispatch behavior.

The current adapter accepts text, resource links, embedded text resources, and
PNG/JPEG/GIF/WebP images. Images are forwarded to the selected model without
requiring `media_inputs` configuration; image-only prompts are supported. Provider
rejections are returned with their error message.

Stdio, HTTP, and SSE MCP servers forwarded by an ACP client are supported.
ACP-transport MCP proxies and additional workspace directories are not advertised
yet. An unavailable, duplicate, or unsupported MCP server is skipped with a session
warning so it cannot prevent a new or restored Aster session from opening.
