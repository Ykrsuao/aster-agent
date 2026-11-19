# Aster terminal workspace

Aster brings coding conversations, tool output, Git changes, and saved sessions
into one terminal workspace. Its Rust / Ratatui frontend connects to Aster's
Python ACP agent. The terminal owns the conversation view, composer, searchable
pickers, tool inspector, and change viewer. Python owns model configuration,
tools, execution, and saved sessions.

Launch a new conversation with `aster` or `aster tui`. Rust sources live in
`packages/aster-tui`; project configuration and sessions live in `.aster/`.

## Build and run

From this repository checkout, with Python 3.12–3.13, Rust 1.88+ and uv installed:

```powershell
uv sync --extra tui
cargo build --locked --release --manifest-path packages/aster-tui/Cargo.toml
uv run --extra tui aster
```

Use `/model [ALIAS]` to switch among configured models. Add or edit models in
`llm_config.yaml`, or run `aster connect` from your shell. See
[model configuration](../../docs/model-configuration.md) for file locations and
credentials.
When the Python launcher is on PATH, `aster` starts directly in the current
workspace. Model selection uses `--model`, then `ASTER_MODEL`, then
`tui.default_model` from layered `settings.yaml`. If no default is saved,
Aster can use a single configured chat model. With several aliases it opens the
model picker. With none, the picker explains how to add a model in
`llm_config.yaml` and restart Aster.
For example, save this in your project's `.aster/settings.yaml`:

```yaml
tui:
  default_model: my-model
```

User defaults can be saved in `~/.config/aster/settings.yaml`. Confirming a model
in the picker or with `/model ALIAS` saves its alias as `tui.default_model` after
the backend successfully prepares it. The save updates the highest-priority
existing settings file, including `ASTER_SETTINGS` overrides; with no settings
file, it creates the project's `.aster/settings.yaml` (or `ASTER_PROJECT_DIR`).
The next launch uses that choice unless `--model` or `ASTER_MODEL` overrides it.
`-w PATH`, `--client-type completion|responses`, `--sandbox`,
`--resume ID` and `--continue` are forwarded to the Rust client. Python is selected
from the environment running `aster`, so the ACP subprocess has the same installed
packages. New conversations prepare their model and session in the background
as the interface opens. You can type immediately. If preparation is still running
when you press Enter, your message appears immediately in the conversation.
The status line shows `Connecting…` or `Starting conversation…` until it can be
sent. Press Esc to bring the waiting message back for editing. Multiple waiting
messages also show a count; paused sending is labeled explicitly.
Messages submitted during an active turn use `queued`.
`--resume ID` reopens a chosen session. `--continue` reopens the most recent
session, or starts a new conversation when there is no history.

The Python wheel does not currently bundle the native executable. For a locally
installed package, build this crate and put `aster` (`aster.exe` on Windows) on
PATH, or set `ASTER_TUI_BIN` to its absolute path.
Launcher discovery checks this override, a bundled `aster_cli/bin` binary, the
checkout's release build, then PATH. It never downloads or compiles code during
startup. Rebuild after changing Rust sources.

The session title sits on the input box's upper-left border. A configurable HUD
below the input shows the model, workspace folder, Git status and usage. Click
the project row or use `/status` to view the full workspace path.

Native execution is the default. To use the optional microsandbox runtime:

```powershell
uv run --extra tui --extra sandbox aster --model my-model --sandbox
```

`ASTER_SANDBOX=true` also enables sandbox execution. `--no-sandbox` explicitly
overrides the environment. See
[sandbox execution](../aster-acp/README.md#sandbox-execution) for runtime setup.

## HUD

The HUD adapts [claude-hud](https://github.com/jarrodwatts/claude-hud)'s status
display ideas to Aster's ACP events, runtime tasks and local workspace. The default
is compact and uses at most three rows: project and usage share a row when the
width allows, followed by tool, plan and background-job activity when space is
available. During a conversation, compact layouts are also limited to one eighth
of the window height (at least one row when space permits). Expanded and full
layouts show the configured rows while reserving space for the conversation and
composer, including after typing or sending a message. Growing drafts,
attachments and queued messages can reduce the available HUD space. Short
windows prioritize usage and cost when only one HUD row fits.
The shortcut hint sits immediately below the HUD, without extra blank rows.
Expanded layouts align metric labels in terminals at least 60 columns wide.
Labels and separators are muted; reported values are brighter, with bold accents
for the model, selected reasoning level and measured context usage. Pending and
unavailable values stay subdued. Activity markers distinguish running, completed
and failed work without adding rows. Narrow layouts drop the label padding.
The welcome page shows the configured HUD rows while reserving space for the
Aster wordmark and menu. It reduces spacing and uses two menu columns when
needed, so the full HUD and complete artwork fit in an 80-column, 24-row window.
When the artwork cannot fit, a compact `ASTER` wordmark remains visible above
the menu.

`/hud` opens a searchable settings menu; `/hud info` shows current values, their
sources, configuration paths and collection errors. Click a HUD row to open the
matching inspector (`/status`, `/context`, `/effort`, `/tools` or `/tasks`); metric rows open
`/hud info`. Dragging selects and copies text instead of activating the row.

| Command | Effect |
| --- | --- |
| `/hud compact` | Combine project and usage on wide terminals |
| `/hud expanded` | Give each enabled element its own row |
| `/hud full` | Enable all elements, Git details and day/week costs; allow up to 16 rows within the window's height budget |
| `/hud off` | Hide the HUD; `/hud expanded` shows it again |
| `/hud show ELEMENT` / `/hud hide ELEMENT` | Add or remove an element |
| `/hud set KEY VALUE` | Change a setting; values accept JSON or a plain string |
| `/hud reset` | Restore built-in defaults |
| `/hud undo` | Restore the previous settings; available after restarting too |
| `/hud reload` | Apply manual edits to global and workspace settings |

Settings load from `<ASTER_USER_DIR>/tui-hud.json` when `ASTER_USER_DIR` is set,
otherwise `$XDG_CONFIG_HOME/aster/tui-hud.json` or `~/.config/aster/tui-hud.json`.
The workspace's `.aster/tui-hud.json` overrides global values. Changes made through
`/hud` save the effective configuration atomically to the workspace file, with
the previous configuration in `.aster/tui-hud.previous.json`. Invalid settings
produce an explanation and preserve the current configuration.

For example:

```text
/hud set elements ["project","usage","tokens","cache","tools","tasks","agents"]
/hud set token_scope request
/hud set context_value both
/hud set path_levels 2
/hud set colors {"model":"#b1b9f9","context":"#93ba85"}
```

| Setting | Default and accepted values |
| --- | --- |
| `layout` | `compact`; also `expanded`, `hidden` |
| `elements` | Ordered array: `project`, `usage`, `tools`, `tasks`, `agents`; additional names: `reasoning`, `tokens`, `cache`, `cost`, `skills`, `memory`, `limits`, `session`, `custom` |
| `path_levels` | `1` trailing workspace component; `0` shows the full path; maximum `8` |
| `max_rows`, `max_width` | `3` rows (1–16), `160` columns (20–400); the row count is also limited by window height and available conversation space |
| `context_value` | `percent`; also `tokens`, `remaining`, `both` |
| `context_bar` | `true`; hide the bar with `false` |
| `token_scope` | `session`; `request` shows token/cache counts for the latest foreground response |
| `warning_percent`, `critical_percent` | `70`, `85`; warning must be below critical, both within 0–100 |
| `git`, `git_details` | `true`, `false`; details add file and line-change counts |
| `refresh_seconds` | `5` (2–120) for background host collection; pending accounting is saved sooner |
| `activity_limit` | `3` (1–8) visible tool/job/skill names per row |
| `daily_cost`, `weekly_cost` | `false`, `false`; show recorded totals in the `cost` row |
| `model_format`, `model_override` | `full` (`short` omits the prefix before the last colon), empty display override |
| `provider` | Empty; use the backend's provider label, or supply a display override |
| `custom_text` | Empty; literal text for the `custom` row |
| `external_usage_path` | Empty; optional absolute or workspace-relative JSON snapshot path |
| `external_usage_ttl_seconds` | `300` (1–86400); freshness threshold for quotas and maximum age of external snapshots and cache-expiry observations |
| `language` | `en`; `zh` uses Chinese HUD row labels |
| `colors` | Empty; overrides for `model`, `project`, `text`, `muted`, `context`, `warning`, `critical`, using strings `#rrggbb`, `0`–`255` or `reset`. An explicit `{}` clears inherited colors. |

`NO_COLOR` overrides HUD color settings. Display overrides do not change the actual
model or provider. The custom row displays literal text and does not execute commands.

### Data sources

Git reports the branch, dirty marker, ahead/behind counts, worktree and conflicts.
Optional details show modified/added/deleted/untracked file counts and tracked
line additions/deletions. Git and host reads run outside the rendering loop.
The memory row measures host RAM on native Windows and Linux. Skills show active
and available skills, registered MCP servers and applicable `AGENTS.md` files
between the Git root and current workspace directory.

The `reasoning` row shows the current runtime reasoning level, falling back to
the level reported by the backend. With no reported level it shows `model default`;
the provider's exact default effort remains unknown. `/hud show reasoning` enables
the row, and `/hud full` includes it automatically. Click it or use `/effort`
to inspect or switch among the current route's declared levels.

Tokens and cache counts come from reported model usage. A field missing from any
participating response makes that session total `n/a`; an explicit zero remains
zero. Cache hit percentage is reported cache-read tokens divided by input tokens.
The latest foreground response supplies context, request token counts, output rate
and provider limits; helper/summary responses contribute to session accounting
without replacing those foreground values. Output rate divides reported output
tokens by the whole completed call duration, including latency and retries. It is
not an instantaneous streaming measurement; calls shorter than half a second or
without reported output usage leave it unavailable.

Magpie connections can set `usage_source: magpie` under their existing model alias
in `llm_config.yaml`. The ACP backend reads the same gateway using that model's
explicit credential, correlates request records by response ID, and fills missing
costs and corrected cache counts before the response is saved. Gateway USD costs
are pricing estimates; account credit balances are shown separately under Limits.
Requests and responses without a unique, consistent record retain their original
usage. Lookups have a two-second deadline and a 2 MiB response limit; lookup failure
does not fail the model reply. Other connections keep the default `response` source.

Once a foreground response identifies the account, its quotas refresh every 30
seconds while the chat is open. Their provider observation time controls freshness,
independently of the last reply time. If the gateway still returns an older snapshot,
the HUD labels its values `last known` with their age. It does not reset that age
when polling; expired quota windows remain unavailable. Raw credentials and account
identifiers are not included in HUD telemetry.

Tool counts and timing use conversation events. Plan progress, waiting reasons
and background jobs use Aster's runtime task snapshots; running job timers advance
from the reported start time, while finished jobs retain measured durations.
Opening `/tasks` exposes individual job controls. Session age uses the earliest
restored model response, or the current ACP session's start when no history exists.
Completed-call totals include saved model responses; failed-call counts cover
only the current connection because failure lifecycle events are not persisted.

Cache expiry and quota windows require explicit provider facts. Supported OpenAI
and Anthropic rate-limit headers can supply request/token windows. Account-level
5-hour/7-day quotas and spend limits can instead come from a local snapshot:

```json
{
  "updated_at": "2026-11-15T08:00:00Z",
  "five_hour": {"used_percentage": 25, "resets_at": "2026-11-15T12:00:00Z"},
  "seven_day": {"used_percentage": 40, "resets_at": "2026-11-16T00:00:00Z"},
  "spend_limit": {"used_percentage": 30},
  "windows": [{"label": "requests", "percent": 10, "resets_at": "2026-11-15T08:01:00Z"}],
  "cache_expires_at": "2026-11-15T08:05:00Z"
}
```

Set `external_usage_path` and enable `limits`. Timestamps accept Unix seconds or
RFC3339; the producer must refresh `updated_at`. Snapshots are limited to 64 KiB
and eight quota windows. The external snapshot replaces provider quota windows
when configured. Missing, invalid, expired or stale external observations display `n/a`,
with collection errors in `/hud info`. The HUD does not infer subscription quotas
from local tokens. The terminal reads local snapshots; optional ACP integrations
obtain provider data using the configured model connection.

Costs with some unpriced calls show the known subtotal prefixed with `≥`, both for
the session and the recorded day/week. Entirely unknown costs remain `n/a`;
explicitly free calls remain zero. Old missing prices are not silently backfilled.
The standard ACP cost field remains a complete session total; the HUD's known
USD subtotal travels separately in Aster metadata so other clients do not mistake
partial accounting for a complete total.

## Context and cost

The HUD's usage row is present from startup. Before the first response, a known
model capacity displays `context ░░░░░░ 0k / 1.0m`, with an empty bar and `0k`
as pending placeholders. The bar respects `context_bar` and available width.
The context inspector still waits for a reported measurement. An unknown capacity
displays `context —`. After a response, the row shows the latest reported input
usage in the selected `context_value` format (percentage or token counts).
In `tokens` and `both` formats, used tokens stay in `k` (thousands), while capacity
uses its compact unit; for example, `12.3k / 1.0m`.
If the window is unknown, it shows the reported token count instead. A missing
usage report displays `context n/a`; it does not imply an empty context.

`/context` opens a matrix of 100 rounded square cells for the latest completed
foreground request. Colors identify used input categories, blue-gray marks output
reserve, and dark gray marks free space. The legend lists system prompts, tool
descriptions and schemas, repository instructions, skills, messages and runtime context,
followed by output reserve and free space. Each cell represents 1% of the model's
context window. A planning reserve is labelled when no explicit output cap is known.

When the terminal reports Sixel support and cell pixel dimensions (including Windows
Terminal 1.24+), the grid uses filled rounded squares with equal pixel width and height
and tight gaps on both axes. It scales with the terminal font and fits the available
panel width. Terminals without complete graphics reports, very small windows and
monochrome mode use a compact character grid without spaces between cells.

The total comes from provider usage. Category and block counts carry `~` because
they are attributed by text size; they are not separate provider measurements.
Image, audio and file inputs retain the provider total with no category estimate.
Unknown input usage or capacity displays an explanation instead of an empty grid.
Unsent drafts and output generated by the latest request are outside that request.

Use arrows to select a category, then Enter or a mouse click to read the actual
text sent to the model. `Latest model output` opens the assistant text and tool
calls returned by that request. `/context all` or Tab adds individually selectable
blocks and messages. In the text view, arrows, PgUp/PgDn and Home/End scroll;
Esc or Left returns to the overview, and Esc from the overview closes the panel.
Drag across the text to select it; releasing the mouse copies the selected part
to the system clipboard. Shift+click extends the selection, and Ctrl+C copies it
again. Copying preserves real newlines and tabs without inserting screen-wrap
newlines. Selections remain attached to the same text when scrolling or resizing.

The panel can open while a request is running and does not send another model
request. The overview follows the latest completed foreground request; an open
text view stays pinned to its selected request. Only the latest request is retained
in the current process. Switching models clears it, and loading a conversation or
restarting waits for a new response. Text is fetched when an item is opened;
recurring usage updates contain only an item index. Input and output captures
each keep up to one million characters, with any truncation shown in the text view.

The ACP backend can read the exact model's published input limit from a configured
OpenAI-compatible gateway's `/models` response. This optional lookup runs when
opening a session or switching models, without sending a prompt. Explicit model
settings take priority, with provider-aware LiteLLM metadata as the fallback.

Session cost is the estimated USD total reported for model responses, including
responses restored from the saved conversation. If only some calls have valid prices,
the HUD shows the known subtotal prefixed with `≥`. When no calls have valid prices,
it displays `cost n/a`. An explicitly reported zero can display `$0.0000`.
Changing models clears the previous context measurement while keeping the current
cost total. Its known capacity appears immediately as `0 / capacity`; the new
model supplies its own context measurement on its next response.

Day/week totals include newly observed usage recorded by this TUI across workspaces,
stored in the user directory's `hud-usage/`. They use the local calendar day and a
Monday-based week. Restoring history updates session totals without charging the
day/week ledger again. Separate connection streams have separate atomic files;
sequence numbers prevent duplicate writes from inflating totals. These totals
cover locally recorded activity, not all activity on the provider account.

Narrow terminals use shorter labels and keep the indicators visible. The
`examples/preview.rs` executable uses synthetic conversation and usage fixtures;
its `welcome` mode has no sample usage. Those images demonstrate rendering, not
measurements from a live model session.

## Controls

| Key | Action |
| --- | --- |
| Enter | Send a message; queue it while preparing or while a prompt is running |
| Alt+S | Send the current text as a correction to the running task |
| Alt+Enter / Shift+Enter | New line (Shift+Enter depends on terminal support) |
| Click / drag inside input | Position the cursor / select text; releasing the mouse copies it, and Shift+click extends the selection |
| Ctrl+C / Ctrl+X with selected input and no popup open | Copy / cut the selection to the system clipboard |
| Ctrl+V / Ctrl+Shift+V / Shift+Insert / Right click | Paste clipboard text or attach a copied image without sending |
| `@` at a word boundary / Alt+A | Find and attach a workspace file |
| Click an attachment name / +N | Preview that attachment / inspect hidden attachments |
| Enter / click the image in the attachment panel | Open the original image in your system image viewer |
| Click × beside an attachment / Alt+Backspace | Remove that attachment / the last attachment from the draft |
| Backspace with empty input | Remove the last attachment from the draft |
| Click Undo / Ctrl+Z immediately after removal | Restore the last removed attachment |
| Up / Down at the first / last input row | Previous / next submitted message; return to your draft |
| `/` at the start of input | Show inline command suggestions |
| Up / Down, Tab, Enter | Select, complete or run an inline command |
| Ctrl+K | Open the command palette without replacing the draft |
| Ctrl+R | Search and restore a conversation |
| Ctrl+F | Search messages, tool output and diffs in this conversation |
| Ctrl+E | Manage queued messages |
| Ctrl+S | Continue a paused message queue |
| Ctrl+T | Browse tool calls and saved output in this conversation |
| Ctrl+G | Inspect Git workspace changes; Tab switches to conversation edits |
| Ctrl+U in Git Changes | Review rollback of the selected file or one hunk |
| Alt+T | Inspect plan steps, waiting reasons and background jobs |
| Ctrl+O | Expand/collapse all message and tool details |
| Click a message / tool / more-lines hint | Expand or collapse that content |
| Alt+B | Toggle the conversation/activity sidebar |
| PgUp / PgDn, mouse wheel | Scroll conversation or inspector |
| Ctrl+End | Clear the text selection and follow newest messages |
| Ctrl+P / Ctrl+N | Previous/next submitted input |
| Drag across conversation, tool output, panels or other visible text | Select text; releasing the mouse copies the selection to the system clipboard |
| Ctrl+C with selected text | Copy that selection again without exiting or changing the draft |
| Ctrl+C twice within 2 seconds with no text selected | Save and exit; the first press shows confirmation |
| Esc | Close suggestions / an overlay; recall a message waiting for preparation; cancel a running task |
| Esc twice within 600 ms, while idle | Clear a nonempty draft; with an empty composer, choose a message to rewind to |
| Ctrl+Z / Ctrl+U in the composer | Undo an input edit, including clearing a draft with double Esc |
| Ctrl+Q | Save and exit |
| Ctrl+X without selected input / F1 (no popup open) | Help |

Pickers support typing to filter, arrow selection, Enter to choose and Esc to
return to the draft. Slash commands stay in the composer, with a filtered list
above it. Tab completes the selected name and inserts a space for arguments;
Enter executes the selected command. Esc hides the list without changing the
draft. The Ctrl+K palette inserts commands with arguments only when the composer
is empty; an existing draft, cursor and undo history are preserved.
Type `/resume ` followed by a session ID or unique prefix. You can draft a
message during execution; Enter queues ordinary messages and shows what is next
above the composer. Messages run one at a time after a successful turn. Cancel or
errors pause the queue; Ctrl+S continues. Ctrl+E opens the queue: Enter takes an
item back for editing when the composer is empty, and Delete removes it. Taking
an item back also pauses the remaining queue. Drafts, queued messages and their
attachment snapshots are autosaved separately from submitted conversation history;
recovered inputs always start paused. See [Recovery](#recovery).
Finish or remove queued messages before switching to another conversation.
The same rule applies to rewinding a conversation.
When the next message starts automatically, the queue panel keeps the same
remaining message selected, including in filtered lists and with duplicate text.

Background preparation leaves the welcome screen and composer available. Esc
recalls the first waiting message into an empty composer and pauses any remaining
queue. If the composer already has text or attachments, both are kept and the paused queue
opens for editing. Preparation continues; a recalled message is never sent unless
you submit it again. Input history and conversation messages are recorded only
when a prompt is dispatched. Unused prepared sessions are cleaned up on normal
close. Preparation errors remain visible and preserve pending input.

Esc cancels the running task from the main conversation. With a panel or slash
suggestions open, Esc closes that view first and keeps the draft. Cancellation
keeps the queue paused until the operation actually finishes. After 20 seconds
without a terminal response, the TUI reports that it is still waiting and stays
open; it does not stop the backend automatically. Press Ctrl+C twice within two
seconds to leave, or use Ctrl+Q or `/exit`.

With no text selected, the first Ctrl+C shows an exit confirmation:
it does not cancel the task or clear your draft. Another key, a paste, or a mouse action dismisses confirmation;
waiting more than two seconds also resets it. Explicit exit requests use bounded
backend cleanup. Saved drafts and queued messages can be recovered after closing;
restoring them does not send a prompt automatically.

While idle, press Esc twice within 600 ms to clear the entire draft as one
undoable edit. With an empty composer, double Esc (or `/rewind`) opens a searchable
list of messages with saved conversation checkpoints, newest first. Enter creates
a separate conversation ending just before the selected message and places its
original text back in the composer. Edit it and press Enter when ready; selecting
a message does not send it. The original conversation remains available through
`/sessions`. This restores conversation context and agent state; workspace files
stay as they are. Checkpoints are recorded for new messages; older messages that
predate this feature cannot be rewound.

An Esc used to cancel work, recall a waiting message or dismiss a panel,
suggestions or search highlighting does not start a double-Esc sequence. Other
keys, paste and mouse input also reset the sequence. This prevents finishing a
cancellation or closing a panel from accidentally clearing a draft or rewinding.

Ctrl+F searches the current conversation, including collapsed tools. Enter jumps
to the selected result and expands its tool if needed; Ctrl+End returns to the
latest output and clears highlighting. The composer and cursor survive searches.
Picker lists support mouse selection and scrolling.
Clipboard shortcuts, right-click in the composer, and bracketed paste insert multiline
text without sending it. In a searchable panel, right-click its search field to paste
into that field. Releasing or dragging the right button does not paste again.
Windows reads native clipboard text and PNG/bitmap images. Linux image paste uses
`wl-paste` on Wayland or `xclip` on X11; text paste also supports `xsel`. Your
terminal's text Paste command works without these utilities. If reading fails,
the draft is kept and an error is shown.
Drag across conversation text, expanded tool output, inspector details, help,
diagnostics, headers or other visible labels to select them. Releasing the mouse
copies the selection to the system clipboard. Soft wrapping adds no extra newlines;
original tabs, hard newlines, Chinese and emoji are retained. Shift+click extends
the selection, and Ctrl+C copies it again without exiting or changing the draft.
After a copy or cut succeeds, the line immediately above the message composer shows
`copied N chars to clipboard` in light blue-violet text for two seconds, including
while an inspector or other panel is open. The message aligns with the composer's
right edge. The count uses Unicode characters
from the copied text; another successful copy refreshes the count and timer.
Single clicks still activate the original control on release; dragging a tool
heading or picker item does not activate it. Selected conversation text stays
steady while new output arrives; Ctrl+End clears the selection and follows the latest messages.

Click the composer to position the cursor; dragging and releasing also copies
selected draft text. With no popup open, Ctrl+X cuts selected input only after copying succeeds.
If the draft changes before copying
finishes, its text is kept. Failed clipboard access also keeps the selection.
Copy, cut and paste requests run in order, including a paste requested while copying.
A pending paste is cancelled if you change its input, move the cursor, or switch conversations.
Long user messages initially occupy at most three visual rows. Expanded tools show
the first two and last three rows of long output; click the omitted-lines hint to
show all rows, or use Ctrl+T for the full tool inspector. Search opens the entire
matching message or tool output before jumping to it. Conversation diffs remain
available in full; Git preview limits are described below.

Commands: `/new`, `/sessions`, `/resume ID`, `/drafts`, `/tree`, `/rewind`, `/search [text]`, `/queue`,
`/steer TEXT`, `/attach [PATH]`, `/attachments`, `/retry`, `/reconnect`, `/model [ALIAS]`,
`/effort [LEVEL|auto|status]`, `/reasoning [LEVEL|default]`, `/mode [code|plan]`,
`/permissions [auto|manual|accept_edits|plan]`, `/sandbox [on|off]`, `/tasks`, `/stop JOB-ID`, `/tools`,
`/changes`, `/undo`, `/status`, `/context [all]`, `/hud`, `/logs`, `/help`, `/cancel`, `/exit` and commands
advertised by the agent.
The interface follows the layout observed in Grok Build 1.0.77: a grayscale
canvas, full-width conversation, compact bordered composer and session title on
its top border. The HUD adds model, project and activity rows below the composer.
The welcome menu centers available
actions; typing immediately focuses the conversation. Alt+B opens the sidebar when the terminal has at
least 104 columns and 20 rows. Smaller windows retain the conversation and composer.
The composer wraps long input and grows up to eight visual rows, including Chinese
and emoji, without inserting newlines into the message. Up/Down and Home/End move
through the visual rows; Shift extends the selection. Taller drafts scroll to keep
the cursor visible. At the first row, Up recalls a submitted message; at the last
row, Down advances through recalled messages and returns to your unsent draft,
including its cursor and undo history. Editing a recalled message makes it a new
draft. Ctrl+P / Ctrl+N recall history from any row.
Switching conversations while browsing input history preserves the hidden draft
and its cursor/undo history; Down past the latest recalled entry returns to it.
Submitted inputs are saved immediately in `.aster/tui-input-history.jsonl`
(up to 1,000 entries and 8 MiB).
Restarting in the same workspace loads this history without resuming a conversation.
On first use, recent user inputs from saved sessions are imported automatically.
Concurrent terminals merge their submissions; drafts and unsent queue entries are
not saved as submitted history. Shortcut hints sit below it and
running status sits above it. Inline commands support mouse selection and wheel
navigation; Ctrl+K, session and search pickers size themselves to their contents.
`NO_COLOR` uses terminal default colors with text markers and emphasis.
See [reference observations](docs/grok-reference.md) for the evidence and remaining differences.

## Steering during execution

Enter queues the draft for the next turn. Alt+S or `/steer TEXT` instead sends a
text correction to the current turn. The receipt distinguishes **received** from
**included in the model request**. The correction joins the next model request at
a safe execution boundary; a cell already executing can finish before it takes
effect. It does not undo that cell's effects. Use Esc to stop the foreground task.

Steering accepts text only. Attachments stay with messages queued for a later turn.
Rejected or interrupted corrections remain available for review, and typing a new
draft while waiting for acknowledgement preserves both inputs.

## Files and images

Type `@` after whitespace or at the beginning of a line, press Alt+A, or use
`/attach` to open the file picker. Search respects Git ignore rules and returns a
bounded list; narrow the query when necessary. `/attach src/parser.py` selects a
relative path, while an explicit absolute path can select a file outside the
workspace. Enter attaches the selection without sending it.

Ctrl+V attaches a copied image in the composer. Click the × beside an attachment
to remove it immediately, or use Alt+Backspace to remove the last attachment.
With an empty composer, Backspace also removes the last attachment. Alt+Backspace
does nothing when there are no attachments. Click a name to preview it, or click
`+N` to inspect hidden attachments. Click Undo or immediately press Ctrl+Z to
restore the last removed item at its original position, including its captured
bytes. This one-step undo lasts until the next text edit, successful attachment
addition, draft replacement or submission, or conversation switch. Inside
`/attachments`, Delete removes the selected item and Ctrl+Z restores it. These controls
affect the current draft; queued and submitted messages keep their snapshots.
Files and images are
captured when attached, so later source-file edits do not change the next prompt.
Both the prompt and saved conversation retain that snapshot. Sending waits for
pending file reads. You can send an image without entering text. The terminal
forwards attached images to the selected model without a per-model image switch.
If the provider does not accept images, its error is shown.

The attachment inspector displays PNGs as pixel images on Windows terminals that
report Sixel support and cell geometry, including Windows Terminal 1.24. Images
fit the available panel without enlargement, within a 1024-pixel limit per side.
Detection does not delay startup; decoding and encoding run in a background worker
and reuse a bounded cache. Other terminals retain the character thumbnail;
`NO_COLOR` retains the metadata-only view.

To inspect full-size detail, click the image or **Open original**, or press Enter with
an image selected. The system image viewer opens the captured original, where
you can zoom in. This also works for JPEG/GIF/WebP, `NO_COLOR`, and images that
cannot be shown as a terminal thumbnail. The original image snapshot is included
when sent; the terminal thumbnail does not reduce its quality.

A message accepts at most eight attachments: UTF-8 text up to 256 KiB per file,
or PNG/JPEG/GIF/WebP images up to 8 MiB each, within a 16 MiB aggregate prompt
limit. Directories, binary text and unsupported image formats produce an error
while preserving the draft. Selecting a file adds
its context; it does not grant permission to execute code.

## Session tree

`/tree` opens a searchable tree of user messages and assistant replies. Shared
history appears once, with alternative paths branching from it; `●` marks the
current node. Type to search, use Up/Down to select, Enter to navigate, and Esc to
close. Search checks complete saved message text, including content beyond the
display preview, and keeps matching nodes' ancestors visible so branches remain
clear.

- Select a user message to restore the context before it and recall its original
  text for editing. Press Enter in the composer when ready to send it.
- Select an assistant reply to restore the context after that reply and continue
  from there.

Existing branches remain available in the tree after navigation and after
reopening the conversation. The model uses the selected path when you next send
a message. Navigation restores saved context without calling the model or
generating a summary, and leaves workspace files unchanged.

Your existing draft is kept. If it would be replaced by recalled text, that text
goes into the paused queue instead. Finish or stop active work and resolve queued
messages before navigating. Input and work state are checked again when the
selected branch finishes loading; a failed input save prevents the switch.
Nodes without a usable saved restore point, including unfinished progress,
background notices and some older messages, remain visible but cannot be
navigated to. A later background notice does not remove the restore point of an
already completed reply.

## Recovery

The composer, cursor, attachment snapshots, queue and interrupted input are saved
separately for each conversation under `<workspace>/.aster/tui-drafts/`. Starting
`aster` or `aster tui` opens a new conversation with an empty composer and queue.
Use `--continue`, `--resume ID` or `/sessions` to reopen a saved conversation and
restore its input.

`/drafts` opens a menu of drafts containing unsent input, newest first. This
includes drafts saved before a first message was sent. Select a draft and press
Enter to restore it.

Restored queues stay paused until Ctrl+S. `/retry` recalls only the current
conversation's interrupted input for editing before Enter sends it. If another
draft is present, the recovered message goes into the paused queue.

`/reconnect` restarts the ACP connection and restores a saved conversation when
available. Stop an active prompt before reconnecting. After an ambiguous delivery
failure, inspect the restored conversation before resending: reconnect never
assumes that an interrupted request did nothing. Autosave is bounded to 128 queued
messages and 32 MiB per saved draft, and reports write conflicts or storage errors.
It preserves newer saved input from another terminal instead of silently replacing it.
Messages whose recovery save fails remain in the composer or paused queue;
autosave retries transient storage errors. Exit also checks that the latest input
was saved, including edits made while the backend was closing. A failed save keeps
the interface open; resolve the storage error and retry exit. If the session has
already closed, `/reconnect` remains available.

## Session settings and approvals

`/model`, `/effort`, `/reasoning`, `/mode`, `/permissions` and `/sandbox` open session controls. Supply an argument to
select a configured model alias, `code` or `plan` mode, or one of the four
permission modes below. A confirmed model becomes the saved startup default and is also
used by new or reopened conversations in the same workspace. Reasoning effort,
mode and execution approval apply to the live session. Finish or stop foreground
and background work before changing settings.

`/sandbox` selects **Native** or **Python sandbox** for the current session;
`/sandbox on` and `/sandbox off` apply either choice directly. Enabling runs a
readiness check before confirming the microVM. Esc cancels preparation, and a
failure or cancellation keeps the previous execution environment. Finish or
stop background jobs first. Chat history and saved agent state remain available;
temporary Python objects are not migrated between environments. The setting
lasts for this live session; new or reconnected sessions use the startup flag.

This isolates generated Python cells only. Shell and file tools still execute
on the host, and permission settings remain separate from Python execution.

`/model` or Alt+P opens the model picker. Use Up/Down to choose a model and
Left/Right to preview its declared reasoning effort. Enter applies both in one
request; Esc discards the preview. Each model shows its own available levels,
including unknown or unsupported capability. The controls follow
[Claude Code's model and effort interaction](https://code.claude.com/docs/en/model-config#adjust-effort-level).
If validation or saving fails, the current model and saved default stay unchanged.
The picker also works before the first session when no default model is selected;
the alias is saved after successful session preparation, when effort controls
also become available.
Add or edit entries through [model configuration](../../docs/model-configuration.md).

`/effort` opens a horizontal slider at the current model's confirmed selection.
Use Left/Right to preview a level and Enter to apply it. `/effort status` reads
the current backend setting and available levels. Use `/effort LEVEL` (for
example, `/effort high` when `high` is listed) to select one directly, or
`/effort auto` to clear the selection and restore the
model's configured request defaults. The HUD changes after the backend accepts
the setting, and subsequent model calls send the declared provider parameters.
Automatic context budgets also follow any reply cap declared for that level.
Unknown routes offer only Model default and explain that selectable levels are
unknown; explicitly unsupported routes report that selection is unavailable.
An invalid choice preserves the current settings. Labels and their request
parameters belong in the model's `reasoning_levels` configuration; the terminal
does not infer them from a model name.
`/reasoning` remains an alias; `default` also clears the selection.

`/permissions` opens **Permissions · this session** with the current mode selected:

| Mode | Behavior |
|---|---|
| Auto | Permit generated code and file changes, including deletion, without asking. |
| Manual | Ask before executing generated code or making changes. |
| Accept edits | Automatically allow supported file reads and edits; ask before commands and other code. |
| Plan | Plan from supplied context with generated code and tool execution disabled. |

Use `/permissions auto`, `/permissions manual`, `/permissions accept_edits`, or
`/permissions plan` to select a mode directly. Choosing Auto, Manual, or Accept edits
also enters Code mode. The HUD updates after the backend confirms the selection.
These settings apply to the live session and reset when it closes or reconnects.
The earlier `/permissions ask` and `/permissions session` commands remain aliases
for the existing approval controls; `/mode code|plan` remains available.
If the backend does not report support for the new permission modes, the menu and
new commands leave permissions unchanged and ask you to update `aster-acp`.
The earlier approval commands and `/mode` remain usable with those backends.

With Manual, execution pauses for a compact review. Literal shell commands and
file operations show their arguments first; Tab opens the complete Python cell.
Dynamic or incomplete previews show full code immediately. PgUp/PgDn scroll the
preview while the choices remain visible. Choose Allow once, Allow all code for
this session, or Reject. The initial selection is Allow once; Enter confirms the
selected choice, and Esc rejects the request. Approval covers the entire Python
cell, including any files or processes it accesses. Session consent permits all
code until the session closes or reconnects.
Approval dialogs close when their task ends or the backend disconnects.
Accept edits uses the same review for commands, unsupported or computed file
operations, and code that combines edits with other execution. Approving the whole
session switches to Auto.

Ordinary conversation can return a written answer without executing code or
requesting approval. A greeting combined with a coding task still follows the
normal tool and approval flow. Native Windows shell commands use PowerShell.

Plan mode disables all generated Python and tool execution. It can reason from
conversation context and supplied file/image snapshots, and cannot inspect new
workspace files through tools. Native and sandbox execution keep their existing
backend behavior; choosing Plan does not create a separate read-only filesystem.

## Reviewing and undoing file changes

Open Git Changes with Ctrl+G, select a file, then press Ctrl+U. Choose all of its
unstaged hunks or one hunk, inspect the proposed patch, and select Apply. The
confirmation starts on Cancel. Foreground and background work must be stopped.

Rollback accepts existing tracked files with unstaged-only UTF-8 modifications,
up to 4 MiB with a complete patch no larger than 256 KiB. It excludes staged,
new, deleted, renamed, conflicted, binary, symlink and custom-filter/encoding
changes. Unselected hunks and the Git index are preserved. If the selected file
or index changes after review, refresh and prepare the rollback again.

Original bytes are saved in `.aster/tui-rollbacks/` before replacement. `/undo`
finds the latest available backup, including after restarting, and asks for
confirmation before restoring it. Undo refuses to overwrite edits made after
rollback. Conversation `/rewind` continues to restore context independently of
these file operations.

## Task progress

Alt+T or `/tasks` displays the agent's actual plan steps, current step, waiting
reason and background jobs with bounded output previews. Ctrl+R refreshes the
panel. Select a running job and press Ctrl+X, or use `/stop JOB-ID`, to stop that
specific background job while keeping the foreground request and other jobs
running. Concurrent jobs sharing a channel retain separate IDs and can be stopped
individually. Finished or removed job IDs cannot stop a newer job with the same name.

Plan steps appear when the agent records them; elapsed time alone does not imply
a completed step or an overall completion percentage. Python cards use a leading
purpose comment when supplied and retain their code, output and measured time.

## Behavior and scope

- Sessions use `<workspace>/.aster/sessions`, shared with other ACP clients. Resume
  replays messages and restores agent context. A session can have only one writer.
  New sessions use the first message as their title unless a title was already set.
  Existing unnamed sessions are not migrated.
- Resume restores saved Python calls, terminal output and file diffs in their
  original order, including failed and interrupted tools. Terminal/file events
  are recorded starting with this version; older sessions can only restore the
  events they actually saved. Restoring does not execute the tools again.
- `/changes` reads the Git working tree in the background, with separate staged,
  unstaged and untracked entries, searchable filenames, and a selected patch.
  Ctrl+R refreshes the snapshot; Tab switches to the conversation's saved edits.
  Opening and refreshing the view only inspect Git; Ctrl+U starts the reviewed
  rollback flow described above. Ignored files are excluded; binary files and
  truncated previews are labelled. Previews are bounded to 500 entries, 256 KiB
  each and 8 MiB total.
- Completion and Responses clients can show final-answer text as provider tool
  arguments arrive. The validated final reply replaces its preview in the same
  message. Cancellation marks the preview incomplete; unvalidated preview text
  is not saved as an answer. Ordinary written final answers use the separate
  `return_result` tool, with `message` first. The coding strategy rejects definite
  literal final answers embedded in Python before executing the cell and asks the
  model to use that reply tool. This also applies to sandbox execution. A rejected
  cell may require one correction request; an ordinary reply needs no extra call.
  Replies actually computed inside Python still arrive together without a model
  rewrite. The routing check recognizes direct literals and simple bindings in a
  cell; arbitrary expressions, branches, and standalone progress messages keep
  normal execution behavior. Providers configured with streaming disabled, or
  returning all arguments in one chunk, continue to send whole replies.
  Internal reasoning and Python source are not shown as answer text.
- Replies render headings, nested lists, quotes, links with visible destinations,
  fenced code and tables with Unicode column widths. Fenced code hides delimiter
  and language-title rows, preserves indentation and blank lines, and highlights
  recognized languages including Rust, Python, JavaScript, JSON and Bash.
  Unknown languages stay readable as plain text; `NO_COLOR` disables syntax colors.
  Durations display seconds with two decimal places. Python and terminal tools
  use backend execution timing, so buffered notifications do not collapse durations
  to zero. Completed-only events without timing omit the duration.
  The display redraws when
  events arrive and updates elapsed time several times per second while working.
- Backend stderr is drained independently and retained in bounded `/logs` output.
  Unexpected backend exits also print the recent diagnostic tail after restoring
  the terminal. Protocol stdout remains JSON-RPC only. Terminal control sequences
  are filtered.
- Closing cancels an active prompt, waits for its terminal result, closes the
  session, then shuts down and reaps the owned backend. Windows Job Objects also
  clean up descendants, including nested jobs. Linux cleanup targets the backend's
  dedicated process group; processes that deliberately detach from that group are
  outside its scope. Hung operations and shutdown have time bounds. Native
  execution is the default; sandbox remains opt-in using the existing Python implementation.

## Development and verification

```powershell
uv sync --extra tui
cargo test --locked --manifest-path packages/aster-tui/Cargo.toml
cargo clippy --locked --manifest-path packages/aster-tui/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path packages/aster-tui/Cargo.toml -- --check
uv run pytest packages/aster-cli/tests/test_tui_command.py
```

Rust tests include controller transitions, CJK/paste/draft retention, TestBackend
rendering at several terminal sizes, noisy/broken subprocesses, bounded cleanup,
attachment snapshots, approvals, runtime control isolation, targeted background
stops, recovery and file rollback. `tests/hud.rs` covers layered configuration,
undo, narrow layouts, input coexistence, accounting deduplication, concurrent
streams, missing prices and external snapshot freshness. `tests/session_controls.rs` exercises the public
controller and protocol boundary. Real Python ACP prompt/replay/cancel flows use
FakeLLMClient (no model calls).
They locate the repository `.venv`, or use `ASTER_TEST_PYTHON` when explicitly set.
`cargo run --manifest-path packages/aster-tui/Cargo.toml --example preview -- 120 36 work`
exports actual rendered cells as JSON for visual checks; modes include `welcome`,
`welcome-hud-full`, `sent-hud-full`, `starting`, `starting-multiple`, `starting-paused`,
`work`, `tools`, `commands`, `slash`, `changes`,
`sessions`, `search`, `queue`,
`markdown`, `sidebar`, `compose`, `compact`, `python-error`, `steering`, `files`,
`attachments`, `image`, `approval`, `approval-expanded`, `approval-fallback`,
`tasks`, `progress`, `recovery`, `rollback`, `context`, `context-all`, `context-pending`,
`context-system`, `context-messages`, `context-output`, `hud`, `hud-full`,
`hud-tokens`, `hud-compact`, `hud-zh`, `hud-monochrome`, `hud-unknown`, `hud-settings`,
`reasoning`, `reasoning-unknown`, `reasoning-unsupported`, `model-effort`,
`model-effort-unknown`, `permissions` and `permissions-plan`.
HUD previews use synthetic ACP and host data with the production renderer.
Model configuration previews use synthetic fields and credentials:
`model-switch` and `model-switch-empty`.
