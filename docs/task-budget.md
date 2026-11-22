# Task budgets

Coding sessions can share one allowance across prompts, child agents, model
changes and history summaries. Enable it explicitly in the usual settings
layers, for example a trusted project's `.aster/settings.yaml`:

```yaml
coding:
  task_budget:
    calls: 100
    tokens: 100000
    time_seconds: 1800
```

Each limit is optional. Values must be finite, nonnegative numbers; calls and
tokens must be integers. Zero prevents the corresponding work from starting.
Omitting `coding.task_budget` leaves a new session's existing behavior unchanged.

Use `/task-budget` to inspect the task ID, consumed calls, observed tokens,
active time, cost and stop reason. `/task-budget new` explicitly creates a new
task using the current settings. It refuses while task work or a provider
attempt is still active. The command is a host action, not an agent tool.
It does not clear other host stops, such as unresolved execution recovery.

The task ID is independent of a prompt's turn ID. Follow-ups, `Done`, `NeedInput`,
`Waiting`, model switches and reopening the same session retain the original
allowance. Changing settings affects a new task, never an existing allowance.
Python skill commands share the same task, including model calls or background
jobs started directly by a command. Budget inspection and new-task commands
remain available after the task stops. The host's `/recovery` command also
remains available to inspect or record a human review of interrupted actions;
reviewing an action does not refund or restart the task allowance. User skill
commands, including a skill that uses the same name, remain budgeted.

## Accounting

- **Calls** count each ASTER provider attempt, including framework retries.
  The host commits the reservation after admission succeeds and before it
  dispatches the request. Queue rejection, queue timeout and queue cancellation
  do not spend a call. A storage failure stops dispatch. SDK retry passthrough
  (`max_retries` or `num_retries` greater than zero) is rejected while a budget
  is bound; use `RetryConfig` for counted retries. This is accounting at the
  framework provider boundary, not a claim to measure every underlying HTTP
  exchange performed by an arbitrary transport.
- **Tokens** are reported input plus output tokens. Cache and reasoning counters
  are subsets and are not added again. Usage is settled after the response,
  including empty-content attempts that will be retried. A response can exceed
  the remaining allowance; this is not a provider token hard cap. Subsequent
  dispatch stops at the limit. Missing, malformed or interrupted usage remains
  unknown. With a token limit, any unknown attempt blocks further dispatch,
  including a retry whose previous failure supplied no usage. Concurrent
  token-limited requests wait for outstanding usage to settle before dispatch.
- **Cost** remains unknown when any attempt lacks a valid price. An explicitly
  reported zero is known; it is not interchangeable with missing cost.
- **Time** is cumulative active task wall time, measured as the union of active
  intervals. Queue waits, retries, tools, summaries and active background jobs
  count; concurrent children do not multiply time. Waiting for the next user
  prompt and a normally closed session do not count. Host deadlines cancel
  asynchronous work and await cleanup, including managed shell commands.
  Blocking synchronous Python still requires its own execution or provider
  timeout: an asyncio deadline cannot preempt code that blocks its event loop.

The actual Completion and Responses clients participate in both synchronous
and asynchronous calls. With a finite time allowance, synchronous provider
dispatch is refused before spending a call; use `acall()` so the host can cancel
it. Calls and token allowances remain supported for synchronous dispatch when
no time limit is configured. Different clients used by nested agents inherit the
same owner through the task context. Budget control errors are terminal for
provider retries, CodeAct and history compaction. They are displayed by ACP as
a stopped turn, with the ledger preserved for inspection.

## Persistence and recovery

SQLite stores the ledger in `task_budgets`, separately from conversation
snapshots. It contains limits, counters, attempt IDs and time state, not model
credentials. Old task records remain when a new task is started.

After an interrupted process, unfinished attempts retain their spent calls and
become unknown usage. An unfinished active interval is conservatively charged
through recovery, including downtime; restarting cannot refund it. A normal
host close cancels and settles local outstanding provider tasks before closing
the store. If a clock rollback makes interrupted time unknown, recovery stops
the task. In-memory agents retain accounting only for their process lifetime.

If a budget stops a Python cell after it has already changed something, its
action remains pending for human recovery review even after saving the session.
Starting a new allowance does not clear that separate recovery requirement.

Rewind and session forks copy the **latest** consumed ledger, even when the
selected conversation checkpoint is older. The copied task is stopped with
`forked_task`; the user must explicitly run `/task-budget new` to proceed with a
new allowance. Restoring a chat snapshot alone cannot change the ledger.

## Process boundaries

The owner is a host runtime object. Sandbox calls through `self.*` execute on
that host and share its accounting. A budgeted sandbox rejects guest-created
model clients. Exporting model clients or admission-broker configurations while
a task budget is bound also fails explicitly, even when those objects were
created before the task. An inherited owner refuses dispatch after a process
fork. The broker protocol does not yet settle usage for independent workers.

Keep model clients on the owning host and invoke host agent methods. Budget
support is opt-in at the framework boundary; arbitrary user programs that call
a provider SDK or endpoint directly are not metered by this mechanism.

For library workflows, bind one owner around the whole active operation:

```python
from aster.task_budget import TaskBudget, TaskBudgetLimits, task_budget_activity

budget = TaskBudget(TaskBudgetLimits(calls=20, tokens=50000, time_seconds=300))
async with task_budget_activity(budget):
    result = await agent.run()
```

That example is in-memory. A persistent host should use
`restore_task_budget(storage, limits)` with the session's SQLite storage, and
close outstanding provider work with `await budget.aclose()` before closing
the store. `task_budget_scope(budget)` supplies a synchronous context for
counting and binding; use the async activity scope for a cancellable workflow
deadline.
