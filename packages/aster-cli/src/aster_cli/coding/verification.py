# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-declared task acceptance that survives turns and session snapshots."""

from __future__ import annotations

import hashlib
import stat
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from aster.interactive import Done
from aster.strategy_validation import InvariantError
from aster.tools.shell_tools import ShellResult, ShellTools
from aster_cli.coding.activity import ActivityShellTools


class CodingDone(Done):
    """Finish a reply, optionally leaving an earlier coding task pending."""

    model_config = ConfigDict(from_attributes=True)

    # The host can append a notice without losing the validated reply's stream identity.
    _original_message: str | None = PrivateAttr(default=None)

    complete_task: bool = Field(
        default=True,
        strict=True,
        description=(
            "Complete the pending coding task after required checks. Use false only for an "
            "unrelated conversational or planning answer; it retains the pending task and "
            "cannot bypass checks for files changed or checks requested in this turn."
        ),
    )


class RequiredCheck(BaseModel):
    """One trusted host command and the explicit files its result covers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str = Field(min_length=1)
    files: tuple[str, ...] = Field(min_length=1)
    timeout: float = Field(default=60, gt=0, allow_inf_nan=False)

    @field_validator("command")
    @classmethod
    def command_not_blank(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("check command must be nonblank and contain no NUL")
        return value

    @field_validator("timeout", mode="before")
    @classmethod
    def numeric_timeout(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("check timeout must be a positive finite number")
        return value

    @field_validator("files")
    @classmethod
    def relative_files(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            path = Path(value)
            windows = PureWindowsPath(value)
            if (
                not value.strip()
                or "\x00" in value
                or any(character in value for character in "*?[")
                or path.is_absolute()
                or windows.drive
                or windows.root
                or ".." in path.parts
                or ".." in windows.parts
            ):
                raise ValueError("check files must be relative paths inside the workspace")
        return values


class VerificationState(BaseModel):
    """Snapshot data only: commands always come from the current host settings."""

    model_config = ConfigDict(extra="forbid")

    workspace: str
    checks: dict[str, RequiredCheck]
    baseline: dict[str, str | None]
    task_id: str | None = None
    pending: bool = False
    passed: dict[str, dict[str, str | None]] = Field(default_factory=dict)
    outcomes: dict[str, Literal["running", "passed", "failed", "interrupted"]] = Field(
        default_factory=dict
    )


class Verification:
    """Keep task evidence while rebuilding shells and attempt ownership each turn."""

    def __init__(self, root: Path, checks: dict[str, RequiredCheck]) -> None:
        self.root = root.resolve()
        self.checks = dict(checks)
        self._files = tuple(
            dict.fromkeys(path for check in checks.values() for path in check.files)
        )
        self.state = VerificationState(
            workspace=str(self.root), checks=self.checks, baseline=self._snapshot()
        )
        self._initial = dict(self.state.baseline)
        self._attempts: dict[str, object] = {}
        self._turn_id: str | None = None
        self._requested = False

    def configure(
        self, checks: dict[str, RequiredCheck], *, state: VerificationState | None = None
    ) -> None:
        """Rebind restored data; a changed workspace or contract invalidates evidence."""
        if state is not None:
            self.state = state
        self.checks = dict(checks)
        self._files = tuple(
            dict.fromkeys(path for check in checks.values() for path in check.files)
        )
        if self.state.workspace != str(self.root) or self.state.checks != self.checks:
            current = self._snapshot()
            self.state.baseline = {
                name: self.state.baseline.get(name, digest) for name, digest in current.items()
            }
            self.state.passed.clear()
            self.state.outcomes.clear()
            self._attempts.clear()
            if self.state.task_id is not None:
                self._activate()
            self.state.workspace = str(self.root)
            self.state.checks = dict(self.checks)

    def _activate(self) -> None:
        if not self.state.pending:
            self.state.task_id = str(uuid.uuid4())
        self.state.pending = True

    def observe(self) -> dict[str, str | None]:
        """Capture pending changes without running commands or accepting the task."""
        current = self._snapshot()
        if current != self.state.baseline:
            self._activate()
        return current

    def status(self) -> dict[str, Any]:
        current = self.observe()
        checks = {
            name: "passed"
            if self.state.passed.get(name) == current
            else "stale"
            if name in self.state.passed
            else self.state.outcomes.get(name, "not_run")
            for name in self.checks
        }
        return {
            "task_id": self.state.task_id,
            "state": "disabled"
            if not self.checks
            else "pending"
            if self.state.pending
            else "verified"
            if self.state.task_id is not None
            else "idle",
            "changed_files": [
                name for name, digest in current.items() if digest != self.state.baseline.get(name)
            ],
            "checks": checks,
        }

    def _snapshot(self) -> dict[str, str | None]:
        state = {}
        for name in self._files:
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"Check file leaves the workspace: {name}")
            try:
                mode = path.stat().st_mode
            except FileNotFoundError:
                state[name] = None
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(f"Check path is not a regular file: {name}")
            with path.open("rb") as stream:
                state[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        return state

    def begin(self, turn_id: str) -> None:
        self._turn_id = turn_id
        self._attempts.clear()
        self._requested = False
        for name, outcome in self.state.outcomes.items():
            if outcome == "running":
                self.state.outcomes[name] = "interrupted"
        self._initial = self.observe()

    async def run(self, name: str, events: Any, *, journal: Any = None) -> ShellResult:
        if name not in self.checks:
            raise ValueError(f"Unknown check {name!r}; choose from {list(self.checks)}")
        if self._turn_id is None:
            raise RuntimeError("Run checks during a coding handle turn")
        self._requested = True
        self._activate()
        self.state.passed.pop(name, None)
        self.state.outcomes[name] = "running"
        attempt = self._attempts[name] = object()
        turn_id = self._turn_id
        before = self._snapshot()
        check = self.checks[name]
        # A fresh shell cannot inherit aliases, functions or cwd from ordinary work cells.
        shell = ShellTools(cwd=str(self.root), default_timeout=check.timeout)
        try:
            try:
                result = await ActivityShellTools(shell, events, journal=journal).run(
                    check.command, timeout=check.timeout
                )
            finally:
                await shell.close()
        except BaseException:
            if self._turn_id == turn_id and self._attempts.get(name) is attempt:
                self.state.outcomes[name] = "interrupted"
            raise
        after = self._snapshot()
        if self._turn_id == turn_id and self._attempts.get(name) is attempt:
            self.state.outcomes[name] = "failed"
            if result.returncode == 0 and not result.timed_out and before == after:
                self.state.passed[name] = after
                self.state.outcomes[name] = "passed"
        return result

    def require_passes(self, *, complete_task: bool = True) -> bool:
        """Accept verified changes, or retain a task while answering unrelated input."""
        if not self.checks:
            return False
        try:
            current = self.observe()
        except (OSError, ValueError) as error:
            raise InvariantError("Cannot verify the configured workspace files") from error
        if not self.state.pending:
            return False
        if not complete_task:
            if self._requested or current != self._initial:
                raise InvariantError(
                    "This turn changed covered files or requested checks. Complete verification "
                    "before returning Done; complete_task=false cannot bypass current work."
                )
            return True
        missing = [name for name in self.checks if self.state.passed.get(name) != current]
        if missing:
            raise InvariantError(
                f"Required checks are missing, failed, or stale: {', '.join(missing)}. "
                "Run await self.verify(check_name) after your final edits, inspect the result, "
                "and fix failures before returning Done. Evidence text does not replace checks. "
                "For an unrelated answer without new edits, use complete_task=false to keep "
                "the earlier task pending."
            )
        self.state.baseline = current
        self.state.pending = False
        return False


def begin_verification(agent: Any, call: Any) -> None:
    if call.method_name == "handle" and (checks := getattr(agent, "_verification", None)):
        agent.refresh_host_context()
        checks.begin(call.id)


def require_verification(agent: Any, result: Any, call: Any) -> None:
    if call.method_name != "handle" or not (checks := getattr(agent, "_verification", None)):
        return
    if not isinstance(result, Done):
        checks.observe()
    else:
        agent.refresh_host_context()
        if checks.require_passes(complete_task=getattr(result, "complete_task", True)):
            missing = [
                name for name, status in checks.status()["checks"].items() if status != "passed"
            ]
            notice = (
                f"The coding task still needs verification: {', '.join(missing)}."
                if missing
                else "The coding task remains open; its required checks have passed."
            )
            original = result.message or result.explanation
            if isinstance(result, CodingDone) and result._original_message is None:
                result._original_message = original
            result.message = f"{original}\n\n{notice}"
