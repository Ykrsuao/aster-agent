# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-declared checks for files changed during a coding turn."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path, PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aster.interactive import Done
from aster.strategy_validation import InvariantError
from aster.tools.shell_tools import ShellResult, ShellTools
from aster_cli.coding.activity import ActivityShellTools


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


class Verification:
    """Ephemeral results; a new turn or restored agent starts without old passes."""

    def __init__(self, root: Path, checks: dict[str, RequiredCheck]) -> None:
        self.root = root.resolve()
        self.checks = dict(checks)
        self._files = tuple(
            dict.fromkeys(path for check in checks.values() for path in check.files)
        )
        self._initial: dict[str, str | None] = {}
        self._passed: dict[str, dict[str, str | None]] = {}
        self._attempts: dict[str, object] = {}
        self._turn_id: str | None = None
        self._requested = False
        self._snapshot()

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
        self._passed.clear()
        self._attempts.clear()
        self._requested = False
        self._initial = self._snapshot()

    async def run(self, name: str, events: Any) -> ShellResult:
        if name not in self.checks:
            raise ValueError(f"Unknown check {name!r}; choose from {list(self.checks)}")
        if self._turn_id is None:
            raise RuntimeError("Run checks during a coding handle turn")
        self._requested = True
        self._passed.pop(name, None)
        attempt = self._attempts[name] = object()
        turn_id = self._turn_id
        before = self._snapshot()
        check = self.checks[name]
        # A fresh shell cannot inherit aliases, functions or cwd from ordinary work cells.
        shell = ShellTools(cwd=str(self.root), default_timeout=check.timeout)
        try:
            result = await ActivityShellTools(shell, events).run(
                check.command, timeout=check.timeout
            )
        finally:
            await shell.close()
        after = self._snapshot()
        if (
            self._turn_id == turn_id
            and self._attempts.get(name) is attempt
            and result.returncode == 0
            and not result.timed_out
            and before == after
        ):
            self._passed[name] = after
        return result

    def require_passes(self) -> None:
        if not self.checks:
            return
        try:
            current = self._snapshot()
        except (OSError, ValueError) as error:
            raise InvariantError("Cannot verify the configured workspace files") from error
        if not self._requested and current == self._initial:
            return
        missing = [name for name in self.checks if self._passed.get(name) != current]
        if missing:
            raise InvariantError(
                f"Required checks are missing, failed, or stale: {', '.join(missing)}. "
                "Run await self.verify(check_name) after your final edits, inspect the result, "
                "and fix failures before returning Done. evidence text does not replace checks."
            )


def begin_verification(agent: Any, call: Any) -> None:
    if call.method_name == "handle" and (checks := getattr(agent, "_verification", None)):
        checks.begin(call.id)


def require_verification(agent: Any, result: Any, call: Any) -> None:
    if (
        call.method_name == "handle"
        and isinstance(result, Done)
        and (checks := getattr(agent, "_verification", None))
    ):
        checks.require_passes()
