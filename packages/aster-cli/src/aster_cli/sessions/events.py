# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable metadata for interactive coding-agent sessions."""

from typing import Any, ClassVar, Literal

from pydantic import Field

from aster.context_blocks import Metadata
from aster.context_blocks.roles import Role

type SteeringState = Literal["accepted", "applied", "cancelled", "interrupted", "turn_finished"]


class SessionStarted(Metadata):
    """Identity and environment recorded once when a session is created."""

    _role: ClassVar[Role] = Role.METADATA

    origin: str = ""
    model: str = ""
    agent: str = ""
    working_directory: str = ""
    provisional: bool = False


class SessionTitleUpdated(Metadata):
    """The latest human- or agent-selected session title."""

    _role: ClassVar[Role] = Role.METADATA

    title: str = ""
    user_set: bool = False


class SessionUserMessage(Metadata):
    """Raw user text accepted by the agent runtime as a conversation turn."""

    _role: ClassVar[Role] = Role.METADATA

    content: str = ""
    turn_id: str | None = None


class SessionTurnFinished(Metadata):
    """Bind a foreground turn to its completed reply, or mark it incomplete."""

    user_message_id: str
    completed_message_id: str | None


class SessionSnapshotInvalidated(Metadata):
    """Bind an interrupted turn to its own branch's last pre-turn state."""

    session_id: str = Field(min_length=1)
    checkpoint_id: str | None = Field(default=None, pattern=r"^user:\d+$")
    checkpoint_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SessionSteering(Metadata):
    """A durable state change for input addressed to an existing foreground turn."""

    message_id: str
    turn_id: str
    text: str
    state: SteeringState
    reason: str | None = None


class SessionActivity(Metadata):
    """A host-visible tool activity snapshot, excluded from model context."""

    activity: dict[str, Any]


class SessionActionIntent(Metadata):
    """A host side effect whose intent was committed before execution."""

    kind: str = Field(min_length=1, max_length=64)
    detail: str = Field(min_length=1, max_length=1024)
    cwd: str = Field(min_length=1, max_length=4096)


class SessionActionOutcome(Metadata):
    """A bounded execution outcome, independent of saving agent progress."""

    action_id: str = Field(min_length=1, max_length=256)
    outcome: str = Field(min_length=1, max_length=256)


class SessionActionsSaved(Metadata):
    """Actions included in a successfully saved current agent snapshot."""

    action_ids: list[str]


class SessionActionReviewed(Metadata):
    """A human's explicit reconciliation of an uncertain host side effect."""

    action_id: str = Field(min_length=1, max_length=256)
    note: str = Field(min_length=1, max_length=1024)


class SessionActionsInherited(Metadata):
    """Retained side effects need a fresh review against the restored state."""

    action_ids: list[str]


SESSION_ACTION_EVENT_TYPES: tuple[type[Metadata], ...] = (
    SessionActionIntent,
    SessionActionOutcome,
    SessionActionsSaved,
    SessionActionReviewed,
    SessionActionsInherited,
)


SESSION_EVENT_TYPES: tuple[type[Metadata], ...] = (
    SessionStarted,
    SessionTitleUpdated,
    SessionUserMessage,
    SessionTurnFinished,
    SessionSnapshotInvalidated,
    SessionSteering,
    SessionActivity,
    *SESSION_ACTION_EVENT_TYPES,
)
