# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make CLI tests independent of the developer's real provider keys.

CI has no provider keys, while a developer shell often has real ones. Several
offline tests leave ``--api-key-env`` blank, so the provider SDKs fall back to
their default variables. Give those SDKs fixed synthetic values (requests go to
mock transports, never to a provider) and remove the other provider keys, so
local runs and CI see the same environment. A test can still set or delete any
of these itself.
"""

import pytest

SYNTHETIC = {"OPENAI_API_KEY": "offline-test-only", "ANTHROPIC_API_KEY": "offline-test-only"}
REMOVED = (
    "NVIDIA_API_KEY",
    "NVIDIA_INFERENCE_API_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
)


@pytest.fixture(autouse=True)
def provider_keys_like_ci(monkeypatch):
    for name, value in SYNTHETIC.items():
        monkeypatch.setenv(name, value)
    for name in REMOVED:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _workspace_trust_isolation(monkeypatch):
    """Keep the folder-trust check from leaking between tests.

    Commands invoked in-process turn the check on for the whole process. These
    suites exercise other behaviour in temporary folders, so every folder is
    trusted here; tests of the check itself remove the variable.
    """
    from aster import workspace_trust

    monkeypatch.setenv("ASTER_TRUST_ALL_WORKSPACES", "1")
    with workspace_trust.enforced(workspace_trust.is_enforced()):
        yield
