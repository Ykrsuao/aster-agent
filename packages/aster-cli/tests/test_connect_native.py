# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import litellm
import pytest
import yaml
from aster_cli.commands.connect import command
from click.testing import CliRunner


@pytest.fixture
def local_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.delenv("ASTER_LLM_CONFIG", raising=False)
    monkeypatch.delenv("ASTER_SECRETS", raising=False)
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    return tmp_path / "user" / "llm_config.yaml"


def test_native_provider_flags_save_the_exact_route_without_inference(local_config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Native setup must not make a model call")

    monkeypatch.setattr(litellm, "acompletion", forbidden)
    result = CliRunner().invoke(
        command,
        [
            "deepseek-flash",
            "--provider",
            "deepseek",
            "--as",
            "official",
            "--no-probe",
            "--yes",
            "--api-key-env",
            "TEST_PROVIDER_KEY",
            "--output",
            str(local_config),
        ],
    )
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(local_config.read_text(encoding="utf-8"))["models"]["official"]
    assert entry["model_name"] == "deepseek/deepseek-flash"
    assert entry["api_key_env"] == "TEST_PROVIDER_KEY"
    assert entry["api_base"] == "https://api.deepseek.com"
    assert "synthetic-provider-key" not in result.output


def test_interactive_provider_menu_contains_all_litellm_providers(local_config, monkeypatch):
    from aster_cli.commands import _connect_native, _connect_prompts

    captured = {}

    def prompt(text, **kwargs):
        if text == "Choose a provider":
            captured.update(kwargs)
            return "deepseek"
        pytest.fail(f"Unexpected prompt {text}")

    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
    monkeypatch.setattr(_connect_native, "run", lambda state: None)
    result = CliRunner().invoke(command, ["--no-probe", "--output", str(local_config)])
    assert result.exit_code == 0, result.output
    assert {
        "litellm:custom" if name == "custom" else name for name in litellm.provider_list
    } <= set(captured["choices"])
    assert captured["searchable"] is True


def test_native_route_does_not_ignore_an_incompatible_interface(local_config):
    result = CliRunner().invoke(
        command,
        [
            "deepseek-flash",
            "--provider",
            "deepseek",
            "--as",
            "official",
            "--no-probe",
            "--yes",
            "--api-style",
            "responses",
            "--output",
            str(local_config),
        ],
    )
    assert result.exit_code != 0
    assert not local_config.exists()
