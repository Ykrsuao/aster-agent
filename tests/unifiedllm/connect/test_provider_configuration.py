# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider discovery, native routing and local model/credential transactions."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import litellm
import pytest
import yaml

from aster.unifiedllm.connect import configuration
from aster.unifiedllm.connect.providers import provider_details, providers
from aster.unifiedllm.registry import client_from_config


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("ASTER_USER_DIR", str(user))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.delenv("ASTER_LLM_CONFIG", raising=False)
    monkeypatch.delenv("ASTER_SECRETS", raising=False)
    original_env = dict(os.environ)
    yield user
    for key in set(os.environ) - set(original_env):
        if key.startswith("ASTER_"):
            os.environ.pop(key, None)
    from aster.unifiedllm.registry import reload_registry

    reload_registry()


def values(provider, **overrides):
    result = {item["id"]: item["default"] for item in configuration.form(provider)["fields"]}
    return {**result, **overrides}


def test_catalogue_contains_every_installed_provider_and_marks_nonchat_services():
    rows = {row["id"]: row for row in providers()}
    assert set(litellm.provider_list) <= set(rows)
    assert "official" in rows["deepseek"]["name"].lower()
    assert rows["bedrock"]["chat"] is True
    assert rows["elevenlabs"]["chat"] is False
    assert "custom-endpoint" in rows


def test_missing_onboarding_metadata_does_not_hide_new_providers(monkeypatch):
    monkeypatch.setattr(litellm, "provider_list", [*litellm.provider_list, "future_provider"])
    row = configuration.form("future_provider")
    assert row["id"] == "future_provider"
    assert {field["id"] for field in row["fields"]} >= {"model", "api_key", "alias"}


def test_deepseek_suggestions_label_current_models_and_deduplicate_native_prefixes(monkeypatch):
    monkeypatch.setitem(
        litellm.models_by_provider,
        "deepseek",
        [
            "deepseek/deepseek-flash",
            "deepseek-flash",
            "deepseek/deepseek-v4-pro",
            "deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
        ],
    )
    result = provider_details("deepseek")
    assert result["models"] == ["deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash"]
    assert "current" in result["modelLabels"]["deepseek-flash"].lower()
    assert "legacy" in result["modelLabels"]["deepseek-v4-flash"].lower()
    assert "LiteLLM" in result["modelsSource"]
    assert "deepseek-flash" in result["modelHint"]


def test_form_reuses_official_key_reference_without_exposing_values(isolated, monkeypatch):
    (isolated / "llm_config.yaml").write_text(
        "models:\n  existing:\n    model_name: deepseek/deepseek-flash\n"
        "    api_base: https://api.deepseek.com\n    api_key_env: PRIVATE_EXISTING_KEY\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PRIVATE_EXISTING_KEY", "synthetic-secret-value")
    result = configuration.form("deepseek")
    key = next(field for field in result["fields"] if field["id"] == "api_key")
    assert key["default"] == "env:PRIVATE_EXISTING_KEY"
    assert "synthetic-secret-value" not in json.dumps(result)


def test_form_groups_optional_overrides_as_advanced_without_losing_save_defaults(isolated):
    fields = configuration.form("deepseek")["fields"]
    assert {field["id"] for field in fields if field.get("advanced")} == {
        "context_window",
        "max_tokens",
        "additional_drop_params",
    }
    defaults = {field["id"]: field["default"] for field in fields}
    _, entry, _ = configuration.prepare(
        "deepseek", {**defaults, "model": "deepseek-flash", "alias": "basic-setup"}
    )
    assert entry["max_tokens"] == 4096
    assert "context_window" not in entry
    assert "additional_drop_params" not in entry


def test_save_native_deepseek_and_reload_without_touching_another_alias(isolated, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Saving a model must not make inference requests")

    monkeypatch.setattr(litellm, "acompletion", no_network)
    path = isolated / "llm_config.yaml"
    original = "# retained\nmodels:\n  other:\n    model_name: openai/other\n"
    path.write_text(original, encoding="utf-8")
    result = configuration.save(
        "deepseek",
        values(
            "deepseek",
            model="deepseek-flash",
            alias="official",
            api_key="synthetic-secret",
            additional_drop_params="response_format",
            context_window="1048576",
        ),
    )
    text = path.read_text(encoding="utf-8")
    assert original.rstrip() in text
    assert "synthetic-secret" not in text
    entry = yaml.safe_load(text)["models"]["official"]
    assert entry["model_name"] == "deepseek/deepseek-flash"
    assert entry["additional_drop_params"] == ["response_format"]
    assert result["tested"] is False
    from aster.unifiedllm.registry import MODELS

    assert MODELS["official"] == entry
    assert os.environ[entry["api_key_env"]] == "synthetic-secret"


@pytest.mark.asyncio
async def test_cloud_fields_reach_the_native_client_and_keys_stay_separate(isolated):
    result = configuration.save(
        "bedrock",
        values(
            "bedrock",
            model="anthropic.claude-test-v1:0",
            alias="aws-test",
            aws_access_key_id="synthetic-access",
            aws_secret_access_key="synthetic-secret",
            aws_region_name="us-east-1",
            api_key="",
        ),
    )
    entry = yaml.safe_load((isolated / "llm_config.yaml").read_text(encoding="utf-8"))["models"][
        result["alias"]
    ]
    assert len(entry["provider_credentials"]) == 2
    assert "synthetic" not in json.dumps(entry)
    client = client_from_config("aws-test", entry)
    try:
        assert client.model.startswith("bedrock/")
        assert client.config["aws_region_name"] == "us-east-1"
        assert client.config["aws_access_key_id"] == "synthetic-access"
        assert client.config["aws_secret_access_key"] == "synthetic-secret"
    finally:
        await client.aclose()


async def test_oci_private_key_is_masked_and_saved_as_a_credential(isolated: Path) -> None:
    private_key = "-----BEGIN PRIVATE KEY-----\nsynthetic-pem\n-----END PRIVATE KEY-----"
    result = configuration.save(
        "oci",
        values(
            "oci",
            model="meta.test",
            alias="oci-test",
            oci_user="synthetic-user",
            oci_fingerprint="synthetic-fingerprint",
            oci_tenancy="synthetic-tenancy",
            oci_region="us-ashburn-1",
            oci_compartment_id="synthetic-compartment",
            oci_key=private_key,
        ),
    )
    entry = yaml.safe_load((isolated / "llm_config.yaml").read_text(encoding="utf-8"))["models"][
        result["alias"]
    ]
    assert "oci_key" not in entry.get("provider_params", {})
    reference = next(item for item in entry["provider_credentials"] if item["param"] == "oci_key")
    stored = yaml.safe_load((isolated / "secrets.yaml").read_text(encoding="utf-8"))
    assert stored["env"][reference["env"]] == private_key
    assert private_key not in json.dumps(entry)
    fields = {item["id"]: item for item in configuration.form("oci")["fields"]}
    assert fields["oci_key"]["secret"] is True
    assert fields["oci_key"]["default"] == ""
    assert fields["oci_fingerprint"]["secret"] is False
    assert fields["oci_region"]["secret"] is False
    client = client_from_config(result["alias"], entry)
    try:
        assert client.config["oci_key"] == private_key
    finally:
        await client.aclose()


def test_unicode_aliases_save_with_independent_credentials(isolated: Path) -> None:
    references = []
    for alias, secret in [
        ("深度求索", "synthetic-first-key"),
        ("日常/推理", "synthetic-second-key"),
    ]:
        result = configuration.save(
            "deepseek",
            values("deepseek", model="deepseek-flash", alias=alias, api_key=secret),
        )
        assert result["alias"] == alias
        entries = yaml.safe_load((isolated / "llm_config.yaml").read_text(encoding="utf-8"))[
            "models"
        ]
        reference = entries[alias]["api_key_env"]
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", reference)
        assert alias not in reference
        assert os.environ[reference] == secret
        references.append(reference)
    assert references[0] != references[1]
    stored = yaml.safe_load((isolated / "secrets.yaml").read_text(encoding="utf-8"))
    assert stored["env"][references[0]] == "synthetic-first-key"
    assert stored["env"][references[1]] == "synthetic-second-key"
    assert {item["id"] for item in configuration.saved_models()} >= {"深度求索", "日常/推理"}


def test_alias_collision_rolls_back_new_credentials(isolated, monkeypatch):
    import aster.unifiedllm.connect as connect

    def conflict(*args, **kwargs):
        raise ValueError("Concurrent alias change")

    monkeypatch.setattr(connect, "write", conflict)
    with pytest.raises(ValueError, match="rolled back"):
        configuration.save(
            "deepseek",
            values(
                "deepseek",
                model="model",
                alias="rollback",
                api_key="synthetic-new-secret",
            ),
        )
    assert not (isolated / "secrets.yaml").exists()
    assert not (isolated / "llm_config.yaml").exists()


def test_invalid_limits_and_nonchat_providers_do_not_write_files(isolated):
    with pytest.raises(ValueError, match="smaller"):
        configuration.save(
            "deepseek",
            values(
                "deepseek",
                model="model",
                alias="invalid",
                context_window="1000",
                max_tokens="4096",
            ),
        )
    with pytest.raises(ValueError, match="chat endpoint"):
        configuration.save("elevenlabs", {"alias": "speech", "model": "speech"})
    assert not (isolated / "llm_config.yaml").exists()
