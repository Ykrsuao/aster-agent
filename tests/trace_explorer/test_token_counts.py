# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A missing usage dimension must survive trace parsing as unknown."""

import pytest

from aster.trace_explorer.explorer import _extract_token_counts


def test_partial_usage_does_not_manufacture_zero_counts():
    counts = _extract_token_counts({"llm.token_count.prompt": 0})
    assert counts == {"prompt": 0}
    assert _extract_token_counts({}) is None


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_invalid_usage_is_omitted_without_changing_a_reported_zero(value):
    assert _extract_token_counts(
        {
            "llm.token_count.prompt": value,
            "llm.token_count.completion": 0,
        }
    ) == {"completion": 0}
