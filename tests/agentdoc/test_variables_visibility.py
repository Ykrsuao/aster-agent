# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The variables helper respects each instance's visibility settings."""

from typing import Annotated

import pytest

from aster.agentdoc import hidden, spec
from aster.agentdoc.introspect import variables


@pytest.mark.parametrize("instance_value", [False, True])
def test_variables_honors_instance_hidden_for_instance_and_class_fields(instance_value):
    class Settings:
        token = "class-marker"
        public = "visible-marker"

    obj = Settings()
    if instance_value:
        obj.token = "instance-marker"
    spec(obj, "token", hidden=True)

    result = variables(obj)

    assert "token" not in result
    assert "class-marker" not in result
    assert "instance-marker" not in result
    assert "visible-marker" in result
    assert "class-marker" in variables(Settings())


def test_variables_keeps_annotated_hidden_and_honors_instance_opt_in():
    class Settings:
        token: Annotated[str, hidden] = "annotated-marker"

    obj = Settings()
    assert "annotated-marker" not in variables(obj)

    spec(obj, "token", hidden=False)
    assert "annotated-marker" in variables(obj)
    assert "annotated-marker" not in variables(Settings())
