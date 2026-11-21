# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Include the shared notices in source and wheel distributions."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        notice = Path(self.root) / "THIRD_PARTY_NOTICES.md"
        if not notice.is_file():
            # Checkouts share the repository notice; sdists carry their own copy.
            notice = Path(self.root).parents[1] / "THIRD_PARTY_NOTICES.md"
        if not notice.is_file():
            raise FileNotFoundError("THIRD_PARTY_NOTICES.md is required to build eval_pipeline")
        build_data["force_include"][str(notice)] = "THIRD_PARTY_NOTICES.md"
