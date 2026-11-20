# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optionally bundle a prebuilt Windows terminal in a platform wheel."""

import os
import struct
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        source = os.environ.get("ASTER_BUNDLE_TUI")
        if not source:
            return
        if version == "editable":
            raise ValueError("ASTER_BUNDLE_TUI is for release wheels, not editable installs")
        binary = Path(source).resolve(strict=True)
        with binary.open("rb") as stream:
            header = stream.read(64)
            if len(header) != 64 or header[:2] != b"MZ":
                raise ValueError("ASTER_BUNDLE_TUI must be a Windows x64 PE executable")
            stream.seek(struct.unpack_from("<I", header, 60)[0])
            if stream.read(6) != b"PE\0\0\x64\x86":
                raise ValueError("ASTER_BUNDLE_TUI must be a Windows x64 PE executable")
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-win_amd64"
        build_data["force_include"][str(binary)] = "aster_cli/bin/aster.exe"
