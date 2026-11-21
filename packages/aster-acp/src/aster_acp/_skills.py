# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only skill metadata; browsing never imports or activates a capability."""

import inspect
from typing import Any

from aster.skill import TextSkill
from aster.skill_registry import SkillRegistry

_MAX_SKILLS = 2000
_MAX_TEXT_CHARS = 65_536
_MAX_DESCRIPTION_CHARS = 512


def skills_snapshot(registry: SkillRegistry, name: str | None = None) -> dict[str, Any]:
    names = registry.discovered()
    if name is not None and name not in names:
        raise KeyError(name)
    loaded = set(registry.loaded())
    active = set(registry.activated())

    def item(skill_name: str, *, detail: bool = False) -> dict[str, Any]:
        skill = registry[skill_name] if skill_name in loaded else None
        kind = "mcp" if skill_name.startswith("mcp.") else "python"
        if isinstance(skill, TextSkill):
            kind = "text"
        raw_text = type(skill).__doc__ if skill is not None else None
        if not isinstance(raw_text, str) or not raw_text.strip():
            raw_text = (
                "No documentation was provided for this skill."
                if skill_name in loaded
                else "Not loaded. Documentation is unavailable until this skill is loaded."
            )
        text = inspect.cleandoc(raw_text[: _MAX_TEXT_CHARS + 1])
        description = text.split("\n", 1)[0][:_MAX_DESCRIPTION_CHARS]
        result: dict[str, Any] = {
            "name": skill_name,
            "description": description,
            "active": skill_name in active,
            "loaded": skill_name in loaded,
            "kind": kind,
        }
        if detail:
            if skill is None:
                entry = registry.entry(skill_name)
                target = getattr(entry.entry_point, "value", None) if entry is not None else None
                if isinstance(target, str) and target:
                    text += f"\n\nImport target: {target}"
            result.update(
                text=text[:_MAX_TEXT_CHARS],
                truncated=len(raw_text) > _MAX_TEXT_CHARS or len(text) > _MAX_TEXT_CHARS,
            )
        return result

    if name is not None:
        return {"skill": item(name, detail=True)}
    return {
        "skills": [item(skill_name) for skill_name in names[:_MAX_SKILLS]],
        "total": len(names),
        "truncated": len(names) > _MAX_SKILLS,
    }
