# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve reported token subtotals without treating missing usage as zero."""

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

TOKEN_DIMENSIONS = ("input_tokens", "output_tokens", "total_tokens")


class TokenUsageSummary(BaseModel):
    """Known counts and completeness for observed turns (or legacy result records)."""

    records: int = Field(default=0, strict=True, ge=0)
    known_input_tokens: int = Field(default=0, strict=True, ge=0)
    known_output_tokens: int = Field(default=0, strict=True, ge=0)
    known_total_tokens: int = Field(default=0, strict=True, ge=0)
    input_tokens_complete: bool = Field(default=False, strict=True)
    output_tokens_complete: bool = Field(default=False, strict=True)
    total_tokens_complete: bool = Field(default=False, strict=True)

    model_config = {"extra": "forbid"}

    @classmethod
    def from_counts(cls, **counts: Any) -> "TokenUsageSummary":
        values: dict[str, Any] = {"records": 1}
        for name in TOKEN_DIMENSIONS:
            value = counts.get(name)
            valid = type(value) is int and value >= 0
            values[f"known_{name}"] = value if valid else 0
            values[f"{name}_complete"] = valid
        return cls(**values)

    @classmethod
    def merge(cls, summaries: Iterable["TokenUsageSummary"]) -> "TokenUsageSummary":
        values: dict[str, Any] = {"records": 0}
        for name in TOKEN_DIMENSIONS:
            values[f"known_{name}"] = 0
            values[f"{name}_complete"] = True
        seen = False
        for summary in summaries:
            seen = True
            values["records"] += summary.records
            for name in TOKEN_DIMENSIONS:
                values[f"known_{name}"] += getattr(summary, f"known_{name}")
                values[f"{name}_complete"] &= getattr(summary, f"{name}_complete")
        return cls(**values) if seen else cls()

    def totals(self) -> dict[str, int | None]:
        return {
            name: getattr(self, f"known_{name}") if getattr(self, f"{name}_complete") else None
            for name in TOKEN_DIMENSIONS
        }
