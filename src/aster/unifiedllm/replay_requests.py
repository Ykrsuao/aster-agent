# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay captured failed LLM requests for debugging.

This module provides tools to replay HTTP requests that were captured
by enable_http_request_logging() with errors_only=True.

Usage:
    uv run python -m aster.unifiedllm.replay_requests eval_errors/llm_errors.jsonl

Custom gateways require an explicit HTTPS origin and credential environment variable:
    --credential https://gateway.example=GATEWAY_API_KEY
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

# Load .env file if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on existing env vars


async def replay_request(
    error_entry: dict[str, Any],
    verbose: bool = True,
    *,
    credentials: dict[str, str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Replay a single request from error log.

    Args:
        error_entry: Dictionary with 'request' and 'response' keys from JSONL
        verbose: Print details about each replay
        credentials: Exact HTTPS origin → API-key environment variable for bearer
            authentication. Never reuse a gateway key for another origin.

    Returns:
        Tuple of (success: bool, response: dict)
        success is True if status < 400, False otherwise
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError(
            "httpx is required for request replay. "
            "Install it in the environment running ASTER with: uv pip install httpx"
        ) from e

    request_data = error_entry["request"]
    original_response = error_entry["response"]

    headers = httpx.Headers(request_data["headers"])
    if "authorization" in headers:
        try:
            url = httpx.URL(request_data["url"])
        except httpx.InvalidURL:
            return False, {"error": "Invalid replay URL"}
        if url.scheme != "https" or url.userinfo:
            return False, {"error": "Replay credentials require HTTPS without URL user information"}
        origin = str(url.copy_with(path="", query=None, fragment=None))
        if credentials and origin in credentials:
            env_names = (credentials[origin],)
        else:
            env_names = {
                "https://api.openai.com": ("OPENAI_API_KEY",),
                "https://api.anthropic.com": ("ANTHROPIC_API_KEY",),
                "https://integrate.api.nvidia.com": ("NVIDIA_API_KEY",),
                "https://inference-api.nvidia.com": (
                    "NVIDIA_INFERENCE_API_KEY",
                    "NVIDIA_INTERNAL_API_KEY",
                ),
            }.get(origin, ())
        api_key = next((os.environ[name] for name in env_names if os.environ.get(name)), None)
        if not api_key:
            error = "No replay credential for this HTTPS origin; use --credential ORIGIN=ENV"
            if verbose:
                print(f"⚠️  {error}")
            return False, {"error": error}
        headers["authorization"] = f"Bearer {api_key}"

    if verbose:
        model = error_entry.get("model", "unknown")
        counter = error_entry.get("counter", "?")
        print(f"\n🔄 Replaying request #{counter} ({model})...")
        print(f"   Original status: {original_response['status_code']}")

    # Make the request
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.request(
                method=request_data["method"],
                url=request_data["url"],
                headers=headers,
                json=request_data["body"],
            )

            success = response.status_code < 400

            if verbose:
                if success:
                    print(f"   ✅ Now succeeds: {response.status_code}")
                else:
                    print(f"   ❌ Still fails: {response.status_code}")

            return success, {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.json() if response.content else None,
            }

        except Exception as e:
            if verbose:
                print(f"   ❌ Exception: {type(e).__name__}")
            return False, {"error": type(e).__name__}


async def replay_from_file(
    error_file: Path | str,
    max_requests: int | None = None,
    delay_seconds: float = 1.0,
    verbose: bool = True,
    *,
    credentials: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Replay multiple requests from error log.

    Args:
        error_file: Path to llm_errors.jsonl file
        max_requests: Maximum number of requests to replay (None = all)
        delay_seconds: Delay between requests to avoid rate limits
        verbose: Print details about each replay
        credentials: Exact HTTPS origin → bearer-key environment variable.

    Returns:
        Summary statistics dictionary with keys:
        - total: Total requests replayed
        - succeeded: Number that now succeed
        - failed: Number that still fail
        - success_rate: Percentage that now succeed
        - results: List of (entry, success, response) tuples
    """
    error_file = Path(error_file)

    if not error_file.exists():
        raise FileNotFoundError(f"Error file not found: {error_file}")

    # Read JSONL file
    entries = []
    with open(error_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if max_requests:
        entries = entries[:max_requests]

    if verbose:
        print(f"📂 Loaded {len(entries)} error entries from {error_file}")
        print(f"{'=' * 60}")

    results = []
    succeeded = 0
    failed = 0

    for i, entry in enumerate(entries):
        if i > 0 and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        success, response = await replay_request(entry, verbose=verbose, credentials=credentials)

        if success:
            succeeded += 1
        else:
            failed += 1

        results.append((entry, success, response))

    total = len(entries)
    success_rate = (succeeded / total * 100) if total > 0 else 0

    if verbose:
        print(f"\n{'=' * 60}")
        print("📊 Replay Summary:")
        print(f"   Total replayed: {total}")
        print(f"   Now succeed: {succeeded} ({success_rate:.1f}%)")
        print(f"   Still fail: {failed} ({100 - success_rate:.1f}%)")
        print("\n💡 Interpretation:")
        if success_rate > 80:
            print("   → Most errors were transient API issues")
        elif success_rate < 20:
            print("   → Most errors are likely payload-related")
        else:
            print("   → Mix of transient and payload issues")

    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate": success_rate,
        "results": results,
    }


async def main():
    """CLI entry point for replaying requests."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay captured LLM error requests")
    parser.add_argument("error_file", type=Path, help="Path to llm_errors.jsonl file")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Maximum number of requests to replay (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between requests (default: 1.0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    parser.add_argument(
        "--credential",
        action="append",
        default=[],
        metavar="ORIGIN=ENV",
        help="Bearer-key environment variable for an exact HTTPS origin (repeatable)",
    )

    args = parser.parse_args()
    import httpx

    credentials = {}
    for value in args.credential:
        origin, separator, env_name = value.partition("=")
        try:
            url = httpx.URL(origin)
        except httpx.InvalidURL:
            parser.error("--credential requires a valid HTTPS origin")
        if (
            not separator
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name)
            or url.scheme != "https"
            or not url.host
            or url.userinfo
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            parser.error(
                "--credential must be HTTPS_ORIGIN=ENV_NAME, without a path or user information"
            )
        credentials[str(url.copy_with(path=""))] = env_name

    stats = await replay_from_file(
        error_file=args.error_file,
        max_requests=args.max_requests,
        delay_seconds=args.delay,
        verbose=not args.quiet,
        credentials=credentials,
    )

    # Exit with non-zero if all requests still fail
    if stats["succeeded"] == 0 and stats["total"] > 0:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
