# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cell execution core used inside the sandbox worker.

``run_cell_source`` uses the same execution-cell source preparation as
``ActorRuntime.execute_code`` (the ``wrap_in_function=True`` path), persists
locals into the namespace, and surfaces ``return_result()`` as an
``ExecutionSignal``.

It runs *inside the worker*, on a plain namespace dict, with no dependency on the
live agent or the parent event loop — those concerns stay on the parent.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from aster.agentdoc import FileBackedTruncatingStringIO, TruncatingStringIO
from aster.events import _NO_RETURN, ExecutionResult, ExecutionSignal
from aster.runtime._python_cell import prepare_cell


class _HookedBuffer:
    """StringIO that also forwards each write to an optional streaming hook."""

    def __init__(self, hook=None, *, limit=50_000, tail_chars=None, file_backed=False):
        kind = FileBackedTruncatingStringIO if file_backed else TruncatingStringIO
        self._buffer = kind(limit=limit, tail_chars=tail_chars)
        self._hook = hook

    def write(self, s: str) -> int:  # type: ignore[override]
        if self._hook and s:
            try:
                self._hook(s)
            except Exception:
                pass
        return self._buffer.write(s)

    def getvalue(self):
        return self._buffer.getvalue()

    def flush(self):
        return self._buffer.flush()

    @property
    def was_truncated(self):
        return self._buffer.was_truncated


async def run_cell_source(
    code: str,
    namespace: dict[str, Any],
    *,
    execution_count: int = 1,
    stdout_hook: Callable[[str], None] | None = None,
    stderr_hook: Callable[[str], None] | None = None,
    capture_config=None,
    max_attachments=100,
) -> ExecutionResult:
    """Execute ``code`` in ``namespace`` with REPL semantics; never raises.

    Returns an :class:`ExecutionResult`. ``ExecutionSignal`` (``return_result``)
    is captured into ``result.signal``; every other exception into
    ``result.error``. ``namespace`` is mutated in place so definitions persist
    across cells (the worker's long-lived REPL state).
    """
    from aster.agentdoc._format_observation import _truncation_reasons
    from aster.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
    from aster.runtime.media_capture import _media_buffer_var, _MediaBuffer

    capture = capture_config or DEFAULT_TRUNCATION_CONFIG.capture
    stdout_buf = _HookedBuffer(
        stdout_hook,
        limit=capture.max_stdout,
        tail_chars=capture.tail,
        file_backed=capture.file_backed,
    )
    stderr_buf = _HookedBuffer(
        stderr_hook,
        limit=capture.max_stderr,
        tail_chars=capture.tail,
        file_backed=capture.file_backed,
    )
    media_buffer = _MediaBuffer(max_attachments=max_attachments)
    media_token = _media_buffer_var.set(media_buffer)
    truncation_reasons: set[str] = set()
    truncation_token = _truncation_reasons.set(truncation_reasons)

    returned_value: Any = _NO_RETURN
    has_explicit_return = False
    wrapper_line_offset = 0
    cell_filename = f"Cell In[{execution_count}]"

    def build_result(*, error=None, signal=None) -> ExecutionResult:
        if stdout_buf.was_truncated:
            truncation_reasons.add("max_string")
        if stderr_buf.was_truncated:
            truncation_reasons.add("max_string")
        return ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            truncation_reasons=sorted(truncation_reasons),
            error=error,
            signal=signal,
            defined_methods=defined_methods,
            returned_value=returned_value,
            explicit_return=has_explicit_return,
            captured_locals={},
            images=media_buffer.blocks,
            wrapper_line_offset=wrapper_line_offset,
        )

    defined_methods: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                cell = prepare_cell(
                    code,
                    cell_filename,
                    variables=namespace,
                    wrap_in_function=True,
                )
            except SyntaxError as exc:
                return build_result(error=exc)

            has_explicit_return = cell.explicit_return
            wrapper_line_offset = cell.wrapper_line_offset
            namespace["__repl_captured_locals__"] = {}
            # Compile here to retain the guest's postponed annotation evaluation.
            exec(compile(cell.execution_source(), cell_filename, "exec"), namespace)
            try:
                result_value = await namespace["__repl_wrapper__"]()
            finally:
                # Keep REPL state even when the cell raises or returns via signal.
                captured = namespace.pop("__repl_captured_locals__", {})
                for k, v in captured.items():
                    namespace[k] = v

            returned_value = cell.returned_value(result_value)

            defined_methods = cell.finish_helpers(namespace)
            return build_result()
    except ExecutionSignal as sig:
        return build_result(signal=sig)
    except (SystemExit, KeyboardInterrupt) as exc:
        wrapped = RuntimeError(
            f"{type(exc).__name__} raised inside generated code. Use return_result() "
            "or a normal return to finish a cell, not sys.exit()/exit()/quit()."
        )
        wrapped.__cause__ = exc
        return build_result(error=wrapped)
    except Exception as exc:
        return build_result(error=exc)
    finally:
        _truncation_reasons.reset(truncation_token)
        _media_buffer_var.reset(media_token)
