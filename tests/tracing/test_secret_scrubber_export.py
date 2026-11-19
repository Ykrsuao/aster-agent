# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secrets must be removed at the export boundary, including error telemetry."""

import json

import pytest
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, SpanContext, Status, StatusCode, TraceFlags

from aster.tracing import _add_exporters
from aster.tracing._hooks_impl import _record_error
from aster.tracing._otlp_file_exporter import OtlpJsonFileExporter
from aster.tracing._secret_scrubber import REDACTED, SecretScrubSpanProcessor

FAKE_KEY = "sk-abcdefghijklmnopqrstuvwxyz123456"


@pytest.mark.parametrize("field", ["name", "status", "event_name", "event_attribute"])
def test_attribute_free_spans_are_scrubbed(field):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SecretScrubSpanProcessor(SimpleSpanProcessor(exporter)))
    try:
        with provider.get_tracer(__name__).start_as_current_span(
            FAKE_KEY if field == "name" else "safe name"
        ) as span:
            if field == "status":
                span.set_status(Status(StatusCode.ERROR, f"request failed: {FAKE_KEY}"))
            elif field == "event_name":
                span.add_event(f"request failed: {FAKE_KEY}")
            elif field == "event_attribute":
                span.add_event("request failed", {"client_secret": "synthetic-short-secret"})
        exported = exporter.get_finished_spans()[0]
        wire = exported.to_json()
        assert FAKE_KEY not in wire
        assert "synthetic-short-secret" not in wire
        assert REDACTED in wire
        assert exported.attributes["secrets.redacted_count"] == 1
    finally:
        provider.shutdown()


def test_framework_errors_are_scrubbed_in_file_and_batch_exports(tmp_path):
    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    _add_exporters(provider, [OtlpJsonFileExporter(tmp_path)])
    provider.add_span_processor(SecretScrubSpanProcessor(BatchSpanProcessor(memory_exporter)))
    try:
        with provider.get_tracer(__name__).start_as_current_span("agent method") as span:
            span.set_attribute("session.id", "test-session")
            span.set_attribute("safe.output", "ordinary output")
            try:
                raise ValueError(f"authentication failed: {FAKE_KEY}")
            except ValueError as error:
                _record_error(span, error)
                span.record_exception(error)
        assert provider.force_flush()
        exported = memory_exporter.get_finished_spans()[0]
        assert FAKE_KEY not in exported.to_json()
        assert exported.status.status_code is StatusCode.ERROR
        assert exported.attributes["safe.output"] == "ordinary output"

        wire = (tmp_path / "test-session.jsonl").read_text(encoding="utf-8")
        assert FAKE_KEY not in wire
        payload = json.loads(wire)
        saved = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert saved["status"]["code"] == 2
        assert REDACTED in saved["status"]["message"]
        assert saved["events"][0]["name"] == "exception"
        assert REDACTED in json.dumps(saved["events"])
    finally:
        provider.shutdown()


def test_scrubbing_preserves_metadata_and_original_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        span_limits=SpanLimits(
            max_span_attributes=1, max_events=1, max_event_attributes=1, max_links=1
        )
    )
    context = SpanContext(123, 456, False, TraceFlags(TraceFlags.SAMPLED))
    span = provider.get_tracer(__name__).start_span(
        "metadata", links=[Link(context), Link(context)], start_time=100
    )
    span.set_attribute("dropped", True)
    span.set_attribute("client_secret", "synthetic-short-secret")
    span.add_event("dropped")
    span.add_event(
        "retained event",
        {"dropped": True, "note": FAKE_KEY},
        timestamp=150,
    )
    span.set_status(Status(StatusCode.ERROR, "normal failure"))
    span.end(end_time=200)
    original = span._readable_span()
    processor = SecretScrubSpanProcessor(SimpleSpanProcessor(exporter))
    try:
        processor.on_end(original)
        scrubbed = exporter.get_finished_spans()[0]
        assert scrubbed is not original
        assert scrubbed.context == original.context
        assert scrubbed.parent == original.parent
        assert scrubbed.kind == original.kind
        assert scrubbed.resource is original.resource
        assert scrubbed.instrumentation_scope is original.instrumentation_scope
        assert scrubbed.links == original.links
        assert (scrubbed.start_time, scrubbed.end_time) == (100, 200)
        assert scrubbed.status.status_code is StatusCode.ERROR
        assert scrubbed.status.description == "normal failure"
        assert scrubbed.events[0].timestamp == 150
        assert scrubbed.events[0].name == "retained event"
        assert scrubbed.dropped_attributes == original.dropped_attributes == 1
        assert scrubbed.dropped_events == original.dropped_events == 1
        assert scrubbed.dropped_links == original.dropped_links == 1
        assert scrubbed.events[0].dropped_attributes == original.events[0].dropped_attributes == 1
        assert scrubbed.attributes["client_secret"] == REDACTED
        assert scrubbed.events[0].attributes["note"] == REDACTED
        assert original.attributes["client_secret"] == "synthetic-short-secret"
        assert original.events[0].attributes["note"] == FAKE_KEY
    finally:
        processor.shutdown()
        provider.shutdown()


def test_clean_span_passes_through_without_replacement():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    span = provider.get_tracer(__name__).start_span("ordinary operation")
    span.set_attribute("result", 42)
    span.add_event("progress", {"step": 1})
    span.end()
    original = span._readable_span()
    processor = SecretScrubSpanProcessor(SimpleSpanProcessor(exporter))
    try:
        processor.on_end(original)
        assert exporter.get_finished_spans() == (original,)
        assert "secrets.redacted_count" not in original.attributes
    finally:
        processor.shutdown()
        provider.shutdown()


def test_nested_scrubbers_preserve_redaction_count():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        SecretScrubSpanProcessor(SecretScrubSpanProcessor(SimpleSpanProcessor(exporter)))
    )
    try:
        with provider.get_tracer(__name__).start_as_current_span("request") as span:
            span.set_attribute("client_secret", "synthetic-short-secret")
            span.add_event("request failed", {"refresh_token": "synthetic-refresh-token"})
        exported = exporter.get_finished_spans()[0]
        assert exported.attributes["secrets.redacted_count"] == 2
        assert exported.attributes["client_secret"] == REDACTED
        assert exported.events[0].attributes["refresh_token"] == REDACTED
    finally:
        provider.shutdown()


def test_scrubbing_failure_drops_span_without_logging_raw_data(monkeypatch, caplog):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SecretScrubSpanProcessor(SimpleSpanProcessor(exporter)))

    def fail(_value):
        raise ValueError(FAKE_KEY)

    monkeypatch.setattr("aster.tracing._secret_scrubber.scrub_value", fail)
    try:
        with provider.get_tracer(__name__).start_as_current_span(FAKE_KEY) as span:
            span.set_attribute("message", FAKE_KEY)
        assert not exporter.get_finished_spans()
        assert "scrubbing failed" in caplog.text
        assert FAKE_KEY not in caplog.text
    finally:
        provider.shutdown()
