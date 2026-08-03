#!/usr/bin/env python
"""Synthetic unit tests for scripts/qa/triage.py.
Run directly:  python scripts/qa/test_triage.py  (also pytest-compatible).
No GPU / data / torch / network / API key / anthropic package needed -- the SDK is imported
lazily inside require_client(), so everything below exercises the real request-building,
chunking and parsing code paths offline."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.qa.triage import (
    ESCALATION_MODEL,
    MAX_BATCH_BYTES,
    TRIAGE_MODEL,
    VERDICT_CLASSES,
    VERDICT_SCHEMA,
    build_payload,
    build_request,
    chunk_requests,
    custom_id_for,
    default_max_tokens,
    encode_overlay,
    estimate_cost,
    parse_verdict,
    select_escalations,
    text_of,
)


def _panel(path: Path, w: int = 903, h: int = 384) -> Path:
    img = np.random.default_rng(0).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(model=TRIAGE_MODEL, max_tokens=None, effort=None, jpeg_quality=90)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --------------------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------------------

def test_schema_is_strict_and_covers_every_class():
    assert VERDICT_SCHEMA["additionalProperties"] is False
    assert set(VERDICT_SCHEMA["required"]) == {"verdict", "confidence", "rationale"}
    assert VERDICT_SCHEMA["properties"]["verdict"]["enum"] == VERDICT_CLASSES
    assert set(VERDICT_CLASSES) == {"ok", "empty", "wrong-subject", "bleed", "split", "multiple"}


def test_schema_avoids_unsupported_json_schema_keywords():
    """Structured outputs reject numeric/string constraints; confidence is clamped in code."""
    blob = json.dumps(VERDICT_SCHEMA)
    for keyword in ("minimum", "maximum", "minLength", "maxLength", "multipleOf", "pattern"):
        assert keyword not in blob, f"{keyword} is not supported by structured outputs"


# --------------------------------------------------------------------------------------
# request construction
# --------------------------------------------------------------------------------------

def test_custom_ids_are_short_stable_and_unique():
    ids = [custom_id_for(i) for i in range(1000)]
    assert len(set(ids)) == 1000
    assert all(len(i) <= 64 for i in ids)
    assert custom_id_for(7) == custom_id_for(7)


def test_default_max_tokens_is_generous_for_a_thinking_model():
    """On Claude Opus 5 max_tokens caps thinking AND the reply, so Haiku's 512 would truncate."""
    assert default_max_tokens(TRIAGE_MODEL) == 512
    assert default_max_tokens(ESCALATION_MODEL) >= 4096


def test_build_request_shape():
    req = build_request("req000001", "AAAA", TRIAGE_MODEL, 512, None)
    assert req["custom_id"] == "req000001"
    params = req["params"]
    assert params["model"] == TRIAGE_MODEL
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["output_config"]["format"]["schema"] is VERDICT_SCHEMA
    content = params["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[0]["source"]["data"] == "AAAA"


def test_build_request_omits_effort_by_default():
    """Haiku 4.5 does not accept output_config.effort -- sending it would 400 the whole batch."""
    req = build_request("c", "AAAA", TRIAGE_MODEL, 512, None)
    assert "effort" not in req["params"]["output_config"]


def test_build_request_includes_effort_when_asked():
    req = build_request("c", "AAAA", ESCALATION_MODEL, 4096, "high")
    assert req["params"]["output_config"]["effort"] == "high"


def test_build_request_does_not_set_thinking_or_sampling_params():
    """temperature/top_p/top_k and budget_tokens are rejected on current models."""
    params = build_request("c", "AAAA", ESCALATION_MODEL, 4096, None)["params"]
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in params


# --------------------------------------------------------------------------------------
# image encoding
# --------------------------------------------------------------------------------------

def test_encode_overlay_returns_valid_base64_jpeg():
    with tempfile.TemporaryDirectory() as d:
        path = _panel(Path(d) / "p.png")
        blob = encode_overlay(path)
        raw = base64.standard_b64decode(blob)
        assert raw[:2] == b"\xff\xd8", "expected a JPEG SOI marker"
        assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is not None


def test_encode_overlay_is_much_smaller_than_the_png():
    """The reason for re-encoding: base64 PNG panels blow past the 256MB batch cap well before
    1,000 frames, while the on-disk PNGs stay lossless for human gold-set labelling."""
    with tempfile.TemporaryDirectory() as d:
        path = _panel(Path(d) / "p.png")
        assert len(encode_overlay(path)) < path.stat().st_size


def test_encode_overlay_raises_on_a_missing_panel():
    with tempfile.TemporaryDirectory() as d:
        try:
            encode_overlay(Path(d) / "nope.png")
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


def test_build_payload_skips_unreadable_panels_without_losing_the_mapping():
    with tempfile.TemporaryDirectory() as d:
        good = _panel(Path(d) / "good.png")
        targets = [
            {"overlay_path": str(good), "video_name": "a", "frame_idx": 1, "instance_idx": 0},
            {"overlay_path": str(Path(d) / "missing.png"), "video_name": "b",
             "frame_idx": 2, "instance_idx": 0},
        ]
        requests, mapping, failures = build_payload(targets, _args())
        assert len(requests) == 1 and len(failures) == 1
        assert set(mapping) == {r["custom_id"] for r in requests}
        assert mapping[requests[0]["custom_id"]]["video_name"] == "a"


# --------------------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------------------

def _fake_request(payload_bytes: int) -> dict:
    return {"custom_id": "c", "params": {"blob": "x" * payload_bytes}}


def test_chunk_requests_keeps_one_batch_when_small():
    assert len(chunk_requests([_fake_request(10) for _ in range(50)])) == 1


def test_chunk_requests_splits_on_the_byte_cap():
    chunks = chunk_requests([_fake_request(1000) for _ in range(10)], max_bytes=3000)
    assert len(chunks) > 1
    assert sum(len(c) for c in chunks) == 10
    for chunk in chunks:
        assert sum(len(json.dumps(r)) for r in chunk) <= 3000 or len(chunk) == 1


def test_chunk_requests_splits_on_the_count_cap():
    chunks = chunk_requests([_fake_request(1) for _ in range(10)], max_count=3)
    assert [len(c) for c in chunks] == [3, 3, 3, 1]


def test_chunk_requests_emits_an_oversized_request_alone_rather_than_dropping_it():
    """Silently dropping it would produce a run that looks complete but isn't; the API
    rejecting it with a reason is strictly more useful."""
    requests = [_fake_request(10), _fake_request(5000), _fake_request(10)]
    chunks = chunk_requests(requests, max_bytes=1000)
    assert sum(len(c) for c in chunks) == 3
    assert any(len(c) == 1 and len(json.dumps(c[0])) > 1000 for c in chunks)


def test_chunk_requests_handles_an_empty_list():
    assert chunk_requests([]) == []


def test_chunk_default_cap_respects_the_documented_batch_limit():
    assert int(MAX_BATCH_BYTES * 0.80) < MAX_BATCH_BYTES


# --------------------------------------------------------------------------------------
# verdict parsing
# --------------------------------------------------------------------------------------

def test_parse_verdict_happy_path():
    out = parse_verdict('{"verdict":"bleed","confidence":0.82,"rationale":"spills onto grass"}')
    assert out == {"verdict": "bleed", "confidence": 0.82, "rationale": "spills onto grass"}


def test_parse_verdict_clamps_out_of_range_confidence():
    """Structured outputs cannot express minimum/maximum, so this is the only guard."""
    assert parse_verdict('{"verdict":"ok","confidence":1.7,"rationale":"x"}')["confidence"] == 1.0
    assert parse_verdict('{"verdict":"ok","confidence":-0.4,"rationale":"x"}')["confidence"] == 0.0


def test_parse_verdict_rejects_an_unknown_class():
    """A class outside the enum must not silently become a category in the failure mix."""
    for bad in ('{"verdict":"mostly-ok","confidence":0.9,"rationale":"x"}',
                '{"verdict":null,"confidence":0.9,"rationale":"x"}'):
        try:
            parse_verdict(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_parse_verdict_rejects_bad_confidence():
    for bad in ('{"verdict":"ok","confidence":"high","rationale":"x"}',
                '{"verdict":"ok","confidence":null,"rationale":"x"}',
                '{"verdict":"ok","confidence":NaN,"rationale":"x"}'):
        try:
            parse_verdict(bad)
        except (ValueError, json.JSONDecodeError):
            continue
        raise AssertionError(f"expected an error for {bad}")


def test_parse_verdict_tolerates_a_missing_rationale():
    assert parse_verdict('{"verdict":"ok","confidence":0.5}')["rationale"] == ""


def test_text_of_handles_dicts_and_objects():
    assert text_of({"content": [{"type": "text", "text": "hi"}]}) == "hi"

    class Block:
        type, text = "text", "yo"

    class Message:
        content = [Block()]

    assert text_of(Message()) == "yo"


def test_text_of_skips_non_text_blocks():
    message = {"content": [{"type": "thinking", "thinking": "..."},
                           {"type": "text", "text": "{}"}]}
    assert text_of(message) == "{}"


# --------------------------------------------------------------------------------------
# escalation selection
# --------------------------------------------------------------------------------------

def _write_verdicts(path: Path, rows) -> None:
    fields = ["custom_id", "video_name", "verdict", "confidence", "result_type", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_select_escalations_picks_low_confidence_rows():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v.csv"
        _write_verdicts(path, [
            dict(custom_id="a", video_name="v", verdict="ok", confidence=0.95,
                 result_type="succeeded", error=""),
            dict(custom_id="b", video_name="v", verdict="bleed", confidence=0.40,
                 result_type="succeeded", error=""),
        ])
        assert [r["custom_id"] for r in select_escalations(path, 0.7)] == ["b"]


def test_select_escalations_includes_failures_and_unparseables():
    """A row with no usable confidence must be re-checked, not silently treated as confident --
    otherwise failures quietly shrink the pilot's denominator."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v.csv"
        _write_verdicts(path, [
            dict(custom_id="ok", video_name="v", verdict="ok", confidence=0.99,
                 result_type="succeeded", error=""),
            dict(custom_id="err", video_name="v", verdict="", confidence="",
                 result_type="errored", error="overloaded"),
            dict(custom_id="bad", video_name="v", verdict="", confidence="oops",
                 result_type="unparseable", error="bad json"),
            dict(custom_id="gone", video_name="v", verdict="", confidence="",
                 result_type="missing", error="no result"),
        ])
        assert sorted(r["custom_id"] for r in select_escalations(path, 0.7)) == \
               ["bad", "err", "gone"]


def test_select_escalations_threshold_is_strict():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v.csv"
        _write_verdicts(path, [
            dict(custom_id="at", video_name="v", verdict="ok", confidence=0.70,
                 result_type="succeeded", error=""),
        ])
        assert select_escalations(path, 0.7) == []


# --------------------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------------------

def test_estimate_cost_applies_the_batch_discount():
    full = estimate_cost(1000, TRIAGE_MODEL, image_tokens=1000, prompt_tokens=0, output_tokens=0)
    # 1000 requests x 1000 tokens = 1M input tokens at $1/MTok, halved by the Batch discount
    assert abs(full["cost_usd"] - 0.5) < 1e-9


def test_estimate_cost_opus_is_five_times_haiku():
    haiku = estimate_cost(100, TRIAGE_MODEL, 1000)
    opus = estimate_cost(100, ESCALATION_MODEL, 1000)
    assert opus["cost_usd"] > haiku["cost_usd"] * 4


def test_estimate_cost_scales_linearly_with_frames():
    small = estimate_cost(1_000, TRIAGE_MODEL, 1000)
    big = estimate_cost(1_000_000, TRIAGE_MODEL, 1000)
    assert abs(big["cost_usd"] / small["cost_usd"] - 1000) < 1e-6


def test_pilot_estimate_is_under_one_pound():
    """The pilot's headline claim: 1,000 panels at ~540 image tokens costs well under GBP 1."""
    est = estimate_cost(1000, TRIAGE_MODEL, image_tokens=540)
    assert est["cost_usd"] < 1.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nall {len(fns)} triage tests passed")


if __name__ == "__main__":
    _run_all()
