#!/usr/bin/env python
"""Stage 3b of the SAM-3 mask QA pilot: classify each rendered mask via the Batch API.

Four subcommands, because a batch can take up to 24h and a login-node shell will not survive
that -- a single blocking call would lose the run to any disconnect:

    estimate   what the run would cost, and how many batches it needs. No API call, no key.
    submit     upload, persist the batch id(s) + custom_id map, exit.
    poll       print processing status. Safe to re-run, or loop under tmux/nohup.
    fetch      stream results into a verdicts CSV. Idempotent, re-runnable after a drop.

**Batch API only, by design.** Falling back to some other calling path when a key is missing
would mean the measured dials came from a different model on a different prompt, quietly making
the 1M extrapolation non-comparable. Absent `ANTHROPIC_API_KEY` this stops and says so.

Results are keyed by `custom_id`. Batch results come back in **arbitrary order**, so position
is never used for anything; the custom_id -> (video_name, frame_idx, instance_idx) map is
written to disk at submit time and is the sole source of truth at fetch time.

Overlays are re-encoded to JPEG in memory before upload. The PNGs on disk stay lossless for
human gold-set labelling, while the API sees a payload that fits: a batch is capped at 256MB,
and base64 PNG panels blow through that well before 1,000 frames.

Escalation re-checks low-confidence verdicts on a stronger model through the same four
subcommands -- `submit --escalate-from <verdicts.csv>` selects them.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.qa.verdicts_schema import VERDICT_CLASSES, VERDICT_FIELDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

TRIAGE_MODEL = "claude-haiku-4-5"
ESCALATION_MODEL = "claude-opus-5"

# Numerical/string constraints (minimum, maximum, minLength) are not supported by structured
# outputs, so confidence is clamped client-side instead of constrained here.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": VERDICT_CLASSES},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "confidence", "rationale"],
    "additionalProperties": False,
}

RUBRIC = """\
You are grading the quality of a segmentation mask, not the photograph.

Each image is one review panel. LEFT: the full video frame, with the mask under review shown as \
a translucent green fill with a cyan outline. Any OTHER masks detected in the same frame are \
outlined in orange and are not what you are grading. RIGHT: a zoomed crop of the same mask, for \
judging its exact boundary.

The intended subject is a person holding a sign or placard, used as a distance reference.

Choose exactly one verdict:

- "ok": the mask covers the intended person (with or without their sign) and essentially \
nothing else. Minor boundary roughness of a few pixels is still "ok".
- "empty": the mask is absent or so small it covers no recognisable subject.
- "wrong-subject": the mask is on something that is not the sign-holding person -- vegetation, \
an animal, equipment, a shadow, or a different person who is not holding the sign.
- "bleed": the mask covers the right person but spills noticeably into the background, the \
ground, or an adjacent object.
- "split": the mask covers only part of the person, or is broken into disconnected pieces that \
should be one region.
- "multiple": this single mask covers two or more separate people or objects at once.

If more than one applies, pick the one that would most mislead a downstream distance estimate: \
"wrong-subject" beats "multiple" beats "bleed" beats "split".

Set "confidence" between 0.0 and 1.0 -- your own probability that this verdict is correct. Be \
honest: low confidence routes the panel to a stronger reviewer, which is the desired outcome \
when the panel is genuinely ambiguous. Keep "rationale" to one short sentence."""

# Batch API limits (a request is rejected wholesale if either is exceeded).
MAX_REQUESTS_PER_BATCH = 100_000
MAX_BATCH_BYTES = 256 * 1024 * 1024
BATCH_BYTES_HEADROOM = 0.80  # leave room for the JSON envelope around each base64 payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--name", default="haiku",
                        help="label for this run; the batch record is <batch-dir>/<name>.json")
        sp.add_argument("--batch-dir", type=Path,
                        default=REPO_ROOT / "outputs" / "qa" / "batches")

    def build_opts(sp):
        sp.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "outputs" / "qa" / "overlay_manifest.csv",
                        help="scripts/qa/render_overlays.py output")
        sp.add_argument("--model", default=TRIAGE_MODEL)
        sp.add_argument("--max-tokens", type=int, default=None,
                        help="default: 512 for Haiku, 4096 for a thinking model (on Claude "
                             "Opus 5 max_tokens caps thinking AND the reply, so a tight value "
                             "truncates the verdict)")
        sp.add_argument("--effort", default=None,
                        help="output_config.effort; omit on Haiku 4.5, which does not accept it")
        sp.add_argument("--jpeg-quality", type=int, default=90,
                        help="re-encode quality for upload; the on-disk PNGs are untouched")
        sp.add_argument("--limit", type=int, default=None, help="first N panels only (smoke test)")
        sp.add_argument("--escalate-from", type=Path, default=None,
                        help="a verdicts CSV; select its low-confidence rows instead of the "
                             "whole manifest")
        sp.add_argument("--confidence-below", type=float, default=0.7,
                        help="escalation threshold; tune this on the gold set's TUNE half only")

    sp_est = sub.add_parser("estimate", help="cost and batch count; no API call, no key needed")
    common(sp_est)
    build_opts(sp_est)

    sp_sub = sub.add_parser("submit", help="upload and record the batch id(s)")
    common(sp_sub)
    build_opts(sp_sub)

    sp_poll = sub.add_parser("poll", help="print processing status")
    common(sp_poll)

    sp_fetch = sub.add_parser("fetch", help="stream results into a verdicts CSV")
    common(sp_fetch)
    sp_fetch.add_argument("--out", type=Path, default=None,
                          help="default: outputs/qa/verdicts.csv (backend-neutral -- both this "
                               "VLM backend and scripts/qa/heuristic_verdicts.py write the same "
                               "verdicts-CSV contract)")

    return p.parse_args()


# --------------------------------------------------------------------------------------
# request construction (no SDK import -- unit-testable without the anthropic package)
# --------------------------------------------------------------------------------------

def default_max_tokens(model: str) -> int:
    """Haiku has no thinking to budget for; Claude Opus 5 thinks by default and `max_tokens`
    caps thinking plus the reply together, so a 512-token ceiling would truncate the verdict."""
    return 512 if "haiku" in model else 4096


def encode_overlay(path: Path, jpeg_quality: int = 90) -> str:
    """PNG on disk -> base64 JPEG for upload. Raises FileNotFoundError if the panel is missing."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read overlay {path}")
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise ValueError(f"failed to encode {path} as JPEG")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def custom_id_for(index: int) -> str:
    """A short, stable id. Deliberately not derived from video_name: those run past the
    custom_id length limit once a camera folder is in the name, and the mapping is persisted
    to disk anyway."""
    return f"req{index:06d}"


def build_request(custom_id: str, image_b64: str, model: str, max_tokens: int,
                  effort: str | None) -> dict:
    """One Batch API request. Plain dicts, which the SDK's TypedDicts accept as-is."""
    output_config: dict = {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}}
    if effort:
        output_config["effort"] = effort
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": RUBRIC,
            "output_config": output_config,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "text", "text": "Grade this mask."},
                ],
            }],
        },
    }


def chunk_requests(requests: list[dict],
                   max_bytes: int = int(MAX_BATCH_BYTES * BATCH_BYTES_HEADROOM),
                   max_count: int = MAX_REQUESTS_PER_BATCH) -> list[list[dict]]:
    """Split into batches that respect both Batch API caps.

    Size, not count, is the binding constraint here: base64 image payloads reach 256MB long
    before 100,000 requests do. A single oversized request is still emitted alone rather than
    silently dropped -- the API will reject it and say why, which is more useful than a
    truncated run that looks complete.
    """
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    for req in requests:
        size = len(json.dumps(req))
        if current and (current_bytes + size > max_bytes or len(current) >= max_count):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(req)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def select_escalations(verdicts_csv: Path, threshold: float) -> list[dict]:
    """Rows whose confidence falls below the threshold, plus any that failed outright.

    A row that errored or returned an unparseable verdict has no usable confidence, so treating
    it as 'confident' would silently drop it from the pilot rather than re-checking it.
    """
    selected: list[dict] = []
    with open(verdicts_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("result_type") != "succeeded" or not row.get("verdict"):
                selected.append(row)
                continue
            try:
                confidence = float(row["confidence"])
            except (TypeError, ValueError, KeyError):
                selected.append(row)
                continue
            if confidence < threshold:
                selected.append(row)
    return selected


def load_manifest(path: Path, limit: int | None = None) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def build_payload(targets: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict, list[str]]:
    """(requests, custom_id -> target map, per-target read failures)."""
    max_tokens = args.max_tokens or default_max_tokens(args.model)
    requests: list[dict] = []
    mapping: dict[str, dict] = {}
    failures: list[str] = []
    for i, target in enumerate(targets):
        overlay = Path(target["overlay_path"])
        try:
            image_b64 = encode_overlay(overlay, args.jpeg_quality)
        except (FileNotFoundError, ValueError) as exc:
            failures.append(str(exc))
            continue
        custom_id = custom_id_for(i)
        requests.append(build_request(custom_id, image_b64, args.model, max_tokens, args.effort))
        mapping[custom_id] = target
    return requests, mapping, failures


# --------------------------------------------------------------------------------------
# verdict parsing
# --------------------------------------------------------------------------------------

def parse_verdict(text: str) -> dict:
    """Structured-output JSON -> a normalised verdict dict.

    Structured outputs guarantee the shape, but this still validates: a schema violation would
    otherwise become a silently miscounted class in the headline failure mix.
    """
    data = json.loads(text)
    verdict = data.get("verdict")
    if verdict not in VERDICT_CLASSES:
        raise ValueError(f"verdict {verdict!r} not in {VERDICT_CLASSES}")
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        raise ValueError(f"unparseable confidence {data.get('confidence')!r}") from None
    if not math.isfinite(confidence):
        raise ValueError(f"non-finite confidence {confidence!r}")
    return {
        "verdict": verdict,
        # clamped rather than schema-constrained: structured outputs do not support
        # minimum/maximum, so a model returning 1.5 or -0.2 is possible
        "confidence": min(1.0, max(0.0, confidence)),
        "rationale": str(data.get("rationale", "")).strip(),
    }


def text_of(message) -> str:
    """Concatenate the text blocks of a Message (SDK object or plain dict)."""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", [])
    parts = []
    for block in content or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype == "text":
            parts.append(block.get("text") if isinstance(block, dict) else getattr(block, "text", ""))
    return "".join(parts)


# --------------------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------------------

# USD per million tokens, standard rates. The Batch API bills at 50% of these.
PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-5": (5.0, 25.0),
}
BATCH_DISCOUNT = 0.5


def estimate_cost(n_requests: int, model: str, image_tokens: float,
                  prompt_tokens: float = 400.0, output_tokens: float = 100.0) -> dict:
    """Batch-rate cost estimate. Image tokens are ~w*h/750; the caller measures a real panel."""
    rate_in, rate_out = PRICING.get(model, PRICING["claude-haiku-4-5"])
    tokens_in = n_requests * (image_tokens + prompt_tokens)
    tokens_out = n_requests * output_tokens
    cost_in = tokens_in / 1e6 * rate_in * BATCH_DISCOUNT
    cost_out = tokens_out / 1e6 * rate_out * BATCH_DISCOUNT
    return {
        "n_requests": n_requests,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_in_usd": cost_in,
        "cost_out_usd": cost_out,
        "cost_usd": cost_in + cost_out,
    }


def measure_image_tokens(targets: list[dict]) -> float:
    """Image tokens for the first readable panel, as w*h/750."""
    for target in targets:
        img = cv2.imread(str(target["overlay_path"]))
        if img is not None:
            h, w = img.shape[:2]
            return w * h / 750.0
    return 0.0


# --------------------------------------------------------------------------------------
# batch record
# --------------------------------------------------------------------------------------

def record_path(batch_dir: Path, name: str) -> Path:
    return batch_dir / f"{name}.json"


def save_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def load_record(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"no batch record at {path} -- run `submit` first")
    with open(path) as f:
        return json.load(f)


def require_client():
    """The anthropic client, or a clear stop. Imported lazily so `estimate` needs neither the
    package nor a key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY is not set.\n"
            "This stage is Batch-API-only on purpose: routing around a missing key would mean "
            "the pilot's measured rates came from a different model on a different prompt path, "
            "and the 1M extrapolation built on them would not be comparable.\n"
            "Set the key and re-run (the full pilot run costs well under GBP 1). Note this must "
            "run somewhere with outbound internet -- on this cluster, the login node."
        )
    try:
        import anthropic
    except ImportError:
        sys.exit("the `anthropic` package is not installed: pip install anthropic")
    return anthropic.Anthropic()


# --------------------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------------------

def collect_targets(args: argparse.Namespace) -> list[dict]:
    if args.escalate_from:
        targets = select_escalations(args.escalate_from, args.confidence_below)
        print(f"{len(targets)} rows below confidence {args.confidence_below} (or failed) "
              f"in {args.escalate_from}")
        return targets[:args.limit] if args.limit else targets
    if not args.manifest.exists():
        sys.exit(f"no manifest at {args.manifest} -- run Stage 3a (render_overlays.py) first")
    targets = load_manifest(args.manifest, args.limit)
    print(f"{len(targets)} panels in {args.manifest}")
    return targets


def cmd_estimate(args: argparse.Namespace) -> None:
    targets = collect_targets(args)
    if not targets:
        sys.exit("nothing to estimate")
    image_tokens = measure_image_tokens(targets)
    est = estimate_cost(len(targets), args.model, image_tokens)
    print(f"\n--- estimate ({args.model}, Batch API at 50%) ---")
    print(f"  panels:        {est['n_requests']}")
    print(f"  image tokens:  ~{image_tokens:.0f} per panel")
    print(f"  input tokens:  ~{est['tokens_in']:,.0f}   -> ${est['cost_in_usd']:.2f}")
    print(f"  output tokens: ~{est['tokens_out']:,.0f}   -> ${est['cost_out_usd']:.2f}")
    print(f"  TOTAL:         ~${est['cost_usd']:.2f}")

    approx_bytes = sum(
        Path(t["overlay_path"]).stat().st_size for t in targets[:50]
        if Path(t["overlay_path"]).exists()
    )
    if approx_bytes:
        per_panel = approx_bytes / min(50, len(targets)) * 1.33 * 0.35  # base64 of a JPEG re-encode
        n_batches = max(1, math.ceil(len(targets) * per_panel / (MAX_BATCH_BYTES * BATCH_BYTES_HEADROOM)))
        print(f"  estimated {n_batches} batch(es) at the 256MB cap")

    per_1m = estimate_cost(1_000_000, args.model, image_tokens)
    print(f"\n  for reference, the same prompt over 1,000,000 frames: ~${per_1m['cost_usd']:,.0f}")


def cmd_submit(args: argparse.Namespace) -> None:
    targets = collect_targets(args)
    if not targets:
        sys.exit("nothing to submit")

    client = require_client()
    requests, mapping, failures = build_payload(targets, args)
    for failure in failures[:10]:
        print(f"  !! {failure}")
    if failures:
        print(f"  !! {len(failures)} panels could not be read and were skipped")
    if not requests:
        sys.exit("no readable panels; nothing submitted")

    chunks = chunk_requests(requests)
    print(f"submitting {len(requests)} requests in {len(chunks)} batch(es) to {args.model}")

    batch_ids = []
    for i, chunk in enumerate(chunks):
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        print(f"  batch {i + 1}/{len(chunks)}: {batch.id} ({len(chunk)} requests)")

    record = {
        "name": args.name,
        "model": args.model,
        "batch_ids": batch_ids,
        "n_requests": len(requests),
        "n_skipped": len(failures),
        "escalated_from": str(args.escalate_from) if args.escalate_from else None,
        "confidence_below": args.confidence_below if args.escalate_from else None,
        "custom_ids": mapping,
    }
    path = record_path(args.batch_dir, args.name)
    save_record(path, record)
    print(f"\nwrote batch record to {path}")
    print(f"  next: python scripts/qa/triage.py poll --name {args.name}")


def cmd_poll(args: argparse.Namespace) -> None:
    record = load_record(record_path(args.batch_dir, args.name))
    client = require_client()
    all_ended = True
    for batch_id in record["batch_ids"]:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  {batch_id}: {batch.processing_status}  "
              f"succeeded={counts.succeeded} errored={counts.errored} "
              f"processing={counts.processing} canceled={counts.canceled} expired={counts.expired}")
        if batch.processing_status != "ended":
            all_ended = False
    print("\nall batches ended" if all_ended else "\nstill processing; re-run poll later")
    if all_ended:
        print(f"  next: python scripts/qa/triage.py fetch --name {args.name}")


def cmd_fetch(args: argparse.Namespace) -> None:
    record = load_record(record_path(args.batch_dir, args.name))
    client = require_client()
    mapping = record["custom_ids"]
    out = args.out or (REPO_ROOT / "outputs" / "qa" / "verdicts.csv")

    rows: list[dict] = []
    seen: set[str] = set()
    for batch_id in record["batch_ids"]:
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            seen.add(custom_id)
            target = mapping.get(custom_id, {})
            row = {
                "custom_id": custom_id,
                "video_name": target.get("video_name", ""),
                "frame_idx": target.get("frame_idx", ""),
                "instance_idx": target.get("instance_idx", ""),
                "site": target.get("site", ""),
                "stratum": target.get("stratum", ""),
                "prefilter_class": target.get("prefilter_class", ""),
                "flags": target.get("flags", ""),
                "overlay_path": target.get("overlay_path", ""),
                "model": record["model"],
                "result_type": result.result.type,
                "verdict": "", "confidence": "", "rationale": "", "error": "",
            }
            if result.result.type == "succeeded":
                try:
                    row.update(parse_verdict(text_of(result.result.message)))
                except (ValueError, json.JSONDecodeError) as exc:
                    row["result_type"] = "unparseable"
                    row["error"] = str(exc)
            else:
                row["error"] = str(getattr(result.result, "error", "") or result.result.type)
            rows.append(row)

    missing = set(mapping) - seen
    if missing:
        print(f"  !! {len(missing)} submitted requests returned no result "
              f"(cancelled or expired); recorded as missing")
        for custom_id in sorted(missing):
            target = mapping[custom_id]
            rows.append({
                "custom_id": custom_id, "model": record["model"], "result_type": "missing",
                "error": "no result returned for this custom_id",
                "verdict": "", "confidence": "", "rationale": "",
                **{k: target.get(k, "") for k in
                   ("video_name", "frame_idx", "instance_idx", "site", "stratum",
                    "prefilter_class", "flags", "overlay_path")},
            })

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VERDICT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} verdicts to {out}")

    counts = Counter(r["verdict"] or f"<{r['result_type']}>" for r in rows)
    print("\n--- verdict mix ---")
    for verdict, count in counts.most_common():
        print(f"  {verdict:<16} {count:>5}  ({count / len(rows):.1%})")
    ok = counts.get("ok", 0)
    scored = sum(c for v, c in counts.items() if not v.startswith("<"))
    if scored:
        print(f"\n  bad-mask rate: {(scored - ok) / scored:.3f} ({scored - ok}/{scored} scored)")


def main() -> None:
    args = parse_args()
    {"estimate": cmd_estimate, "submit": cmd_submit,
     "poll": cmd_poll, "fetch": cmd_fetch}[args.command](args)


if __name__ == "__main__":
    main()
