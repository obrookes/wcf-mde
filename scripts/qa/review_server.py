#!/usr/bin/env python
"""Minimalist human review UI for the Haiku-flagged bad masks — a single-file stdlib HTTP
server. The intended way to run it is LOCALLY, from a self-contained bundle built by
make_review_bundle.py (download the tarball, `python run_review.py`, browse localhost:8765 —
see the bundle's README.txt). It can also run on the login node through an SSH tunnel:

    python scripts/qa/review_server.py \
        --verdicts  .../qa_pilot/verdicts_haiku.csv \
        --frames-dir .../export_test/frames \
        --masks-dir  .../export_test/masks \
        --out       .../qa_pilot/corrections.csv
    # laptop:  ssh -L 8765:localhost:8765 <login-node>  ->  http://localhost:8765

The queue is every verdict row with `result_type=succeeded` and `verdict != ok`, grouped by
failure class. For each mask the page shows ONE large image with tabs: the original SAM-3
mask rendered on the frame (green = the instance under review, yellow = other instances in
the frame), a deterministic morphology auto-fix preview (largest connected component + hole
fill), a zoomed crop of the mask region, and the raw frame. Correction boxes are drawn in an
explicit draw mode (Draw box button → drag → Save). One decision appends a row to the
corrections CSV — flushed per write, so killing the server loses nothing and restarting
resumes where you left off (re-deciding a key appends again; last write wins downstream).

Actions: accept (mask is actually fine — Haiku false positive), autofix (morphology preview is
right), box (re-prompt SAM3 with the drawn box later, via apply_corrections.py once the weights
arrive), discard (unfixable — exclude the frame), skip.

This tool is for the flagged-bad triage queue ONLY. It displays the model verdict, so it must
never be used to label the blind gold set (`report.py goldset`). Prefilter auto-fail rows
(`status=empty_mask`) are not in the verdicts CSV and have no exported frame, so they cannot be
reviewed here — that gap is printed at startup rather than silently dropped.

CPU only; needs numpy, cv2, pycocotools (all in the wcf env). No frameworks.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.masks import load_instance_masks  # noqa: E402

BAD_CLASSES = ["empty", "wrong-subject", "bleed", "split", "multiple"]
ACTIONS = {"accept", "autofix", "box", "discard", "skip"}
CORRECTION_FIELDS = [
    "key", "video_name", "frame_idx", "instance_idx", "haiku_verdict", "action",
    "box_x0", "box_y0", "box_x1", "box_y1", "notes", "decided_at",
]

# --------------------------------------------------------------------------------------
# pure logic (unit-tested in test_review.py)
# --------------------------------------------------------------------------------------


def key_of(row: dict) -> str:
    return f"{row['video_name']}|{row['frame_idx']}|{row['instance_idx']}"


def frame_filename(video_name: str, frame_idx: int | str) -> str:
    """Exported-frame naming convention shared with export_calibrated_frames.py."""
    return f"{video_name}_frame{int(frame_idx):06d}.png"


def build_queue(rows: list[dict], only_classes: list[str] | None = None) -> list[dict]:
    """Verdict-CSV rows -> ordered review queue: succeeded, flagged-bad rows sorted by
    failure class (grouping like failures makes review fast), then video/frame/instance."""
    classes = set(only_classes) if only_classes else set(BAD_CLASSES)
    order = {c: i for i, c in enumerate(BAD_CLASSES)}
    queue = [r for r in rows
             if r.get("result_type") == "succeeded" and r.get("verdict") in classes]
    queue.sort(key=lambda r: (order.get(r["verdict"], len(order)), r["video_name"],
                              int(r["frame_idx"]), int(r["instance_idx"])))
    return queue


def autofix_mask(mask: np.ndarray) -> np.ndarray:
    """Deterministic morphology fix: keep the largest connected component, fill its holes.
    Identity on empty masks. The 1-px zero pad before flood-fill means a mask touching the
    frame corner cannot hijack the background seed."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:  # background only
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = (labels == largest).astype(np.uint8)
    padded = cv2.copyMakeBorder(keep, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 1)
    holes = padded[1:-1, 1:-1] == 0  # zeros the border flood couldn't reach
    return keep.astype(bool) | holes


def clamp_box(box, width: int, height: int) -> list[int] | None:
    """[x0,y0,x1,y1] in any corner order, possibly off-image -> ordered ints clamped to the
    image; None if degenerate (< 2 px a side) after clamping."""
    try:
        x0, x1 = sorted((float(box[0]), float(box[2])))
        y0, y1 = sorted((float(box[1]), float(box[3])))
    except (TypeError, ValueError, IndexError):
        return None
    x0 = max(0, min(int(round(x0)), width - 1))
    x1 = max(0, min(int(round(x1)), width - 1))
    y0 = max(0, min(int(round(y0)), height - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [x0, y0, x1, y1]


def load_decisions(path: Path) -> dict[str, dict]:
    """corrections CSV -> {key: row}, last write wins (rows are append-only)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with open(path, newline="") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def append_decision(path: Path, row: dict) -> None:
    """Append one decision, writing the header iff the file is new. Flushed immediately so a
    killed server loses at most the in-flight keypress."""
    path = Path(path)
    fresh = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CORRECTION_FIELDS)
        if fresh:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CORRECTION_FIELDS})
        f.flush()


def render_autofix_panel(frame_bgr: np.ndarray, before: np.ndarray,
                         after: np.ndarray) -> np.ndarray:
    """Frame with the fixed mask as green fill+contour and the original's dropped extent as a
    thin red contour, so what the fix removed is visible."""
    out = frame_bgr.copy()
    a = after.astype(bool)
    green = np.zeros_like(out)
    green[:] = (0, 200, 0)
    out[a] = (0.55 * out[a] + 0.45 * green[a]).astype(np.uint8)
    cnts, _ = cv2.findContours(before.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 0, 255), 1)
    cnts, _ = cv2.findContours(after.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    return out


def render_mask_panel(frame_bgr: np.ndarray, target: np.ndarray,
                      siblings: list[np.ndarray] | None = None) -> np.ndarray:
    """The mask exactly as SAM-3 produced it: target instance as green fill + contour, any
    other instances in the frame as thin yellow contours for context (split/multiple calls
    need to see the neighbours)."""
    out = frame_bgr.copy()
    for sib in siblings or []:
        cnts, _ = cv2.findContours(sib.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 220, 255), 1)
    t = target.astype(bool)
    green = np.zeros_like(out)
    green[:] = (0, 200, 0)
    out[t] = (0.55 * out[t] + 0.45 * green[t]).astype(np.uint8)
    cnts, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    return out


def zoom_bbox(mask: np.ndarray, margin: float = 0.4,
              min_half: int = 60) -> tuple[int, int, int, int]:
    """Square-ish crop window around the mask's bbox with breathing room, clamped to the
    image; the full image if the mask is empty."""
    h, w = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, w, h
    cx, cy = (int(xs.min()) + int(xs.max())) // 2, (int(ys.min()) + int(ys.max())) // 2
    half = max(int(max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + margin) / 2),
               min_half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    return x0, y0, x1, y1


def render_zoom_panel(rendered: np.ndarray, mask: np.ndarray,
                      out_width: int = 720, max_scale: int = 4) -> np.ndarray:
    """Crop an already-rendered panel to the mask region and upscale with nearest-neighbour
    so the true mask-edge pixels stay visible."""
    x0, y0, x1, y1 = zoom_bbox(mask)
    crop = rendered[y0:y1, x0:x1]
    scale = min(max_scale, max(1, out_width // max(1, crop.shape[1])))
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------------------


class ReviewApp:
    def __init__(self, queue: list[dict], frames_dir: Path, masks_dir: Path, out_csv: Path):
        self.queue = queue
        self.frames_dir = Path(frames_dir)
        self.masks_dir = Path(masks_dir)
        self.out_csv = Path(out_csv)
        self.decisions = load_decisions(out_csv)
        self.lock = threading.Lock()

    def frame_path(self, item: dict) -> Path:
        return self.frames_dir / frame_filename(item["video_name"], item["frame_idx"])

    def instance_mask(self, item: dict) -> np.ndarray:
        target, _ = self.frame_masks(item)
        return target

    def frame_masks(self, item: dict) -> tuple[np.ndarray, list[np.ndarray]]:
        """(target instance mask, other instances in the same frame)."""
        instances = load_instance_masks(self.masks_dir, item["video_name"],
                                        int(item["frame_idx"]))
        want = int(item["instance_idx"])
        target, siblings = None, []
        for inst in instances:
            if inst["instance_idx"] == want:
                target = inst["mask"]
            else:
                siblings.append(inst["mask"])
        if target is None:
            raise KeyError(f"instance {want} not in mask JSON for {key_of(item)}")
        return target, siblings

    def item_json(self, i: int, item: dict) -> dict:
        return {
            "index": i, "key": key_of(item),
            "video_name": item["video_name"], "frame_idx": item["frame_idx"],
            "instance_idx": item["instance_idx"], "site": item.get("site", ""),
            "verdict": item["verdict"], "confidence": item.get("confidence", ""),
            "rationale": item.get("rationale", ""), "flags": item.get("flags", ""),
            "prefilter_class": item.get("prefilter_class", ""),
        }

    def decide(self, payload: dict) -> dict:
        action = payload.get("action")
        if action not in ACTIONS:
            return {"ok": False, "error": f"bad action {action!r}"}
        by_key = {key_of(it): it for it in self.queue}
        item = by_key.get(payload.get("key"))
        if item is None:
            return {"ok": False, "error": f"unknown key {payload.get('key')!r}"}
        box = ["", "", "", ""]
        if action == "box":
            frame = cv2.imread(str(self.frame_path(item)))
            if frame is None:
                return {"ok": False, "error": "frame image missing; cannot record a box"}
            clamped = clamp_box(payload.get("box") or [], frame.shape[1], frame.shape[0])
            if clamped is None:
                return {"ok": False, "error": "box is degenerate or off-image; redraw it"}
            box = clamped
        row = {
            "key": key_of(item), "video_name": item["video_name"],
            "frame_idx": item["frame_idx"], "instance_idx": item["instance_idx"],
            "haiku_verdict": item["verdict"], "action": action,
            "box_x0": box[0], "box_y0": box[1], "box_x1": box[2], "box_y1": box[3],
            "notes": (payload.get("notes") or "").replace("\n", " ").strip(),
            "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self.lock:
            append_decision(self.out_csv, row)
            self.decisions[row["key"]] = row
        return {"ok": True, "row": row}


class Handler(BaseHTTPRequestHandler):
    app: ReviewApp  # set on the class before serving

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _png(self, img: np.ndarray | None) -> None:
        if img is None:
            self._json({"error": "image not found"}, 404)
            return
        okay, buf = cv2.imencode(".png", img)
        if not okay:
            self._json({"error": "encode failed"}, 500)
            return
        self._send(200, buf.tobytes(), "image/png")

    def _item(self, prefix: str) -> dict | None:
        tail = self.path[len(prefix):].split(".")[0]
        try:
            return self.app.queue[int(tail)]
        except (ValueError, IndexError):
            return None

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        app = self.app
        try:
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/queue":
                self._json({
                    "items": [app.item_json(i, it) for i, it in enumerate(app.queue)],
                    "decisions": app.decisions,
                    "classes": BAD_CLASSES,
                })
            elif self.path.startswith("/mask/"):
                item = self._item("/mask/")
                if item is None:
                    self._png(None)
                    return
                frame = cv2.imread(str(app.frame_path(item)))
                if frame is None:
                    self._png(None)
                    return
                target, siblings = app.frame_masks(item)
                self._png(render_mask_panel(frame, target, siblings))
            elif self.path.startswith("/zoom/"):
                item = self._item("/zoom/")
                if item is None:
                    self._png(None)
                    return
                frame = cv2.imread(str(app.frame_path(item)))
                if frame is None:
                    self._png(None)
                    return
                target, siblings = app.frame_masks(item)
                self._png(render_zoom_panel(render_mask_panel(frame, target, siblings),
                                            target))
            elif self.path.startswith("/frame/"):
                item = self._item("/frame/")
                self._png(cv2.imread(str(app.frame_path(item))) if item else None)
            elif self.path.startswith("/autofix/"):
                item = self._item("/autofix/")
                if item is None:
                    self._png(None)
                    return
                frame = cv2.imread(str(app.frame_path(item)))
                before = app.instance_mask(item)
                if frame is None:
                    self._png(None)
                    return
                self._png(render_autofix_panel(frame, before, autofix_mask(before)))
            elif self.path.startswith("/api/autofix/"):
                item = self._item("/api/autofix/")
                if item is None:
                    self._json({"error": "bad index"}, 404)
                    return
                before = app.instance_mask(item)
                after = autofix_mask(before)
                n, *_ = cv2.connectedComponentsWithStats(before.astype(np.uint8))
                self._json({"components_before": int(n) - 1,
                            "area_before": int(before.sum()),
                            "area_after": int(after.sum())})
            else:
                self._json({"error": "not found"}, 404)
        except (OSError, KeyError, FileNotFoundError) as exc:
            self._json({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/decide":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json({"ok": False, "error": "bad JSON"}, 400)
            return
        result = self.app.decide(payload)
        self._json(result, 200 if result["ok"] else 400)


# --------------------------------------------------------------------------------------
# page (embedded single-page app; no external resources)
# --------------------------------------------------------------------------------------

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>mask review</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #16181c; color: #dde;
         font-size: 14px; }
  header { display: flex; gap: 1.2em; align-items: baseline; padding: .5em 1em;
           background: #22252b; position: sticky; top: 0; z-index: 2; }
  header .prog { font-weight: 600; }
  main { padding: 0 1em 2em; max-width: 1100px; margin: 0 auto; }
  #info { display: flex; gap: .8em; align-items: baseline; flex-wrap: wrap;
          padding: .6em .8em; margin: .8em 0 .4em; background: #22252b;
          border-radius: 6px; }
  #info .verdict { padding: .15em .6em; border-radius: 3px; background: #7c2d2d;
                   color: #fff; font-weight: 600; text-transform: uppercase;
                   font-size: 12px; letter-spacing: .04em; }
  #info .conf { color: #9ab; }
  #info .rationale { color: #ccd; flex: 1 1 22em; }
  #info .flags { color: #667; font-size: 12px; }
  .done { color: #6c6; font-weight: 600; }
  .tabs { display: flex; gap: 4px; margin-top: .6em; }
  .tabs button { background: #22252b; color: #9ab; border: 1px solid #333;
                 border-bottom: none; border-radius: 6px 6px 0 0; padding: .4em .9em;
                 cursor: pointer; font: inherit; }
  .tabs button.active { background: #2c313a; color: #fff; font-weight: 600; }
  #cap { color: #99a; font-size: 12px; padding: .4em .6em; background: #2c313a; }
  #imgwrap { position: relative; background: #000; border-radius: 0 0 6px 6px;
             overflow: hidden; }
  #view { max-width: 100%; display: block; }
  #boxcanvas { position: absolute; inset: 0; cursor: crosshair; display: none; }
  #drawhint { display: none; background: #4a4022; color: #ffe8a0; padding: .45em .8em;
              border-radius: 4px; margin: .5em 0; }
  .actions { display: flex; gap: .5em; flex-wrap: wrap; align-items: center;
             margin: .8em 0 .4em; }
  .actions button { font: inherit; padding: .5em .9em; border-radius: 5px;
                    border: 1px solid #444; background: #262a31; color: #dde;
                    cursor: pointer; }
  .actions button:hover { background: #313743; }
  .actions button kbd { margin-left: .5em; }
  .actions .accept  { border-color: #2d6a2d; }
  .actions .fix     { border-color: #2d5a7c; }
  .actions .draw    { border-color: #8a7326; }
  .actions .draw.on { background: #8a7326; color: #fff; }
  .actions .save    { background: #8a7326; color: #fff; display: none; }
  .actions .discard { border-color: #7c2d2d; }
  .actions .nav { margin-left: auto; }
  #notes { width: 24em; background: #22252b; color: #dde; border: 1px solid #444;
           padding: .35em; border-radius: 4px; }
  #msg { color: #fa5; min-height: 1.2em; padding: .2em 0; }
  kbd { background: #333; border-radius: 3px; padding: 0 .4em; font-size: 12px; }
  select { background: #22252b; color: #dde; border: 1px solid #444; }
  .legend { color: #778; padding: .5em 0; font-size: 12px; }
</style></head><body>
<header>
  <span class="prog" id="prog"></span>
  <label>class <select id="filter"><option value="">all</option></select></label>
  <span id="pos" style="color:#9ab"></span>
</header>
<main>
  <div id="info"></div>
  <div id="msg"></div>
  <div class="tabs" id="tabs"></div>
  <div id="cap"></div>
  <div id="imgwrap"><img id="view" alt="panel"><canvas id="boxcanvas"></canvas></div>
  <div id="drawhint">Drag on the image to draw a box around the correct subject, then
    <b>Save box</b> (Enter). <b>Esc</b> cancels.</div>
  <div class="actions">
    <button class="accept" onclick="decide('accept')">Mask is fine<kbd>a</kbd></button>
    <button class="fix" onclick="decide('autofix')">Accept auto-fix<kbd>f</kbd></button>
    <button class="draw" id="drawbtn" onclick="toggleDraw()">Draw box&hellip;<kbd>b</kbd></button>
    <button class="save" id="savebtn" onclick="decide('box')">Save box<kbd>Enter</kbd></button>
    <button class="discard" onclick="decide('discard')">Discard<kbd>d</kbd></button>
    <button onclick="decide('skip')">Skip<kbd>s</kbd></button>
    <button class="nav" onclick="nav(-1)">&larr; Prev</button>
    <button onclick="nav(1)">Next &rarr;</button>
  </div>
  <p><label>notes <input id="notes" placeholder="optional note saved with the decision"></label></p>
  <div class="legend">Green = this instance's SAM-3 mask &nbsp;&middot;&nbsp; yellow outline =
    other instances in the frame &nbsp;&middot;&nbsp; tabs: <kbd>1</kbd>&ndash;<kbd>4</kbd>
    &nbsp;&middot;&nbsp; <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate</div>
</main>
<script>
const TABS = [
  {id: 'mask', label: 'Original mask', url: '/mask/',
   cap: 'The mask exactly as SAM-3 produced it \\u2014 green fill = the instance under ' +
        'review, yellow outline = other instances detected in this frame.'},
  {id: 'autofix', label: 'Auto-fix preview', url: '/autofix/',
   cap: 'Deterministic fix (keep largest component, fill holes) \\u2014 green = result, ' +
        'red outline = original mask extent.'},
  {id: 'zoom', label: 'Zoom', url: '/zoom/',
   cap: 'Magnified crop around the mask, true pixels (same colours as the mask tab).'},
  {id: 'raw', label: 'Raw frame', url: '/frame/',
   cap: 'The exported frame with no overlay.'},
];
let items = [], decisions = {}, order = [], pos = 0, tab = 'mask';
let drawMode = false, box = null, drag = null;
const fixStats = {};
const $ = id => document.getElementById(id);

async function init() {
  const r = await (await fetch('/api/queue')).json();
  items = r.items; decisions = r.decisions;
  for (const c of r.classes) {
    const o = document.createElement('option'); o.value = c; o.textContent = c;
    $('filter').appendChild(o);
  }
  $('filter').onchange = () => { rebuild(); render(); };
  for (const t of TABS) {
    const b = document.createElement('button');
    b.id = 'tab-' + t.id; b.textContent = t.label; b.onclick = () => setTab(t.id);
    $('tabs').appendChild(b);
  }
  rebuild();
  pos = order.findIndex(i => !decisions[items[i].key]);
  if (pos < 0) pos = 0;
  render();
}

function rebuild() {
  const f = $('filter').value;
  order = items.map((it, i) => i).filter(i => !f || items[i].verdict === f);
  pos = Math.min(pos, Math.max(0, order.length - 1));
}

function setTab(t) {
  tab = t;
  if (t !== 'mask' && drawMode) setDraw(false);
  render();
}

function setDraw(on) {
  drawMode = on; box = null; drag = null;
  if (on) tab = 'mask';  // boxes are drawn in full-frame coordinates
  $('drawbtn').classList.toggle('on', on);
  $('drawbtn').innerHTML = on ? 'Cancel<kbd>Esc</kbd>' : 'Draw box&hellip;<kbd>b</kbd>';
  $('savebtn').style.display = on ? '' : 'none';
  $('drawhint').style.display = on ? 'block' : 'none';
  render();
}

function toggleDraw() { setDraw(!drawMode); }

function render() {
  const nDone = order.filter(i => decisions[items[i].key]).length;
  $('prog').textContent = nDone + ' / ' + order.length + ' decided';
  if (!order.length) { $('info').textContent = 'queue empty for this filter'; return; }
  const i = order[pos], it = items[i];
  $('pos').textContent = '#' + (pos + 1) + '  ' + it.key;
  const d = decisions[it.key];
  $('info').innerHTML =
    '<span class="verdict">' + esc(it.verdict) + '</span>' +
    '<span class="conf">conf ' + esc(it.confidence) + '</span>' +
    '<span class="rationale">' + esc(it.rationale) + '</span>' +
    (it.flags ? '<span class="flags">[' + esc(it.flags) + ']</span>' : '') +
    (d ? '<span class="done">decided: ' + esc(d.action) + '</span>' : '');
  const t = TABS.find(t => t.id === tab);
  for (const tb of TABS)
    $('tab-' + tb.id).classList.toggle('active', tb.id === tab);
  $('cap').textContent = t.cap;
  if (tab === 'autofix') {
    if (fixStats[i]) capStats(i);
    else fetch('/api/autofix/' + i).then(r => r.json()).then(s => {
      fixStats[i] = s; if (tab === 'autofix' && order[pos] === i) capStats(i);
    }).catch(() => {});
  }
  $('view').src = t.url + i + '.png';
  $('boxcanvas').style.display = (drawMode && tab === 'mask') ? 'block' : 'none';
  drawBox();
}

function capStats(i) {
  const s = fixStats[i];
  if (s.components_before !== undefined)
    $('cap').textContent = TABS[1].cap + '  (' + s.components_before +
      ' component(s), area ' + s.area_before + ' \\u2192 ' + s.area_after + ')';
}

function esc(s) {
  return String(s || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

function flash(m) { $('msg').textContent = m; setTimeout(() => $('msg').textContent = '', 4000); }

async function decide(action) {
  if (!order.length) return;
  const it = items[order[pos]];
  if (action === 'box' && !box) { flash('draw a box on the image first'); return; }
  const body = { key: it.key, action, notes: $('notes').value,
                 box: action === 'box' ? box : null };
  const r = await (await fetch('/api/decide', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
  if (!r.ok) { flash(r.error); return; }
  decisions[it.key] = r.row; $('notes').value = '';
  if (drawMode) setDraw(false);
  const nxt = order.findIndex((i, j) => j > pos && !decisions[items[i].key]);
  pos = nxt >= 0 ? nxt : Math.min(pos + 1, order.length - 1);
  render();
}

function nav(step) {
  if (!order.length) return;
  pos = Math.max(0, Math.min(pos + step, order.length - 1));
  box = null;
  if (drawMode) setDraw(false); else render();
}

// box drawing: canvas overlays the panel img; coords stored in native frame pixels
const cv = $('boxcanvas'), img = $('view');
function syncCanvas() {
  cv.width = img.clientWidth; cv.height = img.clientHeight;
  cv.style.width = img.clientWidth + 'px'; cv.style.height = img.clientHeight + 'px';
  drawBox();
}
img.addEventListener('load', syncCanvas);
window.addEventListener('resize', syncCanvas);
function toNative(e) {
  const r = cv.getBoundingClientRect();
  const sx = img.naturalWidth / r.width, sy = img.naturalHeight / r.height;
  return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
}
cv.addEventListener('mousedown', e => { drag = toNative(e); });
cv.addEventListener('mousemove', e => {
  if (drag) { box = [...drag, ...toNative(e)]; drawBox(); }
});
window.addEventListener('mouseup', () => { drag = null; });
function drawBox() {
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!box || !img.naturalWidth) return;
  const sx = cv.width / img.naturalWidth, sy = cv.height / img.naturalHeight;
  ctx.strokeStyle = '#ff0'; ctx.lineWidth = 2;
  ctx.strokeRect(Math.min(box[0], box[2]) * sx, Math.min(box[1], box[3]) * sy,
                 Math.abs(box[2] - box[0]) * sx, Math.abs(box[3] - box[1]) * sy);
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const k = e.key;
  if (k === 'a') decide('accept');
  else if (k === 'f') decide('autofix');
  else if (k === 'b') toggleDraw();
  else if (k === 'Enter' && drawMode) decide('box');
  else if (k === 'd') decide('discard');
  else if (k === 's') decide('skip');
  else if (k === '1') setTab('mask');
  else if (k === '2') setTab('autofix');
  else if (k === '3') setTab('zoom');
  else if (k === '4') setTab('raw');
  else if (k === 'ArrowRight') nav(1);
  else if (k === 'ArrowLeft') nav(-1);
  else if (k === 'Escape') { if (drawMode) setDraw(false); else { box = null; drawBox(); } }
});
init();
</script></body></html>
"""


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verdicts", type=Path, required=True,
                   help="triage verdicts CSV (triage.py fetch or merge_verdicts.py output)")
    p.add_argument("--frames-dir", type=Path, required=True,
                   help="exported frame PNGs (<video_name>_frame%%06d.png)")
    p.add_argument("--masks-dir", type=Path, required=True,
                   help="mask JSONs as written by run_calibration_eval.py --save-mask-dir")
    p.add_argument("--out", type=Path, default=None,
                   help="corrections CSV to append decisions to "
                        "(default: corrections.csv next to --verdicts)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--only-class", action="append", choices=BAD_CLASSES, default=None,
                   help="restrict the queue to these verdict classes (repeatable)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_csv = args.out or args.verdicts.parent / "corrections.csv"
    with open(args.verdicts, newline="") as f:
        rows = list(csv.DictReader(f))
    queue = build_queue(rows, args.only_class)
    app = ReviewApp(queue, args.frames_dir, args.masks_dir, out_csv)

    from collections import Counter
    mix = Counter(it["verdict"] for it in queue)
    print(f"queue: {len(queue)} flagged masks "
          f"({', '.join(f'{c} {n}' for c, n in mix.most_common())})", flush=True)
    print(f"decisions so far: {len(app.decisions)} (appending to {out_csv})", flush=True)
    print("note: prefilter auto-fail rows (status=empty_mask) are not in the verdicts CSV and "
          "have no exported frame — they are NOT reviewable here.", flush=True)
    print(f"serving on http://localhost:{args.port}  "
          f"(tunnel with: ssh -L {args.port}:localhost:{args.port} <this-host>)", flush=True)

    Handler.app = app
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
