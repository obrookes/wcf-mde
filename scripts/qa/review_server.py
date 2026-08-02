#!/usr/bin/env python
"""Minimalist human review UI for the Haiku-flagged bad masks — a single-file stdlib HTTP
server meant to run on the login node and be browsed through an SSH tunnel:

    python scripts/qa/review_server.py \
        --verdicts  .../qa_pilot/verdicts_haiku.csv \
        --frames-dir .../export_test/frames \
        --masks-dir  .../export_test/masks \
        --out       .../qa_pilot/corrections.csv
    # locally:  ssh -L 8765:localhost:8765 <login-node>  ->  http://localhost:8765

The queue is every verdict row with `result_type=succeeded` and `verdict != ok`, grouped by
failure class. For each mask the page shows the pilot overlay panel, the raw exported frame
(drag on it to draw a correction box), and a deterministic morphology auto-fix preview
(largest connected component + hole fill). One keypress appends a decision row to the
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
        return self.frames_dir / f"{item['video_name']}_frame{int(item['frame_idx']):06d}.png"

    def instance_mask(self, item: dict) -> np.ndarray:
        instances = load_instance_masks(self.masks_dir, item["video_name"],
                                        int(item["frame_idx"]))
        want = int(item["instance_idx"])
        for inst in instances:
            if inst["instance_idx"] == want:
                return inst["mask"]
        raise KeyError(f"instance {want} not in mask JSON for {key_of(item)}")

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
            elif self.path.startswith("/overlay/"):
                item = self._item("/overlay/")
                self._png(cv2.imread(item["overlay_path"]) if item else None)
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
           background: #22252b; position: sticky; top: 0; }
  header .prog { font-weight: 600; }
  #meta { padding: .4em 1em; color: #aab; }
  #meta .verdict { padding: .1em .5em; border-radius: 3px; background: #7c2d2d; color: #fff;
                   font-weight: 600; margin-right: .6em; }
  .done { color: #6c6; font-weight: 600; margin-left: .6em; }
  main { padding: 0 1em 5em; max-width: 1180px; }
  .imgs img { max-width: 100%; display: block; background: #000; }
  .pair { display: flex; gap: 8px; margin-top: 8px; }
  .pair > div { flex: 1; min-width: 0; }
  .cap { color: #889; font-size: 12px; margin: 2px 0; }
  #framewrap { position: relative; }
  #boxcanvas { position: absolute; inset: 0; cursor: crosshair; }
  #notes { width: 24em; background: #22252b; color: #dde; border: 1px solid #444;
           padding: .3em; }
  #msg { color: #fa5; min-height: 1.2em; padding: .2em 1em; }
  .keys { color: #778; padding: .4em 1em 1em; }
  kbd { background: #333; border-radius: 3px; padding: 0 .4em; }
  select { background: #22252b; color: #dde; border: 1px solid #444; }
</style></head><body>
<header>
  <span class="prog" id="prog"></span>
  <label>class <select id="filter"><option value="">all</option></select></label>
  <span id="pos"></span>
</header>
<div id="msg"></div>
<div id="meta"></div>
<main>
  <div class="imgs">
    <div class="cap">triage overlay (as graded by Haiku)</div>
    <img id="overlay" alt="overlay">
    <div class="pair">
      <div>
        <div class="cap">raw frame — drag to draw a re-prompt box, then <b>b</b></div>
        <div id="framewrap"><img id="frame" alt="frame"><canvas id="boxcanvas"></canvas></div>
      </div>
      <div>
        <div class="cap" id="fixcap">auto-fix preview (green = kept+filled, red = original)</div>
        <img id="autofix" alt="autofix">
      </div>
    </div>
  </div>
  <p><label>notes <input id="notes" placeholder="optional"></label></p>
</main>
<div class="keys">
  <kbd>a</kbd> accept as-is &nbsp; <kbd>f</kbd> accept auto-fix &nbsp;
  <kbd>b</kbd> save drawn box &nbsp; <kbd>d</kbd> discard &nbsp; <kbd>s</kbd> skip &nbsp;
  <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate &nbsp; <kbd>Esc</kbd> clear box
</div>
<script>
let items = [], decisions = {}, order = [], pos = 0, box = null, drag = null;

const $ = id => document.getElementById(id);

async function init() {
  const r = await (await fetch('/api/queue')).json();
  items = r.items; decisions = r.decisions;
  for (const c of r.classes) {
    const o = document.createElement('option'); o.value = c; o.textContent = c;
    $('filter').appendChild(o);
  }
  $('filter').onchange = () => { rebuild(); render(); };
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

function render() {
  const nDone = order.filter(i => decisions[items[i].key]).length;
  $('prog').textContent = nDone + ' / ' + order.length + ' decided';
  if (!order.length) { $('meta').textContent = 'queue empty for this filter'; return; }
  const it = items[order[pos]];
  $('pos').textContent = '#' + (pos + 1) + '  ' + it.key;
  const d = decisions[it.key];
  $('meta').innerHTML = '<span class="verdict">' + it.verdict + '</span>' +
    'conf ' + it.confidence + ' — ' + esc(it.rationale) +
    (it.flags ? ' <span style="color:#667">[' + esc(it.flags) + ']</span>' : '') +
    (d ? '<span class="done">decided: ' + d.action + '</span>' : '');
  const i = order[pos];
  $('overlay').src = '/overlay/' + i + '.png';
  $('frame').src = '/frame/' + i + '.png';
  $('autofix').src = '/autofix/' + i + '.png';
  fetch('/api/autofix/' + i).then(r => r.json()).then(s => {
    if (s.components_before !== undefined)
      $('fixcap').textContent = 'auto-fix preview — ' + s.components_before +
        ' component(s), area ' + s.area_before + ' \\u2192 ' + s.area_after;
  }).catch(() => {});
  box = null; drawBox();
}

function esc(s) {
  return String(s || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

function flash(m) { $('msg').textContent = m; setTimeout(() => $('msg').textContent = '', 4000); }

async function decide(action) {
  if (!order.length) return;
  const it = items[order[pos]];
  if (action === 'box' && !box) { flash('draw a box on the frame first'); return; }
  const body = { key: it.key, action, notes: $('notes').value,
                 box: action === 'box' ? box : null };
  const r = await (await fetch('/api/decide', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
  if (!r.ok) { flash(r.error); return; }
  decisions[it.key] = r.row; $('notes').value = '';
  const nxt = order.findIndex((i, j) => j > pos && !decisions[items[i].key]);
  pos = nxt >= 0 ? nxt : Math.min(pos + 1, order.length - 1);
  render();
}

// box drawing: canvas overlays the frame img; store native-pixel coords
const cv = $('boxcanvas'), img = $('frame');
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
  else if (k === 'b') decide('box');
  else if (k === 'd') decide('discard');
  else if (k === 's') decide('skip');
  else if (k === 'ArrowRight') { pos = Math.min(pos + 1, order.length - 1); render(); }
  else if (k === 'ArrowLeft') { pos = Math.max(pos - 1, 0); render(); }
  else if (k === 'Escape') { box = null; drawBox(); }
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
