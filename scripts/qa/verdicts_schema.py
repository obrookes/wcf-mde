"""The verdicts-CSV contract shared by every stage of the mask-QA pilot that reads or writes one
(triage.py, merge_verdicts.py, review_server.py, make_review_bundle.py, report.py).

Stdlib-only, and importing nothing else in this repo: this module is one of the files
`make_review_bundle.py` copies into the self-contained review bundle (see its `BUNDLED_CODE`
list), which ships to a laptop with no numpy/cv2/pycocotools and no `scripts/` package on
`sys.path` beyond what the bundle itself provides. A single `import numpy` here would break
that bundle silently until someone tried to run it offline.

The contract, in full:

* A verdicts CSV has exactly the columns in `VERDICT_FIELDS`, in that order -- no more, no
  fewer. `triage.py fetch` and `merge_verdicts.py` both write this shape; `report.py` and
  `review_server.py` both read it.
* A row is reviewable (fit to show a human, or to score against gold) only when its
  `result_type` is literally the string `"succeeded"` (see `RESULT_SUCCEEDED`) -- not
  `"errored"`, `"missing"`, `"unparseable"`, or anything else a failed API call or a bad parse
  can leave behind.
* On a succeeded row, `frame_idx` and `instance_idx` are guaranteed int-parseable: the review
  queue (`review_server.build_queue`) sorts on `int(row["frame_idx"])` and
  `int(row["instance_idx"])`, and a row that fails that conversion belongs to a different
  code path, not this one.
* A row whose `verdict` is not one of `VERDICT_CLASSES` is not an error to raise -- the review
  queue silently drops it (an empty `verdict` from an unparseable/errored/missing row included).
* Two distinct notions of "priority" exist here, and they are not the same order:
  - The grading rubric's own tie-break, for the model choosing *one* verdict when a mask could
    arguably fit several: "wrong-subject" beats "multiple" beats "bleed" beats "split" (see the
    RUBRIC text in `triage.py`) -- this picks the verdict that would most mislead a downstream
    distance estimate.
  - The human review queue's *grouping* order is simply `BAD_CLASSES` in order -- `empty`,
    `wrong-subject`, `bleed`, `split`, `multiple` -- with `ok` always sorting last. `BAD_CLASSES`
    is `VERDICT_CLASSES` with `"ok"` removed, order otherwise preserved.
"""
from __future__ import annotations

VERDICT_CLASSES = ["ok", "empty", "wrong-subject", "bleed", "split", "multiple"]

# Same order as VERDICT_CLASSES, minus "ok" -- also the review queue's grouping order (see
# module docstring; distinct from the rubric's verdict-selection tie-break).
BAD_CLASSES = [c for c in VERDICT_CLASSES if c != "ok"]

VERDICT_FIELDS = [
    "custom_id", "video_name", "frame_idx", "instance_idx", "site", "stratum",
    "prefilter_class", "flags", "model", "verdict", "confidence", "rationale",
    "result_type", "error", "overlay_path",
]

RESULT_SUCCEEDED = "succeeded"


def key_of(row: dict) -> str:
    """The join key used everywhere a verdict/correction/gold row needs to line up with the
    others: `video_name|frame_idx|instance_idx`, as strings exactly as stored in the row (no
    int coercion -- callers that need the int form convert it themselves)."""
    return f"{row['video_name']}|{row['frame_idx']}|{row['instance_idx']}"


def blank_verdict_row() -> dict:
    """A VERDICT_FIELDS-shaped row with every column blank, for callers building up a row
    field-by-field rather than via a dict literal."""
    return {f: "" for f in VERDICT_FIELDS}
