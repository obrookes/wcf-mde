"""Mapping a flattened annotation `video_name` to its site.

Split out of `qc_annotations.py` so the QA scripts (and anything else that just needs to bucket
rows by site) don't pull pandas in for a one-line string operation. `qc_annotations.site_of`
still resolves -- it re-exports this -- so existing callers and imports are unaffected.
"""
from __future__ import annotations


def site_of(video_name: str) -> str:
    """First path component of the flattened name, e.g. "mafou_E2_..._DSCF0005" -> "mafou".

    Note this maps the `fello_sounga` site to "fello", since it splits on the first underscore
    and the site directory itself contains one. That is the historical behaviour and matches the
    per-site row counts recorded in CLAUDE.md, so it is kept deliberately rather than fixed --
    changing it would silently renumber every per-site statistic in the repo.
    """
    return video_name.split("_", 1)[0]
