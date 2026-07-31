#!/usr/bin/env python
"""Synthetic unit tests for scripts/qa/report.py.
Run directly:  python scripts/qa/test_report.py  (also pytest-compatible).
No GPU / data / torch / network needed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.qa.report import (
    binary_metrics,
    effective_n,
    extrapolate,
    fit_threshold,
    funnel_summary,
    key_of,
    prf,
    unweighted_rate,
    weighted_rate,
    wilson_interval,
)


# --------------------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------------------

def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(3, 4)
    assert lo < 0.75 < hi


def test_wilson_interval_is_wide_for_the_plans_worked_example():
    """The pilot's own argument for a stratified gold set: 3 of 4 correct is roughly 30-99%,
    so a 150-frame uniform set could not support per-class precision."""
    lo, hi = wilson_interval(3, 4)
    assert lo < 0.35 and hi > 0.95


def test_wilson_interval_narrows_with_more_data():
    narrow = wilson_interval(75, 100)
    wide = wilson_interval(3, 4)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_stays_inside_zero_one_at_the_extremes():
    """Where the normal approximation would run outside [0, 1]."""
    for lo, hi in (wilson_interval(0, 10), wilson_interval(10, 10), wilson_interval(0, 1)):
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_of_no_data_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_effective_n_equals_n_for_equal_weights():
    assert abs(effective_n([1.0] * 10) - 10.0) < 1e-9
    assert abs(effective_n([3.7] * 10) - 10.0) < 1e-9


def test_effective_n_shrinks_under_unequal_weights():
    """Unequal inclusion weights buy less precision than the raw count suggests, which is why
    the weighted interval must not be computed on n."""
    assert effective_n([10.0, 1.0, 1.0, 1.0]) < 4.0


def test_effective_n_of_empty_is_zero():
    assert effective_n([]) == 0.0


# --------------------------------------------------------------------------------------
# rates
# --------------------------------------------------------------------------------------

def test_unweighted_rate_counts_and_brackets():
    rows = [{"x": 1}, {"x": 1}, {"x": 0}, {"x": 0}]
    out = unweighted_rate(rows, lambda r: r["x"] == 1)
    assert out["rate"] == 0.5 and out["hits"] == 2 and out["n"] == 4
    assert out["lo"] < 0.5 < out["hi"]


def test_weighted_rate_recovers_the_corpus_rate():
    """Equal-per-site sampling over-represents small sites; the weighted rate corrects it."""
    rows = ([{"inclusion_weight": "3.0", "bad": False}] * 3
            + [{"inclusion_weight": "3.0", "bad": True}] * 1
            + [{"inclusion_weight": "0.2", "bad": True}] * 3
            + [{"inclusion_weight": "0.2", "bad": False}] * 1)
    out = weighted_rate(rows, lambda r: r["bad"])
    unweighted = unweighted_rate(rows, lambda r: r["bad"])
    assert abs(unweighted["rate"] - 0.5) < 1e-9
    assert abs(out["rate"] - (3.0 * 1 + 0.2 * 3) / (3.0 * 4 + 0.2 * 4)) < 1e-9
    assert out["rate"] < unweighted["rate"]


def test_weighted_rate_interval_uses_effective_sample_size():
    rows = [{"inclusion_weight": "10.0", "bad": True}] + \
           [{"inclusion_weight": "0.1", "bad": False}] * 9
    out = weighted_rate(rows, lambda r: r["bad"])
    assert out["n"] == 10
    assert out["n_eff"] < 10, "unequal weights must shrink the effective n"


def test_weighted_rate_skips_rows_without_a_weight():
    """The unannotated stratum carries a blank weight and must not enter a weighted rate."""
    rows = [{"inclusion_weight": "1.0", "bad": True}, {"inclusion_weight": "", "bad": False}]
    assert weighted_rate(rows, lambda r: r["bad"])["n"] == 1


def test_weighted_rate_of_nothing_is_none():
    assert weighted_rate([], lambda r: True)["rate"] is None


# --------------------------------------------------------------------------------------
# precision / recall
# --------------------------------------------------------------------------------------

def test_prf_perfect_prediction():
    out = prf(["bleed", "ok"], ["bleed", "ok"], "bleed")
    assert out["precision"] == 1.0 and out["recall"] == 1.0 and out["f1"] == 1.0


def test_prf_counts_fp_and_fn():
    pred = ["bleed", "bleed", "ok", "ok"]
    gold = ["bleed", "ok", "bleed", "ok"]
    out = prf(pred, gold, "bleed")
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 1)
    assert out["precision"] == 0.5 and out["recall"] == 0.5


def test_prf_is_none_rather_than_zero_when_a_class_never_appears():
    """A class the model never predicted has undefined precision; reporting 0% would read as
    'the model is bad at it' rather than 'there is no evidence either way'."""
    out = prf(["ok", "ok"], ["ok", "ok"], "split")
    assert out["precision"] is None and out["recall"] is None


def test_binary_metrics_collapses_every_failure_class_to_bad():
    pred = ["ok", "bleed", "split", "ok"]
    gold = ["ok", "split", "multiple", "wrong-subject"]
    out = binary_metrics(pred, gold)
    # predicted bad: bleed, split -> both genuinely bad => precision 1.0
    assert out["precision"] == 1.0
    # gold bad: split, multiple, wrong-subject -> two caught => recall 2/3
    assert abs(out["recall"] - 2 / 3) < 1e-9
    assert abs(out["accuracy"] - 0.75) < 1e-9


def test_binary_metrics_accuracy_carries_an_interval():
    out = binary_metrics(["ok"] * 4, ["ok"] * 4)
    assert out["accuracy"] == 1.0
    assert out["accuracy_ci"][0] < 1.0, "a 4/4 result must not read as certainty"


# --------------------------------------------------------------------------------------
# threshold fitting
# --------------------------------------------------------------------------------------

def _tune_row(pred: str, gold: str, confidence: float) -> dict:
    return {"pred": pred, "gold": gold, "confidence": confidence}


def test_fit_threshold_prefers_catching_wrong_verdicts_cheaply():
    """Wrong verdicts arrive with low confidence and right ones with high, so a threshold
    between the two groups should win."""
    rows = ([_tune_row("ok", "ok", 0.95) for _ in range(8)]
            + [_tune_row("ok", "bleed", 0.30) for _ in range(2)])
    threshold, table = fit_threshold(rows)
    assert 0.30 < threshold <= 0.95
    assert table[threshold]["wrong_caught"] == 1.0
    assert table[threshold]["escalation_rate"] <= 0.2


def test_fit_threshold_stays_low_when_confidence_carries_no_signal():
    """If confidence does not separate right from wrong, escalating is pure cost -- the fitted
    threshold should collapse toward zero rather than escalate everything."""
    rows = ([_tune_row("ok", "ok", 0.5) for _ in range(5)]
            + [_tune_row("ok", "bleed", 0.5) for _ in range(5)])
    threshold, _ = fit_threshold(rows)
    assert threshold <= 0.5


def test_fit_threshold_handles_no_wrong_verdicts():
    rows = [_tune_row("ok", "ok", 0.9) for _ in range(5)]
    threshold, _ = fit_threshold(rows)
    assert threshold == 0.0  # nothing to catch, so escalate nothing


# --------------------------------------------------------------------------------------
# funnel
# --------------------------------------------------------------------------------------

def _pf(video: str, frame: int, inst: int, disposition: str, weight: str = "1.0",
        site: str = "pnt", stratum: str = "annotated") -> dict:
    return {"video_name": video, "frame_idx": str(frame), "instance_idx": str(inst),
            "disposition": disposition, "inclusion_weight": weight, "site": site,
            "stratum": stratum}


def _verdict(row: dict, verdict: str) -> tuple[str, dict]:
    return key_of(row), {"verdict": verdict, "confidence": "0.9", "model": "m",
                         "video_name": row["video_name"], "frame_idx": row["frame_idx"],
                         "instance_idx": row["instance_idx"]}


def test_funnel_counts_auto_pass_as_ok_and_auto_fail_as_empty():
    prefilter = [_pf("v", 1, 0, "pass"), _pf("v", 2, 0, "fail")]
    summary = funnel_summary(prefilter, {})
    assert summary["mix"]["ok"] == 1
    assert summary["mix"]["empty"] == 1
    assert abs(summary["bad_unweighted"]["rate"] - 0.5) < 1e-9


def test_funnel_uses_the_model_verdict_for_vision_rows():
    vision = _pf("v", 3, 0, "vision")
    prefilter = [_pf("v", 1, 0, "pass"), vision]
    key, verdict = _verdict(vision, "bleed")
    summary = funnel_summary(prefilter, {key: verdict})
    assert summary["mix"]["bleed"] == 1
    assert abs(summary["bad_unweighted"]["rate"] - 0.5) < 1e-9


def test_funnel_excludes_pipeline_failures_from_every_rate():
    prefilter = [_pf("v", 1, 0, "pass"), _pf("v", 2, 0, "excluded")]
    summary = funnel_summary(prefilter, {})
    assert summary["n_eligible"] == 1
    assert summary["n_excluded"] == 1
    assert summary["bad_unweighted"]["n"] == 1


def test_funnel_does_not_score_vision_rows_that_have_no_verdict_yet():
    """An unscored row must not be silently counted as 'ok' -- that would understate the
    bad-mask rate by exactly the number of results still missing."""
    prefilter = [_pf("v", 1, 0, "pass"), _pf("v", 2, 0, "vision")]
    summary = funnel_summary(prefilter, {})
    assert summary["n_unscored"] == 1
    assert summary["bad_unweighted"]["n"] == 1  # only the auto-pass was scored


def test_funnel_keeps_the_unannotated_stratum_out_of_the_headline():
    prefilter = [_pf("v", 1, 0, "pass"),
                 _pf("v", 2, 0, "fail", weight="", stratum="unannotated")]
    summary = funnel_summary(prefilter, {})
    assert summary["bad_unweighted"]["n"] == 1
    assert summary["bad_unweighted"]["rate"] == 0.0
    assert summary["unannotated_mix"]["empty"] == 1
    assert summary["unannotated_bad"]["rate"] == 1.0


def test_funnel_f_v_is_the_share_routed_to_vision():
    prefilter = [_pf("v", 1, 0, "pass"), _pf("v", 2, 0, "pass"),
                 _pf("v", 3, 0, "vision"), _pf("v", 4, 0, "fail")]
    summary = funnel_summary(prefilter, {})
    assert abs(summary["f_v_unweighted"]["rate"] - 0.25) < 1e-9


def test_funnel_weighted_and_unweighted_f_v_differ_under_unequal_weights():
    prefilter = ([_pf("v", i, 0, "vision", weight="0.2", site="small") for i in range(3)]
                 + [_pf("v", i + 10, 0, "pass", weight="3.0", site="big") for i in range(3)])
    summary = funnel_summary(prefilter, {})
    assert abs(summary["f_v_unweighted"]["rate"] - 0.5) < 1e-9
    assert summary["f_v_weighted"]["rate"] < 0.2  # the big site is clean and dominates


# --------------------------------------------------------------------------------------
# extrapolation
# --------------------------------------------------------------------------------------

def test_extrapolate_scales_the_triage_line_by_f_v():
    full = extrapolate(1.0, 0.0, 540, 1_000_000, 0.795)
    half = extrapolate(0.5, 0.0, 540, 1_000_000, 0.795)
    assert abs(half["haiku_usd"] / full["haiku_usd"] - 0.5) < 1e-9
    assert abs(full["saved_usd"]) < 1e-9  # no pre-filter benefit at f_v = 1


def test_extrapolate_reports_escalation_as_its_own_line():
    """The plan's correction: escalation is comparable to the whole Haiku pass, not a rounding
    error, so it must never be folded into the triage total."""
    costs = extrapolate(0.5, 0.10, 540, 1_000_000, 0.795)
    assert costs["opus_usd"] > 0
    assert abs(costs["total_usd"] - (costs["haiku_usd"] + costs["opus_usd"])) < 1e-9
    assert costs["escalation_calls"] == 1_000_000 * 0.5 * 0.10


def test_extrapolate_escalation_at_five_percent_is_material():
    costs = extrapolate(1.0, 0.05, 540, 1_000_000, 0.795)
    assert costs["opus_usd"] / costs["haiku_usd"] > 0.2, "Opus is 5x Haiku; 5% is not negligible"


def test_extrapolate_converts_to_gbp():
    costs = extrapolate(0.5, 0.05, 540, 1_000_000, 0.8)
    assert abs(costs["total_gbp"] - costs["total_usd"] * 0.8) < 1e-9


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nall {len(fns)} report tests passed")


if __name__ == "__main__":
    _run_all()
