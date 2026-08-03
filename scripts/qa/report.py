#!/usr/bin/env python
"""Stage 5 of the SAM-3 mask QA pilot: draw the gold set, then report the dials.

Two subcommands:

    goldset    draw the overlays to hand-label, BLIND, and split them tune/holdout
    summarise  f_v, bad-mask rate + mix, verifier precision/recall, and the 1M cost table

**The gold template deliberately omits the model's verdict.** If the labeller can see what the
verifier said, the labels stop being an independent trust anchor and the precision/recall
numbers measure agreement-under-anchoring instead. The template carries only the overlay path
and a blank `gold_verdict` column; `summarise` joins the verdicts back on afterwards.

**The escalation threshold is fitted on the tune half and reported on the holdout half.**
Fitting and reporting on the same labels would make the headline trust number an in-sample fit,
which is exactly the mistake the split exists to prevent.

**Rates carry Wilson intervals.** With ~30 labels per class a point estimate is close to
meaningless -- 3 correct out of 4 is a 95% interval of roughly 30-99% -- and the decision gate
is supposed to be a judgement about evidence, not about a point estimate. For corpus-weighted
rates the interval uses Kish's effective sample size, since unequal weights buy less precision
than their raw count suggests.

CPU only, seconds to run -- fine on a login node. Reads only CSVs; no network, no API key.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.qa.triage import ESCALATION_MODEL, TRIAGE_MODEL, estimate_cost
from scripts.qa.verdicts_schema import VERDICT_CLASSES, key_of

REPO_ROOT = Path(__file__).resolve().parents[2]

GOLD_FIELDS = [
    # everything the labeller needs, and nothing that reveals the model's answer
    "key", "overlay_path", "split", "draw_reason", "gold_verdict", "notes",
]
GOLD_VERDICT_HELP = " | ".join(VERDICT_CLASSES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp_gold = sub.add_parser("goldset", help="draw the blind hand-labelling worksheet")
    sp_gold.add_argument("--prefilter-csv", type=Path,
                         default=REPO_ROOT / "outputs" / "qa" / "prefilter.csv")
    sp_gold.add_argument("--verdicts", type=Path,
                         default=REPO_ROOT / "outputs" / "qa" / "verdicts_haiku.csv")
    sp_gold.add_argument("--manifest", type=Path,
                         default=REPO_ROOT / "outputs" / "qa" / "overlay_manifest.csv")
    sp_gold.add_argument("--per-class", type=int, default=30,
                         help="stratified draw per predicted verdict class -> per-class precision")
    sp_gold.add_argument("--random", type=int, default=50,
                         help="uniform draw over all triaged panels -> recall")
    sp_gold.add_argument("--auto-pass", type=int, default=40,
                         help="draw from pre-filter auto-passes -> silent pre-filter false "
                              "negatives, which nothing else in the funnel would catch")
    sp_gold.add_argument("--tune-frac", type=float, default=0.5)
    sp_gold.add_argument("--seed", type=int, default=20260731)
    sp_gold.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "qa" / "gold_template.csv")

    sp_sum = sub.add_parser("summarise", help="the dials and the 1M extrapolation")
    sp_sum.add_argument("--prefilter-csv", type=Path,
                        default=REPO_ROOT / "outputs" / "qa" / "prefilter.csv")
    sp_sum.add_argument("--verdicts", type=Path,
                        default=REPO_ROOT / "outputs" / "qa" / "verdicts_haiku.csv")
    sp_sum.add_argument("--escalated-verdicts", type=Path, default=None,
                        help="outputs/qa/verdicts_opus.csv, if the escalation pass has run")
    sp_sum.add_argument("--gold", type=Path, default=None,
                        help="the filled-in gold template; omit to report the funnel only")
    sp_sum.add_argument("--confidence-threshold", type=float, default=None,
                        help="escalation threshold; default is fitted on the gold TUNE half")
    sp_sum.add_argument("--corpus-frames", type=float, default=1_000_000,
                        help="population for the extrapolation")
    sp_sum.add_argument("--image-tokens", type=float, default=None,
                        help="per-panel image tokens; measured from a rendered panel if omitted")
    sp_sum.add_argument("--gbp-per-usd", type=float, default=0.795)
    sp_sum.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "qa" / "report.md")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------

def wilson_interval(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the normal approximation because these counts are small and often near
    0 or 1, where the normal interval runs outside [0, 1] and understates uncertainty.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def effective_n(weights: list[float]) -> float:
    """Kish's effective sample size: (sum w)^2 / sum(w^2).

    Unequal inclusion weights buy less precision than the raw count suggests, so an interval
    computed on the raw n would overstate confidence in the corpus-weighted rate.
    """
    if not weights:
        return 0.0
    total = sum(weights)
    sq = sum(w * w for w in weights)
    return (total * total / sq) if sq > 0 else 0.0


def weighted_rate(rows: list[dict], predicate, weight_key: str = "inclusion_weight") -> dict:
    """Corpus-weighted proportion with a Wilson interval on the effective sample size."""
    weights, hits = [], []
    for row in rows:
        try:
            weight = float(row.get(weight_key) or "")
        except (TypeError, ValueError):
            continue
        weights.append(weight)
        hits.append(weight if predicate(row) else 0.0)
    if not weights:
        return {"rate": None, "n": 0, "n_eff": 0.0, "lo": None, "hi": None}
    rate = sum(hits) / sum(weights)
    n_eff = effective_n(weights)
    lo, hi = wilson_interval(rate * n_eff, n_eff)
    return {"rate": rate, "n": len(weights), "n_eff": n_eff, "lo": lo, "hi": hi}


def unweighted_rate(rows: list[dict], predicate) -> dict:
    n = len(rows)
    hits = sum(1 for r in rows if predicate(r))
    lo, hi = wilson_interval(hits, n)
    return {"rate": (hits / n if n else None), "n": n, "hits": hits, "lo": lo, "hi": hi}


def prf(pred: list[str], gold: list[str], label: str) -> dict:
    """Precision / recall / F1 for one class, each with a Wilson interval."""
    tp = sum(1 for p, g in zip(pred, gold) if p == label and g == label)
    fp = sum(1 for p, g in zip(pred, gold) if p == label and g != label)
    fn = sum(1 for p, g in zip(pred, gold) if p != label and g == label)
    n_pred, n_gold = tp + fp, tp + fn
    precision = tp / n_pred if n_pred else None
    recall = tp / n_gold if n_gold else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    return {
        "label": label, "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "precision_ci": wilson_interval(tp, n_pred) if n_pred else None,
        "recall": recall, "recall_ci": wilson_interval(tp, n_gold) if n_gold else None,
        "f1": f1, "n_pred": n_pred, "n_gold": n_gold,
    }


def binary_metrics(pred: list[str], gold: list[str]) -> dict:
    """ok-vs-bad, the number the decision gate actually turns on: does the verifier separate
    usable masks from unusable ones, regardless of which failure class it names."""
    pred_bad = ["bad" if p != "ok" else "ok" for p in pred]
    gold_bad = ["bad" if g != "ok" else "ok" for g in gold]
    metrics = prf(pred_bad, gold_bad, "bad")
    n = len(pred)
    correct = sum(1 for p, g in zip(pred_bad, gold_bad) if p == g)
    metrics["accuracy"] = correct / n if n else None
    metrics["accuracy_ci"] = wilson_interval(correct, n)
    return metrics


def fit_threshold(rows: list[dict]) -> tuple[float, dict]:
    """Pick the escalation confidence threshold on the TUNE rows.

    Objective: catch as many wrong verdicts as possible while escalating as few panels as
    possible -- i.e. maximise (recall of wrong verdicts) - (escalation rate). Ties break toward
    the lower threshold, which escalates less.
    """
    candidates = [i / 20 for i in range(21)]
    best, best_score = 0.0, -math.inf
    table = {}
    for threshold in candidates:
        escalated = [r for r in rows if r["confidence"] < threshold]
        wrong = [r for r in rows if r["pred"] != r["gold"]]
        caught = [r for r in wrong if r["confidence"] < threshold]
        recall = len(caught) / len(wrong) if wrong else 0.0
        rate = len(escalated) / len(rows) if rows else 0.0
        score = recall - rate
        table[threshold] = {"escalation_rate": rate, "wrong_caught": recall, "score": score}
        if score > best_score:
            best, best_score = threshold, score
    return best, table


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_verdicts(path: Path, escalated: Path | None) -> dict[str, dict]:
    """key -> verdict row. Escalated verdicts replace the ones they were escalated from."""
    merged: dict[str, dict] = {}
    for row in read_csv(path):
        merged[key_of(row)] = row
    n_replaced = 0
    if escalated and escalated.exists():
        for row in read_csv(escalated):
            key = key_of(row)
            if row.get("result_type") == "succeeded" and row.get("verdict"):
                if key in merged:
                    n_replaced += 1
                merged[key] = row
        print(f"  {n_replaced} verdicts replaced by the escalation pass")
    return merged


def load_gold(path: Path) -> dict[str, dict]:
    """key -> {gold_verdict, split}, skipping unlabelled rows."""
    gold: dict[str, dict] = {}
    n_blank = 0
    for row in read_csv(path):
        label = (row.get("gold_verdict") or "").strip()
        if not label:
            n_blank += 1
            continue
        if label not in VERDICT_CLASSES:
            print(f"  !! gold label {label!r} for {row.get('key')} is not one of "
                  f"{VERDICT_CLASSES}; skipping")
            continue
        gold[row["key"]] = {"gold_verdict": label, "split": row.get("split", "holdout")}
    if n_blank:
        print(f"  {n_blank} gold rows are still unlabelled and were skipped")
    return gold


# --------------------------------------------------------------------------------------
# goldset
# --------------------------------------------------------------------------------------

def cmd_goldset(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    prefilter = read_csv(args.prefilter_csv)
    manifest = {key_of(r): r["overlay_path"] for r in read_csv(args.manifest)} \
        if args.manifest.exists() else {}
    verdicts = {key_of(r): r for r in read_csv(args.verdicts)} if args.verdicts.exists() else {}

    picked: dict[str, str] = {}  # key -> draw_reason, first reason wins

    def take(pool: list[str], n: int, reason: str) -> None:
        fresh = [k for k in pool if k not in picked]
        if not fresh or n <= 0:
            return
        idx = rng.choice(len(fresh), size=min(n, len(fresh)), replace=False)
        for i in sorted(int(j) for j in idx):
            picked[fresh[i]] = reason

    # 1. stratified on the PREDICTED class -> per-class precision with usable per-class n
    by_class: dict[str, list[str]] = defaultdict(list)
    for key, row in verdicts.items():
        if row.get("verdict"):
            by_class[row["verdict"]].append(key)
    for verdict in VERDICT_CLASSES:
        take(sorted(by_class.get(verdict, [])), args.per_class, f"stratified:{verdict}")

    # 2. uniform over everything triaged -> recall (a stratified draw alone cannot give it)
    take(sorted(verdicts), args.random, "random")

    # 3. pre-filter auto-passes -> silent false negatives; nothing else in the funnel sees these
    auto_pass = sorted(key_of(r) for r in prefilter if r.get("disposition") == "pass")
    take(auto_pass, args.auto_pass, "auto_pass")

    keys = sorted(picked)
    if not keys:
        sys.exit("nothing to draw: no verdicts and no auto-passes found")

    # tune/holdout split, drawn independently of the reason so both halves span every stratum
    order = rng.permutation(len(keys))
    n_tune = int(round(args.tune_frac * len(keys)))
    split_of = {keys[int(j)]: ("tune" if rank < n_tune else "holdout")
                for rank, j in enumerate(order)}

    rows = [{
        "key": key,
        "overlay_path": manifest.get(key, ""),
        "split": split_of[key],
        "draw_reason": picked[key],
        "gold_verdict": "",
        "notes": "",
    } for key in keys]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GOLD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_missing = sum(1 for r in rows if not r["overlay_path"])
    print(f"wrote {len(rows)} gold rows to {args.out}")
    print(f"  tune {sum(1 for r in rows if r['split'] == 'tune')} / "
          f"holdout {sum(1 for r in rows if r['split'] == 'holdout')}")
    print("\n  draw reasons:")
    for reason, count in sorted(Counter(r["draw_reason"] for r in rows).items()):
        print(f"    {reason:<24} {count}")
    if n_missing:
        print(f"\n  !! {n_missing} rows have no overlay path (auto-passes are not rendered by "
              f"default). Re-run render_overlays.py with --dispositions vision pass to see them.")
    print(f"\n  Label the `gold_verdict` column with one of: {GOLD_VERDICT_HELP}")
    print("  The model's verdict is deliberately NOT in this file -- labelling must be blind, "
          "or precision/recall measure anchoring rather than accuracy.")


# --------------------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------------------

def funnel_summary(prefilter: list[dict], verdicts: dict[str, dict]) -> dict:
    """The pilot's headline dials, over the annotated stratum."""
    eligible = [r for r in prefilter
                if r.get("disposition") != "excluded" and r.get("stratum") != "unannotated"]
    unannotated = [r for r in prefilter
                   if r.get("disposition") != "excluded" and r.get("stratum") == "unannotated"]

    def classify(row: dict) -> str:
        """Final class for one instance: pre-filter decides pass/fail, the model decides the rest."""
        disposition = row.get("disposition")
        if disposition == "fail":
            return "empty"
        if disposition == "pass":
            return "ok"
        verdict = verdicts.get(key_of(row), {})
        return verdict.get("verdict") or "<no verdict>"

    for row in eligible + unannotated:
        row["final_class"] = classify(row)

    is_bad = lambda r: r["final_class"] not in ("ok", "<no verdict>")  # noqa: E731
    scored = [r for r in eligible if r["final_class"] != "<no verdict>"]

    return {
        "f_v_unweighted": unweighted_rate(eligible, lambda r: r.get("disposition") == "vision"),
        "f_v_weighted": weighted_rate(eligible, lambda r: r.get("disposition") == "vision"),
        "bad_unweighted": unweighted_rate(scored, is_bad),
        "bad_weighted": weighted_rate(scored, is_bad),
        "mix": Counter(r["final_class"] for r in scored),
        "n_eligible": len(eligible),
        "n_scored": len(scored),
        "n_unscored": len(eligible) - len(scored),
        "n_excluded": sum(1 for r in prefilter if r.get("disposition") == "excluded"),
        "by_site": {
            site: unweighted_rate([r for r in scored if r.get("site") == site], is_bad)
            for site in sorted({r.get("site", "") for r in scored})
        },
        "unannotated_mix": Counter(r["final_class"] for r in unannotated),
        "unannotated_bad": unweighted_rate(
            [r for r in unannotated if r["final_class"] != "<no verdict>"], is_bad),
    }


def gold_metrics(verdicts: dict[str, dict], gold: dict[str, dict],
                 threshold: float | None) -> dict:
    """Per-class and binary metrics, fitted on tune and reported on holdout."""
    paired = []
    for key, label in gold.items():
        verdict = verdicts.get(key)
        if not verdict or not verdict.get("verdict"):
            continue
        try:
            confidence = float(verdict.get("confidence") or "nan")
        except (TypeError, ValueError):
            confidence = float("nan")
        paired.append({
            "key": key, "pred": verdict["verdict"], "gold": label["gold_verdict"],
            "split": label["split"], "confidence": confidence,
            "model": verdict.get("model", ""),
        })

    tune = [r for r in paired if r["split"] == "tune"]
    holdout = [r for r in paired if r["split"] == "holdout"]

    fitted, table = (None, {})
    if threshold is None and tune:
        fitted, table = fit_threshold([r for r in tune if math.isfinite(r["confidence"])])
        threshold = fitted

    pred = [r["pred"] for r in holdout]
    gold_labels = [r["gold"] for r in holdout]
    return {
        "n_paired": len(paired), "n_tune": len(tune), "n_holdout": len(holdout),
        "threshold": threshold, "fitted_on_tune": fitted, "threshold_table": table,
        "per_class": [prf(pred, gold_labels, c) for c in VERDICT_CLASSES],
        "binary": binary_metrics(pred, gold_labels) if holdout else None,
        "holdout_escalation_rate": (
            sum(1 for r in holdout if math.isfinite(r["confidence"]) and r["confidence"] < threshold)
            / len(holdout) if holdout and threshold is not None else None),
        "by_model": {
            model: binary_metrics([r["pred"] for r in holdout if r["model"] == model],
                                  [r["gold"] for r in holdout if r["model"] == model])
            for model in sorted({r["model"] for r in holdout if r["model"]})
        },
    }


def extrapolate(f_v: float, escalation_rate: float, image_tokens: float,
                corpus_frames: float, gbp_per_usd: float) -> dict:
    vision_calls = corpus_frames * f_v
    haiku = estimate_cost(vision_calls, TRIAGE_MODEL, image_tokens)
    opus = estimate_cost(vision_calls * escalation_rate, ESCALATION_MODEL, image_tokens)
    no_filter = estimate_cost(corpus_frames, TRIAGE_MODEL, image_tokens)
    total = haiku["cost_usd"] + opus["cost_usd"]
    return {
        "vision_calls": vision_calls,
        "haiku_usd": haiku["cost_usd"],
        "escalation_calls": vision_calls * escalation_rate,
        "opus_usd": opus["cost_usd"],
        "total_usd": total,
        "total_gbp": total * gbp_per_usd,
        "no_prefilter_usd": no_filter["cost_usd"],
        "saved_usd": no_filter["cost_usd"] - haiku["cost_usd"],
    }


def _fmt(rate: dict | None, pct: bool = True) -> str:
    if not rate or rate.get("rate") is None:
        return "n/a"
    scale = 100 if pct else 1
    suffix = "%" if pct else ""
    return (f"{rate['rate'] * scale:.1f}{suffix} "
            f"[{rate['lo'] * scale:.1f}-{rate['hi'] * scale:.1f}] (n={rate['n']})")


def _fmt_pr(value, ci) -> str:
    if value is None:
        return "n/a"
    lo, hi = ci if ci else (0.0, 1.0)
    return f"{value * 100:.0f}% [{lo * 100:.0f}-{hi * 100:.0f}]"


def cmd_summarise(args: argparse.Namespace) -> None:
    prefilter = read_csv(args.prefilter_csv)
    verdicts = load_verdicts(args.verdicts, args.escalated_verdicts) \
        if args.verdicts.exists() else {}
    funnel = funnel_summary(prefilter, verdicts)

    image_tokens = args.image_tokens
    if image_tokens is None:
        import cv2
        image_tokens = 0.0
        for row in verdicts.values():
            path = row.get("overlay_path")
            if path and Path(path).exists():
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    image_tokens = w * h / 750.0
                    break
        if not image_tokens:
            image_tokens = 540.0
            print("  !! could not measure a panel; assuming 540 image tokens")

    gold = load_gold(args.gold) if args.gold and args.gold.exists() else {}
    metrics = gold_metrics(verdicts, gold, args.confidence_threshold) if gold else None

    escalation_rate = 0.0
    if metrics and metrics["holdout_escalation_rate"] is not None:
        escalation_rate = metrics["holdout_escalation_rate"]
    f_v = funnel["f_v_weighted"]["rate"] or funnel["f_v_unweighted"]["rate"] or 0.0
    costs = extrapolate(f_v, escalation_rate, image_tokens, args.corpus_frames, args.gbp_per_usd)

    lines: list[str] = []
    add = lines.append
    add("# SAM-3 mask QA pilot -- measured dials\n")
    add(f"Instances examined: {funnel['n_eligible']} "
        f"({funnel['n_excluded']} excluded as pipeline failures, not mask defects)\n")

    add("\n## 1. Pre-filter: f_v, the fraction needing a vision call\n")
    add(f"- unweighted: {_fmt(funnel['f_v_unweighted'])}")
    add(f"- **corpus-weighted: {_fmt(funnel['f_v_weighted'])}**  <- the one that scales to 1M")

    add("\n## 2. Bad-mask rate and failure mix\n")
    add(f"- unweighted: {_fmt(funnel['bad_unweighted'])}")
    add(f"- **corpus-weighted: {_fmt(funnel['bad_weighted'])}**")
    if funnel["n_unscored"]:
        add(f"- {funnel['n_unscored']} instances went to vision but have no verdict yet "
            f"(excluded from the rate above)")
    add("\n| class | n | share |")
    add("|---|---:|---:|")
    total_mix = sum(funnel["mix"].values()) or 1
    for name, count in funnel["mix"].most_common():
        add(f"| {name} | {count} | {count / total_mix:.1%} |")

    add("\n### per site (unweighted)\n")
    add("| site | bad-mask rate |")
    add("|---|---|")
    for site, rate in funnel["by_site"].items():
        add(f"| {site} | {_fmt(rate)} |")

    add("\n## 3. Unannotated stratum\n")
    if sum(funnel["unannotated_mix"].values()):
        add(f"- bad-mask rate: {_fmt(funnel['unannotated_bad'])}")
        add(f"- mix: {dict(funnel['unannotated_mix'])}")
        add("\nReported separately and never blended in: on frames with no annotation the "
            "sign-holder is frequently absent, so `empty` is often the *correct* mask rather "
            "than a failure. This is the honest measure of how far the annotated-frame mix "
            "transfers to the population the 1M figure refers to.")
    else:
        add("- not measured (no unannotated stratum in this run)")

    add("\n## 4. Verifier accuracy vs the human gold set\n")
    if not metrics:
        add("- no gold labels supplied; run `report.py goldset`, label it blind, and pass --gold")
    else:
        add(f"- paired labels: {metrics['n_paired']} "
            f"(tune {metrics['n_tune']}, holdout {metrics['n_holdout']})")
        if metrics["fitted_on_tune"] is not None:
            add(f"- escalation threshold **fitted on tune**: {metrics['fitted_on_tune']:.2f}")
        else:
            add(f"- escalation threshold supplied: {metrics['threshold']}")
        add("- everything below is on the **holdout** half, which the threshold never saw\n")
        if metrics["binary"]:
            b = metrics["binary"]
            add(f"**ok vs bad** (the number the gate turns on): "
                f"precision {_fmt_pr(b['precision'], b['precision_ci'])}, "
                f"recall {_fmt_pr(b['recall'], b['recall_ci'])}, "
                f"accuracy {_fmt_pr(b['accuracy'], b['accuracy_ci'])}\n")
        add("| class | precision | recall | n predicted | n gold |")
        add("|---|---|---|---:|---:|")
        for cls in metrics["per_class"]:
            add(f"| {cls['label']} | {_fmt_pr(cls['precision'], cls['precision_ci'])} "
                f"| {_fmt_pr(cls['recall'], cls['recall_ci'])} "
                f"| {cls['n_pred']} | {cls['n_gold']} |")
        if metrics["by_model"]:
            add("\n### by model\n")
            add("| model | ok-vs-bad precision | recall | n |")
            add("|---|---|---|---:|")
            for model, b in metrics["by_model"].items():
                add(f"| {model} | {_fmt_pr(b['precision'], b['precision_ci'])} "
                    f"| {_fmt_pr(b['recall'], b['recall_ci'])} | {b['n_pred'] + b['fn']} |")
        add("\nIntervals are Wilson 95%. Where a class has only a handful of gold labels its "
            "interval is wide by construction -- read the interval, not the point estimate.")

    add(f"\n## 5. Extrapolation to {args.corpus_frames:,.0f} frames\n")
    add(f"Measured inputs: f_v = {f_v:.3f}, escalation rate = {escalation_rate:.3f}, "
        f"{image_tokens:.0f} image tokens per panel.\n")
    add("| line | frames | cost (USD) |")
    add("|---|---:|---:|")
    add(f"| triage, no pre-filter | {args.corpus_frames:,.0f} | ${costs['no_prefilter_usd']:,.0f} |")
    add(f"| triage after pre-filter ({TRIAGE_MODEL}) | {costs['vision_calls']:,.0f} "
        f"| ${costs['haiku_usd']:,.0f} |")
    add(f"| escalation ({ESCALATION_MODEL}) | {costs['escalation_calls']:,.0f} "
        f"| ${costs['opus_usd']:,.0f} |")
    add(f"| **total** | | **${costs['total_usd']:,.0f}"
        f" / GBP {costs['total_gbp']:,.0f}** |")
    add(f"\nThe pre-filter saves ~${costs['saved_usd']:,.0f} on the triage line alone. "
        f"All figures are Batch API rates (50% off standard).")
    add("\n**Assumption, not a measurement.** 1,850 videos x ~500 frames reaches 1M only if "
        "every frame is segmented, whereas this pilot measured on annotated frames. Section 3 "
        "sizes how far the mix shifts off them; if that shift is large, this table is an "
        "upper bound on usefulness rather than a forecast.")

    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"\nwrote {args.out}")


def main() -> None:
    args = parse_args()
    {"goldset": cmd_goldset, "summarise": cmd_summarise}[args.command](args)


if __name__ == "__main__":
    main()
