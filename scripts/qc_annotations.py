#!/usr/bin/env python3
"""QC an annotations CSV for genuine, protocol-independent bugs.

Videos may contain two or more annotated people and distances are sparsely
annotated, so sequence-based heuristics (repeated distances, skipped values,
implied speed, direction reversals) CANNOT distinguish bugs from valid labels
and are deliberately not used here. Only violations that are impossible under
any annotation protocol are flagged.

Reads an annotations CSV (video_name, frame_idx, frame_timestamp, distance) and
data/video_fps.csv (produced by scripts/probe_video_fps.py) and writes
data/qc_flags_<basename>.csv with one row per (annotation row, reason):

    VIDEO_NOT_ON_DISK        video_name has no probed video file
    FRAME_IDX_FPS_MISMATCH   frame_idx != round(frame_timestamp * fps_probed) (+/-1 frame)
    TIMESTAMP_PAST_END       frame_timestamp exceeds probed video duration
    ABSURD_DISTANCE          distance > --max-distance (default 100 m; the dataset's
                             confirmed typos are all >= 207 m)
    ZERO_DISTANCE            distance <= 0

Rows whose (video_name, frame_timestamp) has no match in --compare-export are
reported as an informational count only, not flagged.

With --fix, additionally writes <input>_clean.csv (original never modified):
  - frame_idx recomputed as round(frame_timestamp * fps_probed) for on-disk videos
  - frame_idx blanked for videos missing from disk (or estimated at 24 fps with
    --fix-mode estimate, marked in a frame_idx_estimated column)
  - ABSURD_DISTANCE and ZERO_DISTANCE rows dropped
  - distances are never edited
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def site_of(video_name: str) -> str:
    return video_name.split("_", 1)[0]


def collect_flags(df: pd.DataFrame, fps_table: pd.DataFrame,
                  max_distance: float) -> pd.DataFrame:
    """Return flag rows: original columns + reason + detail."""
    flags: list[pd.DataFrame] = []

    def add(mask_or_idx, reason: str, detail: pd.Series | str = ""):
        sub = df.loc[mask_or_idx].copy()
        if sub.empty:
            return
        sub["reason"] = reason
        sub["detail"] = detail if isinstance(detail, str) else detail.loc[sub.index]
        flags.append(sub)

    m = df.merge(fps_table[["video_name", "fps", "duration_s"]],
                 on="video_name", how="left")
    m.index = df.index

    on_disk = m["fps"].notna()
    add(~on_disk, "VIDEO_NOT_ON_DISK")

    expected = (m["frame_timestamp"] * m["fps"]).round()
    mismatch = on_disk & ((m["frame_idx"] - expected).abs() > 1)
    add(mismatch, "FRAME_IDX_FPS_MISMATCH",
        "expected frame_idx " + expected.fillna(-1).astype(int).astype(str)
        + " at fps " + m["fps"].fillna(0).map("{:g}".format))

    past_end = on_disk & m["duration_s"].notna() \
        & (m["frame_timestamp"] > m["duration_s"] + 0.5)
    add(past_end, "TIMESTAMP_PAST_END",
        "video duration " + m["duration_s"].fillna(0).map("{:.1f}s".format))

    add(df["distance"] > max_distance, "ABSURD_DISTANCE")
    add(df["distance"] <= 0, "ZERO_DISTANCE")

    if not flags:
        return pd.DataFrame(columns=list(df.columns) + ["reason", "detail"])
    out = pd.concat(flags).sort_index()
    return out.reset_index(names="row")


def apply_fixes(df: pd.DataFrame, flags: pd.DataFrame, fps_table: pd.DataFrame,
                fix_mode: str) -> pd.DataFrame:
    clean = df.copy()

    drop_rows = flags.loc[
        flags["reason"].isin(["ABSURD_DISTANCE", "ZERO_DISTANCE"]), "row"].unique()

    fps_map = fps_table.set_index("video_name")["fps"]
    fps = clean["video_name"].map(fps_map)
    clean["frame_idx"] = clean["frame_idx"].astype("float64")
    on_disk = fps.notna()
    clean.loc[on_disk, "frame_idx"] = (
        clean.loc[on_disk, "frame_timestamp"] * fps[on_disk]).round()
    clean["frame_idx_estimated"] = False
    if fix_mode == "estimate":
        clean.loc[~on_disk, "frame_idx"] = (
            clean.loc[~on_disk, "frame_timestamp"] * 24).round()
        clean.loc[~on_disk, "frame_idx_estimated"] = True
    else:
        clean.loc[~on_disk, "frame_idx"] = np.nan

    clean = clean.drop(index=drop_rows)
    clean["frame_idx"] = clean["frame_idx"].astype("Int64")
    return clean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("annotations", type=Path, nargs="?",
                    default=DATA_ROOT / "annotations_20260709_with_fps.csv")
    ap.add_argument("--fps-table", type=Path, default=DATA_ROOT / "video_fps.csv")
    ap.add_argument("--compare-export", type=Path,
                    default=DATA_ROOT / "annotations_export_20260709.csv",
                    help="export CSV to diff timestamps against ('' to skip)")
    ap.add_argument("--max-distance", type=float, default=100.0)
    ap.add_argument("--fix", action="store_true",
                    help="also write <input>_clean.csv")
    ap.add_argument("--fix-mode", choices=["nan", "estimate"], default="nan",
                    help="frame_idx for videos missing from disk: blank (nan) "
                         "or estimate at 24 fps")
    args = ap.parse_args()

    df = pd.read_csv(args.annotations)
    fps_table = pd.read_csv(args.fps_table)

    flags = collect_flags(df, fps_table, args.max_distance)

    out = args.annotations.parent / f"qc_flags_{args.annotations.stem}.csv"
    flags.to_csv(out, index=False)
    print(f"{len(df)} rows, {df['video_name'].nunique()} videos -> "
          f"{len(flags)} flags on {flags['row'].nunique() if len(flags) else 0} rows")
    print(f"wrote {out}\n")
    if len(flags):
        print("flags by reason:")
        print(flags["reason"].value_counts().to_string())
        print("\nflags by site:")
        print(flags["video_name"].map(site_of).value_counts().to_string())

    # informational only: large-but-plausible distances (possible outliers, not bugs)
    big = df[(df["distance"] > 30) & (df["distance"] <= args.max_distance)]
    if len(big):
        print(f"\ninfo: {len(big)} rows with distance in (30, {args.max_distance:g}] m "
              f"(plausible, not flagged): "
              + ", ".join(f"{r.video_name}@{r.frame_timestamp:.1f}s={r.distance}m"
                          for r in big.itertuples()))

    # informational only: timestamps with no counterpart in the raw export
    if args.compare_export and str(args.compare_export) != "" \
            and args.compare_export.exists() \
            and args.compare_export.resolve() != args.annotations.resolve():
        export = pd.read_csv(args.compare_export)
        merged = df.merge(export[["video_name", "frame_timestamp"]].drop_duplicates(),
                          on=["video_name", "frame_timestamp"],
                          how="left", indicator=True)
        n_diff = int((merged["_merge"] == "left_only").sum())
        print(f"\ninfo: {n_diff} rows have no (video_name, frame_timestamp) match "
              f"in {args.compare_export.name} (not flagged)")

    if args.fix:
        clean = apply_fixes(df, flags, fps_table, args.fix_mode)
        clean_path = args.annotations.parent / f"{args.annotations.stem}_clean.csv"
        clean.to_csv(clean_path, index=False)
        n_blank = int(clean["frame_idx"].isna().sum())
        print(f"\nwrote {clean_path}: {len(clean)} rows "
              f"({len(df) - len(clean)} dropped, {n_blank} frame_idx blanked, "
              f"{int(clean['frame_idx_estimated'].sum())} estimated)")


if __name__ == "__main__":
    main()
