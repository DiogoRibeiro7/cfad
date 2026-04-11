"""Command-line interface for cfad."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from cfad import compare_models, detect
from cfad.residue_score import normalise_scores


def _load_returns(
    input_path: pathlib.Path,
    column: str,
    date_col: str | None = None,
) -> pd.Series | np.ndarray:
    """Load returns from CSV as either dated Series or plain ndarray."""
    parse_dates = [date_col] if date_col else False
    df = pd.read_csv(input_path, parse_dates=parse_dates)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in {input_path}")

    values = pd.to_numeric(df[column], errors="coerce")
    if date_col is None:
        return values.dropna().to_numpy(dtype=np.float64)

    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in {input_path}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=np.float64), index=dates, name=column)
    series = series[series.index.notna()].dropna()
    return series


def _detect_to_dataframe(report, include_dates: bool = True) -> pd.DataFrame:
    """Convert detector output to a tabular frame for CSV export."""
    n = len(report.scores)
    alarm_flags = np.zeros(n, dtype=bool)
    valid_alarm_idx = report.alarm_indices[
        (report.alarm_indices >= 0) & (report.alarm_indices < n)
    ]
    alarm_flags[valid_alarm_idx] = True

    out = pd.DataFrame(
        {
            "window_idx": np.arange(n, dtype=np.int64),
            "window_end_index": report.window_end_indices,
            "score": report.scores,
            "score_z": normalise_scores(report.scores, method="zscore"),
            "cusum_pos": report.cusum_pos,
            "cusum_neg": report.cusum_neg,
            "alarm": alarm_flags,
        }
    )
    if include_dates and report.dates is not None and len(report.dates) > 0:
        date_idx = np.clip(
            report.window_end_indices - 1,
            0,
            len(report.dates) - 1,
        )
        out["window_end_date"] = pd.to_datetime(report.dates[date_idx]).astype(str)
    return out


def _run_detect(args: argparse.Namespace) -> int:
    data = _load_returns(args.input, args.column, args.date_col)
    report = detect(
        data,
        window=args.window,
        step=args.step,
        h=args.h,
        xi_range=(args.xi_min, args.xi_max),
        n_xi=args.n_xi,
        height=args.height,
        calibration_frac=args.calibration_frac,
    )

    if args.output is not None:
        out_df = _detect_to_dataframe(report, include_dates=args.date_col is not None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output, index=False)

    if args.format == "json":
        payload = {
            "n_windows": int(len(report.scores)),
            "n_alarms": int(len(report.alarm_indices)),
            "mu0": float(report.mu0),
            "sigma0": float(report.sigma0),
            "alarm_indices": report.alarm_indices.tolist(),
            "output": str(args.output) if args.output is not None else None,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(report.summary())
        if args.output is not None:
            print(f"Saved detector output CSV to: {args.output}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    returns = _load_returns(args.input, args.column, date_col=None)
    result = compare_models(np.asarray(returns, dtype=np.float64))

    if args.format == "json":
        payload = {
            "gaussian": {
                "model": repr(result["gaussian"]["model"]),
                "ecf_l2": float(result["gaussian"]["ecf_l2"]),
                "aic": float(result["gaussian"]["aic"]),
            },
            "nig": {
                "model": repr(result["nig"]["model"]),
                "ecf_l2": float(result["nig"]["ecf_l2"]),
                "aic": float(result["nig"]["aic"]),
            },
            "winner": str(result["winner"]),
        }
        print(json.dumps(payload, indent=2))
    else:
        print("Model comparison")
        print(f"  Gaussian ECF-L2: {result['gaussian']['ecf_l2']:.6g}")
        print(f"  NIG ECF-L2     : {result['nig']['ecf_l2']:.6g}")
        print(f"  Winner         : {result['winner']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="cfad",
        description=(
            "Characteristic Function Anomaly Detector — detect structural "
            "breaks in return series."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- detect subcommand ---
    p_detect = subparsers.add_parser("detect", help="Run detector on a CSV file")
    p_detect.add_argument(
        "input", type=pathlib.Path, help="CSV file with returns column"
    )
    p_detect.add_argument(
        "--column",
        default="return",
        help="Column name for returns (default: return)",
    )
    p_detect.add_argument(
        "--date-col",
        default=None,
        help="Date column name (optional)",
    )
    p_detect.add_argument("--window", type=int, default=60)
    p_detect.add_argument("--step", type=int, default=1)
    p_detect.add_argument("--h", type=float, default=5.0)
    p_detect.add_argument("--xi-min", type=float, default=-10.0)
    p_detect.add_argument("--xi-max", type=float, default=10.0)
    p_detect.add_argument("--n-xi", type=int, default=128)
    p_detect.add_argument("--height", type=float, default=0.2)
    p_detect.add_argument("--calibration-frac", type=float, default=0.3)
    p_detect.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Save scores + CUSUM to CSV (optional)",
    )
    p_detect.add_argument("--format", choices=["text", "json"], default="text")

    # --- compare subcommand ---
    p_compare = subparsers.add_parser("compare", help="Compare Gaussian vs NIG model fit")
    p_compare.add_argument("input", type=pathlib.Path)
    p_compare.add_argument("--column", default="return")
    p_compare.add_argument("--format", choices=["text", "json"], default="text")

    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "detect":
            return _run_detect(args)
        if args.command == "compare":
            return _run_compare(args)
    except Exception as exc:  # pragma: no cover - top-level UX guard
        print(f"cfad: error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
