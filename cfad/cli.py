"""Command-line interface for CFAD."""

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
    """Load returns from CSV as either a dated Series or a NumPy array."""
    parse_dates = [date_col] if date_col else False
    frame = pd.read_csv(input_path, parse_dates=parse_dates)
    if column not in frame.columns:
        raise KeyError(f"Column '{column}' not found in {input_path}")

    values = pd.to_numeric(frame[column], errors="coerce")
    if date_col is None:
        return values.dropna().to_numpy(dtype=np.float64)

    if date_col not in frame.columns:
        raise KeyError(f"Date column '{date_col}' not found in {input_path}")
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=np.float64), index=dates, name=column)
    return series[series.index.notna()].dropna()


def _detect_to_dataframe(report, include_dates: bool = True) -> pd.DataFrame:
    """Convert detector output to a tabular frame for CSV export."""
    n_windows = len(report.scores)
    alarm_flags = np.zeros(n_windows, dtype=bool)
    valid_alarm_idx = report.alarm_indices[
        (report.alarm_indices >= 0) & (report.alarm_indices < n_windows)
    ]
    alarm_flags[valid_alarm_idx] = True

    output = pd.DataFrame(
        {
            "window_idx": np.arange(n_windows, dtype=np.int64),
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
        output["window_end_date"] = pd.to_datetime(report.dates[date_idx]).astype(str)
    return output


def _run_detect(args: argparse.Namespace) -> int:
    """Execute the detect subcommand."""
    data = _load_returns(args.input, args.column, args.date_col)
    report = detect(
        data,
        window=args.window,
        step=args.step,
        h=args.h,
        k=args.k,
        xi_range=(args.xi_min, args.xi_max),
        n_xi=args.n_xi,
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
    """Execute the model-comparison subcommand."""
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
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="cfad",
        description="ECF-based distributional-shape anomaly detection for returns.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Run detector on a CSV file")
    detect_parser.add_argument("input", type=pathlib.Path, help="CSV file with returns column")
    detect_parser.add_argument("--column", default="return", help="Returns column name")
    detect_parser.add_argument("--date-col", default=None, help="Optional date column name")
    detect_parser.add_argument("--window", type=int, default=60)
    detect_parser.add_argument("--step", type=int, default=1)
    detect_parser.add_argument("--k", type=float, default=0.5)
    detect_parser.add_argument("--h", type=float, default=5.0)
    detect_parser.add_argument("--xi-min", type=float, default=-10.0)
    detect_parser.add_argument("--xi-max", type=float, default=10.0)
    detect_parser.add_argument("--n-xi", type=int, default=128)
    detect_parser.add_argument("--calibration-frac", type=float, default=0.3)
    detect_parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional CSV path for scores and CUSUM diagnostics",
    )
    detect_parser.add_argument("--format", choices=["text", "json"], default="text")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare Gaussian and NIG distributional fit",
    )
    compare_parser.add_argument("input", type=pathlib.Path)
    compare_parser.add_argument("--column", default="return")
    compare_parser.add_argument("--format", choices=["text", "json"], default="text")

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
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        print(f"cfad: error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
