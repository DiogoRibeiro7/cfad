from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cfad import detect
from cfad.gof import aic_table
from cfad.utils import load_spy_sample
from cfad.viz import plot_detection_timeline

try:
    from cfad.bootstrap import plot_bootstrap_bands as cfad_plot_bootstrap_bands
except Exception:  # pragma: no cover - optional module
    cfad_plot_bootstrap_bands = None

try:
    from cfad.optimize import recommend_params as cfad_recommend_params
except Exception:  # pragma: no cover - optional module
    cfad_recommend_params = None


def _init_session_state() -> None:
    """Initialise dashboard session keys."""
    defaults: dict[str, Any] = {
        "report": None,
        "returns": None,
        "dates": None,
        "detect_params": None,
        "alarm_table": None,
        "model_table": None,
        "sensitivity": None,
        "bootstrap_fig": None,
        "dataset_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def _load_spy_cached(start: str, end: str) -> pd.Series:
    """Load cached SPY sample returns for the dashboard."""
    return load_spy_sample(start=start, end=end)


def _parse_input_dataframe(
    df: pd.DataFrame,
    return_col: str,
    date_col: Optional[str],
) -> tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
    """Extract return values and optional dates from an input DataFrame."""
    if return_col not in df.columns:
        raise ValueError(
            f"Return column '{return_col}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    ret_series = pd.to_numeric(df[return_col], errors="coerce")
    valid_return = ret_series.notna()
    if int(valid_return.sum()) == 0:
        raise ValueError(
            f"Column '{return_col}' has no numeric return values after parsing."
        )

    returns = ret_series.loc[valid_return].to_numpy(dtype=np.float64)
    dates: Optional[pd.DatetimeIndex] = None

    if date_col is not None:
        if date_col not in df.columns:
            raise ValueError(
                f"Date column '{date_col}' was not found. "
                f"Available columns: {list(df.columns)}"
            )
        date_series = pd.to_datetime(df.loc[valid_return, date_col], errors="coerce")
        valid_date = date_series.notna().to_numpy()
        if np.any(valid_date):
            returns = returns[valid_date]
            dates = pd.DatetimeIndex(date_series.loc[date_series.notna()])

    return returns, dates


def _build_alarm_table(report) -> pd.DataFrame:
    """Build an alarm table with score and CUSUM values at alarm times."""
    columns = [
        "alarm_window_index",
        "alarm_index",
        "alarm_date",
        "score_at_alarm",
        "cusum_pos_at_alarm",
        "cusum_neg_at_alarm",
    ]
    if report is None or len(report.alarm_indices) == 0:
        return pd.DataFrame(columns=columns)

    n_scores = len(report.scores)
    rows: list[dict[str, Any]] = []
    for alarm_window_index in report.alarm_indices:
        idx = int(alarm_window_index)
        if idx < 0 or idx >= n_scores:
            continue

        alarm_index = int(report.window_end_indices[idx] - 1)
        alarm_date = None
        if report.dates is not None and 0 <= alarm_index < len(report.dates):
            alarm_date = report.dates[alarm_index]

        rows.append(
            {
                "alarm_window_index": idx,
                "alarm_index": alarm_index,
                "alarm_date": alarm_date,
                "score_at_alarm": float(report.scores[idx]),
                "cusum_pos_at_alarm": float(report.cusum_pos[idx]),
                "cusum_neg_at_alarm": float(report.cusum_neg[idx]),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _fallback_recommend_params(
    window_df: pd.DataFrame,
    height_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> dict[str, float]:
    """Provide a lightweight recommendation when cfad.optimize is unavailable."""
    best_window = int(window_df.loc[window_df["mean_score"].idxmax(), "window"])
    best_height = float(height_df.loc[height_df["mean_score"].idxmax(), "height"])
    target_alarm_rate = 0.05
    h_idx = (threshold_df["alarm_rate"] - target_alarm_rate).abs().idxmin()
    best_h = float(threshold_df.loc[h_idx, "h"])
    return {"window": best_window, "height": best_height, "h": best_h}


def _run_sensitivity(
    returns: np.ndarray,
    base_params: dict[str, float | int],
) -> dict[str, Any]:
    """Run small-grid parameter sensitivity studies for dashboard feedback."""
    window_values = sorted(
        {
            int(np.clip(base_params["window"] + offset, 20, 200))
            for offset in (-40, -20, 0, 20, 40)
        }
    )
    height_values = sorted(
        {
            float(np.round(np.clip(base_params["height"] + offset, 0.05, 0.5), 2))
            for offset in (-0.10, -0.05, 0.0, 0.05, 0.10)
        }
    )
    h_values = np.linspace(
        max(1.0, float(base_params["h"]) - 2.0),
        min(10.0, float(base_params["h"]) + 2.0),
        7,
    )

    window_rows: list[dict[str, float]] = []
    for w in window_values:
        rep = detect(
            returns,
            window=w,
            step=int(base_params["step"]),
            xi_range=(-float(base_params["xi_max"]), float(base_params["xi_max"])),
            n_xi=int(base_params["n_xi"]),
            height=float(base_params["height"]),
            calibration_frac=float(base_params["calibration_frac"]),
            h=float(base_params["h"]),
        )
        window_rows.append(
            {
                "window": float(w),
                "mean_score": float(np.mean(rep.scores)),
                "alarm_rate": float(len(rep.alarm_indices) / max(1, len(rep.scores))),
            }
        )

    height_rows: list[dict[str, float]] = []
    for height in height_values:
        rep = detect(
            returns,
            window=int(base_params["window"]),
            step=int(base_params["step"]),
            xi_range=(-float(base_params["xi_max"]), float(base_params["xi_max"])),
            n_xi=int(base_params["n_xi"]),
            height=float(height),
            calibration_frac=float(base_params["calibration_frac"]),
            h=float(base_params["h"]),
        )
        height_rows.append(
            {
                "height": float(height),
                "mean_score": float(np.mean(rep.scores)),
                "alarm_rate": float(len(rep.alarm_indices) / max(1, len(rep.scores))),
            }
        )

    threshold_rows: list[dict[str, float]] = []
    for h_value in h_values:
        rep = detect(
            returns,
            window=int(base_params["window"]),
            step=int(base_params["step"]),
            xi_range=(-float(base_params["xi_max"]), float(base_params["xi_max"])),
            n_xi=int(base_params["n_xi"]),
            height=float(base_params["height"]),
            calibration_frac=float(base_params["calibration_frac"]),
            h=float(h_value),
        )
        threshold_rows.append(
            {
                "h": float(h_value),
                "mean_score": float(np.mean(rep.scores)),
                "alarm_rate": float(len(rep.alarm_indices) / max(1, len(rep.scores))),
            }
        )

    window_df = pd.DataFrame(window_rows)
    height_df = pd.DataFrame(height_rows)
    threshold_df = pd.DataFrame(threshold_rows)

    if cfad_recommend_params is not None:
        try:
            recommendation = cfad_recommend_params(returns)
        except Exception:
            recommendation = _fallback_recommend_params(window_df, height_df, threshold_df)
    else:
        recommendation = _fallback_recommend_params(window_df, height_df, threshold_df)

    return {
        "window": window_df,
        "height": height_df,
        "threshold": threshold_df,
        "recommended": recommendation,
    }


def _fallback_bootstrap_plot(
    report,
    returns: np.ndarray,
    n_bootstrap: int,
    params: dict[str, float | int],
):
    """Create bootstrap score bands when cfad.bootstrap is unavailable."""
    rng = np.random.default_rng(123)
    n_scores = len(report.scores)
    score_mat = np.zeros((n_bootstrap, n_scores), dtype=np.float64)

    for b in range(n_bootstrap):
        sample = rng.choice(returns, size=len(returns), replace=True)
        rep = detect(
            sample,
            window=int(params["window"]),
            step=int(params["step"]),
            xi_range=(-float(params["xi_max"]), float(params["xi_max"])),
            n_xi=int(params["n_xi"]),
            height=float(params["height"]),
            calibration_frac=float(params["calibration_frac"]),
            h=float(params["h"]),
        )
        score_mat[b, :] = rep.scores

    lower = np.percentile(score_mat, 2.5, axis=0)
    upper = np.percentile(score_mat, 97.5, axis=0)
    mean_scores = np.mean(score_mat, axis=0)

    x = np.arange(n_scores)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, report.scores, color="tab:blue", linewidth=1.1, label="Observed score")
    ax.plot(x, mean_scores, color="black", linewidth=1.0, label="Bootstrap mean")
    ax.fill_between(x, lower, upper, color="tab:blue", alpha=0.2, label="95% bootstrap band")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Residue score")
    ax.set_title("Bootstrap confidence bands for residue score")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _run_bootstrap_plot(
    report,
    returns: np.ndarray,
    n_bootstrap: int,
    params: dict[str, float | int],
):
    """Run bootstrap plot using cfad.bootstrap when available, else fallback."""
    if cfad_plot_bootstrap_bands is not None:
        try:
            return cfad_plot_bootstrap_bands(
                report=report,
                returns=returns,
                n_bootstrap=n_bootstrap,
            )
        except TypeError:
            try:
                return cfad_plot_bootstrap_bands(report, returns, n_bootstrap)
            except Exception:
                pass
        except Exception:
            pass

    return _fallback_bootstrap_plot(report, returns, n_bootstrap, params)


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="cfad dashboard", layout="wide")
    _init_session_state()

    st.title("CFAD Interactive Dashboard")

    st.sidebar.header("Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload returns CSV",
        type=["csv"],
        help="CSV should contain a return column and optional date column.",
    )
    use_spy = st.sidebar.checkbox("Use SPY sample data (2019–2021)", value=False)

    input_df: Optional[pd.DataFrame] = None
    dataset_name = ""

    if use_spy:
        with st.sidebar:
            with st.spinner("Computing..."):
                try:
                    spy = _load_spy_cached(start="2019-01-01", end="2022-01-01")
                    input_df = spy.rename("return").reset_index()
                    dataset_name = "SPY sample"
                except Exception as exc:
                    st.error(f"Failed to load SPY sample data: {exc}")
    elif uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name
        except Exception as exc:
            st.sidebar.error(f"Could not read uploaded CSV: {exc}")

    if input_df is not None:
        columns = list(input_df.columns)
        if "return" in columns:
            default_return_idx = columns.index("return")
        elif "log_return" in columns:
            default_return_idx = columns.index("log_return")
        else:
            default_return_idx = 0

        return_col = st.sidebar.selectbox(
            "Return column selector",
            options=columns,
            index=default_return_idx,
        )

        date_options = ["(none)"] + columns
        if "Date" in columns:
            default_date_idx = date_options.index("Date")
        elif "date" in columns:
            default_date_idx = date_options.index("date")
        else:
            default_date_idx = 0

        date_col_selected = st.sidebar.selectbox(
            "Date column selector (optional)",
            options=date_options,
            index=default_date_idx,
        )
        date_col = None if date_col_selected == "(none)" else date_col_selected
    else:
        return_col = "return"
        date_col = None
        st.sidebar.info("Upload a CSV or enable SPY sample data.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Detector parameters")
    window = st.sidebar.slider("Window", min_value=20, max_value=200, value=60, step=1)
    step = st.sidebar.slider("Step", min_value=1, max_value=20, value=5, step=1)
    xi_max = st.sidebar.slider("xi_max", min_value=5.0, max_value=20.0, value=10.0, step=0.5)
    n_xi = st.sidebar.slider("n_xi", min_value=64, max_value=256, value=128, step=64)
    height = st.sidebar.slider("Height", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    h_value = st.sidebar.slider("CUSUM h", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    calibration_frac = st.sidebar.slider(
        "Calibration fraction",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
    )

    run_detection = st.sidebar.button("Run detector", type="primary")

    if run_detection:
        if input_df is None:
            st.error("No data available. Upload a CSV or enable SPY sample data first.")
        else:
            try:
                returns, dates = _parse_input_dataframe(input_df, return_col=return_col, date_col=date_col)
                if len(returns) <= window:
                    st.error(
                        f"Not enough observations ({len(returns)}) for window={window}. "
                        "Use a smaller window or provide more data."
                    )
                else:
                    params = {
                        "window": window,
                        "step": step,
                        "xi_max": xi_max,
                        "n_xi": n_xi,
                        "height": height,
                        "h": h_value,
                        "calibration_frac": calibration_frac,
                    }
                    with st.spinner("Computing..."):
                        report = detect(
                            returns,
                            window=window,
                            step=step,
                            xi_range=(-xi_max, xi_max),
                            n_xi=n_xi,
                            height=height,
                            calibration_frac=calibration_frac,
                            h=h_value,
                        )

                    st.session_state["report"] = report
                    st.session_state["returns"] = returns
                    st.session_state["dates"] = dates
                    st.session_state["detect_params"] = params
                    st.session_state["alarm_table"] = _build_alarm_table(report)
                    st.session_state["dataset_name"] = dataset_name
                    st.session_state["model_table"] = None
                    st.session_state["sensitivity"] = None
                    st.session_state["bootstrap_fig"] = None
            except Exception as exc:
                st.error(f"Detector failed: {exc}")

    report = st.session_state.get("report")
    returns = st.session_state.get("returns")
    params = st.session_state.get("detect_params")

    tab_detection, tab_models, tab_sensitivity, tab_bootstrap = st.tabs(
        ["Detection results", "Model comparison", "Parameter sensitivity", "Bootstrap bands"]
    )

    with tab_detection:
        if report is None:
            st.info("Run detector from the sidebar to populate results.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("n_windows", f"{len(report.scores)}")
            c2.metric("n_alarms", f"{len(report.alarm_indices)}")
            c3.metric("mu0", f"{report.mu0:.4f}")

            fig = plot_detection_timeline(report, returns=returns)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            alarm_df = st.session_state.get("alarm_table")
            st.subheader("Alarm table")
            if alarm_df is not None and not alarm_df.empty:
                st.dataframe(alarm_df, use_container_width=True)
            else:
                st.write("No alarms triggered for the selected settings.")

    with tab_models:
        if report is None or returns is None:
            st.info("Run detector first to enable model comparison.")
        else:
            compare_clicked = st.button("Compare CF models")
            if compare_clicked:
                with st.spinner("Computing..."):
                    st.session_state["model_table"] = aic_table(np.asarray(returns, dtype=np.float64))

            model_df = st.session_state.get("model_table")
            if model_df is not None:
                st.dataframe(model_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.bar(model_df["model"], model_df["ecf_l2"], color="tab:blue", alpha=0.85)
                ax.set_ylabel("ECF-L2 distance")
                ax.set_title("Model fit comparison")
                ax.grid(axis="y", alpha=0.25)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                winner_row = model_df.loc[model_df["winner"]].iloc[0]
                st.info(
                    f"Winner: {winner_row['model']} (lowest AIC={winner_row['aic']:.3f}, "
                    f"ECF-L2={winner_row['ecf_l2']:.4f})."
                )

    with tab_sensitivity:
        if report is None or returns is None or params is None:
            st.info("Run detector first to enable sensitivity analysis.")
        else:
            sensitivity_clicked = st.button("Run sensitivity analysis")
            if sensitivity_clicked:
                with st.spinner("Computing..."):
                    st.session_state["sensitivity"] = _run_sensitivity(
                        np.asarray(returns, dtype=np.float64),
                        params,
                    )

            sens = st.session_state.get("sensitivity")
            if sens is not None:
                st.write("Window sensitivity (mean_score vs window)")
                st.line_chart(sens["window"].set_index("window")["mean_score"])

                st.write("Height sensitivity (mean_score vs height)")
                st.line_chart(sens["height"].set_index("height")["mean_score"])

                st.write("Threshold sensitivity (alarm_rate vs h)")
                st.line_chart(sens["threshold"].set_index("h")["alarm_rate"])

                rec = sens["recommended"]
                st.success(
                    "Recommended parameters from recommend_params(): "
                    f"window={rec['window']}, height={rec['height']:.2f}, h={rec['h']:.2f}"
                )

    with tab_bootstrap:
        if report is None or returns is None or params is None:
            st.info("Run detector first to enable bootstrap bands.")
        else:
            n_bootstrap = st.slider(
                "n_bootstrap",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
            )
            bootstrap_clicked = st.button("Run bootstrap")

            if bootstrap_clicked:
                with st.spinner("Computing..."):
                    st.session_state["bootstrap_fig"] = _run_bootstrap_plot(
                        report,
                        np.asarray(returns, dtype=np.float64),
                        n_bootstrap,
                        params,
                    )

            boot_fig = st.session_state.get("bootstrap_fig")
            if boot_fig is not None:
                st.pyplot(boot_fig, use_container_width=True)

    st.markdown("---")
    st.caption(
        "cfad v0.1.0 — Diogo Ribeiro, ESMAD-IPP | github.com/diogoribeiro7/cfad"
    )


if __name__ == "__main__":
    main()
