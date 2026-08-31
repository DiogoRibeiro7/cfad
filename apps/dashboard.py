from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cfad import detect
from cfad.bootstrap import bootstrap_scores
from cfad.gof import aic_table
from cfad.sensitivity import (
    frequency_sensitivity,
    recommend_params,
    threshold_sensitivity,
    window_sensitivity,
)
from cfad.utils import load_spy_sample
from cfad.viz import plot_detection_timeline


def _parse_input_dataframe(
    df: pd.DataFrame,
    return_col: str,
    date_col: Optional[str],
) -> tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
    """Extract numeric returns and optional dates from an input frame."""
    if return_col not in df.columns:
        raise ValueError(f"Return column {return_col!r} was not found")

    numeric = pd.to_numeric(df[return_col], errors="coerce")
    valid = numeric.notna()
    values = numeric.loc[valid].to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError("No numeric return observations were found")

    if date_col is None:
        return values, None
    if date_col not in df.columns:
        raise ValueError(f"Date column {date_col!r} was not found")

    dates = pd.to_datetime(df.loc[valid, date_col], errors="coerce")
    keep = dates.notna().to_numpy()
    return values[keep], pd.DatetimeIndex(dates.loc[dates.notna()])


def _alarm_table(report) -> pd.DataFrame:
    """Return alarm windows with their corresponding observation endpoints."""
    rows: list[dict[str, object]] = []
    for alarm_window in np.asarray(report.alarm_indices, dtype=np.int64):
        if alarm_window < 0 or alarm_window >= len(report.window_end_indices):
            continue
        obs_idx = int(report.window_end_indices[alarm_window] - 1)
        alarm_date = None
        if report.dates is not None and 0 <= obs_idx < len(report.dates):
            alarm_date = report.dates[obs_idx]
        rows.append(
            {
                "window": int(alarm_window),
                "observation_index": obs_idx,
                "date": alarm_date,
                "ecf_shape_score": float(report.scores[alarm_window]),
                "cusum_pos": float(report.cusum_pos[alarm_window]),
                "cusum_neg": float(report.cusum_neg[alarm_window]),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_figure(
    report,
    returns: np.ndarray,
    params: dict[str, float | int],
    n_bootstrap: int,
):
    """Plot pointwise bootstrap bands for the corrected ECF-shape score."""
    summary = bootstrap_scores(
        returns,
        window=int(params["window"]),
        n_bootstrap=n_bootstrap,
        xi_range=(-float(params["xi_max"]), float(params["xi_max"])),
        n_xi=int(params["n_xi"]),
        step=int(params["step"]),
        seed=123,
    )
    x = np.arange(len(report.scores))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, report.scores, label="Observed ECF-shape score", linewidth=1.2)
    ax.plot(x, summary["mean"], label="Bootstrap mean", linewidth=1.0)
    ax.fill_between(
        x,
        summary["lower"],
        summary["upper"],
        alpha=0.2,
        label="95% pointwise band",
    )
    ax.set_xlabel("Window index")
    ax.set_ylabel("Gaussian-reference ECF distance")
    ax.set_title("Bootstrap bands for CFAD score")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="CFAD dashboard", layout="wide")
    st.title("CFAD — ECF Shape Change Dashboard")
    st.caption(
        "Rolling empirical characteristic functions are compared with a fitted "
        "Gaussian reference and monitored with Page-CUSUM."
    )

    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Upload returns CSV", type=["csv"])
    use_spy = st.sidebar.checkbox("Use SPY sample data", value=False)

    input_df: Optional[pd.DataFrame] = None
    if use_spy:
        try:
            spy = load_spy_sample(start="2019-01-01", end="2022-01-01")
            input_df = spy.rename("return").reset_index()
        except Exception as exc:
            st.sidebar.error(f"Could not load SPY sample: {exc}")
    elif uploaded is not None:
        input_df = pd.read_csv(uploaded)

    if input_df is None:
        st.info("Upload a CSV or enable the SPY sample to begin.")
        return

    columns = list(input_df.columns)
    default_return = columns.index("return") if "return" in columns else 0
    return_col = st.sidebar.selectbox("Return column", columns, index=default_return)
    date_options = ["(none)"] + columns
    date_col_selected = st.sidebar.selectbox("Date column", date_options)
    date_col = None if date_col_selected == "(none)" else date_col_selected

    st.sidebar.markdown("---")
    st.sidebar.subheader("Detector parameters")
    window = st.sidebar.slider("Window", 20, 200, 60, 1)
    step = st.sidebar.slider("Step", 1, 20, 5, 1)
    xi_max = st.sidebar.slider("Real-frequency cutoff ξ_max", 2.0, 80.0, 10.0, 1.0)
    n_xi = st.sidebar.slider("Frequency grid size", 32, 512, 128, 32)
    h_value = st.sidebar.slider("CUSUM threshold h", 1.0, 10.0, 5.0, 0.5)
    calibration_frac = st.sidebar.slider(
        "Calibration fraction",
        0.1,
        0.5,
        0.3,
        0.05,
    )

    try:
        returns, dates = _parse_input_dataframe(input_df, return_col, date_col)
    except Exception as exc:
        st.error(str(exc))
        return

    if len(returns) <= window:
        st.error(f"Need more than {window} observations; found {len(returns)}")
        return

    params = {
        "window": window,
        "step": step,
        "xi_max": xi_max,
        "n_xi": n_xi,
        "h": h_value,
        "calibration_frac": calibration_frac,
    }

    if "cfad_report" not in st.session_state:
        st.session_state["cfad_report"] = None

    if st.sidebar.button("Run detector", type="primary"):
        detect_input = (
            pd.Series(returns, index=dates, name="return")
            if dates is not None
            else returns
        )
        with st.spinner("Computing detector scores..."):
            st.session_state["cfad_report"] = detect(
                detect_input,
                window=window,
                xi_range=(-xi_max, xi_max),
                n_xi=n_xi,
                step=step,
                calibration_frac=calibration_frac,
                h=h_value,
            )

    report = st.session_state.get("cfad_report")
    if report is None:
        return

    tab_detection, tab_models, tab_sensitivity, tab_bootstrap = st.tabs(
        ["Detection", "Model comparison", "Sensitivity", "Bootstrap"]
    )

    with tab_detection:
        c1, c2, c3 = st.columns(3)
        c1.metric("Windows", len(report.scores))
        c2.metric("Alarms", len(report.alarm_indices))
        c3.metric("Calibration σ", f"{report.sigma0:.4g}")
        fig = plot_detection_timeline(report, returns=returns)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        alarms = _alarm_table(report)
        st.subheader("Alarm table")
        if alarms.empty:
            st.write("No alarms for the selected settings.")
        else:
            st.dataframe(alarms, use_container_width=True)

    with tab_models:
        if st.button("Fit CF models"):
            with st.spinner("Fitting models..."):
                st.session_state["cfad_models"] = aic_table(returns)
        model_df = st.session_state.get("cfad_models")
        if model_df is not None:
            st.dataframe(model_df, use_container_width=True)
            st.caption(
                "Model comparison is descriptive distributional-fit evidence; "
                "it is not a branch-cut or singularity test."
            )

    with tab_sensitivity:
        if st.button("Run sensitivity analysis"):
            with st.spinner("Evaluating parameter sensitivity..."):
                st.session_state["cfad_window_sens"] = window_sensitivity(
                    returns,
                    h=h_value,
                    step=step,
                )
                st.session_state["cfad_freq_sens"] = frequency_sensitivity(
                    returns,
                    window=window,
                    h=h_value,
                    step=step,
                    n_xi=n_xi,
                )
                st.session_state["cfad_threshold_sens"] = threshold_sensitivity(
                    returns,
                    window=window,
                    step=step,
                    calibration_frac=calibration_frac,
                    xi_max=xi_max,
                )
                st.session_state["cfad_recommendation"] = recommend_params(
                    returns,
                    verbose=False,
                )

        window_df = st.session_state.get("cfad_window_sens")
        freq_df = st.session_state.get("cfad_freq_sens")
        threshold_df = st.session_state.get("cfad_threshold_sens")
        if window_df is not None:
            st.write("Window sensitivity")
            st.line_chart(window_df.set_index("window")["metric_value"])
        if freq_df is not None:
            st.write("Real-frequency cutoff sensitivity")
            st.line_chart(freq_df.set_index("xi_max")["score_std"])
        if threshold_df is not None:
            st.write("CUSUM threshold sensitivity")
            st.line_chart(threshold_df.set_index("h")["alarm_rate"])
        recommendation = st.session_state.get("cfad_recommendation")
        if recommendation is not None:
            st.info(
                "Exploratory recommendation: "
                f"window={recommendation['window']}, "
                f"xi_max={recommendation['xi_max']:.1f}, "
                f"h={recommendation['h']:.1f}. "
                "Use out-of-sample calibration for confirmatory work."
            )

    with tab_bootstrap:
        n_bootstrap = st.slider("Bootstrap replicates", 50, 500, 100, 10)
        if st.button("Run bootstrap"):
            with st.spinner("Bootstrapping score paths..."):
                fig = _bootstrap_figure(report, returns, params, n_bootstrap)
                st.session_state["cfad_bootstrap_fig"] = fig
        boot_fig = st.session_state.get("cfad_bootstrap_fig")
        if boot_fig is not None:
            st.pyplot(boot_fig, use_container_width=True)

    st.markdown("---")
    st.caption(
        "CFAD research software · Faculty of Media Arts and Design, "
        "Technical University of Porto"
    )


if __name__ == "__main__":
    main()
