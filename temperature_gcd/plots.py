from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_time_period_temperature(
    feature_table: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    currents = sorted(
        feature_table["current_mA"].dropna().unique().tolist()
    )

    for current in currents:
        subset = feature_table[
            np.isclose(
                feature_table["current_mA"].to_numpy(dtype=float),
                float(current),
                rtol=1e-6,
                atol=1e-9,
            )
        ].sort_values("temperature_C")

        fig.add_trace(
            go.Scatter(
                x=subset["temperature_C"],
                y=subset["time_period_s"],
                mode="lines+markers",
                name=f"{current:g} mA",
            )
        )

    fig.update_layout(
        xaxis_title="Temperature (°C)",
        yaxis_title="Time period (s)",
        template="plotly_white",
        hovermode="closest",
    )

    return fig


def plot_time_period_current(
    feature_table: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    temperatures = sorted(
        feature_table["temperature_C"].dropna().unique().tolist()
    )

    for temperature in temperatures:
        subset = feature_table[
            np.isclose(
                feature_table["temperature_C"].to_numpy(dtype=float),
                float(temperature),
                rtol=1e-6,
                atol=1e-9,
            )
        ].sort_values("current_mA")

        fig.add_trace(
            go.Scatter(
                x=subset["current_mA"],
                y=subset["time_period_s"],
                mode="lines+markers",
                name=f"{temperature:g} °C",
            )
        )

    fig.update_layout(
        xaxis_title="Current (mA)",
        yaxis_title="Time period (s)",
        template="plotly_white",
        hovermode="closest",
    )

    return fig


def plot_parameter_temperature(
    feature_table: pd.DataFrame,
    branch: str,
) -> go.Figure:
    if branch not in {"charge", "discharge"}:
        raise ValueError("branch 必须是 charge 或 discharge。")

    fig = go.Figure()

    for parameter in ("a", "b", "c"):
        column = f"{branch}_{parameter}"

        summary = (
            feature_table
            .groupby("temperature_C", as_index=False)[column]
            .mean()
            .sort_values("temperature_C")
        )

        fig.add_trace(
            go.Scatter(
                x=summary["temperature_C"],
                y=summary[column],
                mode="lines+markers",
                name=parameter,
            )
        )

    title = (
        "Charging parameters"
        if branch == "charge"
        else "Discharging parameters"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Temperature (°C)",
        yaxis_title="Fitted parameter",
        template="plotly_white",
    )

    return fig


def plot_prediction(
    predicted_curve: pd.DataFrame,
    actual_curve: pd.DataFrame | None = None,
    target_temperature_C: float | None = None,
    target_current_mA: float | None = None,
) -> go.Figure:
    fig = go.Figure()

    if actual_curve is not None and len(actual_curve):
        actual = actual_curve.copy()
        actual["time_s"] = actual["time_s"] - actual["time_s"].iloc[0]

        fig.add_trace(
            go.Scatter(
                x=actual["time_s"],
                y=actual["potential_V"],
                mode="lines",
                name="Experimental",
            )
        )

    charge = predicted_curve[
        predicted_curve["branch"] == "Charging"
    ]

    discharge = predicted_curve[
        predicted_curve["branch"] == "Discharging"
    ]

    fig.add_trace(
        go.Scatter(
            x=charge["time_s"],
            y=charge["potential_V"],
            mode="lines",
            name="Predicted - Charging",
            line={"dash": "dash"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=discharge["time_s"],
            y=discharge["potential_V"],
            mode="lines",
            name="Predicted - Discharging",
            line={"dash": "dash"},
        )
    )

    title_parts = []

    if target_temperature_C is not None:
        title_parts.append(f"{target_temperature_C:g} °C")

    if target_current_mA is not None:
        title_parts.append(f"{target_current_mA:g} mA")

    title = "Predicted GCD"
    if title_parts:
        title += " · " + " · ".join(title_parts)

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Potential (V)",
        template="plotly_white",
        hovermode="closest",
    )

    return fig
