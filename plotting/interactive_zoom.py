import numpy as np
import streamlit as st


# =========================================================
# Fixed zoom step
# =========================================================

FINE_ZOOM_STEP = 0.02


# =========================================================
# Data range
# =========================================================

def _finite_range(
    values,
    padding=0.05,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            0.0,
            1.0,
        )

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    span = (
        maximum
        -
        minimum
    )

    if span <= 0:
        span = max(
            abs(maximum),
            1.0,
        )

    extra = (
        span
        *
        padding
    )

    return (
        minimum - extra,
        maximum + extra,
    )


# =========================================================
# Data signature
# =========================================================

def _data_signature(
    x_values,
    y_values,
):
    x = np.asarray(
        x_values,
        dtype=float,
    )

    y = np.asarray(
        y_values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    y = y[
        np.isfinite(y)
    ]

    if len(x) == 0:
        x_min = 0.0
        x_max = 1.0

    else:
        x_min = float(
            np.min(x)
        )

        x_max = float(
            np.max(x)
        )

    if len(y) == 0:
        y_min = 0.0
        y_max = 1.0

    else:
        y_min = float(
            np.min(y)
        )

        y_max = float(
            np.max(y)
        )

    return (
        len(x),
        round(x_min, 12),
        round(x_max, 12),
        len(y),
        round(y_min, 12),
        round(y_max, 12),
    )


# =========================================================
# Zoom math
# =========================================================

def _scale_range(
    axis_range,
    factor,
):
    minimum = float(
        axis_range[0]
    )

    maximum = float(
        axis_range[1]
    )

    center = (
        minimum
        +
        maximum
    ) / 2.0

    half_span = (
        maximum
        -
        minimum
    ) / 2.0

    half_span = (
        half_span
        *
        factor
    )

    return (
        center - half_span,
        center + half_span,
    )


# =========================================================
# Initialize zoom state
# =========================================================

def initialize_zoom_state(
    key,
    x_values,
    y_values,
    x_padding=0.05,
    y_padding=0.08,
):
    state_key = (
        f"{key}_fine_zoom_state"
    )

    signature = (
        _data_signature(
            x_values,
            y_values,
        )
    )

    x_default = (
        _finite_range(
            x_values,
            padding=x_padding,
        )
    )

    y_default = (
        _finite_range(
            y_values,
            padding=y_padding,
        )
    )

    current = (
        st.session_state.get(
            state_key
        )
    )

    if (
        current is None
        or
        current.get(
            "signature"
        )
        != signature
    ):
        current = {
            "signature":
                signature,

            "x_default":
                x_default,

            "y_default":
                y_default,

            "x_range":
                x_default,

            "y_range":
                y_default,
        }

        st.session_state[
            state_key
        ] = current

    return (
        state_key,
        current,
    )


# =========================================================
# Fine zoom controls
# =========================================================

def fine_zoom_controls(
    key,
    x_values,
    y_values,
):
    """
    Fixed incremental zoom.

    Zoom In:
        visible span becomes 98%
        of current span.

    Zoom Out:
        visible span becomes 102%
        of current span.

    Raw data is never changed.
    """

    state_key, state = (
        initialize_zoom_state(
            key,
            x_values,
            y_values,
        )
    )

    st.markdown(
        "#### Fine Zoom"
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    zoom_in = (
        c1.button(
            "Zoom In 2%",
            key=
                f"{key}_zoom_in",
            width="stretch",
        )
    )

    zoom_out = (
        c2.button(
            "Zoom Out 2%",
            key=
                f"{key}_zoom_out",
            width="stretch",
        )
    )

    reset = (
        c3.button(
            "Reset View",
            key=
                f"{key}_reset",
            width="stretch",
        )
    )

    # -----------------------------------------------------
    # Zoom in
    #
    # 2% smaller visible span
    # -----------------------------------------------------

    if zoom_in:

        state[
            "x_range"
        ] = (
            _scale_range(
                state[
                    "x_range"
                ],
                1.0
                -
                FINE_ZOOM_STEP,
            )
        )

        state[
            "y_range"
        ] = (
            _scale_range(
                state[
                    "y_range"
                ],
                1.0
                -
                FINE_ZOOM_STEP,
            )
        )

        st.session_state[
            state_key
        ] = state

    # -----------------------------------------------------
    # Zoom out
    #
    # 2% larger visible span
    # -----------------------------------------------------

    if zoom_out:

        state[
            "x_range"
        ] = (
            _scale_range(
                state[
                    "x_range"
                ],
                1.0
                +
                FINE_ZOOM_STEP,
            )
        )

        state[
            "y_range"
        ] = (
            _scale_range(
                state[
                    "y_range"
                ],
                1.0
                +
                FINE_ZOOM_STEP,
            )
        )

        st.session_state[
            state_key
        ] = state

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------

    if reset:

        state[
            "x_range"
        ] = (
            state[
                "x_default"
            ]
        )

        state[
            "y_range"
        ] = (
            state[
                "y_default"
            ]
        )

        st.session_state[
            state_key
        ] = state

    st.caption(
        "Fixed zoom step: 2% per click."
    )

    return (
        state[
            "x_range"
        ],
        state[
            "y_range"
        ],
    )


# =========================================================
# Apply range to Plotly
# =========================================================

def apply_fine_zoom(
    fig,
    x_range,
    y_range,
):
    """
    Apply program-controlled ranges
    without altering source data.
    """

    fig.update_xaxes(
        range=[
            float(
                x_range[0]
            ),
            float(
                x_range[1]
            ),
        ],
        autorange=False,
    )

    fig.update_yaxes(
        range=[
            float(
                y_range[0]
            ),
            float(
                y_range[1]
            ),
        ],
        autorange=False,
    )

    return fig
