import json
import uuid

import plotly.graph_objects as go
import plotly.io as pio

import streamlit as st


# =========================================================
# CH Instruments style box zoom
# =========================================================
#
# CHI zoom principle:
#
#     The selected rectangle becomes the NEW visible range.
#
# There is NO:
#
#     MIN_ZOOM_CHANGE
#     ZOOM_RESPONSE_GAIN
#     MAX_ZOOM_CHANGE
#     SMALL_BOX_THRESHOLD
#
# Example:
#
#     selected width = 90%  -> X range becomes 90% -> 1.111x
#     selected width = 80%  -> X range becomes 80% -> 1.25x
#     selected width = 50%  -> X range becomes 50% -> 2x
#     selected width = 20%  -> X range becomes 20% -> 5x
#     selected width = 10%  -> X range becomes 10% -> 10x
#
# X and Y are calculated independently.
# =========================================================


# Ignore an accidental click or microscopic drag.
#
# This is NOT a zoom limit.
# It only prevents 1-3 pixel mouse movement from being
# interpreted as a huge zoom.
MIN_BOX_PIXEL_SIZE = 4


# Default component height.
DEFAULT_PLOT_HEIGHT = 580


# Minimum height allowed.
MIN_PLOT_HEIGHT = 350


# Height of hint line.
HINT_HEIGHT = 32


# Double click restores the original display range.
#
# This is retained as a convenient UI operation.
ENABLE_DOUBLE_CLICK_RESET = True


# =========================================================
# Plotly Configuration
# =========================================================

def _prepare_config(config=None):
    """
    Plotly configuration for CHI-style rectangular zoom.

    Wheel zoom:
        disabled

    Native Zoom In / Zoom Out buttons:
        removed

    Box Zoom mode:
        retained

    Pan:
        retained

    Autoscale / Reset:
        retained

    Actual rectangle handling is performed by custom
    JavaScript.
    """

    result = dict(
        config
        if config is not None
        else {}
    )

    # -----------------------------------------------------
    # Disable mouse-wheel zoom
    # -----------------------------------------------------

    result["scrollZoom"] = False

    # -----------------------------------------------------
    # Modebar
    # -----------------------------------------------------

    result["displayModeBar"] = True
    result["displaylogo"] = False
    result["responsive"] = True

    # We implement our own double-click reset.
    result["doubleClick"] = False

    remove_buttons = set(
        result.get(
            "modeBarButtonsToRemove",
            []
        )
    )

    # Keep rectangular zoom mode.
    remove_buttons.discard("zoom2d")

    # Keep pan.
    remove_buttons.discard("pan2d")

    # Keep autoscale.
    remove_buttons.discard("autoScale2d")

    # Keep reset.
    remove_buttons.discard("resetScale2d")

    # Remove fixed-step zoom buttons and selection tools.
    remove_buttons.update(
        {
            "zoomIn2d",
            "zoomOut2d",
            "select2d",
            "lasso2d",
        }
    )

    result["modeBarButtonsToRemove"] = list(
        remove_buttons
    )

    return result


# =========================================================
# JavaScript
# =========================================================

def _build_custom_zoom_javascript(div_id):
    """
    Build CH Instruments style rectangular zoom.

    Core rule
    ---------

    The rectangle selected by the mouse becomes the new
    X/Y visible range.

    If the current X range is:

        0 ... 1 V

    and the user selects:

        0.2 ... 0.4 V

    the new X axis becomes:

        0.2 ... 0.4 V

    Exactly.

    No response gain.
    No artificial zoom percentage.
    No maximum zoom amount.
    """

    template = r"""
<script>
(function() {

    "use strict";


    // =====================================================
    // Configuration
    // =====================================================

    const GRAPH_ID =
        __GRAPH_ID__;


    const MIN_BOX_PIXEL_SIZE =
        __MIN_BOX_PIXEL_SIZE__;


    const ENABLE_DOUBLE_CLICK_RESET =
        __ENABLE_DOUBLE_CLICK_RESET__;


    // =====================================================
    // Runtime state
    // =====================================================

    let dragging =
        false;


    let startX =
        0;


    let startY =
        0;


    let currentX =
        0;


    let currentY =
        0;


    let startRanges =
        null;


    let selectionBox =
        null;


    // =====================================================
    // General helpers
    // =====================================================

    function clamp(
        value,
        minimum,
        maximum
    ) {

        return Math.min(
            maximum,
            Math.max(
                minimum,
                value
            )
        );
    }


    function finiteNumber(value) {

        return (
            typeof value === "number"
            &&
            Number.isFinite(value)
        );
    }


    function cloneRanges(ranges) {

        if (!ranges) {

            return null;
        }


        return {

            x: [
                ranges.x[0],
                ranges.x[1]
            ],

            y: [
                ranges.y[0],
                ranges.y[1]
            ]
        };
    }


    // =====================================================
    // Plot area geometry
    // =====================================================

    function getPlotArea(graph) {

        if (
            !graph
            ||
            !graph._fullLayout
            ||
            !graph._fullLayout._size
        ) {

            return null;
        }


        const size =
            graph._fullLayout._size;


        const left =
            Number(size.l);


        const top =
            Number(size.t);


        const width =
            Number(size.w);


        const height =
            Number(size.h);


        if (
            !finiteNumber(left)
            ||
            !finiteNumber(top)
            ||
            !finiteNumber(width)
            ||
            !finiteNumber(height)
            ||
            width <= 0
            ||
            height <= 0
        ) {

            return null;
        }


        return {

            left:
                left,

            top:
                top,

            width:
                width,

            height:
                height,

            right:
                left + width,

            bottom:
                top + height
        };
    }


    // =====================================================
    // Mouse coordinates relative to graph
    // =====================================================

    function getMousePosition(
        graph,
        event,
        plotArea
    ) {

        const rect =
            graph.getBoundingClientRect();


        let x =
            event.clientX
            -
            rect.left;


        let y =
            event.clientY
            -
            rect.top;


        x =
            clamp(
                x,
                plotArea.left,
                plotArea.right
            );


        y =
            clamp(
                y,
                plotArea.top,
                plotArea.bottom
            );


        return {

            x:
                x,

            y:
                y
        };
    }


    // =====================================================
    // Is mouse in actual data plotting area?
    // =====================================================

    function isInsidePlot(
        graph,
        event,
        plotArea
    ) {

        const rect =
            graph.getBoundingClientRect();


        const x =
            event.clientX
            -
            rect.left;


        const y =
            event.clientY
            -
            rect.top;


        return (
            x >= plotArea.left
            &&
            x <= plotArea.right
            &&
            y >= plotArea.top
            &&
            y <= plotArea.bottom
        );
    }


    // =====================================================
    // Read current Plotly axis ranges
    // =====================================================

    function readRanges(graph) {

        if (
            !graph
            ||
            !graph._fullLayout
            ||
            !graph._fullLayout.xaxis
            ||
            !graph._fullLayout.yaxis
        ) {

            return null;
        }


        const xr =
            graph._fullLayout.xaxis.range;


        const yr =
            graph._fullLayout.yaxis.range;


        if (
            !xr
            ||
            !yr
            ||
            xr.length < 2
            ||
            yr.length < 2
        ) {

            return null;
        }


        const x0 =
            Number(xr[0]);


        const x1 =
            Number(xr[1]);


        const y0 =
            Number(yr[0]);


        const y1 =
            Number(yr[1]);


        if (
            !finiteNumber(x0)
            ||
            !finiteNumber(x1)
            ||
            !finiteNumber(y0)
            ||
            !finiteNumber(y1)
        ) {

            return null;
        }


        return {

            x: [
                x0,
                x1
            ],

            y: [
                y0,
                y1
            ]
        };
    }


    // =====================================================
    // Selection rectangle
    // =====================================================

    function createSelectionBox(graph) {

        const box =
            document.createElement(
                "div"
            );


        box.style.position =
            "absolute";


        box.style.pointerEvents =
            "none";


        // CHI-like simple rectangular outline.
        box.style.border =
            "1px solid rgba(30, 30, 30, 0.95)";


        box.style.background =
            "rgba(80, 80, 80, 0.08)";


        box.style.boxSizing =
            "border-box";


        box.style.zIndex =
            "100";


        box.style.display =
            "none";


        graph.appendChild(box);


        return box;
    }


    function updateSelectionBox() {

        if (!selectionBox) {

            return;
        }


        const left =
            Math.min(
                startX,
                currentX
            );


        const top =
            Math.min(
                startY,
                currentY
            );


        const width =
            Math.abs(
                currentX
                -
                startX
            );


        const height =
            Math.abs(
                currentY
                -
                startY
            );


        selectionBox.style.left =
            left + "px";


        selectionBox.style.top =
            top + "px";


        selectionBox.style.width =
            width + "px";


        selectionBox.style.height =
            height + "px";


        selectionBox.style.display =
            "block";
    }


    function hideSelectionBox() {

        if (!selectionBox) {

            return;
        }


        selectionBox.style.display =
            "none";


        selectionBox.style.width =
            "0px";


        selectionBox.style.height =
            "0px";
    }


    // =====================================================
    // Pixel -> X coordinate
    // =====================================================
    //
    // Plotly's numeric/log ranges are represented in
    // axis-coordinate space.
    //
    // Therefore linear interpolation of the CURRENT
    // displayed range gives the correct new axis range.
    // =====================================================

    function pixelToDataX(
        pixelX,
        plotArea,
        range
    ) {

        let fraction =
            (
                pixelX
                -
                plotArea.left
            )
            /
            plotArea.width;


        fraction =
            clamp(
                fraction,
                0.0,
                1.0
            );


        return (
            range[0]
            +
            fraction
            *
            (
                range[1]
                -
                range[0]
            )
        );
    }


    // =====================================================
    // Pixel -> Y coordinate
    // =====================================================
    //
    // Screen Y grows downward.
    //
    // Plot Y grows from range[0] at bottom to
    // range[1] at top.
    // =====================================================

    function pixelToDataY(
        pixelY,
        plotArea,
        range
    ) {

        let fraction =
            (
                pixelY
                -
                plotArea.top
            )
            /
            plotArea.height;


        fraction =
            clamp(
                fraction,
                0.0,
                1.0
            );


        return (
            range[1]
            -
            fraction
            *
            (
                range[1]
                -
                range[0]
            )
        );
    }


    // =====================================================
    // CHI rectangular zoom
    // =====================================================
    //
    // IMPORTANT:
    //
    // We DO NOT calculate:
    //
    //     zoomAmount
    //
    // Instead:
    //
    //     selected left   -> new X left
    //     selected right  -> new X right
    //     selected bottom -> new Y bottom
    //     selected top    -> new Y top
    //
    // Thus the selected rectangle fills the plot.
    // =====================================================

    function calculateSelectedRanges(
        plotArea,
        ranges
    ) {

        const leftPixel =
            Math.min(
                startX,
                currentX
            );


        const rightPixel =
            Math.max(
                startX,
                currentX
            );


        const topPixel =
            Math.min(
                startY,
                currentY
            );


        const bottomPixel =
            Math.max(
                startY,
                currentY
            );


        const newXLeft =
            pixelToDataX(
                leftPixel,
                plotArea,
                ranges.x
            );


        const newXRight =
            pixelToDataX(
                rightPixel,
                plotArea,
                ranges.x
            );


        const newYBottom =
            pixelToDataY(
                bottomPixel,
                plotArea,
                ranges.y
            );


        const newYTop =
            pixelToDataY(
                topPixel,
                plotArea,
                ranges.y
            );


        return {

            x: [
                newXLeft,
                newXRight
            ],

            y: [
                newYBottom,
                newYTop
            ]
        };
    }


    // =====================================================
    // Restore initial view
    // =====================================================

    function restoreHomeView(graph) {

        const home =
            graph.__chiHomeRanges;


        if (!home) {

            Plotly.relayout(
                graph,
                {
                    "xaxis.autorange":
                        true,

                    "yaxis.autorange":
                        true,

                    "dragmode":
                        "zoom"
                }
            );

            return;
        }


        Plotly.relayout(
            graph,
            {
                "xaxis.autorange":
                    false,

                "yaxis.autorange":
                    false,

                "xaxis.range":
                    [
                        home.x[0],
                        home.x[1]
                    ],

                "yaxis.range":
                    [
                        home.y[0],
                        home.y[1]
                    ],

                "dragmode":
                    "zoom"
            }
        );
    }


    // =====================================================
    // Attach CHI-style zoom
    // =====================================================

    function attachCustomBoxZoom() {

        const graph =
            document.getElementById(
                GRAPH_ID
            );


        if (!graph) {

            setTimeout(
                attachCustomBoxZoom,
                30
            );

            return;
        }


        if (
            !graph._fullLayout
            ||
            !graph._fullLayout._size
        ) {

            setTimeout(
                attachCustomBoxZoom,
                30
            );

            return;
        }


        // Do not attach twice.
        if (
            graph.__chiExactBoxZoomAttached
        ) {

            return;
        }


        graph.__chiExactBoxZoomAttached =
            true;


        // -------------------------------------------------
        // Make graph a positioning container for the box.
        // -------------------------------------------------

        const computedStyle =
            window.getComputedStyle(
                graph
            );


        if (
            computedStyle.position
            ===
            "static"
        ) {

            graph.style.position =
                "relative";
        }


        // -------------------------------------------------
        // Create box.
        // -------------------------------------------------

        selectionBox =
            createSelectionBox(
                graph
            );


        // -------------------------------------------------
        // Store the original range.
        //
        // This is important:
        // reset returns to the range actually displayed
        // when the chart was first created, rather than
        // inventing a new range.
        // -------------------------------------------------

        const initialRanges =
            readRanges(
                graph
            );


        if (initialRanges) {

            graph.__chiHomeRanges =
                cloneRanges(
                    initialRanges
                );
        }


        // -------------------------------------------------
        // Start in zoom mode.
        // -------------------------------------------------

        Plotly.relayout(
            graph,
            {
                "dragmode":
                    "zoom"
            }
        );


        // =================================================
        // Mouse down
        // =================================================

        graph.addEventListener(
            "mousedown",

            function(event) {

                // -----------------------------------------
                // Left mouse button only.
                // -----------------------------------------

                if (
                    event.button
                    !==
                    0
                ) {

                    return;
                }


                // -----------------------------------------
                // Pan mode:
                //
                // If the user selected Pan from Plotly's
                // toolbar, do not interfere.
                // -----------------------------------------

                if (
                    !graph._fullLayout
                    ||
                    graph._fullLayout.dragmode
                    !==
                    "zoom"
                ) {

                    return;
                }


                const plotArea =
                    getPlotArea(
                        graph
                    );


                if (!plotArea) {

                    return;
                }


                // -----------------------------------------
                // Only intercept drags inside data area.
                //
                // This means Plotly axis interactions
                // outside the data area can still work.
                // -----------------------------------------

                if (
                    !isInsidePlot(
                        graph,
                        event,
                        plotArea
                    )
                ) {

                    return;
                }


                // -----------------------------------------
                // Stop Plotly's native box zoom.
                //
                // Native Plotly box zoom is not allowed
                // to calculate the final ranges because
                // we want deterministic CHI behavior.
                // -----------------------------------------

                event.preventDefault();

                event.stopPropagation();

                event.stopImmediatePropagation();


                const mouse =
                    getMousePosition(
                        graph,
                        event,
                        plotArea
                    );


                const ranges =
                    readRanges(
                        graph
                    );


                if (!ranges) {

                    return;
                }


                startX =
                    mouse.x;


                startY =
                    mouse.y;


                currentX =
                    mouse.x;


                currentY =
                    mouse.y;


                startRanges =
                    cloneRanges(
                        ranges
                    );


                dragging =
                    true;


                updateSelectionBox();

            },

            true
        );


        // =================================================
        // Mouse move
        // =================================================

        window.addEventListener(
            "mousemove",

            function(event) {

                if (!dragging) {

                    return;
                }


                const plotArea =
                    getPlotArea(
                        graph
                    );


                if (!plotArea) {

                    return;
                }


                event.preventDefault();


                const mouse =
                    getMousePosition(
                        graph,
                        event,
                        plotArea
                    );


                currentX =
                    mouse.x;


                currentY =
                    mouse.y;


                updateSelectionBox();

            },

            true
        );


        // =================================================
        // Mouse up
        // =================================================

        window.addEventListener(
            "mouseup",

            function(event) {

                if (!dragging) {

                    return;
                }


                dragging =
                    false;


                const plotArea =
                    getPlotArea(
                        graph
                    );


                if (
                    !plotArea
                    ||
                    !startRanges
                ) {

                    hideSelectionBox();

                    startRanges =
                        null;

                    return;
                }


                const mouse =
                    getMousePosition(
                        graph,
                        event,
                        plotArea
                    );


                currentX =
                    mouse.x;


                currentY =
                    mouse.y;


                updateSelectionBox();


                const selectedWidth =
                    Math.abs(
                        currentX
                        -
                        startX
                    );


                const selectedHeight =
                    Math.abs(
                        currentY
                        -
                        startY
                    );


                // -----------------------------------------
                // Ignore accidental click.
                //
                // This is the ONLY minimum restriction.
                //
                // Once the rectangle exceeds this size,
                // there is no artificial zoom cap.
                // -----------------------------------------

                if (
                    selectedWidth
                    <
                    MIN_BOX_PIXEL_SIZE
                    ||
                    selectedHeight
                    <
                    MIN_BOX_PIXEL_SIZE
                ) {

                    hideSelectionBox();

                    startRanges =
                        null;

                    return;
                }


                // =================================================
                // THIS IS THE CHI ZOOM CALCULATION
                // =================================================
                //
                // Selected rectangle = new visible range.
                //
                // No gain.
                // No percentage response.
                // No zoom cap.
                // No anchor scaling.
                // =================================================

                const selectedRanges =
                    calculateSelectedRanges(
                        plotArea,
                        startRanges
                    );


                // Hide box before Plotly redraw.
                hideSelectionBox();


                // =================================================
                // Apply exact selected range
                // =================================================

                Plotly.relayout(
                    graph,
                    {
                        "xaxis.autorange":
                            false,

                        "yaxis.autorange":
                            false,

                        "xaxis.range":
                            selectedRanges.x,

                        "yaxis.range":
                            selectedRanges.y,

                        "dragmode":
                            "zoom"
                    }
                );


                startRanges =
                    null;

            },

            true
        );


        // =================================================
        // Double click -> original range
        // =================================================

        graph.addEventListener(
            "dblclick",

            function(event) {

                if (
                    !ENABLE_DOUBLE_CLICK_RESET
                ) {

                    return;
                }


                const plotArea =
                    getPlotArea(
                        graph
                    );


                if (!plotArea) {

                    return;
                }


                if (
                    !isInsidePlot(
                        graph,
                        event,
                        plotArea
                    )
                ) {

                    return;
                }


                if (
                    !graph._fullLayout
                    ||
                    graph._fullLayout.dragmode
                    !==
                    "zoom"
                ) {

                    return;
                }


                event.preventDefault();

                event.stopPropagation();

                event.stopImmediatePropagation();


                dragging =
                    false;


                hideSelectionBox();


                startRanges =
                    null;


                restoreHomeView(
                    graph
                );

            },

            true
        );


        // =================================================
        // Escape key -> cancel current rectangle
        // =================================================

        window.addEventListener(
            "keydown",

            function(event) {

                if (
                    event.key
                    !==
                    "Escape"
                ) {

                    return;
                }


                if (!dragging) {

                    return;
                }


                dragging =
                    false;


                hideSelectionBox();


                startRanges =
                    null;

            },

            true
        );

    }


    // =====================================================
    // Start
    // =====================================================

    attachCustomBoxZoom();

})();
</script>
"""

    script = (
        template
        .replace(
            "__GRAPH_ID__",
            json.dumps(div_id)
        )
        .replace(
            "__MIN_BOX_PIXEL_SIZE__",
            json.dumps(
                MIN_BOX_PIXEL_SIZE
            )
        )
        .replace(
            "__ENABLE_DOUBLE_CLICK_RESET__",
            (
                "true"
                if ENABLE_DOUBLE_CLICK_RESET
                else "false"
            )
        )
    )

    return script


# =========================================================
# Render custom Plotly
# =========================================================

def render_box_zoom_plotly(
    figure,
    config=None,
    height=None,
):
    """
    Render Plotly with CH Instruments style box zoom.

    Core behavior
    -------------

    The selected rectangle becomes the new visible
    X/Y range.

    Example:

        Current X:
            0 -> 1

        Selected:
            0.2 -> 0.4

        New X:
            0.2 -> 0.4

    Therefore:

        90% selected -> 1.111x
        80% selected -> 1.25x
        70% selected -> 1.429x
        50% selected -> 2x
        30% selected -> 3.333x
        20% selected -> 5x
        10% selected -> 10x

    X and Y are independent.

    Pan:
        use Plotly Pan toolbar button

    Double-click:
        restore initial axes

    Escape:
        cancel current box selection
    """

    # -----------------------------------------------------
    # Copy figure so original is not modified.
    # -----------------------------------------------------

    fig = go.Figure(
        figure
    )


    # -----------------------------------------------------
    # Ensure the axes are interactive.
    # -----------------------------------------------------

    fig.update_xaxes(
        fixedrange=False
    )


    fig.update_yaxes(
        fixedrange=False
    )


    # -----------------------------------------------------
    # Default mode = rectangular zoom.
    # -----------------------------------------------------

    fig.update_layout(
        dragmode="zoom"
    )


    final_config = (
        _prepare_config(
            config
        )
    )


    div_id = (
        "chi_exact_box_zoom_"
        +
        uuid.uuid4().hex
    )


    # =====================================================
    # Determine height
    # =====================================================

    if height is None:

        layout_height = (
            fig.layout.height
        )


        if (
            layout_height
            is not None
            and
            isinstance(
                layout_height,
                (
                    int,
                    float,
                )
            )
        ):

            height = int(
                layout_height
            )

        else:

            height = (
                DEFAULT_PLOT_HEIGHT
            )


    height = max(
        MIN_PLOT_HEIGHT,
        int(height)
    )


    graph_height = max(
        300,
        height
        -
        HINT_HEIGHT
    )


    # =====================================================
    # Generate Plotly HTML
    # =====================================================

    plot_html = (
        pio.to_html(
            fig,

            include_plotlyjs=True,

            full_html=False,

            config=final_config,

            div_id=div_id,

            default_width="100%",

            default_height=(
                f"{graph_height}px"
            ),
        )
    )


    # =====================================================
    # Custom CHI zoom JavaScript
    # =====================================================

    custom_js = (
        _build_custom_zoom_javascript(
            div_id
        )
    )


    # =====================================================
    # Hint
    # =====================================================

    hint_html = """
<div
    style="
        height: 32px;
        line-height: 26px;
        font-family: Arial, sans-serif;
        font-size: 12px;
        color: #666666;
        padding-left: 4px;
        box-sizing: border-box;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    "
>
    CHI-style Box Zoom
    &nbsp;·&nbsp;
    Selected rectangle fills the plot
    &nbsp;·&nbsp;
    X/Y zoom independently
    &nbsp;·&nbsp;
    Pan = toolbar
    &nbsp;·&nbsp;
    Double-click = reset
</div>
"""


    complete_html = (
        hint_html
        +
        plot_html
        +
        custom_js
    )


    # =====================================================
    # Streamlit component
    # =====================================================

    st.iframe(
        complete_html,

        height=height,

        scrolling=False,
    )


# =========================================================
# Backward compatibility
# =========================================================

def render_mouse_zoom_plotly(
    figure,
    config=None,
    height=None,
):
    """
    Backward-compatible alias.
    """

    return render_box_zoom_plotly(
        figure=figure,
        config=config,
        height=height,
    )


# =========================================================
# Patch st.plotly_chart
# =========================================================

def enable_mouse_anchor_zoom():
    """
    Replace Streamlit's st.plotly_chart with CHI-style
    rectangular zoom for ordinary Plotly figures.

    Intended for:

        CV
        GCD
        EIS
        Multiple EIS / overlays

    Native Streamlit selection behavior is preserved when
    on_select is explicitly enabled.
    """

    current_plotly_chart = (
        st.plotly_chart
    )


    # -----------------------------------------------------
    # Already patched
    # -----------------------------------------------------

    if getattr(
        current_plotly_chart,
        "_chi_exact_box_zoom",
        False,
    ):

        return


    # Capture the genuine Streamlit implementation inside
    # this closure.
    #
    # This is more robust than relying on a module-level
    # global variable across Streamlit reruns.
    original_plotly_chart = (
        current_plotly_chart
    )


    # -----------------------------------------------------
    # Replacement
    # -----------------------------------------------------

    def patched_plotly_chart(
        figure_or_data,
        *args,
        **kwargs,
    ):

        # -------------------------------------------------
        # If native Streamlit selection events are wanted,
        # fall back to the original Streamlit renderer.
        # -------------------------------------------------

        on_select = (
            kwargs.get(
                "on_select",
                "ignore",
            )
        )


        if (
            on_select is not None
            and
            on_select != "ignore"
        ):

            return original_plotly_chart(
                figure_or_data,
                *args,
                **kwargs,
            )


        config = (
            kwargs.get(
                "config"
            )
        )


        # -------------------------------------------------
        # Render CHI-style plot.
        # -------------------------------------------------

        try:

            return render_box_zoom_plotly(
                figure=figure_or_data,
                config=config,
            )


        # -------------------------------------------------
        # Safety fallback.
        # -------------------------------------------------

        except Exception:

            return original_plotly_chart(
                figure_or_data,
                *args,
                **kwargs,
            )


    # Mark patched function.
    patched_plotly_chart._chi_exact_box_zoom = (
        True
    )


    # Keep reference for debugging if necessary.
    patched_plotly_chart._chi_original_plotly_chart = (
        original_plotly_chart
    )


    # Install.
    st.plotly_chart = (
        patched_plotly_chart
    )
