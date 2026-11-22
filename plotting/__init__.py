from .mouse_zoom import (
    enable_mouse_anchor_zoom,
)


# =========================================================
# Enable custom interactive Plotly behavior
# =========================================================

enable_mouse_anchor_zoom()


# =========================================================
# Publication plotting
# =========================================================

from .publication import (
    plot_cv_publication,
    plot_cv_multi_publication,
    plot_gcd_publication,
    plot_gcd_multi_publication,
    plot_nyquist_publication,
    plot_bode_modulus_publication,
    plot_bode_phase_publication,
    figure_to_bytes,
)
