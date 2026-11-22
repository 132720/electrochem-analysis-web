# V3 Reference Notes

## Direct supercapacitor temperature ML baseline

S. Ghosh et al.
Temperature-dependent performance prediction for cerium oxynitride solid-state symmetric supercapacitor using machine learning.
Journal of Energy Storage, 113 (2025), 115562.
DOI: 10.1016/j.est.2025.115562

Use:
- literature RF / decision-tree temperature-current baseline
- temperature-dependent supercapacitor motivation

Do not claim:
- its 99.92% R² is a full GCD curve R²
- the PG-BDL V3 framework is derived directly from this paper

---

## Coordinate transformation for sharp-gradient PINNs

Z. Zhang, X. Su, X. Yuan.
Physics-informed neural networks with coordinate transformation to solve high Reynolds number boundary layer flows.
Journal of Computational Physics, 541 (2025), 114338.
DOI: 10.1016/j.jcp.2025.114338

Use:
- conceptual support for stretching a steep physical region in a transformed coordinate
- motivation for conditioned monotonic GCD time warping

Adaptation boundary:
- GCD is not a fluid boundary layer PDE
- V3 borrows the numerical representation concept, not the governing equations

---

## Latent neural operator learning

K. Kontolati, S. Goswami, G. E. Karniadakis, et al.
Learning nonlinear operators in latent spaces for real-time predictions of complex dynamics in physical systems.
Nature Communications, 15 (2024), 5101.
DOI: 10.1038/s41467-024-49411-w

Use:
- low-dimensional latent-space operator learning motivation
- support for predicting latent coordinates instead of the full high-dimensional output directly

Adaptation boundary:
- V3 uses PCA as the invertible reduced-dimensional representation
- current Temperature + constant-J input is not the same as the full function-input DeepONet setting

---

## Sparse-data GPR hybrid residual modeling

J. B. Fogelquist, X. Lin.
Combining electrochemistry and data-sparse Gaussian process regression for lithium-ion battery hybrid modeling.
Applied Energy, 399 (2025), 126458.
DOI: 10.1016/j.apenergy.2025.126458

Use:
- sparse-data GPR motivation
- hybrid physics/data residual-model concept
- uncertainty-aware small-data modeling rationale

Adaptation boundary:
- paper is lithium-ion battery SPMe + GPR
- V3 uses the sparse-data hybrid modeling principle for supercapacitor GCD

---

## Recommended wording

"Motivated by recent advances in coordinate-transformed physics-informed
learning for sharp-gradient problems, latent-space neural operators, and
sparse-data Gaussian-process hybrid modeling, a temperature- and
current-density-conditioned branch-decoupled latent framework was developed
for full supercapacitor GCD prediction."

Avoid:
- "the first"
- "novel DeepONet algorithm"
- "exact physics-informed model"
unless additional literature review and governing-equation residual support
justify those claims.
