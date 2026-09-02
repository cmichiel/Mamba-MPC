"""Mamba-MPC: sequence models as multi-step predictors inside model predictive control.

Companion code for *Mamba Sequence Modeling meets Model Predictive Control*
(Cevaal, de Jong, Lazar).

Layout
------
`models`         PyTorch definitions: Mamba, LSTM, Transformer
`casadi_models`  CasADi twins of the above, for embedding in the MPC problem
`mpc`            the CasADi/IPOPT optimal control problem builder
`systems`        plant dynamics and excitation signals
`data`           Hankel/sequence construction and checkpoint saving
`plotting`       the paper's figure style, and one builder per paper figure

`plotting` is not imported here: importing it applies the paper's matplotlib
style, which renders all text through LaTeX. Keeping it out of this module
means `import mambampc` works without a TeX installation.
"""

__version__ = "1.0.0"
