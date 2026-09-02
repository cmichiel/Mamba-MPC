# Mamba-MPC

Companion code for **[Mamba Sequence Modeling Meets Model Predictive Control][paper]**
— M. Cevaal, T. O. de Jong, M. Lazar, *IEEE Open Journal of Control Systems*, 2026.

The paper builds model predictive control on a Mamba state-space network used as a
multi-step predictor, and compares it against LSTM and Transformer predictors on two
numerical examples and one physical system.

[paper]: https://ieeexplore.ieee.org/document/11657796

## What is here

| Experiment | Mamba | LSTM | Transformer |
|---|:---:|:---:|:---:|
| Van der Pol — stabilisation from 100 initial conditions | ✓ | — | — |
| Van der Pol — reference tracking | ✓ | ✓ | ✓ |
| Van der Pol — computation time vs horizon | ✓ | ✓ | ✓ |
| Van der Pol — noise (SNR 20 dB) | ✓ | ✓ | — |
| Four Tank — MIMO reference tracking | ✓ | ✓ | — |
| Quanser Aero2 — hardware | ✓ | — | — |

```
mambampc/              importable package — everything reusable lives here
  models/              PyTorch: Mamba, LSTM, Transformer
  casadi_models.py     CasADi twins of all three, for embedding in the optimiser
  mpc.py               the optimal control problem builder
  simulate.py          the closed-loop MPC simulation
  systems.py           plant dynamics and excitation signals
  data.py              Hankel / sequence construction, checkpoint saving
  plotting.py          the paper's figure style, one builder per paper figure
experiments/
  van_der_pol/         notebook + data/ checkpoints/ results/ figures/
  four_tank/           idem
  aero2/               idem, plus the vendored Quanser SDK
```

Each experiment is one notebook that sets up the study and plots it. All the machinery
it calls is in `mambampc`, so the same Mamba definition, the same CasADi translation and
the same MPC problem are shared by every experiment.

## Installation

```bash
git clone https://github.com/cmichiel/Mamba-MPC.git
cd Mamba-MPC
pip install -r requirements.txt
jupyter notebook experiments/van_der_pol/van_der_pol.ipynb
```

Python 3.11 or newer. The notebooks add the repository root to `sys.path`, so no install
step is needed. A GPU helps for training but nothing here requires one — **every figure
can be regenerated on CPU** (Aero2 replots from its stored hardware run).

**The figures need LaTeX.** `mambampc/plotting.py` renders all text with
`text.usetex = True`, which is how the paper's figures were typeset, so producing them
requires a TeX distribution providing `latex` and `dvipng` — TeX Live, MiKTeX or MacTeX.
Everything else — data generation, training, closed-loop control — runs without it;
only the figure cells need it.

The Aero2 experiment additionally needs the Quanser Python SDK (`hal`, `pal`, `pit`,
`qvl`, vendored under `experiments/aero2/`) and the physical rig.

## Reproducing the results

**From the shipped checkpoints (minutes, CPU).** Open a notebook and skip the data
generation and training sections; everything downstream runs from `checkpoints/`. This
is the path to regenerate the figures.

The figures are the paper's, not lookalikes: each experiment's plotting cell calls a
builder in `mambampc/plotting.py` that reproduces the corresponding figure's palette,
layout and typography. Each lands in the experiment's `figures/` as both `.png` and
`.pdf`. Van der Pol produces paper Figures 7, 8, 9 and 11, Four Tank Figure 12, and
Aero2 Figure 14.

**From scratch (hours, GPU recommended).** Run the notebook top to bottom. Data
generation and training are the expensive parts; the closed-loop sections are the same
either way.

## What reproduces, and what does not

This section is deliberately explicit, because parts of the study are reproducible and
parts are not.

**Re-runnable end to end:** the Van der Pol and Four Tank experiments — data generation,
training, closed-loop control, and figures.

**Training is not bit-reproducible.** The shipped checkpoints are the ones behind the
paper's numbers and are the artifacts of record. The code now seeds NumPy and PyTorch, so
a fresh training run is repeatable, but those seeds were added after the paper's models
were trained. Retraining therefore gives a comparable model, not an identical one.

**Closed-loop trajectories reproduce closely, not exactly.** Regenerating the Van der Pol
reference-tracking run matches the stored trajectory to a median of 3.5e-4, with
deviations up to 0.29 confined to transients at reference steps. The nonlinear program is
nonconvex and CasADi/IPOPT has changed versions since the original runs, so exact
numerical reproduction is not achievable. The figures are visually indistinguishable.

**The Transformer is re-plottable but not re-runnable.** Its trained weights were not
recoverable. `results/transformer/` holds the stored closed-loop trajectories behind the
paper's comparison figure, and `mambampc/models/transformer.py` and the CasADi twin are
the model code, so the method is fully inspectable — but the specific network that
produced those trajectories cannot be reloaded. Its configuration was `depth=3, heads=4,
mlp_dim=32, hidden_dim=8, num_frames=50`, SiLU activation, attention temperature 2.5.

**Computation time is replotted, not re-timed.** Paper Figure 9 is drawn from the solve
times recorded for the paper (`results/*/comp_time_vs_N.npy`), which were measured on an
RTX 4070 and a Ryzen 9 7845HX. Re-running would time your machine instead, so the stored
records are the artifacts of record for that figure.

**The two Monte-Carlo studies redraw their random samples.** Paper Figure 7 samples 100
initial states and Figure 11 samples 100 noise realisations; the seeds behind the paper's
runs were not recorded. The arrays shipped in `results/` are the paper's, so the figures
as committed are the paper's — but re-running those sections overwrites them with a fresh
draw. The figure is then the same in every other respect (layout, palette, spread) while
the individual trajectories are not the paper's.

**Aero2 cannot be re-run without the hardware.** Its notebook keeps the outputs from the
original run as the record of the experiment. The model and controller code are shared
with the numerical examples, and every shipped checkpoint was checked to agree with its
CasADi twin to ~1e-15, so everything except the physical run is verifiable offline.

## Citation

```bibtex
@article{cevaal2026mamba,
  title   = {Mamba Sequence Modeling Meets Model Predictive Control},
  author  = {Cevaal, Michiel and de Jong, Thomas O. and Lazar, Mircea},
  journal = {IEEE Open Journal of Control Systems},
  year    = {2026},
  pages   = {1--15},
  doi     = {10.1109/OJCSYS.2026.3724785},
  issn    = {2694-085X}
}
```

## Licence

MIT — see [LICENSE](LICENSE).

The Mamba blocks and the parallel scan come from [mamba.py][mambapy] (MIT, © 2024
Alexandre TL); its licence is reproduced in
[mambampc/models/LICENSE-mamba.py](mambampc/models/LICENSE-mamba.py). The matplotlib
style in `mambampc/plotting.py` is the helper published with the PINNs code of Raissi et
al., credited in that file.

The [Quanser Academic Resources][quanser] SDK (`hal/`, `pal/`, `pit/`, `qvl/`) is vendored
unmodified under `experiments/aero2/` and is redistributed under its own BSD 3-Clause
licence, reproduced with its copyright notice in
[experiments/aero2/LICENSE-Quanser](experiments/aero2/LICENSE-Quanser).
`pit/LaneNet/architecture/` carries a further MIT licence of its own.

[quanser]: https://github.com/quanser/Quanser_Academic_Resources
[mambapy]: https://github.com/alxndrTL/mamba.py

## Contact

Michiel Cevaal — <m.cevaal@tilburguniversity.edu>
Thomas O. de Jong — <t.o.d.jong@tue.nl> · Mircea Lazar — <m.lazar@tue.nl>
