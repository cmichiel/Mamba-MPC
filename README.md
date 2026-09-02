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
