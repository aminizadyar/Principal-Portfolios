# principal_portfolios <!-- omit in toc -->

[![PyPI version](https://img.shields.io/pypi/v/principal_portfolios.svg)](https://pypi.org/project/principal_portfolios)
[![Python](https://img.shields.io/pypi/pyversions/principal_portfolios.svg)](https://pypi.org/project/principal_portfolios)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

> A pure-Python implementation of the **Principal Portfolios** framework introduced by Kelly, Malamud & Pedersen (2023), *The Journal of Finance* for developing optimal trading strategies that exploit both own-asset and cross-asset predictive signals.

---

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methodology in a Nutshell 📐](#methodology-in-a-nutshell-)
  - [1 · Prediction matrix $\Pi$](#1-·-prediction-matrix-\pi)
  - [2 · Optimal linear rule](#2-·-optimal-linear-rule)
  - [3 · Principal Portfolios (PPs)](#3-·-principal-portfolios-pps)
  - [4 · Alpha / Beta symmetry](#4-·-alpha--beta-symmetry)
  - [5 · Asset-pricing test](#5-·-asset-pricing-test)
- [Package Features](#package-features)
- [Project Roadmap](#project-roadmap)
- [Citation](#citation)
- [License](#license)

---

## Installation

```bash
pip install principal_portfolios
```

> Need the dev version?  
> `pip install git+https://github.com/your-github-handle/principal_portfolios`

---

## Quick Start

```python
import pandas as pd
from principal_portfolios import utils, pp

# 1) load panels of excess returns and signals (shape: T × N)
returns = pd.read_parquet("returns.parquet")
signals = pd.read_parquet("signals.parquet")

# 2) build prediction matrix
Pi_hat = utils.prediction_matrix(returns, signals)

# 3) decompose into principal portfolios
pp_obj = pp.decompose(Pi_hat, top_k=5)

# 4) trade the first principal portfolio
weights_t = pp_obj.trade(signals.iloc[-1], k=0)   # position vector for next period
```

A full notebook example lives in [`examples/`](examples/).

---

## Methodology in a Nutshell 

### 1 · Prediction matrix $\Pi$

For excess-return vector $R_{t+1} \in \mathbb{R}^N$ and signal vector $S_t \in \mathbb{R}^N$:

$$
\Pi = \mathbb{E}[R_{t+1} S_t^\top], \quad
\Pi_{ij} = \mathbb{E}[R_{i,t+1} S_{j,t}].
$$

* **Diagonal** elements ($i=j$) capture *own-signal predictability* — how a stock’s *own* signal forecasts its future return.  
* **Off-diagonal** elements ($i\neq j$) capture *cross-predictability* — how the signal of asset $j$ forecasts the future return of asset $i$.  
Collecting every element in a single matrix lets us use *all* this information at once, rather than throwing away the rich cross-asset structure.

---

### 2 · Optimal linear rule

We consider a set of linear strategies of the form $R_{t+1}^w = w_t^\top R_{t+1}$. Define the **position matrix** $L$ that turns signals into weights $w_t = S_t^\top L$. This framework nests the traditional and more familiar cases of simple factor portfolios and long-short portfolios.

* **Simple factor portfolio.**  
  Taking $L = I$ (the identity matrix) gives $w_t = S_t$: each asset is traded only on its *own* signal.  
  This is the classic long-only “characteristic-sorted factor” and uses *only* the diagonal of $\Pi$.

* **Long-short portfolio.**  
  If we restrict $S_t$ to the largest and smallest signals and define $D_{jt}$ where:<br>
  <div align="center">
  <table>
    <tr><td>𝐷<sub>𝑗𝑡</sub> = +1</td><td>if 𝑆<sub>𝑗𝑡</sub> is the maximum value at time 𝑡,</td></tr>
    <tr><td>𝐷<sub>𝑗𝑡</sub> = -1</td><td>if 𝑆<sub>𝑗𝑡</sub> is the minimum value at time 𝑡,</td></tr>
    <tr><td>𝐷<sub>𝑗𝑡</sub> = 0</td><td>otherwise</td></tr>
  </table>
  </div>
  Notably the long-short (LS) portfolio would simply be $LS_{t+1} = D_t^\top R_{t+1}$.

We bound the overall size of the position matrix by ‖L‖ ≤ 1 which represents a bound on the portfolio size. Then we solve the following optimization problem to find the optimal portfolio:

$$
\max_{L}\;\mathbb{E}[S_t^\top L R_{t+1}]
\quad\Longrightarrow\quad
L^* = (\Pi^\top \Pi)^{-1/2}\ \Pi^\top,
$$

The optimal position matrix is denoted by $L^*$ and is expressed in terms of the prediction matrix.

---

### 3 · Principal portfolios via SVD
We apply singular value decomposition (SVD) to derive the principal portfolios:

<div align="center">
  Π = U Σ V<sup>⊤</sup>
</div>

* **Portfolio weights:** 𝑢<sub>𝑘</sub> (columns of 𝑈)  
  𝑃<sub>𝑘,𝑡+1</sub> = 𝑢<sub>𝑘</sub><sup>⊤</sup>𝑅<sub>𝑡+1</sub>

* **Signal factors:** 𝑣<sub>𝑘</sub> (columns of 𝑉)  
  𝑠<sub>𝑘,𝑡</sub> = 𝑣<sub>𝑘</sub><sup>⊤</sup>𝑆<sub>𝑡</sub>

* **Predictive strength:** σ<sub>𝑘</sub> (diagonal of Σ)  
  𝔼[𝑃<sub>𝑘,𝑡+1</sub>𝑠<sub>𝑘,𝑡</sub>] = σ<sub>𝑘</sub>

Because 𝑈 and 𝑉 are orthonormal, the 𝑃<sub>𝑘</sub> and 𝑠<sub>𝑘</sub> series are pairwise uncorrelated. Rank by σ<sub>𝑘</sub> and keep the top 𝐾 to get a low-dimensional, maximally predictable 𝐾-factor strategy.

**Optimal Trading rule:**

<div align="center">
  𝑤<sub>𝑡</sub> = 𝑆<sub>𝑡</sub><sup>⊤</sup>𝑉 𝑈<sup>⊤</sup> = ∑ 𝑠<sub>𝑘,𝑡</sub> 𝑢<sub>𝑘</sub>
</div>

---

### 4 · Principal **Exposure** (PEP) & **Alpha** (PAP) portfolios  

Start by splitting the prediction matrix into its symmetric and antisymmetric parts:  

<div align="center">
  Π<sub>𝑠</sub> = ½(Π + Π<sup>⊤</sup>), Π<sub>𝑎</sub> = ½(Π − Π<sup>⊤</sup>), Π = Π<sub>𝑠</sub> + Π<sub>𝑎</sub>
</div>
<div style="margin-bottom: 20px;"></div>


| Component       | Portfolio Set | Factor β? | Expected Return |
|-----------------|---------------|-----------|-----------------|
| **Symmetric Π<sub>𝑠</sub>** | **PEPs** (principal *exposure* portfolios) | Non-zero | 𝔼[PEP<sub>𝑘</sub>] = λ<sup>𝑠</sup><sub>𝑘</sub> (eigenvalue of Π<sub>𝑠</sub>) |
| **Antisymmetric Π<sub>𝑎</sub>** | **PAPs** (principal *alpha* portfolios) | Zero | 𝔼[PAP<sub>𝑗</sub>] = 2λ<sup>𝑎</sup><sub>𝑗</sub> (eigenvalue of 𝑖Π<sub>𝑎</sub>) |

* **PEPs**  
  Diagonalize Π<sub>𝑠</sub> = 𝑊Λ<sub>𝑠</sub>𝑊<sup>⊤</sup>.  
  Each eigenvector 𝑤<sub>𝑘</sub> gives a PEP:  
  <div align="center">
    PEP<sub>𝑘,𝑡+1</sub> = 𝑆<sub>𝑡</sub><sup>⊤</sup>𝑤<sub>𝑘</sub> 𝑤<sub>𝑘</sub><sup>⊤</sup>𝑅<sub>𝑡+1</sub>
  </div>
  Traded long if λ<sup>𝑠</sup><sub>𝑘</sub> > 0, short otherwise. PEPs are the *factor-bearing* legs.

* **PAPs**  
  For the 𝑗-th purely imaginary eigenpair of Π<sub>𝑎</sub><sup>⊤</sup> (𝑥<sub>𝑗</sub> + 𝑖𝑦<sub>𝑗</sub>), the PAP is:  
  <div align="center">
    𝐿<sub>𝑗</sub> = 𝑥<sub>𝑗</sub>𝑦<sub>𝑗</sub><sup>⊤</sup> − 𝑦<sub>𝑗</sub>𝑥<sub>𝑗</sub><sup>⊤</sup>
  </div>
  with return:  
  <div align="center">
    PAP<sub>𝑗,𝑡+1</sub> = 𝑆<sub>𝑡</sub><sup>⊤</sup>𝐿<sub>𝑗</sub>𝑅<sub>𝑡+1</sub>
  </div>
  PAPs harvest *pure alpha*—zero systematic exposure.

* **Complete Strategy**  
  The optimal linear portfolio combines:  
  <div align="center">
    ∑<sub>𝑘</sub> sign(λ<sup>𝑠</sup><sub>𝑘</sub>) PEP<sub>𝑘</sub> + ∑<sub>𝑗</sub> PAP<sub>𝑗</sub>
  </div>

---

### 4 · Alpha / Beta symmetry

Split

$$
\Pi_s = \tfrac12(\Pi + \Pi^\top), \quad
\Pi_a = \tfrac12(\Pi - \Pi^\top).
$$

| Block                   | Decomposition      | Strategy                       | Property              |
|-------------------------|--------------------|--------------------------------|-----------------------|
| Symmetric $\Pi_s$       | eigenvectors ⇒ **PEPs** | Factor-exposed (beta)         | Return = eigenvalue   |
| Antisymmetric $\Pi_a$   | eigenvectors ⇒ **PAPs** | Beta-neutral (alpha)          | Pure alpha source     |

---

### 5 · Asset-pricing test

If signals are genuine betas to the pricing kernel, no-arbitrage ⇒ $\Pi$ must be *symmetric & positive-semidefinite*.

Violations:
- negative eigenvalues of $\Pi_s$
- non-zero $\Pi_a$

are direct, model-free evidence of mis-pricing.

---

## Package Features

| Module        | What it does                          |
|---------------|---------------------------------------|
| `utils`       | Data prep, windowed estimators, signal scaling |
| `pp.decompose`| SVD / eigendecomp with rank selection and shrinkage |
| `pp.trade`    | Generates period-$t$ weights for PPs, PEPs, PAPs |
| `backtest`    | Simple vectorised back-testing helpers |
| `plotting`    | Performance charts (cumulative PnL, eigenvalue scree, etc.) |

*NumPy-only core; optional extras (`pandas`, `matplotlib`) auto-installed.*

---

## Project Roadmap

- [x] Core decomposition & trading API  
- [ ] Transaction-cost aware back-tests  
- [ ] Online (“rolling”) SVD with forgetting factor  
- [ ] R & Julia ports  
- [ ] Paper replication notebooks  

Contributions via pull requests are welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Citation

If you use this code in academic work, please cite both the package and the original paper:

```bibtex
@software{principal_portfolios,
  author  = {Your Name},
  title   = {principal_portfolios: Python implementation of Principal Portfolios},
  year    = {2025},
  url     = {https://github.com/your-github-handle/principal_portfolios},
  version = {<current_version>}
}

@article{kelly2023principal,
  title   = {Principal Portfolios},
  author  = {Kelly, Bryan and Malamud, Semyon and Pedersen, Lasse Heje},
  journal = {The Journal of Finance},
  volume  = {78},
  number  = {1},
  pages   = {347--392},
  year    = {2023},
  doi     = {10.1111/jofi.13199}
}
```

---

## License

Distributed under the MIT License — see [`LICENSE`](LICENSE) for details.



