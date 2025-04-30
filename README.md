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

**Trading rule:**

<div align="center">
  𝑤<sub>𝑡</sub> = 𝑆<sub>𝑡</sub><sup>⊤</sup>𝑉 𝑈<sup>⊤</sup> = 
  <span style="display: inline-block; vertical-align: middle; height: 3em; line-height: 3em;">
    <span style="display: inline-block; vertical-align: middle; line-height: normal;">
      <span style="display: block; text-align: center;">𝐾</span>
      <span style="display: block; text-align: center;">∑</span>
      <span style="display: block; text-align: center;">𝑘=1</span>
    </span>
  </span>
  𝑠<sub>𝑘,𝑡</sub> 𝑢<sub>𝑘</sub>
</div>

---
### 3 · Principal Portfolios (PPs)

SVD: $\Pi = U\,\Sigma\,V^\top$.

<p align="center">
  <img 
    src="https://latex.codecogs.com/svg.image?\color{magenta}%20L_k%20%3D%20v_k%20u_k%5E%5Ctop%2C%20%5Cquad%20PP_k%28t%2B1%29%20%3D%20S_t%5E%5Ctop%20L_k%20R_%7Bt%2B1%7D%2C%20%5Cquad%20%5Cmathbb%7BE%7D%5BPP_k%5D%20%3D%20%5Csigma_k"  
    alt="L_k = v_k u_k^⊤,  PP_k(t+1) = S_t^⊤ L_k R_{t+1},  E[PP_k] = σ_k" 
  />
</p>

*Timeable portfolios*: the top $k$ singular values pinpoint where predictability is strongest.

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



