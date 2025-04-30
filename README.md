# principal_portfolios

A Python package implementing the **Principal Portfolios** methodology introduced by Kelly, Malamud, and Pedersen (2023), *The Journal of Finance*.

---

## Installation

```bash
pip install principal_portfolios
```

---

## Overview

This package provides tools for constructing and analyzing **Principal Portfolios (PPs)**—linear trading strategies obtained from the singular-value decomposition (SVD) of a **prediction matrix** that captures both own-asset and cross-asset predictive signals.

**Key features**

* Build the prediction matrix from return and signal panels  
* Decompose that matrix into  
  * **Principal Portfolios (PPs)** – timeable portfolios ordered by predictability  
  * **Principal Exposure Portfolios (PEPs)** – factor-exposed (beta) strategies  
  * **Principal Alpha Portfolios (PAPs)** – beta-neutral (alpha) strategies  

---

## Methodology at a glance 📐

### 1&nbsp;· Prediction matrix  

For excess-return vector \(R_{t+1}\in\mathbb R^{N}\) and signal vector \(S_t\in\mathbb R^{N}\)

\[
\Pi = \operatorname{E}\!\bigl[R_{t+1}S_t^{\!\top}\bigr],
\]

so \(\Pi_{ij} = \operatorname{E}[R_{i,t+1}S_{j,t}]\) tells how signal \(j\) forecasts return \(i\).

### 2&nbsp;· Optimal linear rule  

Choose a fixed position matrix \(L\) (‖ L ‖ ≤ 1) to maximise  

\[
\operatorname{E}[S_t^{\!\top} L R_{t+1}]
\;\;\Longrightarrow\;\;
L^\star = (\Pi^{\!\top}\Pi)^{-1/2}\,\Pi^{\!\top},
\]

and the optimal value equals the sum of the singular values \(\{\sigma_k\}\) of \(\Pi\).

### 3&nbsp;· Principal Portfolios (PPs)  

SVD: \(\Pi = U\Sigma V^{\!\top}\).  
For each singular triplet \((u_k, v_k, \sigma_k)\)

\[
L_k = v_k u_k^{\!\top}, \qquad
\text{PP}_k(t+1) = S_t^{\!\top} L_k R_{t+1},
\]

with expected return \(\operatorname{E}\![\text{PP}_k] = \sigma_k\).  
Summing all PPs reproduces the optimal strategy.

### 4&nbsp;· Alpha/Beta symmetry  

Split the prediction matrix

\[
\Pi_s = \tfrac12(\Pi + \Pi^{\!\top}), \quad
\Pi_a = \tfrac12(\Pi - \Pi^{\!\top}).
\]

* **PEPs**: eigenvectors of \(\Pi_s\); returns = eigenvalues; carry factor exposure.  
* **PAPs**: eigenvectors of \(\Pi_a\); beta-neutral by construction; positive returns signal mis-pricing.

### 5&nbsp;· Asset-pricing test  

If signals are true betas to the pricing kernel, no-arbitrage ⇒ \(\Pi\) must be symmetric & positive-semidefinite.  
Therefore, negative eigenvalues in \(\Pi_s\) or non-zero \(\Pi_a\) provide direct evidence of alpha.

---

## Reference

Kelly, B., Malamud, S., & Pedersen, L. H. (2023). [*Principal Portfolios*](https://doi.org/10.1111/jofi.13199). *The Journal of Finance*, 78(1), 347–392.

---

## License

MIT