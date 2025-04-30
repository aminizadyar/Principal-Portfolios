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

- Build the prediction matrix from return and signal panels  
- Decompose that matrix into  
  - **Principal Portfolios (PPs)** – timeable portfolios ordered by predictability  
  - **Principal Exposure Portfolios (PEPs)** – factor-exposed (beta) strategies  
  - **Principal Alpha Portfolios (PAPs)** – beta-neutral (alpha) strategies  

---

## Methodology at a glance 📐

### 1 · Prediction matrix  

For excess-return vector $R_{t+1} \in \mathbb{R}^{N}$ and signal vector $S_t \in \mathbb{R}^{N}$:

$$
\Pi = \mathbb{E}[R_{t+1} S_t^\top]
$$

So $\Pi_{ij} = \mathbb{E}[R_{i,t+1} S_{j,t}]$ tells how signal $j$ forecasts return $i$.

---

### 2 · Optimal linear rule  

Choose a fixed position matrix $L$ (with $\| L \| \leq 1$) to maximise:

$$
\mathbb{E}[S_t^\top L R_{t+1}]
\quad \Rightarrow \quad
L^* = (\Pi^\top \Pi)^{-1/2} \Pi^\top
$$


The optimal value equals the sum of the singular values $\{\sigma_k\}$ of $\Pi$.

---

### 3 · Principal Portfolios (PPs)  

SVD decomposition: $\Pi = U \Sigma V^\top$.  
For each singular triplet $(u_k, v_k, \sigma_k)$:

$$ 
L_k = v_k u_k^\top  \quad
{PP}_k(t+1) = {S_t^\top} {L_k} {R_{t+1}} 
$$

with expected return $\mathbb{E}[\text{PP}_k] = \sigma_k$.  
Summing all PPs reproduces the optimal strategy.

---

### 4 · Alpha/Beta symmetry  

Split the prediction matrix:

$$
\Pi_s = \tfrac{1}{2}(\Pi + \Pi^\top), \quad
\Pi_a = \tfrac{1}{2}(\Pi - \Pi^\top)
$$

- **PEPs**: eigenvectors of $\Pi_s$; returns = eigenvalues; carry factor exposure.  
- **PAPs**: eigenvectors of $\Pi_a$; beta-neutral by construction; positive returns signal mispricing.

---

### 5 · Asset-pricing test  

If signals are true betas to the pricing kernel, no-arbitrage implies:  
$\Pi$ must be symmetric & positive semi-definite.  

Therefore:  
- Negative eigenvalues in $\Pi_s$  
- Non-zero $\Pi_a$  
⇒ provide direct evidence of alpha.

---

## Reference

Kelly, B., Malamud, S., & Pedersen, L. H. (2023). [*Principal Portfolios*](https://doi.org/10.1111/jofi.13199). *The Journal of Finance*, 78(1), 347–392.

---

## License

MIT
```