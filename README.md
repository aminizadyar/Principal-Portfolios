GitHub-flavored Markdown does **not support LaTeX equations** rendered inside `\[ ... \]` or `\( ... \)` environments, nor does it support LaTeX-style alignment or matrix notation out of the box.

To render math properly on your GitHub README, you have two main options:

---

### ✅ **Option 1: Use images for equations**
Best for perfect display, but less flexible/editable.

---

### ✅ **Option 2: Use MathJax via HTML** *(recommended workaround)*

You can make GitHub render LaTeX using a simple MathJax setup with inline HTML tags (`<script>` and `<style>` won’t work on GitHub Pages/README, so this is limited). Instead, you can do the following:

- Use backticks with `math` environments in a simplified way using **dollar signs (`$...$` for inline, `$$...$$` for block)**.
- Equations will still **not render as math** on the main GitHub repo page but **will render** on GitHub Pages if you set up a proper Jekyll site.

But assuming you're staying in `README.md`, your best bet is to:

---

## ✅ **Fix your current README with inline formatting**
Here's your **fixed version** that works best **on GitHub** using **`$` for inline math** and triple backticks for code.

---  
I'll keep your formatting, but change the math parts to be Markdown/GitHub-friendly:

---

```markdown
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
L_k = v_k u_k^\top, \quad
\text{PP}_k(t+1) = S_t^\top L_k R_{t+1}
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

---

You can copy and paste the fixed version above into your `README.md`. It will render properly on GitHub (though the math won't look like LaTeX—it'll be readable and clean).

Would you like help setting up GitHub Pages with MathJax so the equations display beautifully?