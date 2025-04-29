# principal_portfolios

A Python package implementing the **Principal Portfolios** methodology introduced by Kelly, Malamud, and Pedersen (2023), *The Journal of Finance*.

## Installation

Install via:

```bash
pip install principal_portfolios
```

## Overview

This package provides tools for constructing and analyzing **Principal Portfolios** — linear trading strategies derived from the singular value decomposition (SVD) of the **prediction matrix** that captures both own-asset and cross-asset predictive signals.

Key components include:

- Construction of the prediction matrix from asset returns and signals
- Decomposition into:
  - **Principal Portfolios (PPs)**: timeable portfolios ordered by predictability
  - **Principal Exposure Portfolios (PEPs)**: factor-exposed strategies (beta)
  - **Principal Alpha Portfolios (PAPs)**: factor-neutral strategies (alpha)

## Reference

Kelly, B., Malamud, S., & Pedersen, L. H. (2023). [*Principal Portfolios*](https://doi.org/10.1111/jofi.13199). *The Journal of Finance*, 78(1), 347–392.


