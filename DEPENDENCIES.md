# Dependencies

## Python Version
Python 3.8+

## Required Packages

| Package | Used by | Purpose |
|---|---|---|
| `numpy` | `vc2_logistic_regression.py` | Numerical computation |
| `pandas` | both | Data manipulation and time series |
| `matplotlib` | both | Plotting charts and equity curves |
| `scikit-learn` | `vc2_logistic_regression.py` | Logistic Regression, StandardScaler, Pipeline, and metrics |
| `backtrader` | `vc2strategy.py` | Backtesting engine, data feeds, and analyzers |

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn backtrader
```

## Standard Library (no installation needed)
- `os`
- `glob`
- `logging`
- `dataclasses`
- `typing`

## Data Requirements
Place stock CSV files in a `data/` directory (configurable via `DATA_DIR`). Each CSV must have the following columns:

```
date, price, rsi, pb_ratio, pe_ratio, ps_ratio, pcf_ratio, ev_ebitda, shareholder_yield
```
