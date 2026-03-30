# Statistical Arbitrage Backtester - NSE Equities

A pairs trading backtesting framework for NSE (National Stock Exchange of India) equities with cointegration analysis, Z-score signal generation, and realistic transaction cost modeling.

## Overview

This project implements a complete statistical arbitrage trading strategy backtest system:

- **Cointegration Scanner**: Identifies cointegrated pairs using Engle-Granger two-step test
- **Z-Score Signals**: Generates entry/exit signals based on rolling z-score of spread
- **Transaction Cost Model**: Realistic NSE cost structure (brokerage, STT, exchange charges, SEBI fees, GST, stamp duty)
- **Portfolio Backtester**: Multi-pair equal-weighted portfolio performance evaluation

## Features

| Feature | Description |
|---------|-------------|
| Cointegration Analysis | Engle-Granger test with ADF and half-life filtering |
| Signal Generation | Rolling Z-score with entry/exit/stop thresholds |
| Cost Modeling | NSE-specific fees including market impact |
| Performance Metrics | Sharpe, Sortino, Calmar, Drawdown, Win Rate |

## Installation

```bash
pip install numpy pandas scipy statsmodels
```

## Usage

```bash
python main.py
```

## Sample Output

```
════════════════════════════════════════════════════════════════════════
  STATISTICAL ARBITRAGE BACKTESTER  —  NSE EQUITIES
════════════════════════════════════════════════════════════════════════

📋  COINTEGRATED PAIRS IDENTIFIED
Pair                  Beta   Coint p    ADF p  Half-life
────────────────────────────────────────────────────────
AXISBANK/ICICIBANK  0.9433    0.0001   0.0000       6.9d
RELIANCE/HDFCBANK   0.8916    0.0004   0.0001       7.8d
TCS/WIPRO           0.9999    0.0425   0.0111      12.4d
WIPRO/PNB           0.6188    0.0584   0.0162      16.5d
HDFCBANK/AXISBANK   0.5713    0.0699   0.0202      16.8d
HDFCBANK/ICICIBANK   0.5220    0.0718   0.0208      17.2d

💸  TRANSACTION COST BREAKDOWN  (sample: ₹1000/share, 100 shares)
Item                  Buy (₹)   Sell (₹)
──────────────────────────────────────────
  brokerage             30.00      30.00
  stt                    0.00     100.00
  exchange               3.45       3.45
  sebi                   0.10       0.10
  gst                    6.02       6.02
  stamp_duty            15.00       0.00
  half_spread           25.00       25.00
  market_impact         10.00      10.00
  TOTAL...........      89.57     174.57

  ➤ Round-trip cost: 26.41 bps

📊  PER-PAIR PERFORMANCE SUMMARY
Pair                Sharpe  Ann Ret%   MaxDD%  Trades  WinRate%     NetPnL ₹
──────────────────────────────────────────────────────────────────────────
AXISBANK/ICICIBANK  -0.512     -2.84    -7.26       9     44.44      -44,940
RELIANCE/HDFCBANK   -0.323     -2.67   -10.00      12     66.67      -36,257
TCS/WIPRO            0.563      3.68    -4.54       9     66.67       86,864
WIPRO/PNB            1.466      7.94    -2.19      13     76.92      175,914
HDFCBANK/AXISBANK    0.590      6.84    -8.53      12     66.67      153,996
HDFCBANK/ICICIBANK   0.887      9.13    -7.99      10     90.00      195,792

════════════════════════════════════════════════════════════════════════
  PORTFOLIO  (equal-weighted, all pairs combined)
════════════════════════════════════════════════════════════════════════
  Sharpe Ratio          :      0.913
  Sortino Ratio         :      0.563
  Calmar Ratio          :      1.173
  Annual Return         :      3.68%
  Annual Volatility     :      4.03%
  Max Drawdown          :     -3.14%
  Total Trades          :         65
  Win Rate              :     69.23%
  Profit Factor         :      1.795
  Avg Hold (days)       :       13.2
  Total Net P&L         : ₹     531,368
  Total Transaction Costs: ₹     86,363
  Cost Drag             :     1.439%
════════════════════════════════════════════════════════════════════════
```

## Parameter Comparison

Run comparison across different strategy configurations:

```bash
python comparison.py
```

### Results Summary

| Scenario | Lookback | Entry Z | Sharpe | Ann Ret% | MaxDD% | Trades | Win Rate% | Net P&L (₹) |
|----------|----------|---------|--------|----------|--------|--------|-----------|-------------|
| **Conservative** | 90 | 2.5 | 2.062 | 3.90% | -0.92% | 26 | 65.38% | 505,426 |
| **Moderate** | 60 | 2.0 | 0.913 | 3.68% | -3.14% | 65 | 69.23% | 531,368 |
| **Aggressive** | 30 | 1.5 | 1.378 | 6.00% | -5.33% | 173 | 65.32% | 951,330 |

### Key Findings

1. **Conservative**: Lowest risk (0.92% max DD), highest Sharpe (2.062), fewest trades
2. **Moderate**: Balanced approach, highest win rate (69.23%)
3. **Aggressive**: Highest return (6.00%) but highest drawdown (-5.33%) and costs

## Architecture

```
ArbitrageBacktester/
├── main.py           # Main execution & reporting
├── comparison.py    # Parameter comparison analysis
└── README.md         # This file
```

### Core Components

| Class | Purpose |
|-------|---------|
| `CointegrationScanner` | Engle-Granger two-step cointegration test |
| `ZScoreSignal` | Rolling z-score entry/exit signal generation |
| `TransactionCostModel` | NSE fee structure calculation |
| `PairsTradingBacktester` | Single pair backtest engine |
| `PortfolioBacktester` | Multi-pair portfolio aggregation |

## Configuration

Key parameters in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 60 | Rolling window for z-score |
| `entry_z` | 2.0 | Z-score threshold for entry |
| `exit_z` | 0.5 | Z-score threshold for exit |
| `stop_z` | 3.5 | Stop-loss z threshold |
| `capital_per_pair` | 1,000,000 | INR allocation per pair |

## License

MIT
