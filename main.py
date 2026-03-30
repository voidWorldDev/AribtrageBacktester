import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  1. DATA GENERATION  (simulated NSE prices)
# ─────────────────────────────────────────────


def generate_nse_prices(
    tickers: list[str], n_days: int = 504, seed: int = 42
) -> pd.DataFrame:
    """
    Simulate realistic NSE daily price series with embedded cointegrated pairs.
    Uses geometric Brownian motion with mean-reverting spread injection.
    """
    rng = np.random.default_rng(seed)
    trading_days = pd.bdate_range("2022-01-03", periods=n_days, freq="B")

    prices = {}
    n = len(tickers)

    # base log-returns: slight positive drift, sector correlations
    corr_matrix = np.full((n, n), 0.35)
    np.fill_diagonal(corr_matrix, 1.0)
    # tighten pairs (0,1), (2,3), (4,5) to be cointegrated
    for i in range(0, min(6, n), 2):
        corr_matrix[i, i + 1] = corr_matrix[i + 1, i] = 0.92

    L = np.linalg.cholesky(corr_matrix)
    raw = rng.standard_normal((n_days, n))
    corr_returns = raw @ L.T

    annual_vol = 0.25
    daily_vol = annual_vol / np.sqrt(252)
    daily_drift = 0.08 / 252

    init_prices = rng.uniform(200, 3000, n)
    log_prices = np.zeros((n_days, n))
    log_prices[0] = np.log(init_prices)

    for t in range(1, n_days):
        log_prices[t] = log_prices[t - 1] + daily_drift + daily_vol * corr_returns[t]

    # inject mean-reverting spread into cointegrated pairs
    for i in range(0, min(6, n), 2):
        beta = rng.uniform(0.8, 1.2)
        spread_vol = 0.015
        theta = 0.08  # mean reversion speed
        spread = np.zeros(n_days)
        for t in range(1, n_days):
            spread[t] = (
                spread[t - 1]
                - theta * spread[t - 1]
                + spread_vol * rng.standard_normal()
            )
        log_prices[:, i + 1] = beta * log_prices[:, i] + spread

    price_df = pd.DataFrame(np.exp(log_prices), index=trading_days, columns=tickers)
    # add realistic intraday OHLCV columns per ticker (simplified)
    return price_df


# ─────────────────────────────────────────────
#  2. COINTEGRATION SCANNER
# ─────────────────────────────────────────────


class CointegrationScanner:
    """
    Engle-Granger two-step cointegration test on all ticker pairs.
    Stores hedge ratios via OLS regression.
    """

    def __init__(
        self,
        pvalue_threshold: float = 0.05,
        min_half_life: int = 5,
        max_half_life: int = 126,
    ):
        self.pvalue_threshold = pvalue_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def half_life(self, spread: pd.Series) -> float:
        """Ornstein-Uhlenbeck half-life from OLS regression on spread."""
        lag = spread.shift(1).dropna()
        diff = spread.diff().dropna()
        result = OLS(diff, add_constant(lag)).fit()
        lam = result.params.iloc[1]
        if lam >= 0:
            return np.inf
        return -np.log(2) / lam

    def adf_test(self, series: pd.Series) -> tuple[float, float]:
        res = adfuller(series, autolag="AIC")
        return res[0], res[1]

    def scan(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame of cointegrated pairs with statistics."""
        tickers = prices.columns.tolist()
        records = []

        for t1, t2 in combinations(tickers, 2):
            s1 = np.log(prices[t1])
            s2 = np.log(prices[t2])

            # OLS: s1 = alpha + beta * s2
            model = OLS(s1, add_constant(s2)).fit()
            beta = model.params.iloc[1]
            spread = s1 - beta * s2

            # Engle-Granger cointegration test
            _, pvalue, _ = coint(s1, s2)

            # ADF on spread
            adf_stat, adf_p = self.adf_test(spread)

            hl = self.half_life(spread)

            if (
                pvalue < self.pvalue_threshold
                and self.min_half_life <= hl <= self.max_half_life
            ):
                records.append(
                    {
                        "ticker1": t1,
                        "ticker2": t2,
                        "beta": round(beta, 4),
                        "coint_pval": round(pvalue, 4),
                        "adf_stat": round(adf_stat, 4),
                        "adf_pval": round(adf_p, 4),
                        "half_life": round(hl, 1),
                        "spread_std": round(spread.std(), 6),
                    }
                )

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("coint_pval").reset_index(drop=True)
        return df


# ─────────────────────────────────────────────
#  3. SIGNAL GENERATOR  (Z-score)
# ─────────────────────────────────────────────


class ZScoreSignal:
    """
    Rolling z-score of the cointegrated spread.
    Generates long/short entry & exit signals.
    """

    def __init__(
        self,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z  # stop-loss z threshold

    def compute(self, spread: pd.Series) -> pd.DataFrame:
        mu = spread.rolling(self.lookback).mean()
        sigma = spread.rolling(self.lookback).std()
        zscore = (spread - mu) / sigma

        signals = pd.DataFrame(
            {
                "spread": spread,
                "zscore": zscore,
                "mu": mu,
                "sigma": sigma,
            }
        )

        # 1 = long spread, -1 = short spread, 0 = flat
        position = 0
        positions = []
        for z in zscore:
            if np.isnan(z):
                positions.append(0)
                continue
            if position == 0:
                if z <= -self.entry_z:
                    position = 1  # spread too low → long
                elif z >= self.entry_z:
                    position = -1  # spread too high → short
            elif position == 1:
                if z >= -self.exit_z or z <= -self.stop_z:
                    position = 0
            elif position == -1:
                if z <= self.exit_z or z >= self.stop_z:
                    position = 0
            positions.append(position)

        signals["position"] = positions
        signals["trade"] = signals["position"].diff().fillna(0)
        return signals


# ─────────────────────────────────────────────
#  4. TRANSACTION COST & SLIPPAGE MODEL
# ─────────────────────────────────────────────


class TransactionCostModel:
    """
    NSE realistic cost structure:
      - Brokerage:       0.03% per side (discount broker)
      - STT (equity):    0.1% on sell side
      - Exchange txn:    0.00345% per side
      - SEBI charges:    0.0001% per side
      - GST:             18% on brokerage + exchange charges
      - Stamp duty:      0.015% on buy side
      - Slippage:        market-impact model based on ADV fraction
    """

    BROKERAGE_RATE = 0.0003
    STT_RATE = 0.001  # sell side only
    EXCHANGE_RATE = 0.0000345
    SEBI_RATE = 0.000001
    GST_RATE = 0.18
    STAMP_DUTY_RATE = 0.00015  # buy side only
    IMPACT_COEFF = 0.1  # Kyle's lambda approximation

    def __init__(self, avg_spread_bps: float = 5.0, adv_fraction: float = 0.001):
        """
        avg_spread_bps : average bid-ask spread in basis points
        adv_fraction   : trade size as fraction of avg daily volume
        """
        self.avg_spread_bps = avg_spread_bps
        self.adv_fraction = adv_fraction

    def compute_costs(self, price: float, qty: int, side: str) -> dict:
        """
        side: 'buy' or 'sell'
        Returns itemized cost dict and total cost in INR.
        """
        value = price * qty
        brokerage = value * self.BROKERAGE_RATE
        stt = value * self.STT_RATE if side == "sell" else 0.0
        exchange = value * self.EXCHANGE_RATE
        sebi = value * self.SEBI_RATE
        gst = (brokerage + exchange) * self.GST_RATE
        stamp = value * self.STAMP_DUTY_RATE if side == "buy" else 0.0
        half_spread = value * (self.avg_spread_bps / 2) / 10_000
        # market impact: linear in participation rate
        impact = value * self.IMPACT_COEFF * self.adv_fraction

        total = brokerage + stt + exchange + sebi + gst + stamp + half_spread + impact
        return {
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "exchange": round(exchange, 2),
            "sebi": round(sebi, 2),
            "gst": round(gst, 2),
            "stamp_duty": round(stamp, 2),
            "half_spread": round(half_spread, 2),
            "market_impact": round(impact, 2),
            "total": round(total, 2),
            "total_bps": round(total / value * 10_000, 2),
        }

    def round_trip_cost_bps(self, price: float = 1000, qty: int = 100) -> float:
        buy = self.compute_costs(price, qty, "buy")["total"]
        sell = self.compute_costs(price, qty, "sell")["total"]
        return (buy + sell) / (price * qty) * 10_000


# ─────────────────────────────────────────────
#  5. PAIRS TRADING BACKTESTER
# ─────────────────────────────────────────────


class PairsTradingBacktester:
    """
    Full event-driven pairs trading backtest for a single pair.
    Maintains trade log, P&L series, and computes performance metrics.
    """

    def __init__(
        self,
        signal_gen: ZScoreSignal,
        cost_model: TransactionCostModel,
        capital_per_leg: float = 500_000,  # INR 5 lakh per leg
        max_holding_days: int = 30,
    ):
        self.signal_gen = signal_gen
        self.cost_model = cost_model
        self.capital_per_leg = capital_per_leg
        self.max_holding_days = max_holding_days

    def run(self, prices: pd.DataFrame, t1: str, t2: str, beta: float) -> dict:
        """
        prices : DataFrame with at minimum columns [t1, t2]
        t1, t2 : ticker names
        beta   : hedge ratio (t1 vs beta*t2)
        """
        log_s1 = np.log(prices[t1])
        log_s2 = np.log(prices[t2])
        spread = log_s1 - beta * log_s2

        signals = self.signal_gen.compute(spread)
        signals["price1"] = prices[t1]
        signals["price2"] = prices[t2]

        trades = []
        daily_pnl = []
        position = 0  # current position: +1 long spread, -1 short
        entry_date = None
        entry_p1 = entry_p2 = 0.0
        qty1 = qty2 = 0
        hold_days = 0

        for date, row in signals.iterrows():
            sig_pos = int(row["position"])
            p1, p2 = row["price1"], row["price2"]

            # Force exit if max holding exceeded
            if position != 0:
                hold_days += 1
                if hold_days >= self.max_holding_days:
                    sig_pos = 0

            if position == 0 and sig_pos != 0:
                # ENTER
                qty1 = max(1, int(self.capital_per_leg / p1))
                qty2 = max(1, int(self.capital_per_leg / p2))

                if sig_pos == 1:  # long spread: buy t1, short t2
                    c_in = (
                        self.cost_model.compute_costs(p1, qty1, "buy")["total"]
                        + self.cost_model.compute_costs(p2, qty2, "sell")["total"]
                    )
                else:  # short spread: sell t1, buy t2
                    c_in = (
                        self.cost_model.compute_costs(p1, qty1, "sell")["total"]
                        + self.cost_model.compute_costs(p2, qty2, "buy")["total"]
                    )

                position = sig_pos
                entry_date = date
                entry_p1 = p1
                entry_p2 = p2
                hold_days = 0
                daily_pnl.append({"date": date, "pnl": -c_in, "type": "entry_cost"})

            elif position != 0 and sig_pos == 0:
                # EXIT
                if position == 1:
                    gross_pnl = qty1 * (p1 - entry_p1) - qty2 * (p2 - entry_p2) * beta
                    c_out = (
                        self.cost_model.compute_costs(p1, qty1, "sell")["total"]
                        + self.cost_model.compute_costs(p2, qty2, "buy")["total"]
                    )
                else:
                    gross_pnl = qty1 * (entry_p1 - p1) + qty2 * (entry_p2 - p2) * beta
                    c_out = (
                        self.cost_model.compute_costs(p1, qty1, "buy")["total"]
                        + self.cost_model.compute_costs(p2, qty2, "sell")["total"]
                    )

                net_pnl = gross_pnl - c_out
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "direction": "long" if position == 1 else "short",
                        "hold_days": hold_days,
                        "gross_pnl": round(gross_pnl, 2),
                        "total_costs": round(c_out, 2),
                        "net_pnl": round(net_pnl, 2),
                    }
                )
                daily_pnl.append({"date": date, "pnl": net_pnl, "type": "exit"})
                position = 0
                hold_days = 0

            else:
                daily_pnl.append({"date": date, "pnl": 0.0, "type": "hold"})

        pnl_df = pd.DataFrame(daily_pnl).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity = pnl_df["pnl"].cumsum() + self.capital_per_leg * 2

        metrics = self._compute_metrics(pnl_df["pnl"], equity, trades_df)
        return {
            "pair": (t1, t2),
            "beta": beta,
            "signals": signals,
            "trades": trades_df,
            "daily_pnl": pnl_df,
            "equity": equity,
            "metrics": metrics,
        }

    def _compute_metrics(
        self, pnl: pd.Series, equity: pd.Series, trades: pd.DataFrame
    ) -> dict:
        initial_cap = self.capital_per_leg * 2
        returns = pnl / initial_cap

        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = drawdown.min()

        calmar = ann_return / abs(max_dd) if max_dd < 0 else np.inf

        sortino_den = returns[returns < 0].std() * np.sqrt(252)
        sortino = ann_return / sortino_den if sortino_den > 0 else np.inf

        n_trades = len(trades)
        if n_trades > 0:
            win_rate = (trades["net_pnl"] > 0).mean()
            avg_win = (
                trades.loc[trades["net_pnl"] > 0, "net_pnl"].mean()
                if (trades["net_pnl"] > 0).any()
                else 0
            )
            avg_loss = (
                trades.loc[trades["net_pnl"] < 0, "net_pnl"].mean()
                if (trades["net_pnl"] < 0).any()
                else 0
            )
            profit_fac = (
                -avg_win * win_rate / (avg_loss * (1 - win_rate))
                if avg_loss < 0 and win_rate < 1
                else np.inf
            )
            avg_hold = trades["hold_days"].mean()
            total_net = trades["net_pnl"].sum()
            total_cost = trades["total_costs"].sum()
        else:
            win_rate = avg_win = avg_loss = profit_fac = avg_hold = total_net = (
                total_cost
            ) = 0

        return {
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "annual_return_pct": round(ann_return * 100, 2),
            "annual_vol_pct": round(ann_vol * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "n_trades": n_trades,
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win_inr": round(avg_win, 2),
            "avg_loss_inr": round(avg_loss, 2),
            "profit_factor": round(profit_fac, 3),
            "avg_hold_days": round(avg_hold, 1),
            "total_net_pnl_inr": round(total_net, 2),
            "total_costs_inr": round(total_cost, 2),
            "cost_drag_pct": round(total_cost / (initial_cap) * 100, 3),
        }


# ─────────────────────────────────────────────
#  6. PORTFOLIO BACKTESTER  (multi-pair)
# ─────────────────────────────────────────────


class PortfolioBacktester:
    """Run backtests across all cointegrated pairs and aggregate results."""

    def __init__(
        self,
        capital_per_pair: float = 1_000_000,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
        avg_spread_bps: float = 5.0,
    ):
        self.capital_per_pair = capital_per_pair
        self.signal_params = dict(
            lookback=lookback, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z
        )
        self.cost_params = dict(avg_spread_bps=avg_spread_bps)

    def run(self, prices: pd.DataFrame, pairs_df: pd.DataFrame) -> dict:
        results = {}
        for _, row in pairs_df.iterrows():
            t1, t2, beta = row["ticker1"], row["ticker2"], row["beta"]
            signal_gen = ZScoreSignal(**self.signal_params)
            cost_model = TransactionCostModel(**self.cost_params)
            bt = PairsTradingBacktester(
                signal_gen=signal_gen,
                cost_model=cost_model,
                capital_per_leg=self.capital_per_pair / 2,
            )
            res = bt.run(prices, t1, t2, beta)
            results[f"{t1}_{t2}"] = res

        # Portfolio aggregate
        if results:
            all_pnl = pd.concat(
                [v["daily_pnl"]["pnl"].rename(k) for k, v in results.items()], axis=1
            ).fillna(0)
            portfolio_pnl = all_pnl.sum(axis=1)
            total_capital = self.capital_per_pair * len(results)
            portfolio_equity = portfolio_pnl.cumsum() + total_capital

            bt_dummy = PairsTradingBacktester(
                ZScoreSignal(),
                TransactionCostModel(),
                capital_per_leg=total_capital / 2,
            )
            portfolio_metrics = bt_dummy._compute_metrics(
                portfolio_pnl,
                portfolio_equity,
                pd.concat(
                    [v["trades"] for v in results.values() if not v["trades"].empty],
                    ignore_index=True,
                )
                if any(not v["trades"].empty for v in results.values())
                else pd.DataFrame(),
            )
        else:
            portfolio_metrics = {}
            portfolio_pnl = pd.Series(dtype=float)
            portfolio_equity = pd.Series(dtype=float)

        return {
            "pair_results": results,
            "portfolio_pnl": portfolio_pnl,
            "portfolio_equity": portfolio_equity,
            "portfolio_metrics": portfolio_metrics,
        }


# ─────────────────────────────────────────────
#  7. REPORT GENERATOR
# ─────────────────────────────────────────────


def print_report(
    portfolio_result: dict, pairs_df: pd.DataFrame, cost_model: TransactionCostModel
) -> None:
    SEP = "═" * 72

    print(f"\n{SEP}")
    print("  STATISTICAL ARBITRAGE BACKTESTER  —  NSE EQUITIES")
    print(f"{SEP}")

    print("\n📋  COINTEGRATED PAIRS IDENTIFIED")
    print(f"{'Pair':<18} {'Beta':>7} {'Coint p':>9} {'ADF p':>8} {'Half-life':>10}")
    print("─" * 56)
    for _, r in pairs_df.iterrows():
        pair = f"{r.ticker1}/{r.ticker2}"
        print(
            f"{pair:<18} {r.beta:>7.4f} {r.coint_pval:>9.4f} "
            f"{r.adf_pval:>8.4f} {r.half_life:>9.1f}d"
        )

    print(f"\n\n💸  TRANSACTION COST BREAKDOWN  (sample: ₹1000/share, 100 shares)")
    rt = cost_model.round_trip_cost_bps()
    sample_buy = cost_model.compute_costs(1000, 100, "buy")
    sample_sell = cost_model.compute_costs(1000, 100, "sell")
    items = [
        "brokerage",
        "stt",
        "exchange",
        "sebi",
        "gst",
        "stamp_duty",
        "half_spread",
        "market_impact",
    ]
    print(f"{'Item':<18} {'Buy (₹)':>10} {'Sell (₹)':>10}")
    print("─" * 42)
    for it in items:
        print(f"  {it:<16} {sample_buy[it]:>10.2f} {sample_sell[it]:>10.2f}")
    print(
        f"  {'TOTAL':.<16} {sample_buy['total']:>10.2f} {sample_sell['total']:>10.2f}"
    )
    print(f"\n  ➤ Round-trip cost: {rt:.2f} bps")

    print(f"\n\n📊  PER-PAIR PERFORMANCE SUMMARY")
    print(
        f"{'Pair':<18} {'Sharpe':>7} {'Ann Ret%':>9} {'MaxDD%':>8} "
        f"{'Trades':>7} {'WinRate%':>9} {'NetPnL ₹':>12}"
    )
    print("─" * 74)
    for key, res in portfolio_result["pair_results"].items():
        m = res["metrics"]
        p = key.replace("_", "/")
        print(
            f"{p:<18} {m['sharpe_ratio']:>7.3f} {m['annual_return_pct']:>9.2f} "
            f"{m['max_drawdown_pct']:>8.2f} {m['n_trades']:>7} "
            f"{m['win_rate_pct']:>9.2f} {m['total_net_pnl_inr']:>12,.0f}"
        )

    pm = portfolio_result["portfolio_metrics"]
    print(f"\n\n{'═' * 72}")
    print("  PORTFOLIO  (equal-weighted, all pairs combined)")
    print(f"{'═' * 72}")
    if pm:
        print(f"  Sharpe Ratio          : {pm['sharpe_ratio']:>10.3f}")
        print(f"  Sortino Ratio         : {pm['sortino_ratio']:>10.3f}")
        print(f"  Calmar Ratio          : {pm['calmar_ratio']:>10.3f}")
        print(f"  Annual Return         : {pm['annual_return_pct']:>9.2f}%")
        print(f"  Annual Volatility     : {pm['annual_vol_pct']:>9.2f}%")
        print(f"  Max Drawdown          : {pm['max_drawdown_pct']:>9.2f}%")
        print(f"  Total Trades          : {pm['n_trades']:>10}")
        print(f"  Win Rate              : {pm['win_rate_pct']:>9.2f}%")
        print(f"  Profit Factor         : {pm['profit_factor']:>10.3f}")
        print(f"  Avg Hold (days)       : {pm['avg_hold_days']:>10.1f}")
        print(f"  Total Net P&L         : ₹{pm['total_net_pnl_inr']:>12,.0f}")
        print(f"  Total Transaction Costs: ₹{pm['total_costs_inr']:>11,.0f}")
        print(f"  Cost Drag             : {pm['cost_drag_pct']:>9.3f}%")
    print(f"{'═' * 72}\n")


# ─────────────────────────────────────────────
#  8. MAIN
# ─────────────────────────────────────────────


def main():
    print("\n⚙  Generating simulated NSE price data …")

    TICKERS = [
        "RELIANCE",
        "HDFCBANK",  # cointegrated pair 1
        "INFY",
        "TCS",  # cointegrated pair 2
        "AXISBANK",
        "ICICIBANK",  # cointegrated pair 3
        "WIPRO",
        "HCLTECH",  # cointegrated pair 4  (may/may not pass)
        "SBIN",
        "PNB",  # cointegrated pair 5  (may/may not pass)
    ]
    N_DAYS = 504  # ~2 trading years

    prices = generate_nse_prices(TICKERS, n_days=N_DAYS, seed=42)
    print(f"   Price data: {prices.shape[0]} days × {prices.shape[1]} tickers")
    print(f"   Period: {prices.index[0].date()} → {prices.index[-1].date()}")

    print("\n🔍  Running Engle-Granger cointegration scan …")
    scanner = CointegrationScanner(
        pvalue_threshold=0.10, min_half_life=3, max_half_life=120
    )
    pairs_df = scanner.scan(prices)
    print(f"   Found {len(pairs_df)} cointegrated pair(s) at p < 0.10")

    if pairs_df.empty:
        print("   ⚠ No cointegrated pairs found. Try relaxing thresholds.")
        return

    print("\n📈  Running portfolio backtest …")
    cost_model = TransactionCostModel(avg_spread_bps=5.0, adv_fraction=0.001)
    portfolio = PortfolioBacktester(
        capital_per_pair=1_000_000,  # ₹10 lakh per pair
        lookback=60,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,
        avg_spread_bps=5.0,
    )
    result = portfolio.run(prices, pairs_df)

    print_report(result, pairs_df, cost_model)

    # ── quick equity curve to terminal ──────────────────────────────
    eq = result["portfolio_equity"]
    if not eq.empty:
        n_pts = 60
        step = max(1, len(eq) // n_pts)
        sampled = eq.iloc[::step]
        lo, hi = sampled.min(), sampled.max()
        H = 8
        print("  Portfolio Equity Curve (ASCII)\n")
        for row_i in range(H, -1, -1):
            thresh = lo + (hi - lo) * row_i / H
            line = ""
            prev_above = None
            for v in sampled:
                above = v >= thresh
                if above:
                    line += "█" if prev_above else "▐"
                else:
                    line += " "
                prev_above = above
            label = f"₹{thresh / 1e5:.1f}L" if row_i % 2 == 0 else "      "
            print(f"  {label:>8} │{line}")
        print(f"  {'':>8} └{'─' * len(sampled)}")
        print(f"  {'':>8}  {str(eq.index[0].date()):<20}{str(eq.index[-1].date())}")

    print("\n✅  Backtest complete.\n")


if __name__ == "__main__":
    main()
