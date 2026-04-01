#ifndef ARBITRAGE_BACKTESTER_HPP
#define ARBITRAGE_BACKTESTER_HPP

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <functional>

namespace arb {

struct PriceData {
    std::string date;
    std::vector<std::string> tickers;
    std::vector<std::string> dates;
    std::map<std::string, std::vector<double>> prices;
    
    double at(const std::string& ticker, size_t idx) const;
    double at(const std::string& ticker, const std::string& date) const;
    size_t size() const { return dates.size(); }
    size_t num_tickers() const { return tickers.size(); }
};

struct PairResult {
    std::string ticker1;
    std::string ticker2;
    double beta;
    double coint_pval;
    double adf_stat;
    double adf_pval;
    double half_life;
    double spread_std;
};

struct Trade {
    std::string entry_date;
    std::string exit_date;
    std::string direction;
    int hold_days;
    double gross_pnl;
    double total_costs;
    double net_pnl;
};

struct Metrics {
    double sharpe_ratio;
    double sortino_ratio;
    double calmar_ratio;
    double annual_return_pct;
    double annual_vol_pct;
    double max_drawdown_pct;
    int n_trades;
    double win_rate_pct;
    double avg_win_inr;
    double avg_loss_inr;
    double profit_factor;
    double avg_hold_days;
    double total_net_pnl_inr;
    double total_costs_inr;
    double cost_drag_pct;
};

struct PairBacktestResult {
    std::string pair_name;
    double beta;
    std::vector<double> equity;
    std::vector<std::string> equity_dates;
    std::vector<Trade> trades;
    Metrics metrics;
};

struct PortfolioResult {
    std::vector<std::pair<std::string, PairBacktestResult>> pair_results;
    std::vector<double> portfolio_equity;
    std::vector<std::string> equity_dates;
    Metrics portfolio_metrics;
};

PriceData generate_nse_prices(
    const std::vector<std::string>& tickers,
    size_t n_days = 504,
    int seed = 42
);

std::vector<PairResult> scan_cointegrated_pairs(
    const PriceData& prices,
    double pvalue_threshold = 0.05,
    double min_half_life = 5.0,
    double max_half_life = 126.0
);

PortfolioResult run_portfolio_backtest(
    const PriceData& prices,
    const std::vector<PairResult>& pairs,
    double capital_per_pair = 1000000.0,
    int lookback = 60,
    double entry_z = 2.0,
    double exit_z = 0.5,
    double stop_z = 3.5,
    double avg_spread_bps = 5.0
);

void print_report(
    const PortfolioResult& result,
    const std::vector<PairResult>& pairs
);

std::string format_number(double value, int width = 10, int decimals = 2);

}

#endif
