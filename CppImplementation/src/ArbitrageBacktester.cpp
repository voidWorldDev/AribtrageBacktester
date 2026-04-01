#include "../include/ArbitrageBacktester.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <iostream>
#include <cstring>

namespace arb {

double PriceData::at(const std::string& ticker, size_t idx) const {
    auto it = prices.find(ticker);
    if (it != prices.end() && idx < it->second.size()) {
        return it->second[idx];
    }
    return 0.0;
}

double PriceData::at(const std::string& ticker, const std::string& date) const {
    auto it = prices.find(ticker);
    if (it != prices.end()) {
        for (size_t i = 0; i < dates.size(); ++i) {
            if (dates[i] == date) return it->second[i];
        }
    }
    return 0.0;
}

namespace {

std::vector<std::string> generate_business_dates(const std::string& start, size_t n_days) {
    std::vector<std::string> dates;
    
    int year = 2022, month = 1, day = 3;
    
    for (size_t i = 0; i < n_days; ++i) {
        int current_day = day + static_cast<int>(i);
        int current_month = month;
        int current_year = year;
        
        int days_in_month[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        
        while (true) {
            int dim = days_in_month[current_month];
            if (current_month == 2 && (current_year % 4 == 0) && 
                (current_year % 100 != 0 || current_year % 400 == 0)) {
                dim = 29;
            }
            if (current_day <= dim) break;
            current_day -= dim;
            current_month++;
            if (current_month > 12) {
                current_month = 1;
                current_year++;
            }
        }
        
        char buf[20];
        snprintf(buf, sizeof(buf), "%04d-%02d-%02d", current_year, current_month, current_day);
        dates.push_back(std::string(buf));
    }
    return dates;
}

double vector_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double vector_std(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = vector_mean(v);
    double sum_sq = 0.0;
    for (double x : v) {
        sum_sq += (x - m) * (x - m);
    }
    return std::sqrt(sum_sq / (v.size() - 1));
}

std::pair<double, double> linear_regression(const std::vector<double>& y, const std::vector<double>& x) {
    size_t n = y.size();
    if (n != x.size() || n < 2) return {0.0, 0.0};
    
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    
    double denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-10) return {0.0, vector_mean(y)};
    
    double beta = (n * sum_xy - sum_x * sum_y) / denom;
    double alpha = (sum_y - beta * sum_x) / n;
    
    return {alpha, beta};
}

double adf_test_impl(const std::vector<double>& series, int max_lag = 0) {
    size_t n = series.size();
    if (n <= static_cast<size_t>(max_lag) + 1) return 1.0;
    
    std::vector<double> y(n - 1);
    std::vector<double> x(n - 1, 1.0);
    
    for (size_t i = 1; i < n; ++i) {
        y[i - 1] = series[i] - series[i - 1];
        x[i - 1] = series[i - 1];
    }
    
    auto [alpha, lam] = linear_regression(y, x);
    
    std::vector<double> residuals(n - 1);
    double ssr = 0.0;
    double y_mean = vector_mean(y);
    
    for (size_t i = 0; i < n - 1; ++i) {
        residuals[i] = y[i] - alpha - lam * x[i];
        ssr += residuals[i] * residuals[i];
    }
    
    double se = std::sqrt(ssr / (n - 2));
    double t_stat = (se > 1e-10) ? lam / se : 0.0;
    
    double critical_5pct = -2.86;
    if (t_stat < critical_5pct) return 0.01;
    if (t_stat < -2.5) return 0.05;
    if (t_stat < -2.0) return 0.10;
    if (t_stat < -1.5) return 0.20;
    
    return 0.50;
}

double engle_granger_coint_test(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 10) return 1.0;
    
    auto [alpha, beta] = linear_regression(x, y);
    
    std::vector<double> residuals(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        residuals[i] = x[i] - alpha - beta * y[i];
    }
    
    double adf_p = adf_test_impl(residuals, 1);
    
    double critical_5pct = -3.95;
    double t_stat = beta / vector_std(residuals) * std::sqrt(static_cast<double>(x.size()));
    
    if (t_stat < critical_5pct) return 0.01;
    if (t_stat < -3.5) return 0.05;
    if (t_stat < -3.0) return 0.10;
    if (t_stat < -2.5) return 0.20;
    
    return 0.50;
}

double compute_half_life(const std::vector<double>& spread) {
    size_t n = spread.size();
    if (n < 3) return 60.0;
    
    std::vector<double> lag(n - 1), diff(n - 1);
    for (size_t i = 1; i < n; ++i) {
        lag[i - 1] = spread[i - 1];
        diff[i - 1] = spread[i] - spread[i - 1];
    }
    
    auto [alpha, lam] = linear_regression(diff, lag);
    
    if (lam >= 0) return 1000.0;
    
    return -std::log(2.0) / lam;
}

std::vector<std::vector<double>> generate_correlated_returns(
    size_t n_days, size_t n_tickers, int seed,
    const std::vector<std::pair<size_t, size_t>>& paired_indices
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    size_t n = n_tickers;
    std::vector<std::vector<double>> corr_matrix(n, std::vector<double>(n, 0.35));
    for (size_t i = 0; i < n; ++i) {
        corr_matrix[i][i] = 1.0;
    }
    for (const auto& p : paired_indices) {
        if (p.first < n && p.second < n) {
            corr_matrix[p.first][p.second] = 0.92;
            corr_matrix[p.second][p.first] = 0.92;
        }
    }
    
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                L[i][j] = std::sqrt(corr_matrix[i][i] - sum);
            } else {
                L[i][j] = (corr_matrix[i][j] - sum) / L[j][j];
            }
        }
    }
    
    std::vector<std::vector<double>> returns(n_days, std::vector<double>(n));
    for (size_t t = 0; t < n_days; ++t) {
        std::vector<double> z(n);
        for (size_t i = 0; i < n; ++i) {
            z[i] = dist(rng);
        }
        for (size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j <= i; ++j) {
                sum += L[i][j] * z[j];
            }
            returns[t][i] = sum;
        }
    }
    
    return returns;
}

struct CostResult {
    double brokerage;
    double stt;
    double exchange_fee;
    double sebi;
    double gst;
    double stamp_duty;
    double half_spread;
    double market_impact;
    double total;
    double total_bps;
};

CostResult compute_costs(double price, int qty, const std::string& side,
                        double avg_spread_bps = 5.0, double adv_fraction = 0.001) {
    double value = price * qty;
    
    double brokerage = value * 0.0003;
    double stt = (side == "sell") ? value * 0.001 : 0.0;
    double exchange_fee = value * 0.0000345;
    double sebi = value * 0.000001;
    double gst = (brokerage + exchange_fee) * 0.18;
    double stamp = (side == "buy") ? value * 0.00015 : 0.0;
    double half_spread = value * (avg_spread_bps / 2.0) / 10000.0;
    double impact = value * 0.1 * adv_fraction;
    
    double total = brokerage + stt + exchange_fee + sebi + gst + stamp + half_spread + impact;
    
    CostResult r;
    r.brokerage = brokerage;
    r.stt = stt;
    r.exchange_fee = exchange_fee;
    r.sebi = sebi;
    r.gst = gst;
    r.stamp_duty = stamp;
    r.half_spread = half_spread;
    r.market_impact = impact;
    r.total = total;
    r.total_bps = total / value * 10000.0;
    return r;
}

Metrics compute_metrics_impl(const std::vector<double>& pnl, const std::vector<double>& equity,
                              const std::vector<Trade>& trades, double initial_capital) {
    Metrics m;
    
    std::vector<double> returns(pnl.size());
    for (size_t i = 0; i < pnl.size(); ++i) {
        returns[i] = pnl[i] / initial_capital;
    }
    
    double ann_return = vector_mean(returns) * 252.0;
    double ann_vol = vector_std(returns) * std::sqrt(252.0);
    m.sharpe_ratio = (ann_vol > 1e-10) ? ann_return / ann_vol : 0.0;
    
    double roll_max = equity[0];
    double max_dd = 0.0;
    for (double e : equity) {
        if (e > roll_max) roll_max = e;
        double dd = (roll_max - e) / roll_max;
        if (dd > max_dd) max_dd = dd;
    }
    m.max_drawdown_pct = -max_dd * 100.0;
    
    m.calmar_ratio = (max_dd > 1e-10) ? ann_return / max_dd : 0.0;
    
    std::vector<double> neg_returns;
    for (double r : returns) {
        if (r < 0) neg_returns.push_back(r);
    }
    double sortino_den = (neg_returns.size() > 1) ? vector_std(neg_returns) * std::sqrt(252.0) : 1.0;
    m.sortino_ratio = (sortino_den > 1e-10) ? ann_return / sortino_den : 0.0;
    
    m.n_trades = static_cast<int>(trades.size());
    m.annual_return_pct = ann_return * 100.0;
    m.annual_vol_pct = ann_vol * 100.0;
    
    if (!trades.empty()) {
        int wins = 0;
        double total_win = 0.0, total_loss = 0.0;
        double total_net = 0.0, total_cost = 0.0;
        int hold_sum = 0;
        
        for (const auto& t : trades) {
            if (t.net_pnl > 0) {
                wins++;
                total_win += t.net_pnl;
            } else {
                total_loss += t.net_pnl;
            }
            total_net += t.net_pnl;
            total_cost += t.total_costs;
            hold_sum += t.hold_days;
        }
        
        m.win_rate_pct = (100.0 * wins) / trades.size();
        m.avg_win_inr = wins > 0 ? total_win / wins : 0.0;
        m.avg_loss_inr = (trades.size() - wins) > 0 ? total_loss / (trades.size() - wins) : 0.0;
        
        double win_rate = static_cast<double>(wins) / trades.size();
        double avg_loss_val = -m.avg_loss_inr;
        
        if (avg_loss_val > 1e-10 && win_rate < 0.9999) {
            m.profit_factor = (m.avg_win_inr * win_rate) / (avg_loss_val * (1 - win_rate));
        } else {
            m.profit_factor = 0.0;
        }
        
        m.avg_hold_days = static_cast<double>(hold_sum) / trades.size();
        m.total_net_pnl_inr = total_net;
        m.total_costs_inr = total_cost;
    } else {
        m.win_rate_pct = 0.0;
        m.avg_win_inr = 0.0;
        m.avg_loss_inr = 0.0;
        m.profit_factor = 0.0;
        m.avg_hold_days = 0.0;
        m.total_net_pnl_inr = 0.0;
        m.total_costs_inr = 0.0;
    }
    
    m.cost_drag_pct = m.total_costs_inr / initial_capital * 100.0;
    
    return m;
}

}

PriceData generate_nse_prices(const std::vector<std::string>& tickers, size_t n_days, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> price_dist(200.0, 3000.0);
    std::uniform_real_distribution<double> beta_dist(0.8, 1.2);
    
    size_t n = tickers.size();
    std::vector<std::pair<size_t, size_t>> paired_indices;
    for (size_t i = 0; i + 1 < n && i < 6; i += 2) {
        paired_indices.push_back({i, i + 1});
    }
    
    auto raw_returns = generate_correlated_returns(n_days, n, seed, paired_indices);
    
    double annual_vol = 0.25;
    double daily_vol = annual_vol / std::sqrt(252.0);
    double daily_drift = 0.08 / 252.0;
    
    std::vector<double> init_prices(n);
    for (size_t i = 0; i < n; ++i) {
        init_prices[i] = price_dist(rng);
    }
    
    std::vector<std::vector<double>> log_prices(n_days, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        log_prices[0][i] = std::log(init_prices[i]);
    }
    
    for (size_t t = 1; t < n_days; ++t) {
        for (size_t i = 0; i < n; ++i) {
            log_prices[t][i] = log_prices[t - 1][i] + daily_drift + daily_vol * raw_returns[t][i];
        }
    }
    
    std::normal_distribution<double> noise_dist(0.0, 1.0);
    for (size_t i = 0; i + 1 < n && i < 6; i += 2) {
        double beta = beta_dist(rng);
        double spread_vol = 0.015;
        double theta = 0.08;
        double spread = 0.0;
        
        for (size_t t = 0; t < n_days; ++t) {
            spread = spread - theta * spread + spread_vol * noise_dist(rng);
            log_prices[t][i + 1] = beta * log_prices[t][i] + spread;
        }
    }
    
    PriceData data;
    data.tickers = tickers;
    data.dates = generate_business_dates("2022-01-03", n_days);
    
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> prices(n_days);
        for (size_t t = 0; t < n_days; ++t) {
            prices[t] = std::exp(log_prices[t][i]);
        }
        data.prices[tickers[i]] = prices;
    }
    
    return data;
}

std::vector<PairResult> scan_cointegrated_pairs(const PriceData& prices, double pvalue_threshold,
                                                 double min_half_life, double max_half_life) {
    std::vector<PairResult> results;
    
    for (size_t i = 0; i < prices.tickers.size(); ++i) {
        for (size_t j = i + 1; j < prices.tickers.size(); ++j) {
            const std::string& t1 = prices.tickers[i];
            const std::string& t2 = prices.tickers[j];
            
            std::vector<double> s1(prices.size()), s2(prices.size());
            for (size_t k = 0; k < prices.size(); ++k) {
                s1[k] = std::log(prices.at(t1, k));
                s2[k] = std::log(prices.at(t2, k));
            }
            
            auto [alpha, beta] = linear_regression(s1, s2);
            
            std::vector<double> spread(prices.size());
            for (size_t k = 0; k < prices.size(); ++k) {
                spread[k] = s1[k] - beta * s2[k];
            }
            
            double coint_pval = engle_granger_coint_test(s1, s2);
            double adf_pval = adf_test_impl(spread, 1);
            double half_life = compute_half_life(spread);
            
            if (coint_pval < pvalue_threshold && 
                half_life >= min_half_life && half_life <= max_half_life) {
                PairResult pr;
                pr.ticker1 = t1;
                pr.ticker2 = t2;
                pr.beta = beta;
                pr.coint_pval = coint_pval;
                pr.adf_stat = 0.0;
                pr.adf_pval = adf_pval;
                pr.half_life = half_life;
                pr.spread_std = vector_std(spread);
                results.push_back(pr);
            }
        }
    }
    
    std::sort(results.begin(), results.end(), 
        [](const PairResult& a, const PairResult& b) {
            return a.coint_pval < b.coint_pval;
        });
    
    return results;
}

PairBacktestResult run_pair_backtest(const PriceData& prices, const std::string& t1,
                                      const std::string& t2, double beta,
                                      double capital_per_leg, int lookback,
                                      double entry_z, double exit_z, double stop_z,
                                      double avg_spread_bps) {
    size_t n = prices.size();
    
    std::vector<double> log_s1(n), log_s2(n);
    for (size_t i = 0; i < n; ++i) {
        log_s1[i] = std::log(prices.at(t1, i));
        log_s2[i] = std::log(prices.at(t2, i));
    }
    
    std::vector<double> mu(n, 0.0), sigma(n, 0.0), zscore(n, 0.0);
    for (size_t i = lookback; i < n; ++i) {
        double sum = 0.0, sum_sq = 0.0;
        for (size_t j = i - lookback; j < i; ++j) {
            sum += log_s1[j] - beta * log_s2[j];
            sum_sq += (log_s1[j] - beta * log_s2[j]) * (log_s1[j] - beta * log_s2[j]);
        }
        mu[i] = sum / lookback;
        double var = sum_sq / lookback - mu[i] * mu[i];
        sigma[i] = (var > 0) ? std::sqrt(var) : 0.0;
        zscore[i] = (sigma[i] > 1e-10) ? ((log_s1[i] - beta * log_s2[i]) - mu[i]) / sigma[i] : 0.0;
    }
    
    std::vector<int> positions(n, 0);
    int position = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(zscore[i]) || std::isinf(zscore[i])) {
            positions[i] = 0;
            continue;
        }
        
        if (position == 0) {
            if (zscore[i] <= -entry_z) {
                position = 1;
            } else if (zscore[i] >= entry_z) {
                position = -1;
            }
        } else if (position == 1) {
            if (zscore[i] >= -exit_z || zscore[i] <= -stop_z) {
                position = 0;
            }
        } else if (position == -1) {
            if (zscore[i] <= exit_z || zscore[i] >= stop_z) {
                position = 0;
            }
        }
        positions[i] = position;
    }
    
    std::vector<Trade> trades;
    std::vector<double> daily_pnl(n, 0.0);
    
    int current_pos = 0;
    double entry_p1 = 0.0, entry_p2 = 0.0;
    int qty1 = 0, qty2 = 0;
    int hold_days = 0;
    int max_holding_days = 30;
    size_t entry_idx = 0;
    
    for (size_t i = 0; i < n; ++i) {
        if (current_pos != 0) {
            hold_days++;
            if (hold_days >= max_holding_days) {
                positions[i] = 0;
            }
        }
        
        int sig_pos = positions[i];
        
        if (current_pos == 0 && sig_pos != 0) {
            entry_idx = i;
            double p1 = prices.at(t1, i);
            double p2 = prices.at(t2, i);
            
            qty1 = static_cast<int>(std::max(1.0, capital_per_leg / p1));
            qty2 = static_cast<int>(std::max(1.0, capital_per_leg / p2));
            
            double c_in = 0.0;
            if (sig_pos == 1) {
                c_in = compute_costs(p1, qty1, "buy", avg_spread_bps).total +
                       compute_costs(p2, qty2, "sell", avg_spread_bps).total;
            } else {
                c_in = compute_costs(p1, qty1, "sell", avg_spread_bps).total +
                       compute_costs(p2, qty2, "buy", avg_spread_bps).total;
            }
            
            current_pos = sig_pos;
            entry_p1 = p1;
            entry_p2 = p2;
            hold_days = 0;
            daily_pnl[i] = -c_in;
            
        } else if (current_pos != 0 && sig_pos == 0) {
            double p1 = prices.at(t1, i);
            double p2 = prices.at(t2, i);
            
            double gross_pnl = 0.0;
            double c_out = 0.0;
            
            if (current_pos == 1) {
                gross_pnl = qty1 * (p1 - entry_p1) - qty2 * (p2 - entry_p2) * beta;
                c_out = compute_costs(p1, qty1, "sell", avg_spread_bps).total +
                        compute_costs(p2, qty2, "buy", avg_spread_bps).total;
            } else {
                gross_pnl = qty1 * (entry_p1 - p1) + qty2 * (entry_p2 - p2) * beta;
                c_out = compute_costs(p1, qty1, "buy", avg_spread_bps).total +
                        compute_costs(p2, qty2, "sell", avg_spread_bps).total;
            }
            
            double net_pnl = gross_pnl - c_out;
            
            Trade t;
            t.entry_date = prices.dates[entry_idx];
            t.exit_date = prices.dates[i];
            t.direction = (current_pos == 1) ? "long" : "short";
            t.hold_days = hold_days;
            t.gross_pnl = gross_pnl;
            t.total_costs = c_out;
            t.net_pnl = net_pnl;
            trades.push_back(t);
            
            daily_pnl[i] = net_pnl;
            current_pos = 0;
            hold_days = 0;
        }
    }
    
    double initial_capital = capital_per_leg * 2.0;
    std::vector<double> equity(n, initial_capital);
    double cumsum = initial_capital;
    for (size_t i = 0; i < n; ++i) {
        cumsum += daily_pnl[i];
        equity[i] = cumsum;
    }
    
    Metrics metrics = compute_metrics_impl(daily_pnl, equity, trades, initial_capital);
    
    PairBacktestResult result;
    result.pair_name = t1 + "_" + t2;
    result.beta = beta;
    result.equity = equity;
    result.equity_dates = prices.dates;
    result.trades = trades;
    result.metrics = metrics;
    
    return result;
}

PortfolioResult run_portfolio_backtest(const PriceData& prices, const std::vector<PairResult>& pairs,
                                       double capital_per_pair, int lookback,
                                       double entry_z, double exit_z, double stop_z,
                                       double avg_spread_bps) {
    PortfolioResult result;
    
    for (const auto& pair : pairs) {
        PairBacktestResult pair_res = run_pair_backtest(
            prices, pair.ticker1, pair.ticker2, pair.beta,
            capital_per_pair / 2.0, lookback, entry_z, exit_z, stop_z, avg_spread_bps
        );
        result.pair_results.push_back({pair.ticker1 + "/" + pair.ticker2, pair_res});
    }
    
    if (!result.pair_results.empty()) {
        size_t n = prices.size();
        size_t num_pairs = result.pair_results.size();
        
        std::vector<double> portfolio_pnl(n, 0.0);
        for (const auto& pr : result.pair_results) {
            for (size_t i = 0; i < n; ++i) {
                if (i < pr.second.equity.size()) {
                    double pnl_i;
                    if (i == 0) {
                        pnl_i = pr.second.equity[i] - capital_per_pair;
                    } else {
                        pnl_i = pr.second.equity[i] - pr.second.equity[i - 1];
                    }
                    portfolio_pnl[i] += pnl_i;
                }
            }
        }
        
        double total_capital = capital_per_pair * num_pairs;
        std::vector<double> portfolio_equity(n, total_capital);
        double cumsum = total_capital;
        for (size_t i = 0; i < n; ++i) {
            cumsum += portfolio_pnl[i];
            portfolio_equity[i] = cumsum;
        }
        
        std::vector<Trade> all_trades;
        for (const auto& pr : result.pair_results) {
            all_trades.insert(all_trades.end(), 
                             pr.second.trades.begin(), 
                             pr.second.trades.end());
        }
        
        result.portfolio_equity = portfolio_equity;
        result.equity_dates = prices.dates;
        result.portfolio_metrics = compute_metrics_impl(portfolio_pnl, portfolio_equity, 
                                                      all_trades, total_capital);
    }
    
    return result;
}

std::string format_number(double value, int width, int decimals) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals);
    if (std::abs(value) >= 1000000.0) {
        oss << value / 1000000.0 << "M";
    } else if (std::abs(value) >= 1000.0) {
        oss << value / 1000.0 << "K";
    } else {
        oss << value;
    }
    return oss.str();
}

void print_report(const PortfolioResult& result, const std::vector<PairResult>& pairs) {
    std::string sep(72, '=');
    std::string dash(72, '-');
    
    std::cout << "\n" << sep << "\n";
    std::cout << "  STATISTICAL ARBITRAGE BACKTESTER  --  NSE EQUITIES\n";
    std::cout << sep << "\n";
    
    std::cout << "\nCOINTEGRATED PAIRS IDENTIFIED\n";
    std::cout << std::left << std::setw(20) << "Pair" 
              << std::right << std::setw(8) << "Beta" 
              << std::setw(10) << "Coint p"
              << std::setw(10) << "ADF p"
              << std::setw(12) << "Half-life" << "\n";
    std::cout << dash << "\n";
    
    for (const auto& p : pairs) {
        std::cout << std::left << std::setw(20) << (p.ticker1 + "/" + p.ticker2)
                  << std::right << std::setw(8) << std::fixed << std::setprecision(4) << p.beta
                  << std::setw(10) << p.coint_pval
                  << std::setw(10) << p.adf_pval
                  << std::setw(11) << std::fixed << std::setprecision(1) << p.half_life << "d\n";
    }
    
    std::cout << "\n\nTRANSACTION COST BREAKDOWN  (sample: 1000/share, 100 shares)\n";
    CostResult sample_buy = compute_costs(1000.0, 100, "buy");
    CostResult sample_sell = compute_costs(1000.0, 100, "sell");
    
    std::cout << std::left << std::setw(18) << "Item" 
              << std::right << std::setw(10) << "Buy" 
              << std::setw(10) << "Sell" << "\n";
    std::cout << dash << "\n";
    
    std::cout << std::left << std::setw(18) << "  brokerage" 
              << std::right << std::setw(10) << std::fixed << std::setprecision(2) << sample_buy.brokerage
              << std::setw(10) << sample_sell.brokerage << "\n";
    std::cout << std::left << std::setw(18) << "  stt" 
              << std::right << std::setw(10) << sample_buy.stt
              << std::setw(10) << sample_sell.stt << "\n";
    std::cout << std::left << std::setw(18) << "  exchange" 
              << std::right << std::setw(10) << sample_buy.exchange_fee
              << std::setw(10) << sample_sell.exchange_fee << "\n";
    std::cout << std::left << std::setw(18) << "  sebi" 
              << std::right << std::setw(10) << sample_buy.sebi
              << std::setw(10) << sample_sell.sebi << "\n";
    std::cout << std::left << std::setw(18) << "  gst" 
              << std::right << std::setw(10) << sample_buy.gst
              << std::setw(10) << sample_sell.gst << "\n";
    std::cout << std::left << std::setw(18) << "  stamp_duty" 
              << std::right << std::setw(10) << sample_buy.stamp_duty
              << std::setw(10) << sample_sell.stamp_duty << "\n";
    std::cout << std::left << std::setw(18) << "  half_spread" 
              << std::right << std::setw(10) << sample_buy.half_spread
              << std::setw(10) << sample_sell.half_spread << "\n";
    std::cout << std::left << std::setw(18) << "  market_impact" 
              << std::right << std::setw(10) << sample_buy.market_impact
              << std::setw(10) << sample_sell.market_impact << "\n";
    std::cout << std::left << std::setw(18) << "  TOTAL" 
              << std::right << std::setw(10) << sample_buy.total
              << std::setw(10) << sample_sell.total << "\n";
    
    double rt_cost = (sample_buy.total + sample_sell.total) / (1000.0 * 100) * 10000.0;
    std::cout << "\n  Round-trip cost: " << std::fixed << std::setprecision(2) << rt_cost << " bps\n";
    
    std::cout << "\n\nPER-PAIR PERFORMANCE SUMMARY\n";
    std::cout << std::left << std::setw(20) << "Pair"
              << std::right << std::setw(8) << "Sharpe"
              << std::setw(10) << "AnnRet%"
              << std::setw(9) << "MaxDD%"
              << std::setw(8) << "Trades"
              << std::setw(10) << "WinRate%"
              << std::setw(14) << "NetPnL" << "\n";
    std::cout << dash << "\n";
    
    for (const auto& pr : result.pair_results) {
        const auto& m = pr.second.metrics;
        std::cout << std::left << std::setw(20) << pr.first
                  << std::right << std::setw(8) << std::fixed << std::setprecision(3) << m.sharpe_ratio
                  << std::setw(10) << std::fixed << std::setprecision(2) << m.annual_return_pct
                  << std::setw(9) << m.max_drawdown_pct
                  << std::setw(8) << m.n_trades
                  << std::setw(10) << std::fixed << std::setprecision(2) << m.win_rate_pct
                  << std::setw(14) << std::setprecision(0) << m.total_net_pnl_inr << "\n";
    }
    
    const auto& pm = result.portfolio_metrics;
    std::cout << "\n\n" << sep << "\n";
    std::cout << "  PORTFOLIO  (equal-weighted, all pairs combined)\n";
    std::cout << sep << "\n";
    
    std::cout << "  Sharpe Ratio          :" << std::setw(12) << std::fixed << std::setprecision(3) << pm.sharpe_ratio << "\n";
    std::cout << "  Sortino Ratio         :" << std::setw(12) << std::fixed << std::setprecision(3) << pm.sortino_ratio << "\n";
    std::cout << "  Calmar Ratio          :" << std::setw(12) << std::fixed << std::setprecision(3) << pm.calmar_ratio << "\n";
    std::cout << "  Annual Return         :" << std::setw(11) << std::fixed << std::setprecision(2) << pm.annual_return_pct << "%\n";
    std::cout << "  Annual Volatility     :" << std::setw(11) << std::fixed << std::setprecision(2) << pm.annual_vol_pct << "%\n";
    std::cout << "  Max Drawdown          :" << std::setw(11) << std::fixed << std::setprecision(2) << pm.max_drawdown_pct << "%\n";
    std::cout << "  Total Trades          :" << std::setw(12) << pm.n_trades << "\n";
    std::cout << "  Win Rate              :" << std::setw(11) << std::fixed << std::setprecision(2) << pm.win_rate_pct << "%\n";
    std::cout << "  Profit Factor         :" << std::setw(12) << std::fixed << std::setprecision(3) << pm.profit_factor << "\n";
    std::cout << "  Avg Hold (days)       :" << std::setw(12) << std::fixed << std::setprecision(1) << pm.avg_hold_days << "\n";
    std::cout << "  Total Net P&L         :" << std::setw(11) << "Rs " << std::setprecision(0) << pm.total_net_pnl_inr << "\n";
    std::cout << "  Total Transaction Costs:" << std::setw(10) << "Rs " << std::setprecision(0) << pm.total_costs_inr << "\n";
    std::cout << "  Cost Drag             :" << std::setw(11) << std::fixed << std::setprecision(3) << pm.cost_drag_pct << "%\n";
    std::cout << sep << "\n";
    
    if (!result.portfolio_equity.empty()) {
        std::cout << "\n  Portfolio Equity Curve (ASCII)\n\n";
        
        size_t n_pts = 60;
        size_t step = std::max((size_t)1, result.portfolio_equity.size() / n_pts);
        std::vector<double> sampled;
        for (size_t i = 0; i < result.portfolio_equity.size(); i += step) {
            sampled.push_back(result.portfolio_equity[i]);
        }
        
        double lo = sampled[0], hi = sampled[0];
        for (double v : sampled) {
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        int H = 8;
        
        for (int row_i = H; row_i >= 0; --row_i) {
            double thresh = lo + (hi - lo) * row_i / H;
            std::string line;
            bool prev_above = false;
            
            for (double v : sampled) {
                bool above = v >= thresh;
                if (above) {
                    line += prev_above ? "#" : "|";
                } else {
                    line += " ";
                }
                prev_above = above;
            }
            
            char label[20];
            if (row_i % 2 == 0) {
                snprintf(label, sizeof(label), "Rs%.1fL", thresh / 100000.0);
            } else {
                strcpy(label, "       ");
            }
            std::cout << "  " << std::setw(8) << label << " |" << line << "\n";
        }
        
        std::cout << "          +" << std::string(sampled.size(), '-') << "\n";
        std::cout << "          " << result.equity_dates.front() 
                  << std::string(sampled.size() > 40 ? sampled.size() - 40 : 10, ' ')
                  << result.equity_dates.back() << "\n";
    }
    
    std::cout << "\nBacktest complete.\n\n";
}

}
