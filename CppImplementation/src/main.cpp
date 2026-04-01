#include "../include/ArbitrageBacktester.hpp"
#include <iostream>

int main() {
    std::cout << "\nGenerating simulated NSE price data ...\n";
    
    std::vector<std::string> tickers = {
        "RELIANCE", "HDFCBANK",
        "INFY", "TCS",
        "AXISBANK", "ICICIBANK",
        "WIPRO", "HCLTECH",
        "SBIN", "PNB"
    };
    
    size_t n_days = 504;
    
    arb::PriceData prices = arb::generate_nse_prices(tickers, n_days, 42);
    std::cout << "   Price data: " << prices.size() << " days x " << prices.num_tickers() << " tickers\n";
    std::cout << "   Period: " << prices.dates.front() << " -> " << prices.dates.back() << "\n";
    
    std::cout << "\nRunning Engle-Granger cointegration scan ...\n";
    auto pairs = arb::scan_cointegrated_pairs(prices, 0.10, 3.0, 120.0);
    std::cout << "   Found " << pairs.size() << " cointegrated pair(s) at p < 0.10\n";
    
    if (pairs.empty()) {
        std::cout << "   No cointegrated pairs found. Try relaxing thresholds.\n";
        return 0;
    }
    
    std::cout << "\nRunning portfolio backtest ...\n";
    auto result = arb::run_portfolio_backtest(
        prices, pairs,
        1000000.0,
        60, 2.0, 0.5, 3.5,
        5.0
    );
    
    arb::print_report(result, pairs);
    
    return 0;
}
