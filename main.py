#!/usr/bin/env python3
"""
ETF Trading System v2 — Main Entry Point

Strategy: Trend-Filtered Momentum Rotation
Pool: CSI 1000 constituents + all ETFs (filtered)
Data: Sina Finance + Baostock
Targets: Monthly win rate >= 50%, Annual excess >= 5% vs CSI 1000
"""
import sys
import os
import time
import numpy as np
import pandas as pd

from data_fetcher import (fetch_batch_stocks, fetch_batch_etfs, fetch_csi1000_index,
                          build_price_matrix, build_volume_matrix)
from universe import build_universe
from strategy import MomentumRotationStrategy
from backtest import run_backtest
from evaluation import compute_metrics, check_targets, print_report, plot_results
from optimizer import optimize
from report import generate_html_report

BACKTEST_START = "2021-01-01"
BACKTEST_END = "2026-05-01"
DATA_START = "2020-01-01"

# Best parameters from v2 optimization
BEST_PARAMS = {
    "top_n_bull": 25, "top_n_bear": 12,
    "mom_periods": (21, 63, 126, 252),
    "mom_weights": (0.4, 0.3, 0.2, 0.1),
    "score_weights_bull": (0.6, 0.3, 0.1),
    "score_weights_bear": (0.35, 0.5, 0.15),
    "stop_loss": -0.12,
    "bear_cash_ratio": 0.4,
    "vol_target": 0.15,
    "trend_ma": 220,
    "risk_off_threshold": -0.05,
}


def main():
    print("=" * 70)
    print("  ETF TRADING SYSTEM v2 — Trend-Filtered Momentum Rotation")
    print(f"  Universe: CSI 1000 Stocks + All ETFs (filtered)")
    print(f"  Data: Sina Finance + Baostock")
    print(f"  Period: {BACKTEST_START} -> {BACKTEST_END}")
    print("=" * 70)

    # Step 1: Build Universe
    universe = build_universe()
    if universe is None or not universe["all_symbols"]:
        print("[FATAL] Cannot build investment universe.")
        return 1

    stocks = universe["stocks"]
    etfs = universe["etfs"]
    stock_info = {**universe.get("stock_info", {}), **universe.get("etf_info", {})}

    # Step 2: Fetch Data
    print(f"\n=== Fetching Historical Data ===")
    print(f"  Stocks: {len(stocks)}, ETFs: {len(etfs)}")
    t0 = time.time()

    print("\n  [Phase 1] Fetching stock daily data via Baostock...")
    stock_data = fetch_batch_stocks(stocks, DATA_START, BACKTEST_END)

    print("\n  [Phase 2] Fetching ETF daily data via Sina...")
    etf_data = fetch_batch_etfs(etfs, DATA_START, BACKTEST_END)

    print("\n  [Phase 3] Fetching CSI 1000 benchmark...")
    bench_df = fetch_csi1000_index(DATA_START, BACKTEST_END)
    if bench_df.empty:
        print("[FATAL] Cannot fetch benchmark data.")
        return 1

    benchmark_prices = bench_df.set_index("date")["close"].sort_index()
    print(f"  Benchmark: {len(benchmark_prices)} days")

    all_data = {}
    all_data.update(stock_data)
    all_data.update(etf_data)

    if len(all_data) < 10:
        print(f"[FATAL] Insufficient data: {len(all_data)} symbols.")
        return 1

    price_matrix = build_price_matrix(all_data)
    volume_matrix = build_volume_matrix(all_data)
    print(f"\n  Price matrix: {price_matrix.shape}")
    print(f"  Data fetch completed in {time.time() - t0:.0f}s")

    # Step 3: Run
    mode = sys.argv[1] if len(sys.argv) > 1 else "best"

    if mode == "best":
        print(f"\n=== Running with Best Parameters ===")
        strategy = MomentumRotationStrategy(**BEST_PARAMS)
        result = run_backtest(price_matrix, volume_matrix, benchmark_prices,
                              strategy, stock_info=stock_info)
        if result is None:
            print("[FATAL] Backtest failed.")
            return 1
        metrics = compute_metrics(result)
        passed, details = check_targets(metrics)
        best_params = BEST_PARAMS
    elif mode == "optimize":
        print(f"\n=== Running Parameter Optimization ===")
        result, metrics, best_params = optimize(
            price_matrix, volume_matrix, benchmark_prices,
            max_iterations=50, stock_info=stock_info)
        if result is None:
            print("\n[WARN] Optimization failed, trying best params...")
            strategy = MomentumRotationStrategy(**BEST_PARAMS)
            result = run_backtest(price_matrix, volume_matrix, benchmark_prices,
                                  strategy, stock_info=stock_info)
            if result is None:
                print("[FATAL] Backtest also failed.")
                return 1
            metrics = compute_metrics(result)
            passed, details = check_targets(metrics)
            best_params = BEST_PARAMS
        else:
            passed, details = check_targets(metrics)
    else:  # single
        print(f"\n=== Running Single Backtest (Default Strategy) ===")
        strategy = MomentumRotationStrategy()
        result = run_backtest(price_matrix, volume_matrix, benchmark_prices,
                              strategy, stock_info=stock_info)
        if result is None:
            print("[FATAL] Backtest failed.")
            return 1
        metrics = compute_metrics(result)
        passed, details = check_targets(metrics)
        best_params = None

    # Step 4: Report
    print_report(metrics)

    m_ok = "PASS" if details["monthly_ok"] else "FAIL"
    a_ok = "PASS" if details["annual_ok"] else "FAIL"
    print(f"\n  Target Check:")
    print(f"    Monthly win rate >= 50%: {m_ok} ({metrics['monthly_win_rate']:.1%})")
    print(f"    Annual excess >= 5%:    {a_ok}")
    for yr, exc in details.get("annual_excess_details", {}).items():
        print(f"      {yr}: {exc}")

    if passed:
        print(f"\n  *** ALL TARGETS MET ***")
    else:
        print(f"\n  *** TARGETS NOT MET ***")

    # Step 5: Charts & Report
    plot_results(result, metrics)
    generate_html_report(result, metrics, best_params)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    result["daily_stats"].to_csv(os.path.join(out_dir, "daily_stats.csv"), index=False)
    result["monthly_stats"].to_csv(os.path.join(out_dir, "monthly_stats.csv"), index=False)
    result["yearly_stats"].to_csv(os.path.join(out_dir, "yearly_stats.csv"), index=False)
    pd.DataFrame([best_params]).to_csv(os.path.join(out_dir, "best_params.csv"), index=False)

    print(f"\n  All output saved to {out_dir}/")
    print(f"  Open {out_dir}/report.html for interactive dashboard")
    print(f"\nDone.")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
