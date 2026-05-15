"""
Backtesting engine v2.1 — configurable freq (monthly/weekly), look-ahead fixed.
"""
import numpy as np
import pandas as pd


def _build_rebalance_dates(common_dates, freq, start_date=None):
    if freq == "weekly":
        # Every Monday (or next available trading day)
        rebal_dates = []
        seen_weeks = set()
        for d in common_dates:
            if start_date and d < start_date:
                continue
            wk = (d.year, d.isocalendar().week)
            if wk not in seen_weeks:
                rebal_dates.append(d)
                seen_weeks.add(wk)
        # Skip first 12 weeks for warmup
        if len(rebal_dates) > 12:
            rebal_dates = rebal_dates[12:]
        return rebal_dates
    elif freq == "monthly":
        rebal_dates = []
        seen_months = set()
        for ms in pd.date_range(common_dates[0], common_dates[-1], freq="MS"):
            me = ms + pd.offsets.MonthEnd(1)
            mask = common_dates <= me
            if mask.any():
                rd = common_dates[mask][-1]
                if (rd.year, rd.month) not in seen_months:
                    rebal_dates.append(rd)
                    seen_months.add((rd.year, rd.month))
        if len(rebal_dates) > 12:
            rebal_dates = rebal_dates[12:]
        return rebal_dates
    else:
        raise ValueError(f"Unknown rebalance frequency: {freq}")


def run_backtest(price_matrix, volume_matrix, benchmark_prices, strategy,
                 start_date=None, end_date=None, cost_stock=0.001, cost_etf=0.0005,
                 min_history_days=252, stock_info=None):
    freq = getattr(strategy, "rebalance_freq", "monthly")
    common_dates = price_matrix.index
    if start_date:
        common_dates = common_dates[common_dates >= pd.Timestamp(start_date)]
    if end_date:
        common_dates = common_dates[common_dates <= pd.Timestamp(end_date)]
    if len(common_dates) < min_history_days:
        print(f"[ERROR] Not enough data: {len(common_dates)} days")
        return None

    rebal_dates = _build_rebalance_dates(common_dates, freq, start_date)
    if not rebal_dates:
        print("[ERROR] No rebalancing dates available")
        return None

    strategy.set_benchmark(benchmark_prices)
    if stock_info:
        strategy.set_stock_info(stock_info)

    cash = 1_000_000.0
    initial_cash = cash
    holdings = {}
    portfolio_values = []
    trades_log = []

    all_dates = common_dates[common_dates >= rebal_dates[0] - pd.Timedelta(days=30)]
    all_dates = all_dates[all_dates >= all_dates[0]]
    rebal_idx, risk_off, regime = 0, False, "bull"

    for i, date in enumerate(all_dates):
        # Stop-loss check
        stopped = []
        for sym, h in list(holdings.items()):
            if sym in price_matrix.columns and date in price_matrix.index:
                cp = price_matrix.loc[date, sym]
                if pd.notna(cp) and cp > 0:
                    if strategy.check_stop_loss(sym, h["entry_price"], cp):
                        cash += h["shares"] * cp * (1 - cost_stock)
                        stopped.append(sym)
                        trades_log.append({"date": date, "symbol": sym, "action": "stop_loss",
                                           "shares": h["shares"], "price": cp,
                                           "pnl_pct": (cp / h["entry_price"]) - 1})
        for sym in stopped:
            del holdings[sym]

        # Rebalance
        if rebal_idx < len(rebal_dates) and date == rebal_dates[rebal_idx]:
            rebal_idx += 1

            equity_value = _compute_equity(holdings, price_matrix, date)
            pv = cash + equity_value

            port_ret_3m = _trailing_return(portfolio_values, date, 63)
            bench_ret_3m = _trailing_return_bench(benchmark_prices, date, 63)
            risk_off = strategy.should_risk_off(port_ret_3m, bench_ret_3m)

            scores = strategy.compute_scores(price_matrix, volume_matrix, date)
            if not scores.empty:
                target_weights, regime = strategy.select_portfolio(scores, date)
                pos_mult = strategy.compute_position_size(risk_off, regime)
                target_weights = {k: v * pos_mult for k, v in target_weights.items()}

                port_rets = _portfolio_daily_returns(portfolio_values)
                target_weights = strategy.apply_vol_target(target_weights, port_rets)

                # Sell all old holdings
                for sym, h in list(holdings.items()):
                    if sym in price_matrix.columns and date in price_matrix.index:
                        sp = price_matrix.loc[date, sym]
                        if pd.notna(sp) and sp > 0:
                            cash += h["shares"] * sp * (1 - cost_stock)
                    del holdings[sym]

                # Buy new holdings
                equity_value = _compute_equity(holdings, price_matrix, date)
                pv = cash + equity_value

                for sym, weight in target_weights.items():
                    if weight <= 0 or sym not in price_matrix.columns:
                        continue
                    if date not in price_matrix.index:
                        continue
                    bp = price_matrix.loc[date, sym]
                    if pd.isna(bp) or bp <= 0:
                        continue
                    alloc = pv * weight
                    cost = cost_etf if sym.startswith(("51", "15", "56", "58")) else cost_stock
                    shares = int(alloc / bp / 100) * 100
                    if shares > 0:
                        cost_amt = shares * bp * cost
                        if shares * bp + cost_amt <= cash:
                            cash -= (shares * bp + cost_amt)
                            holdings[sym] = {"shares": shares, "entry_price": bp, "weight": weight}
                            trades_log.append({"date": date, "symbol": sym, "action": "buy",
                                               "shares": shares, "price": bp, "weight": weight})

        equity_value = _compute_equity(holdings, price_matrix, date)
        portfolio_value = cash + equity_value
        bench_price = benchmark_prices.get(date, np.nan)
        portfolio_values.append({"date": date, "portfolio_value": portfolio_value,
                                 "cash": cash, "equity": equity_value,
                                 "num_holdings": len(holdings),
                                 "benchmark_price": bench_price,
                                 "risk_off": risk_off, "regime": regime})

    daily_stats = pd.DataFrame(portfolio_values)
    daily_stats["portfolio_return"] = daily_stats["portfolio_value"].pct_change()
    daily_stats["benchmark_return"] = daily_stats["benchmark_price"].pct_change()
    daily_stats["portfolio_cum"] = daily_stats["portfolio_value"] / initial_cash
    daily_stats["benchmark_cum"] = (daily_stats["benchmark_price"] /
                                    daily_stats["benchmark_price"].iloc[0])

    daily_stats["year_month"] = daily_stats["date"].dt.to_period("M")
    monthly = daily_stats.groupby("year_month").agg(
        portfolio_return=("portfolio_return", lambda x: (1 + x).prod() - 1),
        benchmark_return=("benchmark_return", lambda x: (1 + x).prod() - 1),
    ).reset_index()
    monthly["excess_return"] = monthly["portfolio_return"] - monthly["benchmark_return"]

    monthly["year"] = monthly["year_month"].astype(str).str[:4]
    yearly = monthly.groupby("year").agg(
        portfolio_return=("portfolio_return", lambda x: (1 + x).prod() - 1),
        benchmark_return=("benchmark_return", lambda x: (1 + x).prod() - 1),
        months_won=("excess_return", lambda x: (x > 0).sum()),
        total_months=("excess_return", "count"),
    ).reset_index()
    yearly["excess_return"] = yearly["portfolio_return"] - yearly["benchmark_return"]

    return {"daily_stats": daily_stats, "monthly_stats": monthly, "yearly_stats": yearly,
            "trades": trades_log, "initial_value": initial_cash,
            "final_value": daily_stats["portfolio_value"].iloc[-1] if len(daily_stats) > 0 else initial_cash}


def _compute_equity(holdings, price_matrix, date):
    total = 0.0
    for sym, h in holdings.items():
        if sym in price_matrix.columns and date in price_matrix.index:
            price = price_matrix.loc[date, sym]
            if pd.notna(price) and price > 0:
                total += h["shares"] * price
    return total


def _trailing_return(portfolio_values, current_date, window):
    vals = [pv for pv in portfolio_values if pv["date"] <= current_date]
    if len(vals) < window:
        return None
    sv, ev = vals[-window]["portfolio_value"], vals[-1]["portfolio_value"]
    return (ev / sv) - 1 if sv > 0 else None


def _trailing_return_bench(benchmark_prices, current_date, window):
    if benchmark_prices is None:
        return None
    prices = benchmark_prices[benchmark_prices.index <= current_date]
    if len(prices) < window:
        return None
    return (prices.iloc[-1] / prices.iloc[-window]) - 1


def _portfolio_daily_returns(portfolio_values):
    if len(portfolio_values) < 2:
        return None
    returns = []
    for i in range(1, len(portfolio_values)):
        p, c = portfolio_values[i - 1]["portfolio_value"], portfolio_values[i]["portfolio_value"]
        if p > 0:
            returns.append((c / p) - 1)
    return pd.Series(returns)
