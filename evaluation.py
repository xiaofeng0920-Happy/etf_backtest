"""
Performance evaluation & charting — v2.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_metrics(result):
    daily = result["daily_stats"]
    monthly = result["monthly_stats"]
    yearly = result["yearly_stats"]

    monthly_wins = (monthly["excess_return"] > 0).sum()
    monthly_total = len(monthly)
    monthly_win_rate = monthly_wins / monthly_total if monthly_total > 0 else 0

    yearly_excess = yearly["excess_return"].values
    annual_excess_met = all(e >= 0.05 for e in yearly_excess) if len(yearly_excess) > 0 else False
    avg_annual_excess = np.mean(yearly_excess) if len(yearly_excess) > 0 else 0

    excess_daily = daily["portfolio_return"] - daily["benchmark_return"]
    ann_excess_val = excess_daily.mean() * 252
    ann_vol_val = excess_daily.std() * np.sqrt(252)
    sharpe = ann_excess_val / ann_vol_val if ann_vol_val > 0 else 0

    port_cum = daily["portfolio_cum"].values
    peak = np.maximum.accumulate(port_cum)
    max_dd = (port_cum - peak) / peak
    max_dd_val = max_dd.min()

    bench_cum = daily["benchmark_cum"].values
    bench_peak = np.maximum.accumulate(bench_cum)
    bench_max_dd = ((bench_cum - bench_peak) / bench_peak).min()

    total_return = port_cum[-1] - 1 if len(port_cum) > 0 else 0
    years = len(daily) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    port_ann_vol = daily["portfolio_return"].std() * np.sqrt(252)

    bench_total = bench_cum[-1] / bench_cum[0] - 1 if len(bench_cum) > 1 else 0
    bench_ann = (1 + bench_total) ** (1 / years) - 1 if years > 0 else 0

    calmar = ann_return / abs(max_dd_val) if max_dd_val != 0 else 0
    ir = ann_excess_val / ann_vol_val if ann_vol_val > 0 else 0

    return {
        "monthly_win_rate": monthly_win_rate, "monthly_wins": monthly_wins,
        "monthly_total": monthly_total, "annual_excess_met": annual_excess_met,
        "avg_annual_excess": avg_annual_excess, "sharpe": sharpe,
        "max_drawdown": max_dd_val, "benchmark_max_drawdown": bench_max_dd,
        "ann_return": ann_return, "bench_ann_return": bench_ann,
        "ann_excess": ann_excess_val, "port_ann_vol": port_ann_vol,
        "total_return": total_return, "bench_total_return": bench_total,
        "calmar": calmar, "information_ratio": ir,
        "num_trades": len(result.get("trades", [])), "years": years,
        "yearly_stats": yearly, "monthly_stats": monthly,
    }


def check_targets(metrics):
    monthly_ok = metrics["monthly_win_rate"] >= 0.50
    annual_ok = metrics["annual_excess_met"]
    return monthly_ok and annual_ok, {
        "monthly_ok": monthly_ok, "annual_ok": annual_ok,
        "monthly_win_rate": metrics["monthly_win_rate"],
        "annual_excess_details": {
            row["year"]: f"{row['excess_return']:.2%}"
            for _, row in metrics["yearly_stats"].iterrows()},
        "avg_annual_excess": metrics["avg_annual_excess"],
    }


def print_report(metrics):
    print("\n" + "=" * 70)
    print("  ETF TRADING SYSTEM v2 — PERFORMANCE REPORT")
    print("=" * 70)
    print(f"\n  Period: {metrics['years']:.1f} years | "
          f"Monthly Win: {metrics['monthly_wins']}/{metrics['monthly_total']} "
          f"({metrics['monthly_win_rate']:.1%})")
    print(f"\n  {'Metric':<30} {'Portfolio':>15} {'Benchmark':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Annualized Return':<30} {metrics['ann_return']:>14.2%}  {metrics['bench_ann_return']:>14.2%}")
    print(f"  {'Annual Excess':<30} {metrics['ann_excess']:>14.2%}  {'':>15}")
    print(f"  {'Annualized Volatility':<30} {metrics['port_ann_vol']:>14.2%}  {'':>15}")
    print(f"  {'Sharpe Ratio':<30} {metrics['sharpe']:>14.2f}  {'':>15}")
    print(f"  {'Max Drawdown':<30} {metrics['max_drawdown']:>14.2%}  {metrics['benchmark_max_drawdown']:>14.2%}")
    print(f"  {'Calmar Ratio':<30} {metrics['calmar']:>14.2f}  {'':>15}")
    print(f"  {'Information Ratio':<30} {metrics['information_ratio']:>14.2f}  {'':>15}")
    print(f"  {'Total Return':<30} {metrics['total_return']:>14.2%}  {metrics['bench_total_return']:>14.2%}")
    print(f"  {'Number of Trades':<30} {metrics['num_trades']:>15}  {'':>15}")
    print(f"\n  Yearly Returns:")
    print(f"  {'Year':<8} {'Portfolio':>12} {'Benchmark':>12} {'Excess':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for _, row in metrics["yearly_stats"].iterrows():
        exc = row["excess_return"]
        flag = " +" if exc >= 0.05 else (" ✓" if exc > 0 else " ✗")
        print(f"  {row['year']:<8} {row['portfolio_return']:>11.2%}  "
              f"{row['benchmark_return']:>11.2%}  {exc:>11.2%}{flag}")
    monthly = metrics["monthly_stats"]
    monthly["year"] = monthly["year_month"].astype(str).str[:4]
    print(f"\n  Monthly Win Rate by Year:")
    for year, grp in monthly.groupby("year"):
        wins = (grp["excess_return"] > 0).sum()
        print(f"  {year}: {wins}/{len(grp)} ({wins/len(grp):.0%})")
    print("=" * 70)


def plot_results(result, metrics, filename="backtest_result.png"):
    daily = result["daily_stats"]
    monthly = metrics["monthly_stats"]
    yearly = metrics["yearly_stats"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ETF Trading System v2 — 5-Year Backtest Results", fontsize=14, fontweight="bold")

    # 1. Cumulative Returns
    ax = axes[0, 0]
    ax.plot(daily["date"], daily["portfolio_cum"], label="Strategy", linewidth=1.5, color="#1f77b4")
    ax.plot(daily["date"], daily["benchmark_cum"], label="CSI 1000", linewidth=1.5, color="#ff7f0e")
    ax.set_title("Cumulative Return")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    pc, bc = daily["portfolio_cum"].values, daily["benchmark_cum"].values
    pp, bp = np.maximum.accumulate(pc), np.maximum.accumulate(bc)
    ax.fill_between(daily["date"], 0, (pc - pp) / pp, alpha=0.5, label="Strategy DD", color="#d62728")
    ax.fill_between(daily["date"], 0, (bc - bp) / bp, alpha=0.5, label="CSI 1000 DD", color="#ff9896")
    ax.set_title("Drawdown")
    ax.legend(loc="lower left")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    # 3. Regime over time
    ax = axes[0, 2]
    regime_map = {"bull": 1, "bear": -1, "unknown": 0}
    regime_vals = daily["regime"].map(regime_map).fillna(0).values
    ax.fill_between(daily["date"], 0, regime_vals, where=(np.array(regime_vals) > 0),
                    color="green", alpha=0.3, label="Bull")
    ax.fill_between(daily["date"], 0, regime_vals, where=(np.array(regime_vals) < 0),
                    color="red", alpha=0.3, label="Bear")
    ax.set_title("Market Regime (Trend Filter)")
    ax.legend(loc="upper right")
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)

    # 4. Monthly Excess Returns
    ax = axes[1, 0]
    monthly = monthly.copy()
    monthly["ym_str"] = monthly["year_month"].astype(str)
    colors = ["#2ca02c" if x > 0 else "#d62728" for x in monthly["excess_return"]]
    ax.bar(range(len(monthly)), monthly["excess_return"] * 100, color=colors, width=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Monthly Excess Return vs CSI 1000")
    ax.set_ylabel("Excess Return (%)")
    step = max(1, len(monthly) // 12)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels(monthly["ym_str"].iloc[::step], rotation=45, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. Annual Excess Returns
    ax = axes[1, 1]
    colors_yr = ["#2ca02c" if x >= 0.05 else ("#ff7f0e" if x > 0 else "#d62728")
                 for x in yearly["excess_return"]]
    ax.bar(yearly["year"].astype(str), yearly["excess_return"] * 100, color=colors_yr, width=0.6)
    ax.axhline(y=5, color="green", linestyle="--", linewidth=1, label="5% Target")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Annual Excess Return vs CSI 1000")
    ax.set_ylabel("Excess Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Rolling 12M Excess
    ax = axes[1, 2]
    daily2 = daily.copy()
    daily2["excess_daily"] = daily2["portfolio_return"] - daily2["benchmark_return"]
    daily2["roll_12m_excess"] = daily2["excess_daily"].rolling(252).mean() * 252
    ax.plot(daily2["date"], daily2["roll_12m_excess"] * 100, linewidth=1.5, color="#1f77b4")
    ax.axhline(y=5, color="green", linestyle="--", linewidth=1, label="5% Target")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Rolling 12-Month Annualized Excess")
    ax.set_ylabel("Excess Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart saved to: {filepath}")
    return filepath
