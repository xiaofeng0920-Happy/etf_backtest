"""
Parameter optimization via grid search — v2.
"""
import itertools
import numpy as np
from strategy import MomentumRotationStrategy
from backtest import run_backtest
from evaluation import compute_metrics, check_targets

PARAM_GRID = {
    "top_n_bull": [25, 30, 35],
    "top_n_bear": [12, 18],
    "mom_periods": [(21, 63, 126, 252), (10, 42, 63, 126)],
    "mom_weights": [(0.5, 0.25, 0.15, 0.1), (0.4, 0.3, 0.2, 0.1)],
    "score_weights_bull": [(0.7, 0.2, 0.1), (0.6, 0.3, 0.1)],
    "score_weights_bear": [(0.4, 0.45, 0.15), (0.35, 0.5, 0.15)],
    "stop_loss": [-0.10, -0.12, -0.15],
    "bear_cash_ratio": [0.4, 0.5, 0.6],
    "vol_target": [0.12, 0.15],
    "trend_ma": [180, 220],
    "risk_off_threshold": [-0.03, -0.05],
}


def optimize(price_matrix, volume_matrix, benchmark_prices, max_iterations=50, stock_info=None):
    keys = list(PARAM_GRID.keys())
    all_combos = list(itertools.product(*PARAM_GRID.values()))
    total = len(all_combos)

    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION v2: {total} combos (max {max_iterations} iters)")
    print(f"{'='*70}")

    best_result = best_metrics = best_params = None
    best_score = -999
    rng = np.random.RandomState(42)
    indices = rng.permutation(total)[:max_iterations]

    for attempt, idx in enumerate(indices):
        combo = all_combos[idx]
        params = dict(zip(keys, combo))

        print(f"\n--- Iteration {attempt + 1}/{min(total, max_iterations)} ---")
        print(f"    bull_n={params['top_n_bull']} bear_n={params['top_n_bear']} "
              f"stop={params['stop_loss']} vol_tgt={params['vol_target']}")
        print(f"    MA={params['trend_ma']} bear_cash={params['bear_cash_ratio']}")

        strategy = MomentumRotationStrategy(
            top_n_bull=params["top_n_bull"], top_n_bear=params["top_n_bear"],
            mom_periods=params["mom_periods"],
            mom_weights=list(params["mom_weights"]),
            score_weights_bull=list(params["score_weights_bull"]),
            score_weights_bear=list(params["score_weights_bear"]),
            stop_loss=params["stop_loss"],
            risk_off_threshold=params["risk_off_threshold"],
            vol_target=params["vol_target"],
            trend_ma=params["trend_ma"],
            bear_cash_ratio=params["bear_cash_ratio"],
        )

        result = run_backtest(price_matrix, volume_matrix, benchmark_prices,
                              strategy, stock_info=stock_info)
        if result is None:
            print("    [SKIP] No result")
            continue

        metrics = compute_metrics(result)
        passed, details = check_targets(metrics)

        composite = (metrics["monthly_win_rate"] * 0.3 +
                     metrics["avg_annual_excess"] * 5 +
                     metrics["sharpe"] * 0.3 +
                     (1 if passed else 0) * 2 +
                     metrics["calmar"] * 0.2)

        print(f"    Monthly: {metrics['monthly_win_rate']:.1%} | "
              f"Ann Exc: {metrics['avg_annual_excess']:.2%} | "
              f"Sharpe: {metrics['sharpe']:.2f} | MaxDD: {metrics['max_drawdown']:.2%}")
        print(f"    Passed: {passed} | Score: {composite:.3f}")

        if passed:
            print(f"    *** TARGETS MET! ***")
        if composite > best_score:
            best_score, best_result, best_metrics, best_params = composite, result, metrics, params
        if passed:
            print(f"\n  Converged at iteration {attempt + 1}")
            break

    if best_result is None:
        print("\n[ERROR] No valid backtest results.")
        return None, None, None

    print(f"\n  Best score: {best_score:.3f}, params: {best_params}")
    return best_result, best_metrics, best_params
