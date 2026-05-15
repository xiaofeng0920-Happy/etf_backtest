"""
Strategy v2.1 — Trend-Filtered Momentum Rotation (conservative upgrade).

v3 Hermes skill learnings applied:
  - Look-ahead bias fixed: factor computation excludes today's close
  - New optional factors: MA slope, MA alignment, vol-price correlation
  - Cross-sectional rank normalization (same as old v2, verified correct)
  - Configurable rebalance frequency
"""
import numpy as np
import pandas as pd

FACTOR_KEYS = [
    "momentum", "vol_score", "liq_score",
    "ma20_slope", "ma60_slope", "ma_score", "vp_corr_20d",
]


class MomentumRotationStrategy:
    def __init__(
        self,
        top_n=25, top_n_bull=25, top_n_bear=12,
        mom_periods=(21, 63, 126, 252),
        mom_weights=(0.4, 0.3, 0.2, 0.1),
        vol_window=60,
        score_weights_bull=(0.6, 0.3, 0.1),
        score_weights_bear=(0.35, 0.5, 0.15),
        score_weights=(0.6, 0.3, 0.1),
        stop_loss=-0.12,
        risk_off_threshold=-0.05,
        risk_off_cash=0.5,
        vol_target=0.15,
        max_single_weight=0.10,
        trend_ma=220,
        bear_cash_ratio=0.4,
        # v2.1 new params
        rebalance_freq="monthly",
        extra_factor_weights=None,
    ):
        self.top_n = top_n
        self.top_n_bull = top_n_bull
        self.top_n_bear = top_n_bear
        self.mom_periods = mom_periods
        self.mom_weights = np.array(mom_weights) / np.sum(mom_weights)
        self.vol_window = vol_window
        self.score_weights_bull = score_weights_bull
        self.score_weights_bear = score_weights_bear
        self.score_weights = score_weights
        self.stop_loss = stop_loss
        self.risk_off_threshold = risk_off_threshold
        self.risk_off_cash = risk_off_cash
        self.vol_target = vol_target
        self.max_single_weight = max_single_weight
        self.trend_ma = trend_ma
        self.bear_cash_ratio = bear_cash_ratio
        self.rebalance_freq = rebalance_freq
        self._benchmark_prices = None

        # Extra factor weights (new in v2.1, default 0 = pure v2 behavior)
        if extra_factor_weights is not None:
            self.extra_weights = extra_factor_weights
        else:
            self.extra_weights = {
                "ma20_slope": 0.0, "ma60_slope": 0.0,
                "ma_score": 0.0, "vp_corr_20d": 0.0,
            }

    def set_benchmark(self, benchmark_prices):
        self._benchmark_prices = benchmark_prices

    def set_stock_info(self, stock_info):
        pass

    def detect_regime(self, current_date):
        if self._benchmark_prices is None:
            return "bull"
        prices = self._benchmark_prices[self._benchmark_prices.index <= current_date]
        if len(prices) < self.trend_ma + 5:
            return "bull"
        return "bull" if prices.iloc[-1] > prices.iloc[-self.trend_ma:].mean() else "bear"

    def compute_scores(self, price_matrix, volume_matrix, current_date):
        prices = price_matrix.loc[:current_date].copy()
        if len(prices) < self.vol_window + 21:
            return pd.DataFrame()

        # Look-ahead fix: exclude today's close from factor computation
        # We decide what to buy based on data up to T-1, then execute at T close
        prices_factor = prices.iloc[:-1].copy()
        if len(prices_factor) < max(self.mom_periods) + 21:
            return pd.DataFrame()

        regime = self.detect_regime(current_date)
        sw = self.score_weights_bull if regime == "bull" else self.score_weights_bear
        returns = prices_factor.pct_change(fill_method=None).dropna(how="all")

        scores = []
        for sym in prices_factor.columns:
            sym_prices = prices_factor[sym].dropna()
            if len(sym_prices) < max(self.mom_periods) + 21:
                continue
            sym_rets = returns[sym].dropna()
            if len(sym_rets) < 20:
                continue

            # === Core momentum (unchanged from v2) ===
            mom_components = []
            for period in self.mom_periods:
                if len(sym_prices) >= period:
                    mom_components.append(sym_prices.iloc[-1] / sym_prices.iloc[-period] - 1)
                else:
                    mom_components.append(0.0)
            momentum = np.dot(self.mom_weights, mom_components)

            # === Volatility score (unchanged) ===
            ann_vol = 0.0
            if len(sym_rets) >= self.vol_window:
                vol_series = sym_rets.iloc[-self.vol_window:]
                ann_vol = vol_series.std() * np.sqrt(252)
                vol_score = 1.0 / (ann_vol * 100) if ann_vol > 0 else 0.0
            else:
                vol_score = 0.0

            # === Liquidity score (unchanged) ===
            liq_score = 0.0
            if volume_matrix is not None and sym in volume_matrix.columns:
                sv = volume_matrix[sym].dropna()
                sv = sv[sv.index <= current_date]
                # Exclude current date from volume data
                sv = sv.iloc[:-1] if len(sv) > 1 and sv.index[-1] == current_date else sv
                if len(sv) >= 20:
                    avg = sv.iloc[-20:].mean()
                    if avg > 0:
                        liq_score = np.log10(avg)

            # === New factors (v2.1) ===
            ma20 = sym_prices.rolling(20).mean()
            ma60 = sym_prices.rolling(60).mean()
            ma120 = sym_prices.rolling(120).mean()

            ma20_slope = 0.0
            ma60_slope = 0.0
            if len(ma20.dropna()) >= 6:
                ma20_slope = (ma20.iloc[-1] - ma20.iloc[-6]) / max(abs(ma20.iloc[-6]), 0.001)
            if len(ma60.dropna()) >= 6:
                ma60_slope = (ma60.iloc[-1] - ma60.iloc[-6]) / max(abs(ma60.iloc[-6]), 0.001)

            ma_score_val = 1.5
            if len(ma20.dropna()) > 0 and len(ma60.dropna()) > 0 and len(ma120.dropna()) > 0:
                s = 0
                if sym_prices.iloc[-1] > ma20.iloc[-1]: s += 1
                if ma20.iloc[-1] > ma60.iloc[-1]: s += 1
                if ma60.iloc[-1] > ma120.iloc[-1]: s += 1
                ma_score_val = s

            vp_corr = 0.0
            if volume_matrix is not None and sym in volume_matrix.columns:
                vol_all = volume_matrix[sym].dropna()
                vol_all = vol_all[vol_all.index <= current_date]
                vol_all = vol_all.iloc[:-1] if len(vol_all) > 1 else vol_all
                vol_chg = vol_all.pct_change().dropna()
                ci = sym_rets.index.intersection(vol_chg.index)
                if len(ci) >= 20:
                    vp_corr = sym_rets.loc[ci[-20:]].corr(vol_chg.loc[ci[-20:]])
                    if pd.isna(vp_corr):
                        vp_corr = 0.0

            scores.append({
                "symbol": sym,
                "momentum": momentum, "vol_score": vol_score, "liq_score": liq_score,
                "ma20_slope": ma20_slope, "ma60_slope": ma60_slope,
                "ma_score": ma_score_val, "vp_corr_20d": vp_corr,
                "price": sym_prices.iloc[-1], "ann_vol": ann_vol,
            })

        if not scores:
            return pd.DataFrame()

        scores_df = pd.DataFrame(scores)

        # Rank normalize core factors
        for col in ["momentum", "vol_score", "liq_score"]:
            s = scores_df[col]
            scores_df[col + "_norm"] = s.rank(pct=True) if s.std() > 1e-9 else 0.5

        # Rank normalize extra factors (for optional use)
        for col in ["ma20_slope", "ma60_slope", "ma_score", "vp_corr_20d"]:
            s = scores_df[col]
            scores_df[col + "_norm"] = s.rank(pct=True) if s.std() > 1e-9 else 0.5

        # Composite score: core v2 weights + optional extra weights
        w_m, w_v, w_l = sw
        scores_df["composite"] = (w_m * scores_df["momentum_norm"] +
                                  w_v * scores_df["vol_score_norm"] +
                                  w_l * scores_df["liq_score_norm"])

        for ek, ew in self.extra_weights.items():
            nk = ek + "_norm"
            if ew != 0.0 and nk in scores_df.columns:
                scores_df["composite"] += ew * scores_df[nk]

        scores_df = scores_df.sort_values("composite", ascending=False)
        scores_df["rank"] = range(1, len(scores_df) + 1)
        return scores_df

    def select_portfolio(self, scores_df, current_date):
        if scores_df.empty:
            return {}, "unknown"
        regime = self.detect_regime(current_date)
        tn = self.top_n_bull if regime == "bull" else self.top_n_bear
        top = scores_df.head(tn)
        n = len(top)
        if n == 0:
            return {}, regime
        eq_w = 1.0 / n
        weights = {row["symbol"]: eq_w for _, row in top.iterrows()}
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}, regime

    def check_stop_loss(self, symbol, entry_price, current_price):
        if entry_price <= 0:
            return False
        return (current_price / entry_price) - 1 <= self.stop_loss

    def should_risk_off(self, port_ret_3m, bench_ret_3m):
        if port_ret_3m is None or bench_ret_3m is None:
            return False
        return (port_ret_3m - bench_ret_3m) < self.risk_off_threshold

    def compute_position_size(self, is_risk_off, regime):
        if is_risk_off:
            return 1.0 - self.risk_off_cash
        if regime == "bear":
            return 1.0 - self.bear_cash_ratio
        return 1.0

    def apply_vol_target(self, weights, recent_returns):
        if not weights or self.vol_target <= 0:
            return weights
        if recent_returns is not None and len(recent_returns) > 20:
            pv = recent_returns.std() * np.sqrt(252)
            if pv > 0:
                s = np.clip(self.vol_target / pv, 0.3, 1.5)
                return {k: v * s for k, v in weights.items()}
        return weights
