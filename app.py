#!/usr/bin/env python3
"""
ETF交易系统 v2 — 交互式回测与实盘信号面板
支持数据源: 自动(Sina+Baostock) | Tushare | Sina金融 | Baostock
"""
import sys
import os
import io
import time
import traceback
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_fetcher import (
    fetch_batch_stocks, fetch_batch_etfs, fetch_benchmark_index,
    fetch_batch_stocks_tushare, fetch_batch_etfs_tushare, fetch_benchmark_index_tushare,
    fetch_realtime_quotes, build_price_matrix, build_volume_matrix,
)
from universe import build_universe
from strategy import MomentumRotationStrategy
from backtest import run_backtest
from evaluation import compute_metrics

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="ETF交易系统 v2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- 默认参数 ----------
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

BENCHMARK_OPTIONS = {
    "中证1000 (CSI 1000)": "000852",
    "沪深300 (CSI 300)": "000300",
    "中证500 (CSI 500)": "000905",
    "深证成指 (SZSE)": "399001",
    "上证综指 (SSE)": "000001",
}

DATA_SOURCE_OPTIONS = {
    "自动 (Sina+Baostock)": "auto",
    "Tushare (专业数据)": "tushare",
    "Sina 金融": "sina",
    "Baostock": "baostock",
}


# ---------- 数据加载 (带缓存) ----------
@st.cache_data(show_spinner=False, ttl=3600)
def load_universe():
    return build_universe()


@st.cache_data(show_spinner=False, ttl=7200)
def fetch_stock_data(stocks, data_start, backtest_end, source="auto"):
    if source == "tushare":
        return fetch_batch_stocks_tushare(stocks, data_start, backtest_end)
    elif source == "sina":
        # Sina doesn't support stocks well, use as ETF-like fetch
        return fetch_batch_etfs(stocks, data_start, backtest_end)
    else:
        return fetch_batch_stocks(stocks, data_start, backtest_end)


@st.cache_data(show_spinner=False, ttl=7200)
def fetch_etf_data(etfs, data_start, backtest_end, source="auto"):
    if source == "tushare":
        return fetch_batch_etfs_tushare(etfs, data_start, backtest_end)
    else:
        return fetch_batch_etfs(etfs, data_start, backtest_end)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_bench_data(code, data_start, backtest_end, source="auto"):
    if source == "tushare":
        return fetch_benchmark_index_tushare(code, data_start, backtest_end)
    else:
        return fetch_benchmark_index(code, data_start, backtest_end)


def run_full_backtest(start_date, end_date, bench_code, params, data_source="auto"):
    """运行完整回测流水线。"""
    data_start = (pd.Timestamp(start_date) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

    # 构建投资池
    universe = load_universe()
    if universe is None or not universe["all_symbols"]:
        return None, "无法构建投资池，请检查网络连接。"

    stocks = universe["stocks"]
    etfs = universe["etfs"]
    stock_info = {**universe.get("stock_info", {}), **universe.get("etf_info", {})}

    progress_text = st.empty()

    # 获取数据
    progress_text.text("⏳ 正在获取股票日线数据...")
    stock_data = fetch_stock_data(stocks, data_start, end_date, source=data_source)

    progress_text.text("⏳ 正在获取ETF日线数据...")
    etf_data = fetch_etf_data(etfs, data_start, end_date, source=data_source)

    progress_text.text("⏳ 正在获取基准指数数据...")
    bench_df = fetch_bench_data(bench_code, data_start, end_date, source=data_source)
    if bench_df.empty:
        return None, "无法获取基准指数数据，请检查网络。"

    benchmark_prices = bench_df.set_index("date")["close"].sort_index()

    all_data = {}
    all_data.update(stock_data)
    all_data.update(etf_data)

    if len(all_data) < 10:
        return None, f"数据不足：仅获取到 {len(all_data)} 个标的的数据。"

    progress_text.text("⏳ 正在构建价格矩阵...")
    price_matrix = build_price_matrix(all_data)
    volume_matrix = build_volume_matrix(all_data)

    progress_text.text("⏳ 正在运行回测...")
    strategy = MomentumRotationStrategy(**params)
    result = run_backtest(
        price_matrix, volume_matrix, benchmark_prices,
        strategy, start_date=start_date, end_date=end_date,
        stock_info=stock_info,
    )

    progress_text.empty()

    if result is None:
        return None, "回测引擎返回空结果，请尝试调整日期范围。"

    metrics = compute_metrics(result)

    bench_name = [k for k, v in BENCHMARK_OPTIONS.items() if v == bench_code][0]
    output = {
        "result": result,
        "metrics": metrics,
        "price_matrix": price_matrix,
        "benchmark_prices": benchmark_prices,
        "bench_name": bench_name,
        "universe": universe,
        "data_source": data_source,
    }
    return output, None


def get_live_signals(params):
    """从实时行情生成实盘交易信号。"""
    universe = load_universe()
    if universe is None:
        return None, "无法加载投资池。"

    all_codes = universe["all_symbols"]

    progress_text = st.empty()
    progress_text.text("⏳ 正在获取实时行情...")
    quotes = fetch_realtime_quotes(all_codes, batch_size=50)
    progress_text.empty()

    if not quotes:
        return None, "暂无实时行情数据（可能非交易时间），请在工作日 9:30-15:00 尝试。"

    stocks = universe["stocks"]
    etfs = universe["etfs"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

    stock_data = fetch_stock_data(stocks, start_date, end_date, source="auto")
    etf_data = fetch_etf_data(etfs, start_date, end_date, source="auto")
    all_data = {}
    all_data.update(stock_data)
    all_data.update(etf_data)

    price_matrix = build_price_matrix(all_data)
    volume_matrix = build_volume_matrix(all_data)

    # 将实时价格追加为最新一行
    if not price_matrix.empty:
        today = pd.Timestamp.now().normalize()
        if today not in price_matrix.index:
            new_row = {}
            for sym, info in quotes.items():
                if info["price"] > 0:
                    new_row[sym] = info["price"]
            if new_row:
                price_matrix.loc[today] = new_row

    bench_df = fetch_bench_data("000852", start_date, end_date, source="auto")
    if bench_df.empty:
        return None, "无法获取基准数据。"

    bench_prices = bench_df.set_index("date")["close"].sort_index()
    strategy = MomentumRotationStrategy(**params)
    strategy.set_benchmark(bench_prices)
    current_date = price_matrix.index[-1] if not price_matrix.empty else pd.Timestamp.now()

    scores = strategy.compute_scores(price_matrix, volume_matrix, current_date)
    if scores.empty:
        return None, "无法计算当前得分，需要更多历史数据。"

    weights, regime = strategy.select_portfolio(scores, current_date)
    pos_mult = strategy.compute_position_size(False, regime)

    signals = []
    for _, row in scores.iterrows():
        sym = row["symbol"]
        quote = quotes.get(sym, {})
        ref_price = quote.get("price", row.get("price", 0))
        bid = quote.get("bid", 0)
        ask = quote.get("ask", 0)
        name = quote.get("name", "")
        target_weight = weights.get(sym, 0) * pos_mult
        signal_type = "买入" if target_weight > 0 else "持有"
        signals.append({
            "排名": int(row["rank"]),
            "代码": sym,
            "名称": name,
            "最新价": ref_price,
            "买一": bid,
            "卖一": ask,
            "动量得分": row["momentum"],
            "低波得分": row["vol_score"],
            "流动性得分": row["liq_score"],
            "综合得分": row["composite"],
            "目标权重": target_weight,
            "信号": signal_type,
        })

    signals_df = pd.DataFrame(signals).sort_values("排名")
    return signals_df, regime


# ---------- 图表函数 ----------

def plot_cumulative_returns(result, bench_name):
    daily = result["daily_stats"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["portfolio_cum"],
        name="策略", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["benchmark_cum"],
        name=bench_name, line=dict(color="#ff7f0e", width=2),
    ))
    fig.update_layout(
        title="累计收益曲线",
        yaxis=dict(tickformat=".0%", title=""),
        xaxis=dict(title=""),
        margin=dict(t=40, b=20, l=50, r=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=400,
    )
    return fig


def plot_drawdown(result, bench_name):
    daily = result["daily_stats"]
    pc = daily["portfolio_cum"].values
    pp = np.maximum.accumulate(pc)
    port_dd = (pc - pp) / pp
    bc = daily["benchmark_cum"].values
    bp = np.maximum.accumulate(bc)
    bench_dd = (bc - bp) / bp

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=port_dd,
        name="策略回撤", fill="tozeroy",
        fillcolor="rgba(214,39,40,0.3)",
        line=dict(color="#d62728"),
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=bench_dd,
        name=f"{bench_name} 回撤", fill="tozeroy",
        fillcolor="rgba(255,152,150,0.2)",
        line=dict(color="#ff9896"),
    ))
    fig.update_layout(
        title="回撤曲线",
        yaxis=dict(tickformat=".0%", title=""),
        xaxis=dict(title=""),
        margin=dict(t=40, b=20, l=50, r=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=400,
    )
    return fig


def plot_monthly_excess(result):
    monthly = result["monthly_stats"].copy()
    colors = ["#2ca02c" if x > 0 else "#d62728" for x in monthly["excess_return"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["year_month"].astype(str),
        y=monthly["excess_return"] * 100,
        marker_color=colors,
        name="月度超额收益",
    ))
    fig.add_hline(y=0, line_color="black", line_width=0.5)
    fig.update_layout(
        title="月度超额收益（相对基准）",
        yaxis=dict(title="超额收益 (%)"),
        xaxis=dict(title=""),
        margin=dict(t=40, b=40, l=50, r=20),
        showlegend=False,
        height=350,
    )
    return fig


def plot_yearly_excess(result):
    yearly = result["yearly_stats"]
    colors_yr = [
        "#2ca02c" if x >= 0.05 else ("#ff7f0e" if x > 0 else "#d62728")
        for x in yearly["excess_return"]
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly["year"].astype(str),
        y=yearly["excess_return"] * 100,
        marker_color=colors_yr,
        name="年度超额收益",
    ))
    fig.add_hline(y=5, line_color="green", line_dash="dash", line_width=1,
                  annotation_text="5% 目标线")
    fig.add_hline(y=0, line_color="black", line_width=0.5)
    fig.update_layout(
        title="年度超额收益（相对基准）",
        yaxis=dict(title="超额收益 (%)"),
        xaxis=dict(title=""),
        margin=dict(t=40, b=20, l=50, r=20),
        showlegend=False,
        height=350,
    )
    return fig


def plot_regime(result):
    daily = result["daily_stats"].copy()
    regime_map = {"bull": 1, "bear": -1, "unknown": 0}
    daily["regime_val"] = daily["regime"].map(regime_map).fillna(0)

    fig = go.Figure()
    mask_bull = daily["regime_val"] > 0
    mask_bear = daily["regime_val"] < 0

    if mask_bull.any():
        fig.add_trace(go.Scatter(
            x=daily["date"][mask_bull],
            y=daily["regime_val"][mask_bull],
            fill="tozeroy", fillcolor="rgba(44,160,44,0.3)",
            line=dict(color="green", width=0),
            name="牛市",
        ))
    if mask_bear.any():
        fig.add_trace(go.Scatter(
            x=daily["date"][mask_bear],
            y=daily["regime_val"][mask_bear],
            fill="tozeroy", fillcolor="rgba(214,39,40,0.3)",
            line=dict(color="red", width=0),
            name="熊市",
        ))

    fig.update_layout(
        title="市场状态（趋势过滤）",
        yaxis=dict(tickvals=[-1, 1], ticktext=["熊市", "牛市"], range=[-1.5, 1.5]),
        xaxis=dict(title=""),
        margin=dict(t=40, b=20, l=50, r=20),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=300,
    )
    return fig


# ================================================================
#                          侧 边 栏
# ================================================================
st.sidebar.title("📈 ETF交易系统 v2")
st.sidebar.markdown("**趋势过滤 + 动量轮动**")
st.sidebar.markdown("---")

# 日期范围
st.sidebar.subheader("📅 回测区间")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("起始日期", date(2021, 1, 1),
                               min_value=date(2016, 1, 1),
                               max_value=date(2026, 12, 31))
with col2:
    end_date = st.date_input("结束日期", date(2026, 5, 1),
                             min_value=date(2016, 1, 1),
                             max_value=date(2026, 12, 31))

if start_date >= end_date:
    st.sidebar.error("起始日期必须早于结束日期")

# 数据源
st.sidebar.subheader("📡 数据源")
data_source_label = st.sidebar.selectbox(
    "选择数据源",
    list(DATA_SOURCE_OPTIONS.keys()),
    index=0,
    help="自动模式：Sina获取ETF/指数 + Baostock获取股票（推荐）。Tushare需专业版token。",
)
data_source = DATA_SOURCE_OPTIONS[data_source_label]

# 基准指数
st.sidebar.subheader("🎯 对标基准")
bench_label = st.sidebar.selectbox(
    "选择对标基准",
    list(BENCHMARK_OPTIONS.keys()),
    index=0,
)
bench_code = BENCHMARK_OPTIONS[bench_label]

# 策略参数
st.sidebar.subheader("⚙️ 策略参数")
show_params = st.sidebar.checkbox("展开高级设置")
if show_params:
    params = {}
    params["top_n_bull"] = st.sidebar.slider("牛市政持仓数", 5, 50, 25, 5)
    params["top_n_bear"] = st.sidebar.slider("熊市政持仓数", 3, 30, 12, 3)
    params["trend_ma"] = st.sidebar.slider("趋势均线(日)", 100, 300, 220, 20,
                                          help="指数价格 vs N日均线判断牛熊")
    params["stop_loss"] = st.sidebar.slider("止损线", -0.25, -0.03, -0.12, 0.01,
                                           help="单笔持仓最大亏损比例")
    params["bear_cash_ratio"] = st.sidebar.slider("熊市现金比例", 0.1, 0.8, 0.4, 0.1,
                                                  help="熊市时保留的现金比例")
    params["vol_target"] = st.sidebar.slider("波动率目标", 0.05, 0.30, 0.15, 0.05,
                                            help="年化波动率目标，用于仓位缩放")
    params["risk_off_threshold"] = st.sidebar.slider("风控触发阈值", -0.15, 0.0, -0.05, 0.01,
                                                     help="近3月超额收益低于此阈值时启动风控")
    params["mom_periods"] = BEST_PARAMS["mom_periods"]
    params["mom_weights"] = BEST_PARAMS["mom_weights"]
    params["score_weights_bull"] = BEST_PARAMS["score_weights_bull"]
    params["score_weights_bear"] = BEST_PARAMS["score_weights_bear"]
else:
    params = BEST_PARAMS.copy()

# 运行按钮
st.sidebar.markdown("---")
run_backtest_btn = st.sidebar.button(
    "🚀 开始回测", type="primary", use_container_width=True
)

# 实盘信号
st.sidebar.markdown("---")
st.sidebar.subheader("🔴 实盘交易信号")
st.sidebar.caption("基于实时行情 + 历史动量，生成当前买入建议")
fetch_live_btn = st.sidebar.button(
    "📡 获取实时信号", use_container_width=True
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "ETF交易系统 v2  |  趋势过滤动量轮动\n\n"
    "投资池: 中证1000成分股 + 全市场ETF\n"
    "数据: Sina金融 + Baostock + Tushare + AKShare"
)


# ================================================================
#                          主 内 容
# ================================================================
st.title("📈 ETF交易系统 v2")
st.markdown(
    "**趋势过滤 + 动量轮动策略**  |  "
    "中证1000成分股 + 全市场ETF  |  "
    "月频调仓 · 日频止损 · 波动率目标控制"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 业绩概览", "📈 回测图表", "🔴 实盘信号", "📋 交易记录"
])

# 会话状态
if "backtest_output" not in st.session_state:
    st.session_state.backtest_output = None
if "signals_df" not in st.session_state:
    st.session_state.signals_df = None
if "signals_regime" not in st.session_state:
    st.session_state.signals_regime = "unknown"
if "last_backtest_error" not in st.session_state:
    st.session_state.last_backtest_error = None
if "last_signal_error" not in st.session_state:
    st.session_state.last_signal_error = None

# 执行回测
if run_backtest_btn:
    with st.spinner("正在运行回测... 首次运行需从网络获取数据，请耐心等待。"):
        try:
            output, error = run_full_backtest(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                bench_code, params, data_source=data_source,
            )
            if error:
                st.session_state.last_backtest_error = error
                st.session_state.backtest_output = None
            else:
                st.session_state.backtest_output = output
                st.session_state.last_backtest_error = None
        except Exception as e:
            st.session_state.last_backtest_error = f"{e}\n{traceback.format_exc()}"
            st.session_state.backtest_output = None

# 获取实时信号
if fetch_live_btn:
    with st.spinner("正在获取实时行情并计算信号..."):
        try:
            signals_df, regime = get_live_signals(params)
            if signals_df is None:
                st.session_state.last_signal_error = "暂无数据（可能非交易时间）。"
                st.session_state.signals_df = None
            else:
                st.session_state.signals_df = signals_df
                st.session_state.signals_regime = regime
                st.session_state.last_signal_error = None
        except Exception as e:
            st.session_state.last_signal_error = f"{e}"
            st.session_state.signals_df = None

output = st.session_state.backtest_output


# ================================================================
#                      Tab 1: 业绩概览
# ================================================================
with tab1:
    if output is None:
        st.info("👈 请在左侧边栏点击 **开始回测** 按钮，或直接使用默认参数运行。")
        if st.session_state.last_backtest_error:
            st.error(f"回测失败: {st.session_state.last_backtest_error}")
    else:
        metrics = output["metrics"]
        result = output["result"]

        # 指标卡片
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        col1.metric("累计收益", f"{metrics['total_return']:.1%}")
        col2.metric("年化收益", f"{metrics['ann_return']:.1%}")
        col3.metric("年化超额", f"{metrics['ann_excess']:.2%}")
        col4.metric("夏普比率", f"{metrics['sharpe']:.2f}")
        col5.metric("最大回撤", f"{metrics['max_drawdown']:.1%}")
        col6.metric("月度胜率", f"{metrics['monthly_win_rate']:.1%}")
        col7.metric("卡玛比率", f"{metrics['calmar']:.2f}")
        col8.metric("信息比率", f"{metrics['information_ratio']:.2f}")

        st.markdown("---")

        # 年度收益表
        st.subheader("📋 年度收益明细")
        yearly = metrics["yearly_stats"]
        yearly_display = yearly.copy()
        yearly_display["策略收益"] = yearly_display["portfolio_return"].apply(lambda x: f"{x:.2%}")
        yearly_display["基准收益"] = yearly_display["benchmark_return"].apply(lambda x: f"{x:.2%}")
        yearly_display["超额收益"] = yearly_display["excess_return"].apply(
            lambda x: f"{x:.2%} {'✅' if x >= 0.05 else ('⚠️' if x > 0 else '❌')}"
        )
        yearly_display["月度胜率"] = yearly_display.apply(
            lambda r: f"{int(r['months_won'])}/{int(r['total_months'])}", axis=1
        )
        st.dataframe(
            yearly_display[["year", "策略收益", "基准收益", "超额收益", "月度胜率"]]
            .rename(columns={"year": "年份"}),
            use_container_width=True, hide_index=True,
        )

        # 目标检测
        st.markdown("---")
        st.subheader("🎯 目标检测")
        monthly_ok = metrics["monthly_win_rate"] >= 0.50
        annual_ok = metrics["annual_excess_met"]
        c1, c2 = st.columns(2)
        with c1:
            if monthly_ok:
                st.success(f"✅ 月度胜率 ≥ 50% — 当前 {metrics['monthly_win_rate']:.1%}")
            else:
                st.error(f"❌ 月度胜率 ≥ 50% — 当前 {metrics['monthly_win_rate']:.1%}")
        with c2:
            if annual_ok:
                st.success(f"✅ 年度超额 ≥ 5% — 平均 {metrics['avg_annual_excess']:.2%}")
            else:
                st.error(f"❌ 年度超额 ≥ 5% — 平均 {metrics['avg_annual_excess']:.2%}")

        # 风险对比
        st.markdown("---")
        st.subheader("🛡️ 风险对比")
        c1, c2, c3 = st.columns(3)
        c1.metric("策略最大回撤", f"{metrics['max_drawdown']:.1%}")
        c2.metric("基准最大回撤", f"{metrics['benchmark_max_drawdown']:.1%}")
        if metrics['benchmark_max_drawdown'] != 0:
            dd_ratio = abs(metrics['max_drawdown'] / metrics['benchmark_max_drawdown'])
            c3.metric("回撤比", f"{dd_ratio:.2f}x",
                      delta="防守更好" if dd_ratio < 1 else "防守不足",
                      delta_color="normal" if dd_ratio < 1 else "inverse")


# ================================================================
#                      Tab 2: 回测图表
# ================================================================
with tab2:
    if output is None:
        st.info("运行回测后可查看交互式图表。")
    else:
        result = output["result"]
        bench_name = output["bench_name"]

        st.subheader("📈 收益与风险分析")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_cumulative_returns(result, bench_name), use_container_width=True)
        with col2:
            st.plotly_chart(plot_drawdown(result, bench_name), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_monthly_excess(result), use_container_width=True)
        with col4:
            st.plotly_chart(plot_yearly_excess(result), use_container_width=True)

        st.plotly_chart(plot_regime(result), use_container_width=True)


# ================================================================
#                      Tab 3: 实盘信号
# ================================================================
with tab3:
    st.subheader("🔴 实时交易信号")
    st.caption("基于 Sina 实时行情 + 历史动量计算，仅供参考，不构成投资建议。")

    if st.session_state.signals_df is not None:
        signals_df = st.session_state.signals_df
        regime = st.session_state.signals_regime

        # 市场状态标签
        if regime == "bull":
            st.markdown(
                "**当前市场状态:** "
                "<span style='color:white;background:#2ca02c;padding:4px 16px;"
                "border-radius:12px;font-weight:bold;'>🐂 牛 市</span>"
                " — 动量因子主导，高仓位运行",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "**当前市场状态:** "
                "<span style='color:white;background:#d62728;padding:4px 16px;"
                "border-radius:12px;font-weight:bold;'>🐻 熊 市</span>"
                " — 低波因子主导，防御仓位",
                unsafe_allow_html=True,
            )

        buy_signals = signals_df[signals_df["信号"] == "买入"].head(params.get("top_n_bull", 25))
        all_ranked = signals_df.head(50)

        st.markdown("---")
        st.markdown(f"### 📌 买入建议 (共 {len(buy_signals)} 只)")

        # 买入标的卡片
        buy_cols = st.columns(min(len(buy_signals), 5))
        for i, (_, row) in enumerate(buy_signals.iterrows()):
            with buy_cols[i % 5]:
                name_display = row["名称"][:8] if row["名称"] else row["代码"]
                price = row["最新价"]
                st.metric(
                    f"{name_display} ({row['代码']})",
                    f"¥{price:.2f}" if price > 0 else "N/A",
                    delta=f"排名 #{row['排名']} | 权重 {row['目标权重']:.1%}",
                )
                st.caption(
                    f"买一: {row['买一']:.2f} | 卖一: {row['卖一']:.2f}\n"
                    f"动量: {row['动量得分']:.2%} | 低波: {row['低波得分']:.3f}"
                )

        st.markdown("---")
        st.subheader("📋 完整信号排名 (前50)")

        display_df = all_ranked.copy()
        display_df["动量得分"] = display_df["动量得分"].apply(lambda x: f"{x:.2%}")
        display_df["目标权重"] = display_df["目标权重"].apply(lambda x: f"{x:.2%}")
        display_df["最新价"] = display_df["最新价"].apply(lambda x: f"¥{x:.2f}" if x > 0 else "N/A")
        display_df["买一"] = display_df["买一"].apply(lambda x: f"¥{x:.2f}" if x > 0 else "-")
        display_df["卖一"] = display_df["卖一"].apply(lambda x: f"¥{x:.2f}" if x > 0 else "-")

        st.dataframe(
            display_df[["排名", "代码", "名称", "最新价", "买一", "卖一",
                         "动量得分", "目标权重", "信号"]],
            use_container_width=True, hide_index=True,
        )

        st.info(
            "💡 **参考买入价 = 卖一价**。实际成交价取决于市场流动性和挂单深度。"
            "建议在开盘后30分钟内分批建仓，避免集合竞价期间下单。"
        )

    elif st.session_state.last_signal_error:
        st.warning(f"⚠️ {st.session_state.last_signal_error}")
        st.info("实盘信号仅在工作日 9:30-15:00 (北京时间) 可用。")
    else:
        st.info("👈 点击左侧边栏 **获取实时信号** 按钮，获取当前买入建议。")


# ================================================================
#                      Tab 4: 交易记录
# ================================================================
with tab4:
    st.subheader("📋 交易记录")

    if output is None:
        st.info("运行回测后可查看和导出交易记录。")
    else:
        trades = output["result"].get("trades", [])
        if not trades:
            st.info("回测期间没有产生交易记录。")
        else:
            trades_df = pd.DataFrame(trades)
            trades_df["date"] = pd.to_datetime(trades_df["date"])

            # 筛选器
            col1, col2, col3 = st.columns(3)
            with col1:
                action_map = {"buy": "买入", "sell": "卖出", "stop_loss": "止损"}
                action_options = ["买入", "卖出", "止损"]
                action_filter = st.multiselect(
                    "操作类型", action_options, default=action_options,
                )
                action_filter_en = [k for k, v in action_map.items() if v in action_filter]
            with col2:
                symbols = sorted(trades_df["symbol"].unique())
                sym_filter = st.multiselect("标的代码", symbols, default=[])
            with col3:
                search = st.text_input("搜索代码", placeholder="如 600000")

            filtered = trades_df[trades_df["action"].isin(action_filter_en)]
            if sym_filter:
                filtered = filtered[filtered["symbol"].isin(sym_filter)]
            if search:
                filtered = filtered[filtered["symbol"].str.contains(search)]

            st.metric("交易笔数", len(filtered))

            display_trades = filtered.copy()
            display_trades["date"] = display_trades["date"].dt.strftime("%Y-%m-%d")
            display_trades["价格"] = display_trades["price"].apply(lambda x: f"¥{x:.2f}")
            display_trades["操作"] = display_trades["action"].map({
                "buy": "买入", "sell": "卖出", "stop_loss": "止损"
            })

            cols = ["date", "symbol", "操作", "shares", "价格"]
            rename = {"date": "日期", "symbol": "代码", "shares": "股数"}
            if "weight" in display_trades.columns:
                display_trades["权重"] = display_trades["weight"].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else ""
                )
                cols.append("权重")
            if "pnl_pct" in display_trades.columns:
                display_trades["盈亏"] = display_trades["pnl_pct"].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else ""
                )
                cols.append("盈亏")
                rename["pnl_pct"] = "盈亏"

            st.dataframe(
                display_trades[cols].rename(columns=rename),
                use_container_width=True, hide_index=True,
            )

            # 导出
            st.markdown("---")
            st.subheader("⬇️ 数据导出")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                csv_trades = trades_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 导出交易记录 (CSV)",
                    data=csv_trades,
                    file_name=f"交易记录_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_b:
                csv_daily = output["result"]["daily_stats"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 导出每日统计 (CSV)",
                    data=csv_daily,
                    file_name=f"每日统计_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_c:
                csv_monthly = output["result"]["monthly_stats"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 导出月度统计 (CSV)",
                    data=csv_monthly,
                    file_name=f"月度统计_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ---------- 底部 ----------
st.markdown("---")
st.caption(
    f"ETF交易系统 v2 | 数据源: Sina金融 + Baostock + Tushare + AKShare | "
    f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
    "⚠️ 仅供研究参考，不构成投资建议"
)
