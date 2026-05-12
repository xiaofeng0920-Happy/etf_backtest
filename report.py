"""
HTML Dashboard Report — v2.
"""
import os
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_html_report(result, metrics, strategy_params=None):
    daily = result["daily_stats"]
    monthly = result["monthly_stats"]
    yearly = result["yearly_stats"]

    port_cum = daily["portfolio_cum"].tolist()
    bench_cum = daily["benchmark_cum"].tolist()
    dates = daily["date"].astype(str).tolist()

    pc = daily["portfolio_cum"].values
    pp = np.maximum.accumulate(pc)
    port_dd = ((pc - pp) / pp * 100).tolist()
    bc = daily["benchmark_cum"].values
    bp = np.maximum.accumulate(bc)
    bench_dd = ((bc - bp) / bp * 100).tolist()

    monthly_excess = (monthly["excess_return"] * 100).tolist()
    monthly_labels = monthly["year_month"].astype(str).tolist()

    yearly_data = []
    for _, row in yearly.iterrows():
        exc = row["excess_return"]
        yearly_data.append({
            "year": str(row["year"]),
            "portfolio": f"{row['portfolio_return']:.2%}",
            "benchmark": f"{row['benchmark_return']:.2%}",
            "excess": f"{exc:.2%}",
            "exc_val": float(exc) * 100,
            "met": "✓" if exc >= 0.05 else ("~" if exc > 0 else "✗"),
        })

    # Monthly win per year
    monthly["year"] = monthly["year_month"].astype(str).str[:4]
    mw_data = []
    for year, grp in monthly.groupby("year"):
        wins = (grp["excess_return"] > 0).sum()
        mw_data.append({"year": year, "wins": int(wins), "total": len(grp),
                        "rate": f"{wins/len(grp):.0%}"})

    params_html = ""
    if strategy_params:
        params_html = "<h2>Strategy Parameters</h2><table>"
        for k, v in strategy_params.items():
            params_html += f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        params_html += "</table>"

    yearly_rows = ""
    for y in yearly_data:
        c = "#2ca02c" if y["met"] == "✓" else ("#ff7f0e" if y["met"] == "~" else "#d62728")
        yearly_rows += (f"<tr><td>{y['year']}</td><td>{y['portfolio']}</td>"
                        f"<td>{y['benchmark']}</td>"
                        f"<td style='color:{c}'>{y['excess']} {y['met']}</td></tr>")

    mw_rows = ""
    for m in mw_data:
        mw_rows += f"<tr><td>{m['year']}</td><td>{m['wins']}/{m['total']}</td><td>{m['rate']}</td></tr>"

    # Regime distribution
    regime_pct = daily["regime"].value_counts(normalize=True).to_dict()
    bull_pct = regime_pct.get("bull", 0) * 100
    bear_pct = regime_pct.get("bear", 0) * 100

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETF Trading System v2 — Backtest Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
       background: #f0f2f5; color: #333; }}
.container {{ max-width: 1300px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #1f77b4, #2c3e50); color: white; padding: 30px;
           border-radius: 12px; margin-bottom: 20px; text-align: center; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header p {{ opacity: 0.8; font-size: 14px; }}
.card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.card h2 {{ font-size: 16px; color: #1f77b4; margin-bottom: 15px; padding-bottom: 8px;
            border-bottom: 2px solid #e8e8e8; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }}
.metric {{ text-align: center; padding: 15px; background: linear-gradient(135deg, #f8f9fa, #e9ecef);
           border-radius: 8px; border: 1px solid #eee; }}
.metric .value {{ font-size: 26px; font-weight: 700; color: #1f77b4; }}
.metric .label {{ font-size: 12px; color: #888; margin-top: 4px; }}
.metric.green .value {{ color: #2ca02c; }}
.metric.red .value {{ color: #d62728; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid #eee; }}
th {{ background: #f8f9fa; color: #666; font-weight: 600; font-size: 12px; text-transform: uppercase; }}
td:first-child, th:first-child {{ text-align: left; }}
.chart {{ width: 100%; height: 420px; }}
.row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 900px) {{ .row {{ grid-template-columns: 1fr; }} }}
.pass {{ color: #2ca02c; font-weight: bold; }}
.fail {{ color: #d62728; font-weight: bold; }}
.warn {{ color: #ff7f0e; font-weight: bold; }}
.footer {{ text-align: center; color: #999; font-size: 12px; padding: 15px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px;
          font-weight: 600; }}
.badge-bull {{ background: #d4edda; color: #155724; }}
.badge-bear {{ background: #f8d7da; color: #721c24; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>ETF Trading System v2</h1>
  <p>5-Year Backtest (2021–2026) | CSI 1000 + All ETFs | Trend-Filtered Momentum Rotation</p>
  <p style="margin-top:8px;">
    <span class="badge badge-bull">Bull {bull_pct:.0f}%</span>
    <span class="badge badge-bear" style="margin-left:8px;">Bear {bear_pct:.0f}%</span>
  </p>
</div>

<div class="card">
  <h2>Performance Summary</h2>
  <div class="metrics">
    <div class="metric"><div class="value">{metrics['total_return']:.1%}</div><div class="label">Total Return</div></div>
    <div class="metric"><div class="value">{metrics['ann_return']:.1%}</div><div class="label">Ann. Return</div></div>
    <div class="metric"><div class="value">{metrics['ann_excess']:.2%}</div><div class="label">Ann. Excess</div></div>
    <div class="metric"><div class="value">{metrics['sharpe']:.2f}</div><div class="label">Sharpe Ratio</div></div>
    <div class="metric"><div class="value">{metrics['max_drawdown']:.1%}</div><div class="label">Max Drawdown</div></div>
    <div class="metric"><div class="value">{metrics['monthly_win_rate']:.1%}</div><div class="label">Monthly Win Rate</div></div>
    <div class="metric"><div class="value">{metrics['calmar']:.2f}</div><div class="label">Calmar Ratio</div></div>
    <div class="metric"><div class="value">{metrics['information_ratio']:.2f}</div><div class="label">Info Ratio</div></div>
  </div>
</div>

<div class="row">
  <div class="card"><h2>Cumulative Return</h2><div id="chart_cum" class="chart"></div></div>
  <div class="card"><h2>Drawdown</h2><div id="chart_dd" class="chart"></div></div>
</div>

<div class="row">
  <div class="card">
    <h2>Yearly Returns</h2>
    <table>{yearly_rows}</table>
    <div id="chart_yearly" class="chart" style="height:300px;"></div>
  </div>
  <div class="card">
    <h2>Monthly Win Rate by Year</h2>
    <table>{mw_rows}</table>
    <div id="chart_monthly" class="chart" style="height:300px;"></div>
  </div>
</div>

{params_html}

<div class="footer">
  ETF Trading System v2 | Data: Sina Finance + Baostock | Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</div>
</div>

<script>
Plotly.newPlot('chart_cum', [
  {{y: {port_cum}, x: {dates}, name: 'Strategy', type: 'scatter', line: {{color: '#1f77b4', width: 2}}}},
  {{y: {bench_cum}, x: {dates}, name: 'CSI 1000', type: 'scatter', line: {{color: '#ff7f0e', width: 2}}}}
], {{title: '', yaxis: {{tickformat: '.0%'}}, margin: {{t:10,b:40,l:50,r:20}}, hovermode: 'x',
     legend: {{x:0.01,y:0.99}}}});

Plotly.newPlot('chart_dd', [
  {{y: {port_dd}, x: {dates}, name: 'Strategy DD', type: 'scatter', fill: 'tozeroy',
    fillcolor: 'rgba(214,39,40,0.3)', line: {{color: '#d62728'}}}},
  {{y: {bench_dd}, x: {dates}, name: 'CSI 1000 DD', type: 'scatter', fill: 'tozeroy',
    fillcolor: 'rgba(255,152,150,0.2)', line: {{color: '#ff9896'}}}}
], {{title: '', margin: {{t:10,b:40,l:50,r:20}}, hovermode: 'x', legend: {{x:0.01,y:0.01}}}});

var yrData = {yearly_data};
Plotly.newPlot('chart_yearly', [
  {{y: yrData.map(function(d){{return d.exc_val;}}), x: yrData.map(function(d){{return d.year;}}),
    name: 'Excess', type: 'bar', marker: {{color: yrData.map(function(d){{return d.exc_val>=5?'#2ca02c':(d.exc_val>0?'#ff7f0e':'#d62728');}})}}}}
], {{title: '', yaxis: {{title: 'Excess (%)'}}, margin: {{t:10,b:40,l:50,r:20}}, showlegend: false}});

var excVals = {monthly_excess};
var excColors = excVals.map(function(v){{return v>0?'#2ca02c':'#d62728';}});
Plotly.newPlot('chart_monthly', [
  {{y: excVals, x: {monthly_labels}, type: 'bar', marker: {{color: excColors}}}}
], {{title: '', yaxis: {{title: 'Excess (%)'}}, margin: {{t:10,b:40,l:50,r:20}}, showlegend: false}});
</script>
</body></html>"""

    filepath = os.path.join(OUTPUT_DIR, "report.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML dashboard saved to: {filepath}")
    return filepath
