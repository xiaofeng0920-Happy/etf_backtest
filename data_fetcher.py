"""
Data acquisition with caching — v2 (Sina Finance + baostock + Tushare hybrid).
Sina: ETF daily, index daily, real-time quotes
Baostock: stock daily with 前复权
AKShare: index constituents, ETF list
Tushare: stock daily (前复权), ETF daily, index daily, fundamentals
"""
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import baostock as bs
import akshare as ak
import tushare as ts
import pandas as pd
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_sina_session = None
_tushare_pro = None
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")


def _get_sina_session():
    global _sina_session
    if _sina_session is None:
        _sina_session = requests.Session()
        _sina_session.trust_env = False
        _sina_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "https://finance.sina.com.cn",
        })
    return _sina_session


def _cache_path(key):
    return os.path.join(CACHE_DIR, f"{key}.pkl")


def _load_cache(key, max_age_hours=72):
    path = _cache_path(key)
    if os.path.exists(path):
        age = (time.time() - os.path.getmtime(path)) / 3600
        if age < max_age_hours:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    return None


def _save_cache(key, data):
    with open(_cache_path(key), "wb") as f:
        pickle.dump(data, f)


def _to_sina_code(code):
    code = str(code).replace(".", "")
    if code.startswith("sh") or code.startswith("sz"):
        return code
    if code.startswith("6") or code.startswith("5"):
        return "sh" + code
    return "sz" + code


def _to_baostock_code(code):
    code = str(code).replace(".", "")
    if code.startswith("sh"):
        return "sh." + code[2:]
    if code.startswith("sz"):
        return "sz." + code[2:]
    if code.startswith("6") or code.startswith("5"):
        return "sh." + code
    return "sz." + code


def _to_plain_code(code):
    return str(code).replace(".", "").replace("sh", "").replace("sz", "")


def _get_tushare_pro():
    global _tushare_pro
    if _tushare_pro is None:
        ts.set_token(TUSHARE_TOKEN)
        _tushare_pro = ts.pro_api()
    return _tushare_pro


def _to_tushare_code(code):
    code = str(code).replace(".", "")
    if code.startswith("sh"):
        return code[2:] + ".SH"
    if code.startswith("sz"):
        return code[2:] + ".SZ"
    if code.startswith("6") or code.startswith("5"):
        return code + ".SH"
    return code + ".SZ"


# ============ CSI 1000 Constituents ============

def fetch_csi1000_constituents():
    cached = _load_cache("csi1000_constituents", max_age_hours=48)
    if cached is not None:
        return cached
    try:
        df = ak.index_stock_cons(symbol="000852")
        if "品种代码" in df.columns:
            codes = df["品种代码"].tolist()
        else:
            codes = df.iloc[:, 0].tolist()
        codes = [str(c).zfill(6) for c in codes if str(c) != "nan"]
        _save_cache("csi1000_constituents", codes)
        print(f"  [CSI 1000] {len(codes)} constituents")
        return codes
    except Exception as e:
        print(f"  [WARN] fetch_csi1000_constituents: {e}")
        return []


# ============ ETF Data ============

def fetch_all_etfs():
    cached = _load_cache("all_etfs", max_age_hours=48)
    if cached is not None:
        return cached
    try:
        df = ak.fund_etf_category_sina(symbol="ETF基金")
        records = []
        for _, row in df.iterrows():
            code = str(row.get("代码", "")).replace(".", "")
            name = str(row.get("名称", ""))
            volume = float(row.get("成交量", 0) or 0)
            amount = float(row.get("成交额", 0) or 0)
            price = float(row.get("最新价", 0) or 0)
            records.append({"code": code, "name": name,
                            "volume_shares": volume, "amount_yuan": amount, "price": price})
        _save_cache("all_etfs", records)
        print(f"  [ETF] {len(records)} ETFs fetched")
        return records
    except Exception as e:
        print(f"  [WARN] fetch_all_etfs: {e}")
        return []


# ============ Stock Fundamentals (Baostock) ============

def fetch_stock_fundamentals(codes):
    cached = _load_cache("stock_fundamentals", max_age_hours=24)
    if cached is not None:
        return cached

    result = {}
    bs.login()
    try:
        for i, code in enumerate(codes):
            bs_code = _to_baostock_code(code)
            try:
                rs = bs.query_stock_basic(bs_code)
                name, is_st = "", False
                if rs.error_code == "0":
                    df = rs.get_data()
                    if not df.empty:
                        name = str(df["code_name"].iloc[0])
                        is_st = "ST" in name.upper()

                total_share = 0.0
                for yr in [2026, 2025, 2024]:
                    for q in [1, 2, 3, 4]:
                        rs2 = bs.query_profit_data(bs_code, yr, q)
                        if rs2.error_code == "0":
                            df2 = rs2.get_data()
                            if not df2.empty and "totalShare" in df2.columns:
                                ts = float(df2["totalShare"].iloc[0])
                                if ts > 0:
                                    total_share = ts
                                    break
                    if total_share > 0:
                        break

                result[_to_plain_code(code)] = {
                    "name": name, "total_share": total_share, "is_st": is_st}
            except Exception:
                result[_to_plain_code(code)] = {"name": "", "total_share": 0.0, "is_st": False}

            if (i + 1) % 200 == 0:
                print(f"  ... fundamentals {i+1}/{len(codes)}")
    finally:
        bs.logout()

    _save_cache("stock_fundamentals", result)
    print(f"  [Fundamentals] {len(result)} stocks processed")
    return result


# ============ Daily KLine ============

def fetch_daily_sina(symbol, start_date=None, end_date=None, max_datalen=1500):
    """Fetch daily OHLCV from Sina (for ETFs and index)."""
    cache_key = f"sina_daily_{symbol}"
    cached = _load_cache(cache_key, max_age_hours=72)
    if cached is not None:
        df = cached
        if start_date and end_date:
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if not df.empty:
            return df

    try:
        url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={max_datalen}")
        r = _get_sina_session().get(url, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not data or not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.rename(columns={"day": "date", "open": "open", "high": "high",
                                "low": "low", "close": "close", "volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        df["amount"] = df["close"] * df["volume"]
        _save_cache(cache_key, df)
        if start_date and end_date:
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        return df
    except Exception:
        return pd.DataFrame()


def fetch_batch_etfs(symbols, start_date, end_date, max_workers=6):
    """Parallel ETF daily data fetch via Sina."""
    results, total, done = {}, len(symbols), 0
    sina_codes = {_to_sina_code(s): s for s in symbols}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_daily_sina, sc, start_date, end_date): orig
                   for sc, orig in sina_codes.items()}
        for fut in as_completed(futures):
            orig_sym = futures[fut]
            try:
                df = fut.result()
                if not df.empty:
                    results[orig_sym] = df
            except Exception:
                pass
            done += 1
            if done % 100 == 0:
                print(f"  ... ETFs {done}/{total} ({len(results)} with data)")
    print(f"  ETFs: {len(results)}/{total} with valid data")
    return results


def fetch_batch_stocks(symbols, start_date, end_date):
    """Serial stock daily data fetch via baostock (前复权)."""
    results, total = {}, len(symbols)
    bs.login()
    try:
        for i, sym in enumerate(symbols):
            cache_key = f"bs_daily_{sym}"
            cached = _load_cache(cache_key, max_age_hours=72)
            if cached is not None:
                df = cached
                s_dt = pd.Timestamp(start_date) if start_date else df["date"].min()
                e_dt = pd.Timestamp(end_date) if end_date else df["date"].max()
                df = df[(df["date"] >= s_dt) & (df["date"] <= e_dt)]
                if not df.empty:
                    results[sym] = df
                    if (i + 1) % 200 == 0:
                        print(f"  ... stocks {i+1}/{total} ({len(results)} valid, cached)")
                    continue

            bs_code = _to_baostock_code(sym)
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code, "date,open,high,low,close,volume,amount",
                    start_date=start_date, end_date=end_date,
                    frequency="d", adjustflag="2")
                if rs.error_code == "0":
                    df = rs.get_data()
                    if not df.empty:
                        df["date"] = pd.to_datetime(df["date"])
                        for c in ["open", "high", "low", "close", "volume", "amount"]:
                            if c in df.columns:
                                df[c] = pd.to_numeric(df[c], errors="coerce")
                        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                        _save_cache(cache_key, df)
                        results[sym] = df
            except Exception:
                pass
            if (i + 1) % 100 == 0:
                print(f"  ... stocks {i+1}/{total} ({len(results)} valid)")
    finally:
        bs.logout()
    print(f"  Stocks: {len(results)}/{total} with valid data")
    return results


# ============ Benchmark ============

def fetch_csi1000_index(start_date, end_date):
    """Fetch CSI 1000 index via Sina."""
    cached = _load_cache("csi1000_index", max_age_hours=48)
    if cached is not None:
        df = cached
        df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
        if not df.empty:
            return df

    df = fetch_daily_sina("sh000852")
    if df.empty:
        bs.login()
        rs = bs.query_history_k_data_plus(
            "sh.000852", "date,open,high,low,close,volume,amount",
            start_date=start_date, end_date=end_date, frequency="d", adjustflag="2")
        df = rs.get_data() if rs.error_code == "0" else pd.DataFrame()
        bs.logout()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    if not df.empty:
        _save_cache("csi1000_index", df)
    return df


# ============ Real-time Quotes (Sina) ============

def fetch_realtime_quotes(codes, batch_size=50):
    """Fetch real-time bid/ask prices from Sina for live trading signals.
    Returns dict: {code: {"name": str, "price": float, "open": float,
                           "high": float, "low": float, "volume": int,
                           "bid": float, "ask": float, "time": str}}
    """
    results = {}
    sina_map = {_to_sina_code(c): c for c in codes}
    sina_codes = list(sina_map.keys())

    for i in range(0, len(sina_codes), batch_size):
        batch = sina_codes[i:i + batch_size]
        try:
            url = f"https://hq.sinajs.cn/list={','.join(batch)}"
            r = _get_sina_session().get(url, timeout=15)
            if r.status_code != 200:
                continue
            text = r.text
            for line in text.strip().split("\n"):
                line = line.strip()
                if not line or "=" not in line:
                    continue
                parts = line.split('="', 1)
                if len(parts) != 2:
                    continue
                scode = parts[0].split("_str_")[-1]
                raw = parts[1].rstrip('";')
                fields = raw.split(",")
                if len(fields) < 32:
                    continue
                plain = _to_plain_code(scode)
                name = fields[0]
                try:
                    open_p, close_p = float(fields[1]), float(fields[2])
                    price = close_p if close_p > 0 else open_p
                    high = float(fields[4])
                    low = float(fields[5])
                    bid = float(fields[6])
                    ask = float(fields[7])
                    volume = int(float(fields[8]))
                    results[plain] = {"name": name, "price": price, "open": open_p,
                                      "high": high, "low": low, "volume": volume,
                                      "bid": bid, "ask": ask, "time": fields[31]}
                except (ValueError, IndexError):
                    continue
        except Exception:
            continue
    return results


def fetch_benchmark_index(code, start_date, end_date):
    """Fetch any index via Sina. Supports CSI 300 (sh000300), CSI 500 (sh000905), CSI 1000 (sh000852)."""
    sina_code_map = {"000300": "sh000300", "000905": "sh000905", "000852": "sh000852",
                     "399001": "sz399001", "000001": "sh000001"}
    if code in sina_code_map:
        code = sina_code_map[code]
    cache_key = f"sina_daily_{code}"
    return fetch_daily_sina(code, start_date, end_date)


# ============ Tushare Data Source ============

def _ts_adjust_prices(df):
    """Convert Tushare unadjusted prices to 前复权 using adj_factor.
    Tushare daily data: close = 后复权 / adj_factor.
    前复权 close = close * adj_factor / latest_adj_factor.
    """
    if df.empty or "adj_factor" not in df.columns:
        return df
    latest_adj = df["adj_factor"].iloc[-1] if len(df) > 0 else 1
    if latest_adj > 0:
        ratio = df["adj_factor"] / latest_adj
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col] * ratio
    return df


def fetch_daily_tushare(symbol, start_date, end_date):
    """Fetch daily OHLCV from Tushare (股票/ETF). Returns 前复权 prices."""
    cache_key = f"ts_daily_{symbol}"
    cached = _load_cache(cache_key, max_age_hours=72)
    if cached is not None:
        df = cached
        s_dt = pd.Timestamp(start_date) if start_date else df["date"].min()
        e_dt = pd.Timestamp(end_date) if end_date else df["date"].max()
        df = df[(df["date"] >= s_dt) & (df["date"] <= e_dt)]
        if not df.empty:
            return df

    try:
        pro = _get_tushare_pro()
        ts_code = _to_tushare_code(symbol)
        s = start_date.replace("-", "") if start_date else "20180101"
        e = end_date.replace("-", "") if end_date else "20261231"

        # Fetch daily data with adj_factor for 前复权
        df = pro.daily(ts_code=ts_code, start_date=s, end_date=e,
                       fields="ts_code,trade_date,open,high,low,close,vol,amount,adj_factor")
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close", "volume", "amount", "adj_factor"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = _ts_adjust_prices(df)
        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

        _save_cache(cache_key, df)
        if start_date and end_date:
            df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
        return df
    except Exception:
        return pd.DataFrame()


def fetch_index_tushare(code, start_date, end_date):
    """Fetch index daily from Tushare."""
    cache_key = f"ts_index_{code}"
    cached = _load_cache(cache_key, max_age_hours=72)
    if cached is not None:
        df = cached
        s_dt = pd.Timestamp(start_date) if start_date else df["date"].min()
        e_dt = pd.Timestamp(end_date) if end_date else df["date"].max()
        df = df[(df["date"] >= s_dt) & (df["date"] <= e_dt)]
        if not df.empty:
            return df

    try:
        pro = _get_tushare_pro()
        s = start_date.replace("-", "") if start_date else "20180101"
        e = end_date.replace("-", "") if end_date else "20261231"

        # Map common index codes to Tushare format
        ts_code_map = {
            "000852": "000852.SH", "000300": "000300.SH", "000905": "000905.SH",
            "399001": "399001.SZ", "000001": "000001.SH",
        }
        ts_code = ts_code_map.get(code, code + ".SH" if code.startswith(("0", "6")) else code + ".SZ")

        df = pro.index_daily(ts_code=ts_code, start_date=s, end_date=e,
                             fields="ts_code,trade_date,open,high,low,close,vol,amount")
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

        _save_cache(cache_key, df)
        if start_date and end_date:
            df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
        return df
    except Exception:
        return pd.DataFrame()


def fetch_batch_stocks_tushare(symbols, start_date, end_date, max_workers=8):
    """Parallel stock daily data via Tushare."""
    results, total, done = {}, len(symbols), 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_daily_tushare, s, start_date, end_date): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
                if not df.empty:
                    results[sym] = df
            except Exception:
                pass
            done += 1
            if done % 100 == 0:
                print(f"  ... Tushare stocks {done}/{total} ({len(results)} valid)")
    print(f"  [Tushare] Stocks: {len(results)}/{total} with valid data")
    return results


def fetch_batch_etfs_tushare(symbols, start_date, end_date, max_workers=8):
    """Parallel ETF daily data via Tushare."""
    results, total, done = {}, len(symbols), 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_daily_tushare, s, start_date, end_date): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
                if not df.empty:
                    results[sym] = df
            except Exception:
                pass
            done += 1
            if done % 100 == 0:
                print(f"  ... Tushare ETFs {done}/{total} ({len(results)} valid)")
    print(f"  [Tushare] ETFs: {len(results)}/{total} with valid data")
    return results


def fetch_benchmark_index_tushare(code, start_date, end_date):
    """Fetch benchmark index via Tushare."""
    return fetch_index_tushare(code, start_date, end_date)


# ============ Matrix Builders ============

def build_price_matrix(daily_data_dict):
    series = {}
    for sym, df in daily_data_dict.items():
        if df is None or df.empty:
            continue
        s = df.set_index("date")["close"]
        s = s[~s.index.duplicated()]
        series[sym] = s
    return pd.DataFrame(series).sort_index() if series else pd.DataFrame()


def build_volume_matrix(daily_data_dict):
    series = {}
    for sym, df in daily_data_dict.items():
        if df is None or df.empty:
            continue
        vol_col = "volume" if "volume" in df.columns else None
        if vol_col is None:
            continue
        s = df.set_index("date")[vol_col]
        s = s[~s.index.duplicated()]
        series[sym] = s
    return pd.DataFrame(series).sort_index() if series else pd.DataFrame()
