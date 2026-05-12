"""
Investment universe construction — v2.
Pool: CSI 1000 constituents + all ETFs
Filter: ST stocks, market cap < 2B, ETF daily turnover < 20M
"""
from data_fetcher import (
    fetch_csi1000_constituents, fetch_all_etfs, fetch_stock_fundamentals, _to_plain_code)


def build_universe():
    print("\n=== Building Investment Universe ===")

    csi1000_codes = set(fetch_csi1000_constituents())
    if not csi1000_codes:
        print("[ERROR] Cannot fetch CSI 1000 constituents.")
        return None

    print("  Fetching stock fundamentals...")
    fundamentals = fetch_stock_fundamentals(list(csi1000_codes))

    stock_info = {}
    removed_st, removed_mcap = [], []
    for code in csi1000_codes:
        plain = _to_plain_code(code)
        info = fundamentals.get(plain, {})
        name = info.get("name", "")
        total_share = info.get("total_share", 0)
        is_st = info.get("is_st", False)

        if is_st:
            removed_st.append(plain)
            continue
        if total_share > 0:
            if total_share < 20_000_000:
                removed_mcap.append(plain)
                continue
            min_est_mcap = total_share * 2
            if min_est_mcap < 2e9 and total_share * 100 < 2e9:
                removed_mcap.append(plain)
                continue

        stock_info[plain] = {"name": name, "total_share": total_share, "type": "stock"}

    stocks = sorted(stock_info.keys())
    print(f"  Stocks: {len(csi1000_codes)} CSI1000 -> {len(stocks)} after filter")
    print(f"    Removed ST: {len(removed_st)}, Market cap < 20B: {len(removed_mcap)}")

    all_etfs = fetch_all_etfs()
    etf_info, removed_etf_vol = {}, []
    for etf in all_etfs:
        code = _to_plain_code(etf["code"])
        name = etf.get("name", "")
        amount = etf.get("amount_yuan", 0)
        if amount > 0 and amount < 20_000_000:
            removed_etf_vol.append(code)
            continue
        etf_info[code] = {"name": name, "amount_yuan": amount, "type": "etf"}

    etfs = sorted(etf_info.keys())
    print(f"  ETFs: {len(all_etfs)} total -> {len(etfs)} after volume filter")
    print(f"    Removed low volume: {len(removed_etf_vol)}")

    all_symbols = stocks + etfs
    all_symbols = list(dict.fromkeys(all_symbols))

    universe = {"stocks": stocks, "etfs": etfs, "stock_info": stock_info,
                "etf_info": etf_info, "all_symbols": all_symbols}
    print(f"  Total universe: {len(all_symbols)} symbols ({len(stocks)} stocks + {len(etfs)} ETFs)")
    return universe
