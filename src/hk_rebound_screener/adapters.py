from __future__ import annotations

import io
import random
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .strategy import _parse_datetime_series, _parse_hong_kong_datetime, normalize_code


HKEX_SECURITIES_URL = (
    "https://www.hkex.com.hk/eng/services/trading/securities/"
    "securitieslists/ListOfSecurities.xlsx"
)
YFINANCE_BATCH_SIZE = 100
YFINANCE_RETRY_DELAYS = (2.0, 5.0, 10.0)
YFINANCE_CACHE_DIR = Path("data") / "yfinance_cache"
AKSHARE_HISTORY_CACHE_FILE = Path("data") / "akshare_cache" / "hk_history.csv"


def fetch_hkex_security_master(
    ordinary_only: bool = True,
    include_name: bool = False,
) -> pd.DataFrame:
    import requests

    response = requests.get(HKEX_SECURITIES_URL, timeout=30)
    response.raise_for_status()
    frame = pd.read_excel(io.BytesIO(response.content), header=2)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Stock Code", "Category", "Sub-Category", "Board Lot"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"HKEX 证券主数据缺少字段: {sorted(missing)}")
    frame["code"] = frame["Stock Code"].astype(str).str.extract(r"(\d+)")[0].map(normalize_code)
    board_lot = frame["Board Lot"].astype(str).str.replace(",", "", regex=False)
    frame["lot_size"] = pd.to_numeric(board_lot, errors="coerce")
    equity = frame["Category"].astype(str).str.strip().eq("Equity")
    subcategory = frame["Sub-Category"].astype(str).str.strip()
    if ordinary_only:
        equity &= subcategory.str.match(r"Equity Securities \((Main Board|GEM)\)", na=False)
    else:
        equity &= subcategory.str.contains("Equity Securities", na=False)
    columns = ["code", "lot_size"]
    result = frame.loc[equity & frame["lot_size"].gt(0), columns].drop_duplicates("code")
    if include_name:
        names = frame.loc[equity & frame["lot_size"].gt(0), ["code", "Name of Securities"]].copy()
        names = names.rename(columns={"Name of Securities": "name"}).drop_duplicates("code")
        result = result.merge(names, on="code", how="left")
        result = result[["code", "name", "lot_size"]]
    return result


def _yahoo_symbol(code: str, market: str) -> str:
    normalized = normalize_code(code)
    if market.upper() == "HK" and normalized.isdigit():
        return f"{int(normalized):04d}.HK"
    return normalized


def fetch_us_common_stock_master() -> pd.DataFrame:
    """Build a common-stock-only US universe from Nasdaq Trader symbol files."""
    import requests

    sources = {
        "nasdaq": "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
        "other": "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt",
    }
    frames: list[pd.DataFrame] = []
    for source, url in sources.items():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text), sep="|", dtype=str)
        frame = frame.loc[~frame.iloc[:, 0].astype(str).str.startswith("File Creation Time")].copy()
        if source == "nasdaq":
            frame = frame.rename(columns={"Symbol": "code", "Security Name": "name"})
            frame["etf"] = frame.get("ETF", "")
            frame["test_issue"] = frame.get("Test Issue", "")
            frame["exchange"] = "NASDAQ"
        else:
            frame = frame.rename(columns={"NASDAQ Symbol": "code", "Security Name": "name"})
            frame["code"] = frame["code"].fillna(frame.get("ACT Symbol", ""))
            frame["etf"] = frame.get("ETF", "")
            frame["test_issue"] = frame.get("Test Issue", "")
            frame["exchange"] = frame.get("Exchange", "")
        frames.append(frame[["code", "name", "etf", "test_issue", "exchange"]])
    result = pd.concat(frames, ignore_index=True)
    result["code"] = result["code"].fillna("").map(normalize_code)
    result["name"] = result["name"].fillna("").astype(str).str.strip()
    result["etf"] = result["etf"].fillna("").astype(str).str.upper()
    result["test_issue"] = result["test_issue"].fillna("").astype(str).str.upper()
    excluded_name = result["name"].str.contains(
        r"WARRANT|RIGHT|UNIT|PREFERRED|DEPOSITARY|\bADS\b|NOTE|BOND|FUND|TRUST|ETF|BLANK CHECK",
        case=False,
        na=False,
    )
    valid_symbol = result["code"].str.match(r"^[A-Z][A-Z0-9.\-]*$", na=False)
    result = result.loc[
        valid_symbol & result["etf"].ne("Y") & result["test_issue"].ne("Y") & ~excluded_name,
        ["code", "name", "exchange"],
    ].copy()
    result["lot_size"] = 1.0
    result["enabled"] = True
    return result.drop_duplicates("code").sort_values("code").reset_index(drop=True)


def fetch_usd_hkd_rate() -> float:
    """Return the latest USD/HKD quote from Yahoo Finance."""
    import yfinance as yf

    history = yf.Ticker("HKD=X").history(period="5d", interval="1d", auto_adjust=False)
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if close.empty or float(close.iloc[-1]) <= 0:
        raise ValueError("无法取得 USD/HKD 汇率")
    return float(close.iloc[-1])


def enrich_yfinance_industry(
    universe: pd.DataFrame,
    market: str,
    cache_path: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fill industry metadata from Yahoo's paginated sector screens."""
    cache_file = pd.io.common.stringify_path(cache_path)
    cache = pd.DataFrame(columns=["code", "industry", "name"])
    try:
        cache = pd.read_csv(cache_file, dtype={"code": str})
    except FileNotFoundError:
        pass
    cache = cache.reindex(columns=["code", "industry", "name"], fill_value="")
    cache["code"] = cache["code"].map(normalize_code)
    cache = cache.drop_duplicates("code").set_index("code")
    result = universe.copy()
    result["industry"] = result["industry"].fillna("").astype(str)
    result["name"] = result["name"].fillna("").astype(str)
    missing = result["industry"].str.strip().eq("")
    if not refresh:
        cached_industry = result["code"].map(cache["industry"] if "industry" in cache else pd.Series(dtype=str))
        result.loc[missing, "industry"] = cached_industry.loc[missing].fillna("")
        result.loc[result["name"].str.strip().eq(""), "name"] = result["code"].map(cache["name"]).fillna("")
        missing = result["industry"].str.strip().eq("")
    if missing.any():
        import yfinance as yf
        from yfinance import EquityQuery
        from yfinance.screener.query import EQUITY_SCREENER_EQ_MAP

        allowed_codes = set(result.loc[missing, "code"])
        symbol_to_code = {_yahoo_symbol(code, market): code for code in allowed_codes}
        for code in allowed_codes:
            symbol_to_code[code.replace(".", "-")] = code
            symbol_to_code[code.replace(".", "/")] = code
        industries = sorted({
            str(industry)
            for values in EQUITY_SCREENER_EQ_MAP.get("industry", {}).values()
            for industry in values
        })
        region = "hk" if market.upper() == "HK" else "us"
        for index, industry in enumerate(industries, start=1):
            offset = 0
            total = None
            while total is None or offset < total:
                try:
                    query = EquityQuery("and", [
                        EquityQuery("eq", ["region", region]),
                        EquityQuery("eq", ["industry", industry]),
                    ])
                    payload = yf.screen(
                        query,
                        size=250,
                        offset=offset,
                        sortField="ticker",
                        sortAsc=True,
                    )
                    total = int(payload.get("total", 0))
                    quotes = payload.get("quotes", [])
                except Exception as error:  # noqa: BLE001 - other industries remain usable
                    print(f"WARN industry screen {market}/{industry}: {error}", flush=True)
                    break
                if not quotes:
                    break
                for quote in quotes:
                    symbol = str(quote.get("symbol", "")).upper()
                    code = symbol_to_code.get(symbol)
                    if code is None and market.upper() == "HK" and symbol.endswith(".HK"):
                        code = normalize_code(symbol[:-3])
                    if code not in allowed_codes:
                        continue
                    result.loc[result["code"].eq(code), "industry"] = industry
                    if result.loc[result["code"].eq(code), "name"].iloc[0].strip() == "":
                        result.loc[result["code"].eq(code), "name"] = str(
                            quote.get("shortName") or quote.get("longName") or code
                        ).strip()
                offset += len(quotes)
                if len(quotes) < 250:
                    break
            print(f"行业分类查询: {index}/{len(industries)} {industry}", flush=True)
    result["industry"] = result["industry"].fillna("").astype(str).str.strip()
    result["name"] = result["name"].fillna("").astype(str).str.strip()
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    result[["code", "industry", "name"]].drop_duplicates("code").to_csv(
        cache_file,
        index=False,
        encoding="utf-8-sig",
    )
    missing_count = int(result["industry"].eq("").sum())
    if missing_count:
        print(f"行业元数据缺失: {missing_count} 条；这些股票将按严格规则排除", flush=True)
    return result


def build_full_universe(
    market: str,
    seed_path: str,
    metadata_cache: str,
    refresh_metadata: bool = False,
) -> pd.DataFrame:
    market = market.upper()
    if market == "HK":
        master = fetch_hkex_security_master(ordinary_only=True, include_name=True)
    elif market == "US":
        master = fetch_us_common_stock_master()
    else:
        raise ValueError(f"暂不支持 market={market}")
    try:
        seed = pd.read_csv(seed_path, dtype={"code": str})
        seed["code"] = seed["code"].map(normalize_code)
    except FileNotFoundError:
        seed = pd.DataFrame(columns=["code", "name", "industry"])
    seed = seed.reindex(columns=["code", "name", "industry"], fill_value="")
    seed = seed.drop_duplicates("code").rename(columns={"name": "seed_name", "industry": "seed_industry"})
    result = master.merge(seed, on="code", how="left")
    result["name"] = result["seed_name"].where(
        result["seed_name"].fillna("").str.strip().ne(""),
        result["name"],
    ).replace("", pd.NA).fillna(result["code"])
    result["industry"] = result["seed_industry"].fillna("")
    result["enabled"] = True
    result = result[["code", "name", "industry", "lot_size", "enabled"]]
    return enrich_yfinance_industry(result, market, metadata_cache, refresh=refresh_metadata)


def fetch_yfinance_history_bulk(
    codes: list[str],
    market: str,
    start_date: date,
    end_date: date,
    auto_adjust: bool = True,
    chunk_size: int = YFINANCE_BATCH_SIZE,
) -> pd.DataFrame:
    """Fetch many US/HK daily bars in chunks to reduce per-symbol requests."""
    import yfinance as yf

    columns = ["date", "code", "open", "close", "volume"]
    cache_columns = columns + ["auto_adjust", "adjustment_factor"]
    empty = pd.DataFrame(columns=columns)
    rows: list[pd.DataFrame] = []
    code_map = {_yahoo_symbol(code, market): normalize_code(code) for code in codes}
    symbols = list(code_map)
    requested_codes = set(code_map.values())
    cached = pd.DataFrame(columns=cache_columns)
    cache_file: Path | None = None

    def parse_bool(value: object) -> bool | None:
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "auto", "adjusted"}:
            return True
        if text in {"0", "false", "no", "n", "raw", "unadjusted"}:
            return False
        return None

    if market.upper() in {"US", "HK"}:
        cache_file = YFINANCE_CACHE_DIR / f"{market.lower()}_history.csv"
        try:
            cached = pd.read_csv(cache_file, dtype={"code": str})
            required = set(cache_columns)
            if not required.issubset(cached.columns):
                raise ValueError(f"缺少字段: {sorted(required.difference(cached.columns))}")
            cached = cached[cache_columns].copy()
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.normalize()
            cached["code"] = cached["code"].map(normalize_code)
            for column in ["open", "close", "volume"]:
                cached[column] = pd.to_numeric(cached[column], errors="coerce")
            cached["auto_adjust"] = cached["auto_adjust"].map(parse_bool)
            cached["adjustment_factor"] = pd.to_numeric(cached["adjustment_factor"], errors="coerce")
            cached = cached.dropna(
                subset=["date", "code", "close", "volume", "auto_adjust", "adjustment_factor"]
            )
            if cached.empty:
                raise ValueError("缓存没有有效行情")
            cached = cached.drop_duplicates(["date", "code"], keep="last")
            requested_mask = cached["code"].isin(requested_codes)
            cached = cached.loc[
                ~requested_mask | cached["auto_adjust"].eq(bool(auto_adjust))
            ].copy()
            print(
                f"使用 yfinance 缓存: {cache_file}；已校验复权口径 auto_adjust={bool(auto_adjust)}",
                flush=True,
            )
        except FileNotFoundError:
            pass
        except Exception as error:  # noqa: BLE001 - a bad cache falls back to a full download
            cached = pd.DataFrame(columns=cache_columns)
            print(f"WARN yfinance cache {cache_file}: {error}; fallback to full download", flush=True)

    download_starts: dict[str, date] = {}
    for code in requested_codes:
        code_cache = cached.loc[cached["code"].eq(code)]
        if code_cache.empty:
            download_starts[code] = start_date
            continue
        cache_min = code_cache["date"].min().date()
        cache_max = code_cache["date"].max().date()
        if cache_min > start_date:
            download_starts[code] = start_date
        elif cache_max < end_date:
            download_starts[code] = max(start_date, cache_max - timedelta(days=6))
        else:
            download_starts[code] = max(start_date, end_date - timedelta(days=6))
    if download_starts:
        print(
            f"yfinance 各股票请求窗口: {min(download_starts.values())} 至 {end_date}",
            flush=True,
        )

    def extract_rows(frame: pd.DataFrame, batch: list[str]) -> tuple[list[pd.DataFrame], list[str]]:
        extracted: list[pd.DataFrame] = []
        missing: list[str] = []
        if frame.empty:
            return extracted, list(batch)
        for symbol in batch:
            try:
                if isinstance(frame.columns, pd.MultiIndex):
                    if symbol in frame.columns.get_level_values(0):
                        subframe = frame[symbol]
                    else:
                        subframe = frame.xs(symbol, level=1, axis=1)
                else:
                    subframe = frame
                if "Close" not in subframe or "Volume" not in subframe:
                    missing.append(symbol)
                    continue
                subframe = subframe.copy()
                subframe["date"] = _yfinance_local_date(subframe.index).values
                subframe["code"] = code_map[symbol]
                subframe["open"] = pd.to_numeric(subframe.get("Open"), errors="coerce")
                subframe["close"] = pd.to_numeric(subframe["Close"], errors="coerce")
                subframe["volume"] = pd.to_numeric(subframe["Volume"], errors="coerce")
                if "Adj Close" in subframe:
                    adjusted_close = pd.to_numeric(subframe["Adj Close"], errors="coerce")
                    subframe["adjustment_factor"] = adjusted_close / subframe["close"]
                else:
                    subframe["adjustment_factor"] = 1.0
                subframe["auto_adjust"] = bool(auto_adjust)
                subframe = subframe[cache_columns].dropna(
                    subset=["date", "code", "close", "volume"]
                )
                subframe["adjustment_factor"] = pd.to_numeric(
                    subframe["adjustment_factor"], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
                if subframe.empty:
                    missing.append(symbol)
                else:
                    extracted.append(subframe)
            except Exception as error:  # noqa: BLE001 - one bad symbol must not abort a batch
                print(f"WARN bulk symbol {symbol}: {error}", flush=True)
                missing.append(symbol)
        return extracted, missing

    def download_batch(
        batch: list[str],
        starts: dict[str, date],
    ) -> tuple[list[pd.DataFrame], list[str]]:
        remaining = list(batch)
        collected: list[pd.DataFrame] = []
        batch_start = min(starts[code_map[symbol]] for symbol in batch)
        for attempt in range(len(YFINANCE_RETRY_DELAYS) + 1):
            try:
                frame = yf.download(
                    tickers=remaining,
                    start=batch_start,
                    end=end_date + timedelta(days=1),
                    interval="1d",
                    auto_adjust=auto_adjust,
                    actions=False,
                    group_by="ticker",
                    threads=False,
                    timeout=30,
                    progress=False,
                )
                batch_rows, missing = extract_rows(frame, remaining)
                collected.extend(batch_rows)
                remaining = missing
                if not remaining:
                    return collected, []
            except Exception as error:  # noqa: BLE001 - retry/split keeps other batches usable
                print(
                    f"WARN bulk history batch {len(remaining)} symbols, attempt {attempt + 1}: {error}",
                    flush=True,
                )
            if attempt < len(YFINANCE_RETRY_DELAYS):
                delay = YFINANCE_RETRY_DELAYS[attempt] * random.uniform(0.9, 1.1)
                time.sleep(delay)
        if len(remaining) > 1:
            middle = len(remaining) // 2
            print(f"WARN split yfinance batch: {len(remaining)} -> {middle}+{len(remaining) - middle}", flush=True)
            left_rows, left_missing = download_batch(remaining[:middle], starts)
            right_rows, right_missing = download_batch(remaining[middle:], starts)
            return collected + left_rows + right_rows, left_missing + right_missing
        return collected, remaining

    def download_symbols(symbols_to_fetch: list[str], starts: dict[str, date]) -> list[str]:
        failed: list[str] = []
        grouped: dict[date, list[str]] = {}
        for symbol in symbols_to_fetch:
            grouped.setdefault(starts[code_map[symbol]], []).append(symbol)
        completed = 0
        for group_symbols in grouped.values():
            for batch_start in range(0, len(group_symbols), chunk_size):
                batch = group_symbols[batch_start : batch_start + chunk_size]
                batch_rows, batch_failed = download_batch(batch, starts)
                rows.extend(batch_rows)
                failed.extend(batch_failed)
                completed += len(batch)
                print(f"批量历史行情: {completed}/{len(symbols_to_fetch)}")
                if completed < len(symbols_to_fetch):
                    time.sleep(random.uniform(0.5, 1.5))
        return failed

    failed_symbols = download_symbols(symbols, download_starts)
    if failed_symbols:
        print(f"WARN yfinance failed symbols ({len(failed_symbols)}): {','.join(failed_symbols)}", flush=True)

    fresh = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cache_columns)
    if market.upper() in {"US", "HK"}:
        factor_changed: set[str] = set()
        if not cached.empty and not fresh.empty:
            cached_factors = cached.merge(
                fresh[["date", "code", "adjustment_factor"]],
                on=["date", "code"],
                how="inner",
                suffixes=("_cached", "_fresh"),
            )
            changed = ~np.isclose(
                cached_factors["adjustment_factor_cached"],
                cached_factors["adjustment_factor_fresh"],
                rtol=1e-4,
                atol=1e-6,
                equal_nan=False,
            )
            factor_changed = set(cached_factors.loc[changed, "code"])
        if factor_changed:
            print(
                f"WARN yfinance 复权因子变化，重新同步: {','.join(sorted(factor_changed))}",
                flush=True,
            )
            cached = cached.loc[~cached["code"].isin(factor_changed)].copy()
            fresh = fresh.loc[~fresh["code"].isin(factor_changed)].copy()
            resync_starts = {code: start_date for code in factor_changed}
            resync_symbols = [symbol for symbol, code in code_map.items() if code in factor_changed]
            failed_symbols.extend(download_symbols(resync_symbols, resync_starts))
            fresh = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cache_columns)
        combined = pd.concat([cached, fresh], ignore_index=True)
        if not combined.empty:
            combined = combined.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"])
            try:
                assert cache_file is not None
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                combined.to_csv(cache_file, index=False, encoding="utf-8-sig")
            except Exception as error:  # noqa: BLE001 - cache write failure must not stop a scan
                print(f"WARN yfinance cache write {cache_file}: {error}", flush=True)
        fresh = combined[columns]
    if fresh.empty:
        return empty
    return fresh.loc[
        fresh["code"].isin(requested_codes)
        & fresh["date"].ge(pd.Timestamp(start_date))
        & fresh["date"].le(pd.Timestamp(end_date))
    ].sort_values(["code", "date"]).reset_index(drop=True)


def build_full_market_prices(
    codes: list[str],
    market: str,
    history_days: int,
    auto_adjust: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    today = date.today()
    start = today - timedelta(days=history_days * 2)
    history = fetch_yfinance_history_bulk(codes, market, start, today, auto_adjust=auto_adjust)
    if market.upper() == "HK":
        spot = fetch_akshare_spot()
        spot = spot.loc[spot["code"].isin({normalize_code(code) for code in codes})].copy()
        prices = _merge_hk_history_and_spot(history, spot)
    else:
        latest = history.sort_values(["code", "date"]).drop_duplicates("code", keep="last")
        spot = latest[["date", "code", "close", "volume"]].copy()
        spot["timestamp"] = (
            pd.to_datetime(spot["date"]).dt.tz_localize("America/New_York")
            + pd.Timedelta(hours=16)
        )
        spot["name"] = spot["code"]
        prices = history
    return prices.sort_values(["code", "date"]).reset_index(drop=True), spot.reset_index(drop=True)


def _merge_hk_history_and_spot(history: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """Merge HK daily bars without letting a stale snapshot replace a complete bar."""
    if history.empty:
        return spot[["date", "code", "close", "volume"]].copy()
    if spot.empty:
        return history.copy()

    daily = history.copy()
    daily["date"] = daily["date"].map(_parse_hong_kong_datetime).dt.normalize()
    daily["code"] = daily["code"].map(normalize_code)
    daily = daily.dropna(subset=["date", "code", "close", "volume"])

    snapshots = spot.copy()
    snapshots["timestamp"] = snapshots["timestamp"].map(_parse_hong_kong_datetime)
    snapshots["date"] = snapshots["timestamp"].dt.normalize()
    snapshots["code"] = snapshots["code"].map(normalize_code)
    snapshots["close"] = pd.to_numeric(snapshots["close"], errors="coerce")
    snapshots["volume"] = pd.to_numeric(snapshots["volume"], errors="coerce")
    snapshots = snapshots.dropna(subset=["timestamp", "date", "code", "close", "volume"])
    if snapshots.empty:
        return daily
    snapshots = snapshots.sort_values(["code", "date", "timestamp"]).drop_duplicates(
        ["code", "date"], keep="last"
    )

    latest_daily = daily.groupby("code")["date"].max().to_dict()
    current_hk_date = pd.Timestamp.now(tz="Asia/Hong_Kong").date()
    accepted: list[pd.Series] = []
    for _, snapshot in snapshots.iterrows():
        code = snapshot["code"]
        snapshot_date = snapshot["date"]
        daily_date = latest_daily.get(code)
        if daily_date is not None and snapshot_date < daily_date:
            continue
        if daily_date is not None and snapshot_date == daily_date:
            same_day = daily.loc[(daily["code"] == code) & (daily["date"] == snapshot_date)]
            complete = not same_day.empty and same_day[["close", "volume"]].notna().all(axis=None)
            if complete and snapshot_date.date() != current_hk_date:
                continue
            if complete and "timestamp" in same_day:
                daily_timestamp = same_day["timestamp"].map(_parse_hong_kong_datetime).max()
                if pd.notna(daily_timestamp) and snapshot["timestamp"] <= daily_timestamp:
                    continue
        accepted.append(snapshot)

    if not accepted:
        return daily
    accepted_frame = pd.DataFrame(accepted)
    live_rows = accepted_frame[["date", "code", "close", "volume"]]
    turnover_column = "turnover_hkd" if "turnover_hkd" in accepted_frame else "turnover"
    if turnover_column in accepted_frame:
        live_rows["turnover"] = pd.to_numeric(
            accepted_frame[turnover_column], errors="coerce"
        ).to_numpy()
    return pd.concat([daily, live_rows], ignore_index=True).drop_duplicates(
        ["date", "code"], keep="last"
    )


def fetch_akshare_spot() -> pd.DataFrame:
    import akshare as ak

    frame = ak.stock_hk_spot()
    rename = {
        "日期时间": "timestamp",
        "代码": "code",
        "中文名称": "name",
        "最新价": "close",
        "昨收": "prev_close",
        "成交量": "volume",
        "成交额": "turnover_hkd",
    }
    frame = frame.rename(columns=rename)
    required = {"timestamp", "code", "name", "close", "prev_close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"AKShare 港股实时接口缺少字段: {sorted(missing)}")
    frame["code"] = frame["code"].map(normalize_code)
    frame["timestamp"] = frame["timestamp"].map(_parse_hong_kong_datetime)
    for column in ("close", "prev_close", "volume", "turnover_hkd"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = frame["timestamp"].dt.normalize()
    return frame.dropna(subset=["date", "code", "close", "volume"])


def fetch_akshare_history(
    codes: list[str],
    start_date: date,
    end_date: date,
    adjust: str = "",
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    import akshare as ak

    columns = ["date", "code", "close", "volume"]
    cache_columns = columns + ["adjust"]
    adjust_key = str(adjust or "").strip()
    cached = pd.DataFrame(columns=cache_columns)
    try:
        cached = pd.read_csv(AKSHARE_HISTORY_CACHE_FILE, dtype={"code": str})
        required = set(cache_columns)
        if not required.issubset(cached.columns):
            raise ValueError(f"缺少字段: {sorted(required.difference(cached.columns))}")
        cached = cached[cache_columns].copy()
        cached["date"] = cached["date"].map(_parse_hong_kong_datetime).dt.normalize()
        cached["code"] = cached["code"].map(normalize_code)
        for column in ("close", "volume"):
            cached[column] = pd.to_numeric(cached[column], errors="coerce")
        cached["adjust"] = cached["adjust"].fillna("").astype(str).str.strip()
        cached = cached.loc[cached["adjust"].eq(adjust_key)].dropna(
            subset=["date", "code", "close", "volume"]
        )
    except FileNotFoundError:
        pass
    except Exception as error:  # noqa: BLE001 - invalid cache falls back to AKShare
        cached = pd.DataFrame(columns=cache_columns)
        print(f"WARN AKShare history cache {AKSHARE_HISTORY_CACHE_FILE}: {error}; fallback to full download")

    requested_codes = [normalize_code(code) for code in codes]
    download_starts: dict[str, date] = {}
    for code in requested_codes:
        code_cache = cached.loc[cached["code"].eq(code)]
        if code_cache.empty:
            download_starts[code] = start_date
            continue
        cache_min = code_cache["date"].min().date()
        cache_max = code_cache["date"].max().date()
        if cache_min > start_date:
            download_starts[code] = start_date
        elif cache_max < end_date:
            download_starts[code] = max(start_date, cache_max - timedelta(days=6))

    rows: list[pd.DataFrame] = []
    for code in requested_codes:
        code_start = download_starts.get(code)
        if code_start is None:
            continue
        try:
            frame = ak.stock_hk_daily(
                symbol=code,
                adjust=adjust,
            ).rename(columns={"date": "date", "close": "close", "volume": "volume"})
            frame["date"] = frame["date"].map(_parse_hong_kong_datetime).dt.normalize()
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
            frame = frame.loc[
                (frame["date"].dt.date >= code_start) & (frame["date"].dt.date <= end_date)
            ].copy()
            frame["code"] = code
            frame["adjust"] = adjust_key
            rows.append(frame[cache_columns].dropna(subset=["date", "code", "close", "volume"]))
        except Exception as error:  # noqa: BLE001 - one bad symbol must not abort a batch
            print(f"WARN history {code}: {error}")
        time.sleep(pause_seconds)
    fresh = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cache_columns)
    combined = pd.concat([cached, fresh], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(["date", "code", "adjust"], keep="last").sort_values(
            ["code", "date"]
        )
        try:
            AKSHARE_HISTORY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(AKSHARE_HISTORY_CACHE_FILE, index=False, encoding="utf-8-sig")
        except Exception as error:  # noqa: BLE001 - cache write failure must not stop a scan
            print(f"WARN AKShare history cache write {AKSHARE_HISTORY_CACHE_FILE}: {error}")
    return combined.loc[
        combined["code"].isin(set(requested_codes))
        & combined["adjust"].eq(adjust_key)
        & combined["date"].ge(pd.Timestamp(start_date))
        & combined["date"].le(pd.Timestamp(end_date)),
        columns,
    ].reset_index(drop=True)


def fetch_akshare_news(codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import akshare as ak

    news_rows: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    for code in codes:
        try:
            frame = ak.stock_news_em(symbol=normalize_code(code))
            frame = frame.rename(
                columns={
                    "关键词": "code",
                    "新闻标题": "title",
                    "新闻内容": "body",
                    "发布时间": "published_at",
                    "新闻链接": "url",
                }
            )
            frame["code"] = normalize_code(code)
            frame["published_at"] = _parse_datetime_series(frame["published_at"])
            frame["title"] = frame["title"].fillna("").astype(str)
            frame["body"] = frame["body"].fillna("").astype(str)
            news_rows.append(frame[["code", "published_at", "title", "body", "url"]])
            statuses.append({"code": normalize_code(code), "news_fetch_ok": True})
        except Exception as error:  # noqa: BLE001 - status is used to fail closed
            print(f"WARN news {code}: {error}")
            statuses.append({"code": normalize_code(code), "news_fetch_ok": False})
    news = pd.concat(news_rows, ignore_index=True) if news_rows else pd.DataFrame(
        columns=["code", "published_at", "title", "body", "url"]
    )
    return news, pd.DataFrame(statuses)


def build_live_prices(
    codes: list[str],
    history_days: int,
    adjust: str = "",
    pause_seconds: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    today = date.today()
    start = today - timedelta(days=history_days * 2)
    history = fetch_akshare_history(codes, start, today, adjust=adjust, pause_seconds=pause_seconds)
    spot = fetch_akshare_spot()
    spot = spot.loc[spot["code"].isin({normalize_code(code) for code in codes})].copy()
    if "turnover_hkd" in spot:
        spot["turnover"] = pd.to_numeric(spot["turnover_hkd"], errors="coerce")
    prices = _merge_hk_history_and_spot(history, spot)
    if "turnover" in prices:
        prices["turnover"] = pd.to_numeric(prices["turnover"], errors="coerce")
    return prices.reset_index(drop=True), spot


def _yfinance_local_date(index: pd.Index) -> pd.Series:
    values = pd.to_datetime(index, errors="coerce")
    if getattr(values, "tz", None) is not None:
        values = values.tz_localize(None)
    return pd.Series(values).dt.normalize()


def _yfinance_news_datetime(value: object) -> pd.Timestamp:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        stamp = pd.to_datetime(value, unit="s", errors="coerce", utc=True)
        if pd.isna(stamp):
            return pd.NaT
        return stamp.tz_convert("Asia/Hong_Kong").tz_localize(None)
    return _parse_hong_kong_datetime(value)


def fetch_yfinance_history(
    codes: list[str],
    start_date: date,
    end_date: date,
    auto_adjust: bool = True,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Fetch US daily bars from Yahoo Finance through yfinance."""
    import yfinance as yf

    rows: list[pd.DataFrame] = []
    for code in codes:
        symbol = normalize_code(code)
        try:
            frame = yf.Ticker(symbol).history(
                start=start_date,
                end=end_date + timedelta(days=1),
                interval="1d",
                auto_adjust=auto_adjust,
                actions=False,
            )
            if frame.empty:
                continue
            frame = frame.reset_index()
            date_column = "Date" if "Date" in frame.columns else "Datetime"
            frame["date"] = _yfinance_local_date(frame[date_column].tolist()).values
            frame["code"] = symbol
            frame["close"] = pd.to_numeric(frame["Close"], errors="coerce")
            frame["volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
            frame["open"] = pd.to_numeric(frame["Open"], errors="coerce")
            rows.append(frame[["date", "code", "open", "close", "volume"]])
        except Exception as error:  # noqa: BLE001 - one bad symbol must not abort a batch
            print(f"WARN US history {symbol}: {error}")
        time.sleep(pause_seconds)
    if not rows:
        return pd.DataFrame(columns=["date", "code", "open", "close", "volume"])
    return pd.concat(rows, ignore_index=True).dropna(subset=["date", "code", "close", "volume"])


def fetch_yfinance_spot(
    codes: list[str],
    auto_adjust: bool = True,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Fetch the latest available intraday bar; fall back to the latest daily bar."""
    import yfinance as yf

    rows: list[dict[str, object]] = []
    for code in codes:
        symbol = normalize_code(code)
        try:
            ticker = yf.Ticker(symbol)
            frame = ticker.history(period="1d", interval="1m", auto_adjust=auto_adjust, prepost=False)
            intraday = not frame.empty
            if frame.empty:
                frame = ticker.history(period="5d", interval="1d", auto_adjust=auto_adjust, actions=False)
            if frame.empty:
                continue
            valid = frame.dropna(subset=["Close", "Volume"]).copy()
            if valid.empty:
                continue
            if intraday:
                local_dates = _yfinance_local_date(valid.index)
                latest_date = local_dates.max()
                session = valid.loc[local_dates.to_numpy() == latest_date]
                row = session.iloc[-1]
                timestamp = pd.Timestamp(session.index[-1])
                volume = pd.to_numeric(session["Volume"], errors="coerce").sum(min_count=1)
            else:
                row = valid.iloc[-1]
                timestamp = pd.Timestamp(valid.index[-1])
                volume = pd.to_numeric(pd.Series([row["Volume"]]), errors="coerce").iloc[0]
            rows.append({
                "timestamp": timestamp,
                "date": _yfinance_local_date([timestamp]).iloc[0],
                "code": symbol,
                "name": symbol,
                "close": float(row["Close"]),
                "volume": float(volume),
            })
        except Exception as error:  # noqa: BLE001 - status is represented by returned rows
            print(f"WARN US spot {symbol}: {error}")
        time.sleep(pause_seconds)
    return pd.DataFrame(rows, columns=["timestamp", "date", "code", "name", "close", "volume"])


def fetch_yfinance_news(codes: list[str], count: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Yahoo Finance news and normalize both old and new yfinance payloads."""
    import yfinance as yf

    news_rows: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    for code in codes:
        symbol = normalize_code(code)
        try:
            ticker = yf.Ticker(symbol)
            if hasattr(ticker, "get_news"):
                items = ticker.get_news(count=count)
            else:
                items = ticker.news
            for item in items or []:
                content = item.get("content", item)
                title = content.get("title", item.get("title", ""))
                body = content.get("summary", item.get("summary", ""))
                published = content.get("pubDate", item.get("providerPublishTime"))
                canonical = content.get("canonicalUrl", {})
                url = canonical.get("url", "") if isinstance(canonical, dict) else content.get("link", "")
                news_rows.append({
                    "code": symbol,
                    "published_at": _yfinance_news_datetime(published),
                    "title": title or "",
                    "body": body or "",
                    "url": url or "",
                })
            statuses.append({"code": symbol, "news_fetch_ok": True})
        except Exception as error:  # noqa: BLE001 - status is used to fail closed
            print(f"WARN US news {symbol}: {error}")
            statuses.append({"code": symbol, "news_fetch_ok": False})
        time.sleep(0.15)
    news = pd.DataFrame(news_rows, columns=["code", "published_at", "title", "body", "url"])
    return news, pd.DataFrame(statuses)


def build_us_live_prices(
    codes: list[str],
    history_days: int,
    auto_adjust: bool = True,
    pause_seconds: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    today = date.today()
    start = today - timedelta(days=history_days * 2)
    history = fetch_yfinance_history(
        codes,
        start,
        today,
        auto_adjust=auto_adjust,
        pause_seconds=pause_seconds,
    )
    spot = fetch_yfinance_spot(codes, auto_adjust=auto_adjust, pause_seconds=pause_seconds)
    live_rows = spot[["date", "code", "close", "volume"]].copy()
    prices = pd.concat([history, live_rows], ignore_index=True)
    prices = prices.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"])
    return prices.reset_index(drop=True), spot


def enrich_candidate_fundamentals(codes: list[str], market: str) -> pd.DataFrame:
    """批量或按需为初筛通过的候选股获取总市值、PE与PB指标。"""
    import yfinance as yf

    rows: list[dict[str, Any]] = []
    for code in codes:
        symbol = _yahoo_symbol(code, market)
        norm_code = normalize_code(code)
        market_cap = None
        pe = None
        pb = None
        try:
            ticker = yf.Ticker(symbol)
            fast = getattr(ticker, "fast_info", None)
            if fast:
                market_cap = getattr(fast, "market_cap", None)
            info = getattr(ticker, "info", {}) or {}
            if not market_cap:
                market_cap = info.get("marketCap")
            pe = info.get("trailingPE") or info.get("forwardPE")
            pb = info.get("priceToBook")
        except Exception as error:  # noqa: BLE001
            print(f"WARN 获取估值指标失败 {symbol}: {error}")
        rows.append({
            "code": norm_code,
            "market_cap": market_cap,
            "pe": pe,
            "pb": pb,
        })
    return pd.DataFrame(rows)
