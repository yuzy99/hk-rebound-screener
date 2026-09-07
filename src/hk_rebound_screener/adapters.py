from __future__ import annotations

import io
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .strategy import normalize_code


HKEX_SECURITIES_URL = (
    "https://www.hkex.com.hk/eng/services/trading/securities/"
    "securitieslists/ListOfSecurities.xlsx"
)


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
    chunk_size: int = 200,
) -> pd.DataFrame:
    """Fetch many US/HK daily bars in chunks to reduce per-symbol requests."""
    import yfinance as yf

    rows: list[pd.DataFrame] = []
    code_map = {_yahoo_symbol(code, market): normalize_code(code) for code in codes}
    symbols = list(code_map)
    for start in range(0, len(symbols), chunk_size):
        batch = symbols[start : start + chunk_size]
        try:
            frame = yf.download(
                tickers=batch,
                start=start_date,
                end=end_date + timedelta(days=1),
                interval="1d",
                auto_adjust=auto_adjust,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as error:  # noqa: BLE001 - keep other chunks usable
            print(f"WARN bulk history {start + 1}-{start + len(batch)}: {error}")
            continue
        if frame.empty:
            continue
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
                    continue
                subframe = subframe.copy()
                subframe["date"] = _yfinance_local_date(subframe.index).values
                subframe["code"] = code_map[symbol]
                subframe["open"] = pd.to_numeric(subframe.get("Open"), errors="coerce")
                subframe["close"] = pd.to_numeric(subframe["Close"], errors="coerce")
                subframe["volume"] = pd.to_numeric(subframe["Volume"], errors="coerce")
                rows.append(subframe[["date", "code", "open", "close", "volume"]])
            except Exception as error:  # noqa: BLE001 - one bad symbol must not abort a batch
                print(f"WARN bulk symbol {symbol}: {error}")
        print(f"批量历史行情: {min(start + len(batch), len(symbols))}/{len(symbols)}")
    if not rows:
        return pd.DataFrame(columns=["date", "code", "open", "close", "volume"])
    return pd.concat(rows, ignore_index=True).dropna(subset=["date", "code", "close", "volume"])


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
        live_rows = spot[["date", "code", "close", "volume"]].copy()
        if "turnover_hkd" in spot:
            live_rows["turnover"] = pd.to_numeric(spot["turnover_hkd"], errors="coerce")
        prices = pd.concat([history, live_rows], ignore_index=True)
        prices = prices.drop_duplicates(["date", "code"], keep="last")
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
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
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

    rows: list[pd.DataFrame] = []
    for code in codes:
        try:
            frame = ak.stock_hk_daily(
                symbol=normalize_code(code),
                adjust=adjust,
            ).rename(columns={"date": "date", "close": "close", "volume": "volume"})
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            frame = frame.loc[(frame["date"].dt.date >= start_date) & (frame["date"].dt.date <= end_date)]
            frame["code"] = normalize_code(code)
            rows.append(frame[["date", "code", "close", "volume"]])
        except Exception as error:  # noqa: BLE001 - one bad symbol must not abort a batch
            print(f"WARN history {code}: {error}")
        time.sleep(pause_seconds)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "code", "close", "volume"])


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
            frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
            frame["published_at"] = frame["published_at"].dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
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
    live_cols = ["date", "code", "close", "volume"]
    if "turnover_hkd" in spot:
        spot["turnover"] = pd.to_numeric(spot["turnover_hkd"], errors="coerce")
        live_cols.append("turnover")
    live_rows = spot[live_cols].copy()
    prices = pd.concat([history, live_rows], ignore_index=True)
    prices = prices.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"])
    return prices.reset_index(drop=True), spot


def _yfinance_local_date(index: pd.Index) -> pd.Series:
    values = pd.to_datetime(index, errors="coerce")
    if getattr(values, "tz", None) is not None:
        values = values.tz_localize(None)
    return pd.Series(values).dt.normalize()


def _yfinance_news_datetime(value: object) -> pd.Timestamp:
    if isinstance(value, (int, float)):
        stamp = pd.to_datetime(value, unit="s", errors="coerce", utc=True)
    else:
        stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(stamp):
        return pd.NaT
    return stamp.tz_convert("Asia/Hong_Kong").tz_localize(None)


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
            if frame.empty:
                frame = ticker.history(period="5d", interval="1d", auto_adjust=auto_adjust, actions=False)
            if frame.empty:
                continue
            row = frame.dropna(subset=["Close", "Volume"]).iloc[-1]
            timestamp = pd.Timestamp(frame.dropna(subset=["Close", "Volume"]).index[-1])
            rows.append({
                "timestamp": timestamp,
                "date": _yfinance_local_date([timestamp]).iloc[0],
                "code": symbol,
                "name": symbol,
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
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
