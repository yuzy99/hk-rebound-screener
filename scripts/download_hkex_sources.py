"""Download official HKEX market data and HKEXnews announcement indexes.

The two sources deliberately write to different cache directories.  HKEX's
public Daily Quotations pages expose only the current web window; the script
records the requested/available coverage instead of pretending that older
dates were downloaded.  HKEXnews is queried month by month and keeps the
official document URL rather than downloading every PDF body.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

from hk_rebound_screener.adapters import HKEX_SECURITIES_URL, fetch_hkex_security_master
from hk_rebound_screener.strategy import normalize_code


HKEX_DAILY_MAIN_INDEX_URL = "https://www.hkex.com.hk/eng/stat/smstat/dayquot/qtn.asp"
HKEX_DAILY_GEM_INDEX_URL = "https://www.hkex.com.hk/eng/stat/smstat/dayquot/gem/qtn.asp"
HKEX_BASE_URL = "https://www.hkex.com.hk"
HKEXNEWS_PAGE_URL = "https://www.hkexnews.hk/search/titlesearch.xhtml"
HKEXNEWS_SERVLET_URL = "https://www.hkexnews.hk/search/titleSearchServlet.do"
HKEXNEWS_BASE_URL = "https://www.hkexnews.hk"
USER_AGENT = "hk-rebound-screener-official-data/1.0"

HKEX_CACHE_DIR = Path("data") / "hkex_cache"
HKEXNEWS_CACHE_DIR = Path("data") / "hkexnews_cache"

HKEX_QUOTE_COLUMNS = [
    "date",
    "code",
    "name",
    "board",
    "currency",
    "previous_close",
    "ask",
    "high",
    "close",
    "bid",
    "low",
    "volume",
    "turnover",
    "status",
    "source_url",
]

HKEXNEWS_COLUMNS = [
    "published_at",
    "published_date",
    "code",
    "stock_name",
    "security_scope",
    "news_id",
    "title",
    "headline_category",
    "category",
    "file_type",
    "file_info",
    "file_url",
    "source_query_url",
    "retrieved_at",
]

NUMBER_TOKEN = re.compile(r"(?:N/A|-|\d[\d,]*(?:\.\d+)?)")
QUOTE_RECORD = re.compile(
    r"^\s*(?P<code>\d{1,5})(?P<suffix>#[A-Z0-9-]+)?\s+"
    r"(?P<name>.+?)\s+(?P<currency>HKD|CNY|USD|RMB)\s+(?P<rest>.*)$"
)
DATE_IN_DAILY_URL = re.compile(r"(?:d(\d{6})e|e_G(\d{6}))\.htm$", re.IGNORECASE)


def _request_headers(referer: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/json"}
    if referer:
        headers["Referer"] = referer
    return headers


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.reindex(columns=columns).to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def _get_with_retries(session: Any, url: str, *, timeout: int = 120, attempts: int = 3) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, headers=_request_headers(), timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as error:  # noqa: BLE001 - retry transient upstream errors
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"下载失败 {url}: {last_error}") from last_error


def _html_lines(content: bytes) -> list[str]:
    text = content.decode("iso-8859-1", errors="replace")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\r", "")
    return text.splitlines()


def _parse_number(value: str) -> float:
    value = value.strip()
    if value in {"", "-", "N/A"}:
        return float("nan")
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return float("nan")


def _daily_date_from_url(url: str) -> date | None:
    match = DATE_IN_DAILY_URL.search(urlparse(url).path)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    try:
        return datetime.strptime(value, "%y%m%d").date()
    except ValueError:
        return None


def _daily_links(index_url: str, start_date: date, end_date: date) -> list[tuple[date, str]]:
    import requests

    response = requests.get(index_url, headers=_request_headers(), timeout=60)
    response.raise_for_status()
    links: list[tuple[date, str]] = []
    for match in re.finditer(r"(?:href)\s*=\s*['\"]([^'\"]+\.htm)['\"]", response.text, re.I):
        url = urljoin(index_url, match.group(1))
        link_date = _daily_date_from_url(url)
        if link_date is not None and start_date <= link_date <= end_date:
            links.append((link_date, url))
    return sorted(set(links))


def _parse_daily_page(content: bytes, source_url: str, board: str, ordinary_codes: set[str]) -> pd.DataFrame:
    lines = _html_lines(content)
    date_match = re.search(
        r"DATE\s*:\s*(\d{2})\s+([A-Za-z]{3})\s+(\d{4})",
        "\n".join(lines),
        re.IGNORECASE,
    )
    if not date_match:
        raise ValueError(f"HKEX 日报缺少日期: {source_url}")
    report_date = datetime.strptime(
        f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}",
        "%d %b %Y",
    ).date()

    start = None
    for index, line in enumerate(lines):
        if line.strip() == "QUOTATIONS" and any(
            "CODE" in candidate and "NAME" in candidate for candidate in lines[index : index + 10]
        ):
            start = index + 3
            break
    if start is None:
        raise ValueError(f"HKEX 日报找不到 QUOTATIONS 区段: {source_url}")
    end = next(
        (index for index in range(start, len(lines)) if lines[index].strip() == "SALES RECORDS FOR ALL STOCKS"),
        len(lines),
    )

    rows: list[dict[str, object]] = []
    index = start
    while index < end:
        match = QUOTE_RECORD.match(lines[index])
        if not match:
            index += 1
            continue
        code = normalize_code(match.group("code"))
        suffix = match.group("suffix") or ""
        name = re.sub(r"\s+", " ", match.group("name")).strip()
        currency = match.group("currency")
        rest = match.group("rest").strip()
        status = "ok"
        first_values = re.findall(NUMBER_TOKEN, rest)
        second_values: list[str] = []
        if "TRADING" in rest or "HALTED" in rest or "SUSPENDED" in rest:
            status = "suspended_or_halted"
            index += 1
        else:
            if index + 1 < end and not QUOTE_RECORD.match(lines[index + 1]):
                second_values = re.findall(NUMBER_TOKEN, lines[index + 1])
                index += 2
            else:
                index += 1
            if len(first_values) < 4 or len(second_values) < 4:
                status = "incomplete_quote"
        if suffix or code not in ordinary_codes:
            continue
        first_values = (first_values + ["-"] * 4)[:4]
        second_values = (second_values + ["-"] * 4)[:4]
        rows.append(
            {
                "date": report_date,
                "code": code,
                "name": name,
                "board": board,
                "currency": currency,
                "previous_close": _parse_number(first_values[0]),
                "ask": _parse_number(first_values[1]),
                "high": _parse_number(first_values[2]),
                "close": _parse_number(second_values[0]),
                "bid": _parse_number(second_values[1]),
                "low": _parse_number(second_values[2]),
                "volume": _parse_number(first_values[3]),
                "turnover": _parse_number(second_values[3]),
                "status": status,
                "source_url": source_url,
            }
        )
    return pd.DataFrame(rows, columns=HKEX_QUOTE_COLUMNS)


def download_hkex(start_date: date, end_date: date, workers: int = 2) -> dict[str, object]:
    import requests

    retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    master = fetch_hkex_security_master(ordinary_only=True, include_name=True)
    master = master.copy()
    master["code"] = master["code"].map(normalize_code)
    master["retrieved_at"] = retrieved_at
    master["source_url"] = HKEX_SECURITIES_URL
    _atomic_write_csv(
        master[["code", "name", "lot_size", "retrieved_at", "source_url"]],
        HKEX_CACHE_DIR / "securities_master.csv",
        ["code", "name", "lot_size", "retrieved_at", "source_url"],
    )
    ordinary_codes = set(master["code"])

    sources: list[tuple[str, str, str]] = [
        ("Main Board", HKEX_DAILY_MAIN_INDEX_URL, "main"),
        ("GEM", HKEX_DAILY_GEM_INDEX_URL, "gem"),
    ]
    links: list[tuple[date, str, str]] = []
    for board, index_url, board_key in sources:
        for link_date, url in _daily_links(index_url, start_date, end_date):
            links.append((link_date, url, board))
    links = sorted(set(links), key=lambda item: (item[0], item[2]))

    def fetch_one(item: tuple[date, str, str]) -> pd.DataFrame:
        link_date, url, board = item
        with requests.Session() as session:
            response = _get_with_retries(session, url)
        return _parse_daily_page(response.content, url, board, ordinary_codes)

    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    completed = 0
    if links:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(fetch_one, item): item for item in links}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    frames.append(future.result())
                except Exception as error:  # noqa: BLE001 - keep other dates usable
                    errors.append({"date": item[0].isoformat(), "board": item[2], "url": item[1], "error": str(error)})
                completed += 1
                print(f"HKEX 日报 {completed}/{len(links)}: {item[0]} {item[2]}", flush=True)

    quotes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HKEX_QUOTE_COLUMNS)
    if not quotes.empty:
        quotes["date"] = pd.to_datetime(quotes["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        quotes = quotes.drop_duplicates(["date", "code", "board"], keep="last").sort_values(
            ["date", "board", "code"]
        )
    _atomic_write_csv(quotes, HKEX_CACHE_DIR / "daily_quotations.csv", HKEX_QUOTE_COLUMNS)

    available_dates = sorted({item[0].isoformat() for item in links})
    quality = {
        "source": "HKEX Daily Quotations and Securities List",
        "retrieved_at": retrieved_at,
        "requested_start": start_date.isoformat(),
        "requested_end": end_date.isoformat(),
        "available_start": available_dates[0] if available_dates else None,
        "available_end": available_dates[-1] if available_dates else None,
        "available_date_count": len(available_dates),
        "link_count": len(links),
        "completed_link_count": len(links) - len(errors),
        "error_count": len(errors),
        "errors": errors,
        "securities_master_rows": int(len(master)),
        "quote_rows": int(len(quotes)),
        "quote_columns": HKEX_QUOTE_COLUMNS,
        "quote_unique_keys": int(quotes[["date", "code", "board"]].drop_duplicates().shape[0]) if not quotes.empty else 0,
        "quote_null_counts": {column: int(quotes[column].isna().sum()) for column in HKEX_QUOTE_COLUMNS if column in quotes},
        "quote_invalid_ohlc_rows": int(
            ((pd.to_numeric(quotes["low"], errors="coerce") > pd.to_numeric(quotes["high"], errors="coerce"))
             | (pd.to_numeric(quotes["close"], errors="coerce") < pd.to_numeric(quotes["low"], errors="coerce"))
             | (pd.to_numeric(quotes["close"], errors="coerce") > pd.to_numeric(quotes["high"], errors="coerce"))).fillna(False).sum()
        ) if not quotes.empty else 0,
        "coverage_note": "HKEX public Daily Quotations index exposes the current web window; older requested dates are reported as unavailable.",
    }
    _atomic_write_text(HKEX_CACHE_DIR / "quality.json", json.dumps(quality, ensure_ascii=False, indent=2) + "\n")
    return quality


def _month_ranges(start_date: date, end_date: date) -> list[tuple[date, date, str]]:
    current = date(start_date.year, start_date.month, 1)
    ranges: list[tuple[date, date, str]] = []
    while current <= end_date:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        month_end = next_month - timedelta(days=1)
        left = max(start_date, current)
        right = min(end_date, month_end)
        if left <= right:
            ranges.append((left, right, f"{current.year:04d}{current.month:02d}"))
        current = next_month
    return ranges


def _query_hkexnews(
    session: Any,
    scope: str,
    start_date: date,
    end_date: date,
    row_range: int = 50000,
    t1code: str = "-2",
) -> tuple[list[dict[str, Any]], str, int]:
    params = {
        "sortDir": "0",
        "sortByOptions": "DateTime",
        "category": scope,
        "market": "SEHK",
        "stockId": "-1",
        "documentType": "-1",
        "fromDate": start_date.strftime("%Y%m%d"),
        "toDate": end_date.strftime("%Y%m%d"),
        "title": "",
        "searchType": "0",
        "t1code": t1code,
        "t2Gcode": "-2",
        "t2code": "-2",
        "rowRange": str(row_range),
        "lang": "E",
    }
    response = None
    payload: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            response = session.get(
                HKEXNEWS_SERVLET_URL,
                params=params,
                headers={
                    **_request_headers(HKEXNEWS_PAGE_URL),
                    "Accept": "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    if response is None or payload is None:
        raise RuntimeError("HKEXnews 没有返回结果")
    raw_result = payload.get("result")
    records = [] if raw_result in (None, "null", "") else json.loads(raw_result)
    records = records if isinstance(records, list) else []
    record_count = int(payload.get("recordCnt") or 0)
    return records, response.url, record_count


def _query_hkexnews_complete(
    session: Any,
    scope: str,
    start_date: date,
    end_date: date,
    t1_codes: list[str],
) -> list[tuple[list[dict[str, Any]], str, date, date, str]]:
    """Return complete result partitions without crossing the 10,000-row API cap."""
    records, query_url, record_count = _query_hkexnews(session, scope, start_date, end_date)
    if len(records) >= record_count:
        return [(records, query_url, start_date, end_date, "-2")]
    if start_date < end_date:
        midpoint = start_date + timedelta(days=(end_date - start_date).days // 2)
        return _query_hkexnews_complete(session, scope, start_date, midpoint, t1_codes) + _query_hkexnews_complete(
            session, scope, midpoint + timedelta(days=1), end_date, t1_codes
        )
    partitions: list[tuple[list[dict[str, Any]], str, date, date, str]] = []
    for t1code in t1_codes:
        records, query_url, record_count = _query_hkexnews(
            session, scope, start_date, end_date, row_range=10000, t1code=t1code
        )
        if len(records) < record_count:
            raise RuntimeError(
                f"HKEXnews 单日/类别仍被截断: {start_date} scope={scope} t1code={t1code} "
                f"{len(records)}/{record_count}"
            )
        if records:
            partitions.append((records, query_url, start_date, end_date, t1code))
    return partitions


def _fetch_hkexnews_tier_one_codes(session: Any) -> list[str]:
    url = "https://www.hkexnews.hk/ncms/script/eds/tierone_e.json"
    response = session.get(url, headers=_request_headers(HKEXNEWS_PAGE_URL), timeout=60)
    response.raise_for_status()
    payload = response.json()
    codes = [str(item.get("code")) for item in payload if item.get("code")]
    if not codes:
        raise RuntimeError("HKEXnews 一级公告类别清单为空")
    return codes


def _split_html_values(value: object) -> list[str]:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return [part.strip() for part in text.splitlines() if part.strip()]


def _flatten_hkexnews_record(
    record: dict[str, Any],
    scope: str,
    source_query_url: str,
    retrieved_at: str,
) -> list[dict[str, object]]:
    published_text = str(record.get("DATE_TIME") or "").strip()
    published = pd.to_datetime(published_text, format="%d/%m/%Y %H:%M", errors="coerce")
    codes = _split_html_values(record.get("STOCK_CODE")) or [""]
    names = _split_html_values(record.get("STOCK_NAME"))
    headline_category = re.sub(r"\s+", " ", html.unescape(str(record.get("LONG_TEXT") or "")).strip())
    category = headline_category.split(" - [", 1)[0].strip() if headline_category else ""
    file_link = str(record.get("FILE_LINK") or "").strip()
    file_url = urljoin(HKEXNEWS_BASE_URL, file_link) if file_link else ""
    rows: list[dict[str, object]] = []
    for index, code in enumerate(codes):
        rows.append(
            {
                "published_at": published.strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(published) else "",
                "published_date": published.strftime("%Y-%m-%d") if not pd.isna(published) else "",
                "code": normalize_code(code) if code else "",
                "stock_name": names[index] if index < len(names) else (names[0] if names else ""),
                "security_scope": "current" if scope == "0" else "delisted",
                "news_id": str(record.get("NEWS_ID") or ""),
                "title": html.unescape(str(record.get("TITLE") or "")).strip(),
                "headline_category": headline_category,
                "category": category,
                "file_type": str(record.get("FILE_TYPE") or "").strip(),
                "file_info": str(record.get("FILE_INFO") or "").strip(),
                "file_url": file_url,
                "source_query_url": source_query_url,
                "retrieved_at": retrieved_at,
            }
        )
    return rows


def download_hkexnews(
    start_date: date,
    end_date: date,
    pause_seconds: float = 0.2,
) -> dict[str, object]:
    import requests

    retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rows: list[dict[str, object]] = []
    query_log: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    ranges = _month_ranges(start_date, end_date)
    with requests.Session() as session:
        session.get(HKEXNEWS_PAGE_URL, headers=_request_headers(), timeout=60)
        t1_codes = _fetch_hkexnews_tier_one_codes(session)
        total_queries = len(ranges) * 2
        completed = 0
        for month_start, month_end, month_key in ranges:
            for scope in ("0", "1"):
                try:
                    partitions = _query_hkexnews_complete(session, scope, month_start, month_end, t1_codes)
                    for records, query_url, partition_start, partition_end, t1code in partitions:
                        for record in records:
                            rows.extend(_flatten_hkexnews_record(record, scope, query_url, retrieved_at))
                        query_log.append(
                            {
                                "month": month_key,
                                "scope": "current" if scope == "0" else "delisted",
                                "start_date": partition_start.isoformat(),
                                "end_date": partition_end.isoformat(),
                                "t1code": t1code,
                                "record_count": len(records),
                            }
                        )
                except Exception as error:  # noqa: BLE001 - preserve completed months
                    errors.append(
                        {
                            "month": month_key,
                            "scope": "current" if scope == "0" else "delisted",
                            "error": str(error),
                        }
                    )
                completed += 1
                print(f"HKEXnews {completed}/{total_queries}: {month_key} {'current' if scope == '0' else 'delisted'}", flush=True)
                if pause_seconds > 0:
                    time.sleep(pause_seconds)

    announcements = pd.DataFrame(rows, columns=HKEXNEWS_COLUMNS)
    if not announcements.empty:
        announcements = announcements.drop_duplicates(
            ["news_id", "code", "security_scope"], keep="last"
        ).sort_values(["published_at", "news_id", "code"], ascending=[False, True, True])
    _atomic_write_csv(announcements, HKEXNEWS_CACHE_DIR / "announcements.csv", HKEXNEWS_COLUMNS)

    quality = {
        "source": "HKEXnews Listed Company Information Title Search",
        "retrieved_at": retrieved_at,
        "requested_start": start_date.isoformat(),
        "requested_end": end_date.isoformat(),
        "query_count": len(query_log),
        "error_count": len(errors),
        "errors": errors,
        "rows": int(len(announcements)),
        "unique_news_ids": int(announcements["news_id"].nunique()) if not announcements.empty else 0,
        "unique_keys": int(announcements[["news_id", "code", "security_scope"]].drop_duplicates().shape[0]) if not announcements.empty else 0,
        "missing_news_id_rows": int(announcements["news_id"].eq("").sum()) if not announcements.empty else 0,
        "missing_code_rows": int(announcements["code"].eq("").sum()) if not announcements.empty else 0,
        "bad_published_at_rows": int(
            pd.to_datetime(announcements["published_at"], errors="coerce").isna().sum()
        ) if not announcements.empty else 0,
        "published_min": announcements["published_date"].min() if not announcements.empty else None,
        "published_max": announcements["published_date"].max() if not announcements.empty else None,
        "scope_counts": announcements["security_scope"].value_counts().to_dict() if not announcements.empty else {},
        "category_counts": announcements["category"].value_counts().head(20).to_dict() if not announcements.empty else {},
        "query_log": query_log,
        "columns": HKEXNEWS_COLUMNS,
        "content_note": "This cache is an announcement index with official PDF links; PDF bodies are not bulk-downloaded.",
    }
    _atomic_write_text(HKEXNEWS_CACHE_DIR / "quality.json", json.dumps(quality, ensure_ascii=False, indent=2) + "\n")
    return quality


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_args() -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(description="下载官方 HKEX 日报与 HKEXnews 公告索引")
    parser.add_argument("--start", default=(today - timedelta(days=1461)).isoformat(), help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=today.isoformat(), help="结束日期 YYYY-MM-DD")
    parser.add_argument("--hkex-workers", type=int, default=2, help="HKEX 日报并发下载数")
    parser.add_argument("--hkexnews-pause", type=float, default=0.2, help="HKEXnews 查询间隔秒数")
    parser.add_argument("--only", choices=("all", "hkex", "hkexnews"), default="all")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    if start_date > end_date:
        raise SystemExit("--start 不能晚于 --end")
    if args.only in {"all", "hkex"}:
        quality = download_hkex(start_date, end_date, workers=max(1, args.hkex_workers))
        print(json.dumps({"hkex": quality}, ensure_ascii=False, indent=2))
    if args.only in {"all", "hkexnews"}:
        quality = download_hkexnews(start_date, end_date, pause_seconds=max(0.0, args.hkexnews_pause))
        print(json.dumps({"hkexnews": quality}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
