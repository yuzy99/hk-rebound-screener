from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_rebound_screener.adapters import (
    fetch_hkex_security_master,
    fetch_yfinance_history_bulk,
)
from hk_rebound_screener.strategy import normalize_code


HISTORY_FILE = Path("data/yfinance_cache/hk_history.csv")
QUALITY_FILE = Path("data/yfinance_cache/hk_history_quality.json")
REQUIRED_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "auto_adjust",
    "adjustment_factor",
]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "adjustment_factor"]


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def default_start(today: date) -> date:
    try:
        return today.replace(year=today.year - 10)
    except ValueError:
        return today.replace(year=today.year - 10, day=28)


def cached_codes() -> set[str]:
    if not HISTORY_FILE.exists():
        return set()
    frame = pd.read_csv(HISTORY_FILE, usecols=["code"], dtype={"code": "string"})
    return {
        normalize_code(code)
        for code in frame["code"].dropna().astype(str)
        if normalize_code(code).isdigit()
    }


def write_quality_manifest(
    frame: pd.DataFrame,
    requested_codes: set[str],
    master_codes: set[str],
    requested_start: date,
    requested_end: date,
) -> dict:
    frame = frame.copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].map(normalize_code)

    required_null_cells = int(frame[REQUIRED_COLUMNS].isna().sum().sum())
    valid_ohlc = frame[["open", "high", "low", "close"]].notna().all(axis=1)
    ohlc_anomaly = valid_ohlc & (
        frame["high"].lt(frame[["open", "close"]].max(axis=1))
        | frame["low"].gt(frame[["open", "close"]].min(axis=1))
        | frame["high"].lt(frame["low"])
    )
    date_counts = frame.groupby("date", dropna=True)["code"].nunique()
    code_counts = frame.groupby("code", dropna=True)["date"].size()
    actual_codes = set(frame["code"].dropna().astype(str))
    actual_dates = frame["date"].dropna()
    latest_date = actual_dates.max() if not actual_dates.empty else None
    latest_count = int(date_counts.get(latest_date, 0)) if latest_date is not None else 0

    quality = {
        "dataset": "Hong Kong historical OHLCV cache",
        "as_of": date.today().isoformat(),
        "requested_window": {
            "start": requested_start.isoformat(),
            "end": requested_end.isoformat(),
            "years": 10,
        },
        "path_from_repository_root": str(HISTORY_FILE).replace("\\", "/"),
        "source": "yfinance",
        "encoding": "UTF-8",
        "bom": False,
        "delimiter": ",",
        "grain": ["code", "date"],
        "rows": int(len(frame)),
        "unique_codes": int(frame["code"].nunique(dropna=True)),
        "unique_dates": int(frame["date"].nunique(dropna=True)),
        "date_min": actual_dates.min().date().isoformat() if not actual_dates.empty else None,
        "date_max": actual_dates.max().date().isoformat() if not actual_dates.empty else None,
        "columns": [
            {"name": "date", "type": "date", "required": True, "format": "YYYY-MM-DD"},
            {"name": "code", "type": "string", "required": True, "format": "5 digits"},
            *[{"name": column, "type": "float", "required": True} for column in NUMERIC_COLUMNS[:5]],
            {"name": "auto_adjust", "type": "boolean", "required": True},
            {"name": "adjustment_factor", "type": "float", "required": True},
        ],
        "quality_checks": {
            "required_null_cells": required_null_cells,
            "invalid_dates": int(frame["date"].isna().sum()),
            "invalid_codes": int((~frame["code"].fillna("").str.fullmatch(r"\d{5}")).sum()),
            "duplicate_code_date_rows": int(frame.duplicated(["date", "code"]).sum()),
            "non_positive_close_rows": int(frame["close"].le(0).sum()),
            "negative_volume_rows": int(frame["volume"].lt(0).sum()),
            "non_positive_adjustment_factor_rows": int(frame["adjustment_factor"].le(0).sum()),
            "non_finite_numeric_cells": int(
                (~np.isfinite(frame[NUMERIC_COLUMNS])).sum().sum()
            ),
            "weekend_rows": int(frame["date"].dt.dayofweek.ge(5).sum()),
            "ohlc_structure_anomaly_rows": int(ohlc_anomaly.sum()),
            "ohlc_structure_anomaly_rate": round(float(ohlc_anomaly.mean()), 5),
            "ohlc_rule": "high >= max(open, close), low <= min(open, close), high >= low",
        },
        "coverage_notes": {
            "latest_date": latest_date.date().isoformat() if latest_date is not None else None,
            "latest_date_code_count": latest_count,
            "latest_date_code_count_requested": len(requested_codes),
            "codes_with_history": len(actual_codes),
            "requested_codes_without_history": len(requested_codes - actual_codes),
            "requested_missing_sample": sorted(requested_codes - actual_codes)[:30],
            "current_hkex_master_codes": len(master_codes),
            "master_codes_without_history": len(master_codes - actual_codes),
            "master_missing_sample": sorted(master_codes - actual_codes)[:30],
            "history_codes_not_in_current_master": len(actual_codes - master_codes),
            "history_not_in_master_sample": sorted(actual_codes - master_codes)[:30],
            "codes_with_fewer_than_500_rows": int((code_counts < 500).sum()),
            "codes_with_fewer_than_20_rows": int((code_counts < 20).sum()),
            "date_count_min": int(date_counts.min()) if not date_counts.empty else 0,
            "date_count_max": int(date_counts.max()) if not date_counts.empty else 0,
        },
        "use_boundary": {
            "close_volume_screening": "usable only after excluding non_positive_close_rows and checking coverage",
            "intraday_high_low_backtest": "requires filtering or flagging non_positive_close_rows and ohlc_structure_anomaly_rows",
            "raw_values_overwritten": False,
            "realtime": False,
        },
        "related_files": {
            "splits": "data/yfinance_cache/hk_splits.csv",
            "splits_checked": "data/yfinance_cache/hk_splits_checked.csv",
            "industry": "data/cache/hk_industry.csv",
            "market_index": "data/cache/hk_market_index.csv",
            "documentation": "data/yfinance_cache/README.md",
        },
    }
    QUALITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_FILE.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return quality


def main() -> int:
    today = date.today()
    parser = argparse.ArgumentParser(description="将港股 yfinance 日线缓存补齐到指定历史窗口")
    parser.add_argument("--start", type=parse_date, default=default_start(today))
    parser.add_argument("--end", type=parse_date, default=today)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start 不能晚于 --end")

    old_codes = cached_codes()
    try:
        master = fetch_hkex_security_master(ordinary_only=True, include_name=False)
        master_codes = {
            normalize_code(code)
            for code in master["code"].dropna().astype(str)
            if normalize_code(code).isdigit()
        }
    except Exception as error:  # noqa: BLE001 - existing cache remains a safe fallback
        master_codes = set()
        print(f"WARN 当前港交所主表读取失败，将只沿用现有缓存代码: {error}", flush=True)

    requested_codes = old_codes | master_codes
    if not requested_codes:
        raise RuntimeError("没有可下载的港股代码")
    print(
        f"港股十年补齐: {args.start} 至 {args.end}；缓存代码 {len(old_codes)}，"
        f"当前主表 {len(master_codes)}，请求并集 {len(requested_codes)}",
        flush=True,
    )
    started = datetime.now()
    fetch_yfinance_history_bulk(
        sorted(requested_codes),
        "HK",
        args.start,
        args.end,
        auto_adjust=True,
        include_high_low=True,
    )
    elapsed = datetime.now() - started
    if not HISTORY_FILE.exists():
        raise RuntimeError("下载结束但港股历史文件不存在")
    frame = pd.read_csv(HISTORY_FILE, dtype={"code": "string"})
    quality = write_quality_manifest(
        frame,
        requested_codes,
        master_codes,
        args.start,
        args.end,
    )
    print(
        f"港股十年补齐完成: {quality['rows']:,} 行，{quality['unique_codes']:,} 只代码，"
        f"{quality['date_min']} 至 {quality['date_max']}，耗时 {elapsed}",
        flush=True,
    )
    print(
        f"质量检查: 空值 {quality['quality_checks']['required_null_cells']}，"
        f"重复键 {quality['quality_checks']['duplicate_code_date_rows']}，"
        f"无历史代码 {quality['coverage_notes']['requested_codes_without_history']}，"
        f"OHLC 结构异常 {quality['quality_checks']['ohlc_structure_anomaly_rows']:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
