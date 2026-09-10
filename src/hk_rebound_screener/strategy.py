from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CODE_WIDTH = 5


def normalize_code(value: object) -> str:
    text = str(value).strip().upper().replace(".HK", "")
    if text.isdigit():
        return text.zfill(CODE_WIDTH)
    return text


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_universe(path: str | Path, market: str = "HK") -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    required = {"code", "name", "industry", "enabled"}
    if market.upper() == "HK":
        required.add("lot_size")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"universe.csv 缺少字段: {sorted(missing)}")
    if "lot_size" not in frame:
        # 美股通常可按 1 股下单；港股仍必须显式维护 board lot。
        frame["lot_size"] = 1.0
    frame["code"] = frame["code"].map(normalize_code)
    frame["lot_size"] = pd.to_numeric(frame["lot_size"], errors="coerce")
    frame["enabled"] = frame["enabled"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    return frame.loc[frame["enabled"]].copy()


def load_prices(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    required = {"date", "code", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"prices.csv 缺少字段: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].map(normalize_code)
    for column in ("open", "close", "volume", "turnover"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "code", "close", "volume"])
    return frame.sort_values(["code", "date"]).reset_index(drop=True)


def load_news(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    for column in ("code", "published_at", "title", "body", "url"):
        if column not in frame:
            frame[column] = ""
    frame["code"] = frame["code"].map(normalize_code)
    frame["published_at"] = _parse_datetime_series(frame["published_at"])
    frame["title"] = frame["title"].fillna("").astype(str)
    frame["body"] = frame["body"].fillna("").astype(str)
    return frame


def load_industry_returns(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "official_industry", "index_code", "daily_return_pct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"行业指数文件缺少字段: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["official_industry"] = frame["official_industry"].fillna("").astype(str).str.strip()
    frame["daily_return_pct"] = pd.to_numeric(frame["daily_return_pct"], errors="coerce")
    if "source" not in frame:
        frame["source"] = "official_industry_index"
    return frame.dropna(subset=["date", "daily_return_pct"])


def _parse_hong_kong_datetime(value: object) -> pd.Timestamp:
    """Parse a timestamp, treating a timezone-less value as Hong Kong time."""
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Hong_Kong")
    else:
        stamp = stamp.tz_convert("Asia/Hong_Kong")
    return stamp.tz_localize(None)


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    return pd.Series(
        [_parse_hong_kong_datetime(value) for value in values],
        index=values.index,
        name=values.name,
    )


def _asof_end(asof: object) -> pd.Timestamp:
    stamp = pd.Timestamp(asof)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Hong_Kong").tz_localize(None)
    if stamp.hour == 0 and stamp.minute == 0 and stamp.second == 0:
        return stamp.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return stamp


def _asof_date(asof: object, market: str) -> pd.Timestamp:
    stamp = pd.Timestamp(asof)
    if stamp.tzinfo is not None:
        timezone = "America/New_York" if market.upper() == "US" else "Asia/Hong_Kong"
        stamp = stamp.tz_convert(timezone).tz_localize(None)
    return stamp.normalize()


def _news_scores(news: pd.DataFrame, asof: object, config: dict[str, Any]) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["code", "negative_news_score", "negative_news_hits"])
    end = _asof_end(asof)
    start = end - pd.Timedelta(hours=float(config["news_lookback_hours"]))
    window = news.loc[(news["published_at"] >= start) & (news["published_at"] <= end)].copy()
    rules = config.get("news_rules", {})
    results: list[dict[str, object]] = []
    for row in window.itertuples(index=False):
        text = f"{row.title} {row.body}".lower()
        matched: list[str] = []
        score = 0.0
        for category, rule in rules.items():
            terms = [str(term).lower() for term in rule.get("terms", [])]
            if any(term in text for term in terms):
                matched.append(category)
                score += float(rule.get("weight", 0.0))
        if matched:
            results.append({
                "code": row.code,
                "negative_news_score": score,
                "negative_news_hits": ";".join(matched),
            })
    if not results:
        return pd.DataFrame(columns=["code", "negative_news_score", "negative_news_hits"])
    return (
        pd.DataFrame(results)
        .groupby("code", as_index=False)
        .agg(
            negative_news_score=("negative_news_score", "sum"),
            negative_news_hits=("negative_news_hits", lambda values: ";".join(sorted(set(";".join(values).split(";"))))),
        )
    )


def _prepare_prices(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    frame = prices.copy().sort_values(["code", "date"])
    if frame.empty:
        return frame

    if "turnover" not in frame:
        frame["turnover"] = frame["close"] * frame["volume"]
    else:
        frame["turnover"] = pd.to_numeric(frame["turnover"], errors="coerce").fillna(
            frame["close"] * frame["volume"]
        )

    market_dates = sorted(frame["date"].dropna().unique())
    all_codes = frame["code"].dropna().unique()
    lookback = int(config.get("volume_lookback", 20))

    if len(market_dates) > 0 and len(all_codes) > 0:
        first_trade_dates = frame.dropna(subset=["close"]).groupby("code")["date"].min()
        indexed = frame.set_index(["code", "date"])
        indexed = indexed[~indexed.index.duplicated(keep="last")]
        full_index = pd.MultiIndex.from_product([all_codes, market_dates], names=["code", "date"])
        aligned = indexed.reindex(full_index)

        first_dates = aligned.index.get_level_values("code").map(first_trade_dates)
        active_mask = aligned.index.get_level_values("date") >= first_dates
        aligned = aligned.loc[active_mask].copy()

        aligned["close"] = aligned.groupby(level="code")["close"].ffill()
        if "open" in aligned.columns:
            aligned["open"] = aligned.groupby(level="code")["open"].ffill()
        aligned["volume"] = aligned["volume"].fillna(0.0)
        aligned["turnover"] = aligned["turnover"].fillna(0.0)

        grouped = aligned.groupby(level="code", group_keys=False)
        aligned["daily_return_pct"] = grouped["close"].pct_change() * 100.0

        aligned["prior_volume_median"] = grouped["volume"].transform(
            lambda values: values.shift(1).rolling(lookback, min_periods=lookback).median()
        )
        aligned["prior_turnover_median"] = grouped["turnover"].transform(
            lambda values: values.shift(1).rolling(lookback, min_periods=lookback).median()
        )
        aligned["prior_traded_days"] = grouped["volume"].transform(
            lambda values: values.shift(1).rolling(lookback, min_periods=lookback).apply(
                lambda window: float((window > 0).sum()), raw=True
            )
        )
        aligned["volume_ratio"] = aligned["volume"] / aligned["prior_volume_median"]
        aligned["volume_anomaly"] = (aligned["volume_ratio"] - 1.0).clip(lower=0.0)
        frame = aligned.reset_index().dropna(subset=["date", "code", "close"])
    else:
        grouped = frame.groupby("code", group_keys=False)
        frame["daily_return_pct"] = grouped["close"].pct_change() * 100.0
        frame["prior_volume_median"] = grouped["volume"].transform(
            lambda values: values.shift(1).rolling(lookback, min_periods=lookback).median()
        )
        frame["prior_turnover_median"] = grouped["turnover"].transform(
            lambda values: values.shift(1).rolling(lookback, min_periods=lookback).median()
        )
        frame["prior_traded_days"] = grouped["volume"].transform(
            lambda values: values.shift(1).rolling(lookback, min_periods=lookback).apply(
                lambda window: float((window > 0).sum()), raw=True
            )
        )
        frame["volume_ratio"] = frame["volume"] / frame["prior_volume_median"]
        frame["volume_anomaly"] = (frame["volume_ratio"] - 1.0).clip(lower=0.0)

    return frame.sort_values(["code", "date"]).reset_index(drop=True)


def _compute_industry_benchmark(
    result: pd.DataFrame,
    method: str = "median",
) -> tuple[pd.Series, pd.Series]:
    """计算行业基准涨跌幅与同行有效股票数。

    参数 method 支持：
    - 'median'（默认）：同行中位数。彻底免疫小盘仙股、除权未复权异常或个别妖股单日暴涨暴跌对行业均值的失真拉扯。
    - 'trimmed_mean'：同行截断均值。同行数 >= 4 时剔除两端各 10% 极值后计算算术均值。
    - 'turnover_weighted'：同行成交额加权平均值（资金体量加权）。
    - 'mean'：传统同行等权算术平均值（兼容旧版）。
    """
    clean_industry = result["industry"].fillna("").astype(str).str.strip()
    peer_count = pd.Series(0, index=result.index, dtype=int)
    industry_benchmark = pd.Series(np.nan, index=result.index, dtype=float)

    for industry_name, group in result.groupby(clean_industry):
        if not industry_name:
            continue
        n = len(group)
        if n <= 1:
            continue
        returns = pd.to_numeric(group["daily_return_pct"], errors="coerce").to_numpy(dtype=float)

        if method == "mean":
            benchmarks = np.full(n, np.nan, dtype=float)
            for i in range(n):
                peer_vals = np.delete(returns, i)
                valid = peer_vals[np.isfinite(peer_vals)]
                peer_count.loc[group.index[i]] = len(valid)
                if len(valid) > 0:
                    benchmarks[i] = np.mean(valid)
        elif method == "trimmed_mean":
            benchmarks = np.zeros(n, dtype=float)
            for i in range(n):
                peer_vals = np.delete(returns, i)
                valid = peer_vals[np.isfinite(peer_vals)]
                peer_count.loc[group.index[i]] = len(valid)
                if len(valid) >= 4:
                    low, high = np.percentile(valid, [10, 90])
                    trimmed = valid[(valid >= low) & (valid <= high)]
                    benchmarks[i] = np.mean(trimmed) if len(trimmed) > 0 else np.mean(valid)
                elif len(valid) > 0:
                    benchmarks[i] = np.mean(valid)
                else:
                    benchmarks[i] = np.nan
        elif method == "turnover_weighted":
            benchmarks = np.zeros(n, dtype=float)
            turnovers = (
                pd.to_numeric(group["turnover"], errors="coerce").to_numpy(dtype=float)
                if "turnover" in group
                else np.ones(n, dtype=float)
            )
            for i in range(n):
                peer_vals = np.delete(returns, i)
                peer_to = np.delete(turnovers, i)
                peer_count.loc[group.index[i]] = int(np.isfinite(peer_vals).sum())
                mask = np.isfinite(peer_vals) & np.isfinite(peer_to) & (peer_to > 0)
                if np.any(mask):
                    benchmarks[i] = np.average(peer_vals[mask], weights=peer_to[mask])
                else:
                    valid = peer_vals[np.isfinite(peer_vals)]
                    benchmarks[i] = np.median(valid) if len(valid) > 0 else np.nan
        else:  # default: "median"
            benchmarks = np.zeros(n, dtype=float)
            for i in range(n):
                peer_vals = np.delete(returns, i)
                valid = peer_vals[np.isfinite(peer_vals)]
                peer_count.loc[group.index[i]] = len(valid)
                benchmarks[i] = np.median(valid) if len(valid) > 0 else np.nan

        industry_benchmark.loc[group.index] = benchmarks

    return industry_benchmark, peer_count


def evaluate_signal(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    news: pd.DataFrame,
    config: dict[str, Any],
    asof: object | None = None,
    news_status: pd.DataFrame | None = None,
    industry_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prepared = _prepare_prices(prices, config)
    available_dates = sorted(prepared["date"].dropna().unique())
    if not available_dates:
        return pd.DataFrame()
    market = str(config.get("market", "HK")).upper()
    asof_date = _asof_date(asof, market) if asof is not None else pd.Timestamp(available_dates[-1])
    dates = [date for date in available_dates if pd.Timestamp(date) <= asof_date]
    if len(dates) < 2:
        return pd.DataFrame()
    today_date, prior_date = dates[-1], dates[-2]
    strategy_mode = str(config.get("strategy_mode", "rebound")).lower()
    if strategy_mode in {"two_day_drop", "industry_lag_rebound"} and len(dates) < 3:
        return pd.DataFrame()
    today = prepared.loc[prepared["date"] == today_date].copy()
    prior = prepared.loc[prepared["date"] == prior_date, ["code", "daily_return_pct"]].rename(
        columns={"daily_return_pct": "prior_return_pct"}
    )
    result = today.merge(prior, on="code", how="inner")
    if strategy_mode in {"two_day_drop", "industry_lag_rebound"}:
        two_days_ago_date = dates[-3]
        two_days_ago = prepared.loc[
            prepared["date"] == two_days_ago_date, ["code", "daily_return_pct"]
        ].rename(columns={"daily_return_pct": "two_days_ago_return_pct"})
        result = result.merge(two_days_ago, on="code", how="inner")
    result = result.merge(universe.drop(columns=["enabled"], errors="ignore"), on="code", how="left")
    if result.empty:
        return result

    industry_method = str(config.get("industry_avg_method", "median")).lower()
    if strategy_mode == "industry_lag_rebound":
        result["industry_avg_return_pct"] = np.nan
        result["peer_count"] = 0
    else:
        result["industry_avg_return_pct"], result["peer_count"] = _compute_industry_benchmark(
            result, method=industry_method
        )
    result["lag_vs_industry_pct"] = result["industry_avg_return_pct"] - result["daily_return_pct"]

    if strategy_mode == "industry_lag_rebound":
        result["official_industry"] = result.get("official_industry", pd.Series("", index=result.index))
        result["official_industry"] = result["official_industry"].fillna("").astype(str).str.strip()
        official_today = pd.DataFrame(columns=[
            "official_industry", "industry_index_code", "industry_return_pct", "industry_return_source",
        ])
        if industry_returns is not None and not industry_returns.empty:
            official_today = industry_returns.loc[
                industry_returns["date"].eq(pd.Timestamp(today_date)),
                ["official_industry", "index_code", "daily_return_pct", "source"],
            ].rename(columns={
                "index_code": "industry_index_code",
                "daily_return_pct": "industry_return_pct",
                "source": "industry_return_source",
            }).drop_duplicates("official_industry")
        result = result.merge(official_today, on="official_industry", how="left")
        result["industry_avg_return_pct"] = result["industry_return_pct"]
        result["lag_vs_industry_pct"] = result["industry_return_pct"] - result["daily_return_pct"]
        result["yesterday_return_pct"] = result["prior_return_pct"]
        result["day_before_yesterday_return_pct"] = result["two_days_ago_return_pct"]
        result["today_return_pct"] = result["daily_return_pct"]
        result["industry_return_source"] = result["industry_return_source"].fillna("unavailable")
        large_drop_max_pct = float(config["large_drop_max_pct"])
        other_day_max_pct = float(config["other_prior_day_return_max_pct"])
        result["prior_drop_ok"] = (
            (result["yesterday_return_pct"].le(large_drop_max_pct)
             & result["day_before_yesterday_return_pct"].le(other_day_max_pct))
            | (result["day_before_yesterday_return_pct"].le(large_drop_max_pct)
               & result["yesterday_return_pct"].le(other_day_max_pct))
        )
        result["industry_rebound_ok"] = result["industry_return_pct"] >= float(config["industry_return_min_pct"])
        result["today_not_up_ok"] = result["today_return_pct"] <= float(config["today_return_max_pct"])
        result["lag_ok"] = result["lag_vs_industry_pct"] >= float(config["lag_min_pct"])
        result["signal_day_volume"] = pd.to_numeric(result["volume"], errors="coerce")
        result["signal_volume_ok"] = result["signal_day_volume"].gt(
            float(config.get("min_signal_day_volume_shares", 500000))
        )
        result["industry_data_ok"] = (
            result["official_industry"].ne("")
            & result["yesterday_return_pct"].notna()
            & result["day_before_yesterday_return_pct"].notna()
            & result["today_return_pct"].notna()
            & result["industry_return_pct"].notna()
        )
        result["industry_relationship_checked"] = result["industry_data_ok"]
        industry_rule_ok = result["industry_rebound_ok"] & (result["today_not_up_ok"] | result["lag_ok"])
        result["passes"] = (
            result["prior_drop_ok"]
            & result["signal_volume_ok"]
            & ((result["industry_data_ok"] & industry_rule_ok)
               | (~result["industry_data_ok"] & result["today_not_up_ok"]))
        )
        result["score"] = result["lag_vs_industry_pct"].where(result["industry_data_ok"], 0.0)
        columns = [
            "code", "name", "industry", "official_industry", "industry_index_code", "close", "day_before_yesterday_return_pct",
            "yesterday_return_pct", "today_return_pct",
            "industry_return_pct", "lag_vs_industry_pct", "industry_return_source",
            "prior_drop_ok", "industry_rebound_ok", "today_not_up_ok", "lag_ok", "industry_data_ok",
            "industry_relationship_checked",
            "signal_day_volume", "signal_volume_ok", "score", "passes",
        ]
        return result[[column for column in columns if column in result]].sort_values("score", ascending=False)

    news_cutoff = _asof_end(asof) if asof is not None else _asof_end(today_date)
    scores = _news_scores(news, news_cutoff, config)
    result = result.merge(scores, on="code", how="left")
    result["negative_news_score"] = result["negative_news_score"].fillna(0.0)
    result["negative_news_hits"] = result["negative_news_hits"].fillna("")
    if news_status is not None and not news_status.empty:
        status = news_status.copy()
        status["code"] = status["code"].map(normalize_code)
        result = result.merge(status[["code", "news_fetch_ok"]], on="code", how="left")
    else:
        result["news_fetch_ok"] = True
    result["news_fetch_ok"] = result["news_fetch_ok"].fillna(False).astype(bool)
    currency = str(config.get("currency", "HKD" if market == "HK" else "USD")).lower()
    lot_value_column = f"lot_value_{currency}"
    result[lot_value_column] = result["close"] * result["lot_size"]
    if currency == "usd" and "max_lot_value_hkd" in config:
        usd_hkd_rate = config.get("usd_hkd_rate")
        if usd_hkd_rate is None or float(usd_hkd_rate) <= 0:
            raise ValueError("美股策略需要有效的 usd_hkd_rate 才能换算港币 30,000 等值")
        max_lot_value = float(config["max_lot_value_hkd"]) / float(usd_hkd_rate)
        result["usd_hkd_rate"] = float(usd_hkd_rate)
    else:
        max_lot_value = float(
            config.get(
                f"max_lot_value_{currency}",
                config.get("max_lot_value", 30000.0),
            )
        )

    large_drop_max_pct = float(config.get("large_drop_max_pct", config.get("prior_drop_max_pct", -5.0)))
    other_day_max_pct = float(config.get("other_day_return_max_pct", config.get("today_return_max_pct", 0.5)))

    # 两日内任一日大跌且另一日涨幅 <= other_day_max_pct (默认0.5%)
    drop_yesterday = (result["prior_return_pct"] <= large_drop_max_pct) & (result["daily_return_pct"] <= other_day_max_pct)
    drop_today = (result["daily_return_pct"] <= large_drop_max_pct) & (result["prior_return_pct"] <= other_day_max_pct)
    result["two_day_drop_ok"] = drop_yesterday | drop_today
    result["drop_strength_pct"] = result[["prior_return_pct", "daily_return_pct"]].min(axis=1).abs()

    weights = config.get("score_weights", {})
    drop_weight = float(weights.get("drop_strength", weights.get("prior_drop_abs", 1.0)))
    result["score"] = (
        drop_weight * result["drop_strength_pct"]
        + float(weights.get("lag", 1.0)) * result["lag_vs_industry_pct"]
        + float(weights.get("volume_anomaly", 1.0)) * result["volume_anomaly"].fillna(0.0)
        - float(weights.get("negative_news", 1.0)) * result["negative_news_score"]
    )

    if strategy_mode == "two_day_drop":
        # 保留旧字段名以兼容已有输出，但策略窗口改为最近三个交易日。
        result["drop_strength_pct"] = result[["prior_return_pct", "daily_return_pct", "two_days_ago_return_pct"]].min(axis=1).abs()
        result["score"] = (
            float(weights.get("drop_strength", 1.0)) * result["drop_strength_pct"]
            + float(weights.get("volume_anomaly", 1.0)) * result["volume_anomaly"].fillna(0.0)
            - float(weights.get("negative_news", 1.0)) * result["negative_news_score"]
        )

    news_clean = result["negative_news_score"] < float(config["negative_news_threshold"])
    if config.get("strict_news_fetch", True):
        news_clean &= result["news_fetch_ok"]
    lot_known = result["lot_size"].notna() & (result["lot_size"] > 0)
    if config.get("strict_lot_size", True):
        lot_known &= result[lot_value_column].notna()
    common_filters = (
        (result[lot_value_column] <= max_lot_value)
        & news_clean
        & lot_known
    )
    if strategy_mode != "two_day_drop":
        industry_known = result["industry"].notna() & result["industry"].astype(str).str.strip().ne("")
        common_filters &= (
            (result["peer_count"] >= int(config["min_industry_peers"]))
            & industry_known
            & (result["industry_avg_return_pct"] > float(config.get("industry_mean_min_pct", 0.0)))
        )
    if strategy_mode == "two_day_drop":
        liquidity = config.get("liquidity_filter", {})
        min_current_turnover = float(liquidity.get("min_current_turnover", 0.0))
        min_prior_median_turnover = float(liquidity.get("min_prior_median_turnover", 0.0))
        min_traded_days = int(liquidity.get("min_traded_days", 0))
        three_day_returns = result[[
            "two_days_ago_return_pct", "prior_return_pct", "daily_return_pct",
        ]]
        result["large_drop_ok"] = (
            three_day_returns.le(float(config.get("large_drop_max_pct", -5.0))).any(axis=1)
        )
        result["other_days_ok"] = (
            three_day_returns.le(other_day_max_pct).all(axis=1)
        )
        result["three_day_drop_ok"] = result["large_drop_ok"] & result["other_days_ok"]
        # 兼容旧报告/下游脚本：旧字段反映新的三日条件。
        result["two_day_drop_ok"] = result["three_day_drop_ok"]
        result["consecutive_down_ok"] = result["other_days_ok"]
        result["liquidity_ok"] = (
            result["volume"].gt(0)
            & result["turnover"].ge(min_current_turnover)
            & result["prior_turnover_median"].ge(min_prior_median_turnover)
            & result["prior_traded_days"].ge(min_traded_days)
            & np.isfinite(result["prior_turnover_median"])
        )
        result["passes"] = (
            common_filters
            & result["three_day_drop_ok"]
            & result["liquidity_ok"]
        )
    else:
        lag_condition = (
            (result["lag_vs_industry_pct"] >= float(config.get("lag_min_pct", 1.0)))
            | (result["daily_return_pct"] > 0.0)
        )
        result["passes"] = (
            common_filters
            & result["two_day_drop_ok"]
            & lag_condition
        )
    columns = [
        "code", "name", "industry", "close", "lot_size", "lot_value_hkd", "prior_return_pct",
        "daily_return_pct", "drop_strength_pct", "two_day_drop_ok", "industry_avg_return_pct", "lag_vs_industry_pct", "peer_count",
        "volume_ratio", "volume_anomaly", "negative_news_score", "negative_news_hits", "news_fetch_ok", "usd_hkd_rate", "score", "passes",
    ]
    if strategy_mode == "two_day_drop":
        columns.extend([
            "two_days_ago_return_pct", "turnover", "prior_turnover_median",
            "prior_traded_days", "large_drop_ok", "other_days_ok", "three_day_drop_ok",
            "consecutive_down_ok", "liquidity_ok",
        ])
    if lot_value_column not in columns:
        columns.insert(6, lot_value_column)
    return result[[column for column in columns if column in result]].sort_values("score", ascending=False)


def append_forward_observations(
    result: pd.DataFrame,
    prices: pd.DataFrame,
    asof: object,
    trading_days: int,
) -> pd.DataFrame:
    """Append post-signal prices for historical checks without affecting selection."""
    if result.empty or trading_days <= 0:
        return result

    prepared = _prepare_prices(prices, {})
    signal_date = pd.Timestamp(asof).normalize()
    future_dates = [
        pd.Timestamp(value)
        for value in sorted(prepared["date"].dropna().unique())
        if pd.Timestamp(value) > signal_date
    ][:trading_days]
    enriched = result.copy()
    enriched["signal_date"] = signal_date.date().isoformat()
    for offset in range(1, trading_days + 1):
        date_column = f"forward_t{offset}_date"
        close_column = f"forward_t{offset}_close"
        return_column = f"forward_t{offset}_return_pct"
        if offset > len(future_dates):
            enriched[date_column] = pd.NA
            enriched[close_column] = np.nan
            enriched[return_column] = np.nan
            continue
        future_date = future_dates[offset - 1]
        observation = prepared.loc[
            prepared["date"].eq(future_date), ["code", "close", "daily_return_pct"]
        ].rename(columns={"close": close_column, "daily_return_pct": return_column})
        observation[date_column] = future_date.date().isoformat()
        enriched = enriched.merge(observation, on="code", how="left")
    return enriched


def run_backtest(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    news: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    prepared = _prepare_prices(prices, config)
    dates = sorted(prepared["date"].dropna().unique())
    warmup = max(2, int(config["volume_lookback"]) + 1)
    trades: list[pd.DataFrame] = []
    for index in range(warmup, len(dates) - 1):
        signal_date = pd.Timestamp(dates[index])
        picks = evaluate_signal(prepared, universe, news, config, asof=signal_date)
        picks = picks.loc[picks["passes"]].head(int(config["top_n"]))
        if picks.empty:
            continue
        next_date = pd.Timestamp(dates[index + 1])
        next_bars = prepared.loc[prepared["date"] == next_date, ["code", "open", "close"]].rename(
            columns={"open": "entry_price", "close": "exit_price"}
        )
        picks = picks.merge(next_bars, on="code", how="inner")
        if picks.empty:
            continue
        picks["signal_date"] = signal_date.date().isoformat()
        picks["entry_date"] = next_date.date().isoformat()
        picks["holding_return_pct"] = (picks["exit_price"] / picks["entry_price"] - 1.0) * 100.0
        trades.append(picks)
    trade_frame = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    returns = trade_frame["holding_return_pct"] if not trade_frame.empty else pd.Series(dtype=float)
    summary = {
        "signals": float(len(trade_frame)),
        "mean_holding_return_pct": float(returns.mean()) if not returns.empty else 0.0,
        "win_rate_pct": float((returns > 0).mean() * 100.0) if not returns.empty else 0.0,
    }
    return trade_frame, summary
