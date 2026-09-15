"""对任意自选名单做短线体检，并把「这个信号在这批票上到底行不行」当场验证一遍。

存在的理由：扫描器每天给的是**全市场**筛出来的名单，而人真正会盯的往往是十来只
自己挑的票。这两件事的风险完全不同——全市场等权篮子的优势来自宽度，手挑名单常常
是同一主题、高相关、高波动的几只，把同一个信号套上去未必成立，甚至符号相反。
没有这个脚本时，只能凭印象争论；有了它，结论当场就能算出来。

**它不做预测，也不调参。** 只回答两个问题：
  1. 这批票现在各自处于什么状态（用主流程同一套判据，不重写因子）
  2. 这个信号在**这批票自己的历史上**表现如何（扣全市场中位的超额）

⚠️ 历史研究那部分会读整份行情缓存来算全市场基准，慢（分钟级）。样本小的时候
没有任何结论价值——脚本会自己提示。

用法：
    PYTHONPATH=src python scripts/watchlist_review.py --codes SMR,OKLO,ORCL
    PYTHONPATH=src python scripts/watchlist_review.py --file mylist.txt
    PYTHONPATH=src python scripts/watchlist_review.py --codes SMR,OKLO --backtest
    PYTHONPATH=src python scripts/watchlist_review.py --codes SMR,OKLO --refresh
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 这些是模块内部件，但体检必须和主流程共用同一套复权与因子实现。自己照抄一遍
# 迟早会漂移（README 记了两起由此引发的假数事故）。
from hk_rebound_screener.strategy import _apply_split_adjustment, evaluate_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = PROJECT_ROOT / "data" / "yfinance_cache" / "us_history.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "yfinance_cache" / "us_splits.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "config.us.two_day_drop.json"
DEFAULT_INDUSTRY = PROJECT_ROOT / "data" / "cache" / "us_industry.csv"

READ_CHUNK = 500_000
MAX_HORIZON = 5
RISK_LOOKBACK = 20
RANGE_LOOKBACK = 252
# 基准样本太少的交易日（早年缓存稀疏）会把中位数算得没有意义。
MIN_BENCH_CODES = 100
MIN_STUDY_SIGNALS = 30


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自选名单短线体检 + 历史验证")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--codes", default=None, help="逗号分隔的代码，如 SMR,OKLO,ORCL")
    source.add_argument("--file", default=None, help="每行一个代码的文件（# 开头为注释）")
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--industry", default=str(DEFAULT_INDUSTRY))
    parser.add_argument("--asof", default=None, help="体检截止日 YYYY-MM-DD，默认用数据里的最新一天")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="联网抓最新行情（缓存会落后几天；抓取可能被限流）",
    )
    parser.add_argument("--backtest", action="store_true", help="额外跑历史信号验证（慢）")
    parser.add_argument("--horizon", type=int, default=3, help="持有到 T+n 收盘，默认 3")
    return parser.parse_args()


def _read_codes(args: argparse.Namespace) -> list[str]:
    if args.codes:
        raw = args.codes.replace("，", ",").split(",")
    else:
        text = Path(args.file).read_text(encoding="utf-8")
        raw = [line.split("#")[0] for line in text.splitlines()]
    codes = []
    for item in raw:
        code = item.strip().upper()
        if code and code not in codes:
            codes.append(code)
    if not codes:
        raise SystemExit("没有解析出任何代码")
    return codes


def _load_prices(codes: list[str], history_path: Path, splits_path: Path, refresh: bool) -> pd.DataFrame:
    """取这批代码的行情。

    默认读本地缓存；`--refresh` 时联网抓一份近 400 天的覆盖在上面——缓存是一天
    多次全市场扫描攒出来的，收盘后往往还没更新到最新交易日。
    """
    parts: list[pd.DataFrame] = []
    if history_path.exists():
        for chunk in pd.read_csv(
            history_path,
            encoding="utf-8",
            chunksize=READ_CHUNK,
            usecols=["date", "code", "open", "close", "volume"],
            dtype={"code": "string"},
            parse_dates=["date"],
        ):
            hit = chunk.loc[chunk["code"].isin(codes)]
            if not hit.empty:
                parts.append(hit)
    cached = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if refresh:
        from hk_rebound_screener.adapters import fetch_yfinance_history

        end = date.today()
        fresh = fetch_yfinance_history(
            codes, end - timedelta(days=400), end, auto_adjust=True, pause_seconds=0.3
        )
        fresh["code"] = fresh["code"].astype("string").str.upper()
        if not cached.empty:
            # 缓存与新抓都来自 yfinance auto_adjust，重叠日的收盘价应当一致；对不上
            # 说明两次抓取之间发生了拆股，那段时间不能混着用。
            merged = cached.merge(
                fresh[["date", "code", "close"]].rename(columns={"close": "close_fresh"}),
                on=["date", "code"],
                how="inner",
            )
            if not merged.empty:
                deviation = (merged["close_fresh"] / merged["close"] - 1.0).abs()
                overlap = float(deviation.median())
                print(f"缓存与新抓的重叠日校验：中位偏差 {overlap:.4%}（{len(merged)} 个重叠点）")
                if overlap > 0.01:
                    print("  ⚠️ 偏差偏大，可能期间有拆股；结果请谨慎对待")
            # 只保留缓存里没有的新日期，避免同一根 K 线混两种来源。
            newest = cached["date"].max()
            fresh = fresh.loc[fresh["date"] > newest]
        cached = pd.concat([cached, fresh], ignore_index=True)

    if cached.empty:
        raise SystemExit(f"没有取到任何行情；确认代码拼写，或加 --refresh 联网抓取")

    splits = pd.read_csv(splits_path) if splits_path.exists() else pd.DataFrame()
    frame = _apply_split_adjustment(cached, splits)
    frame = frame.loc[pd.to_numeric(frame["close"], errors="coerce") > 0]
    frame["code"] = frame["code"].astype("string").str.upper()
    return frame.sort_values(["code", "date"]).reset_index(drop=True)


def _universe(codes: list[str], industry_path: Path) -> pd.DataFrame:
    if industry_path.exists():
        ind = pd.read_csv(industry_path, dtype={"code": "string"})
        ind["code"] = ind["code"].astype("string").str.upper()
        lookup = ind.drop_duplicates("code").set_index("code")
    else:
        lookup = pd.DataFrame()
    names, industries = [], []
    for code in codes:
        if code in lookup.index:
            names.append(str(lookup.at[code, "name"]))
            industries.append(str(lookup.at[code, "industry"]))
        else:
            names.append(code)
            industries.append("")
    return pd.DataFrame(
        {"code": codes, "name": names, "industry": industries, "lot_size": 1.0, "enabled": True}
    )


def _snapshot(prices: pd.DataFrame, codes: list[str], config: dict, industry: Path,
              asof: str | None) -> pd.DataFrame:
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])
    return evaluate_signal(prices, _universe(codes, industry), news, config, asof=asof)


def _benchmark(history_path: Path, splits_path: Path) -> pd.DataFrame:
    """全市场等权中位日收益，以及 T+1..T+n 的复利累乘。

    口径与 scripts/ledger_review.py 一致：中位日收益逐日复利，而不是「区间涨跌幅的
    中位」——后者会被停牌和长度不齐的窗口污染。
    """
    if not history_path.exists():
        return pd.DataFrame(columns=["date"])
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        history_path,
        encoding="utf-8",
        chunksize=READ_CHUNK,
        usecols=["date", "code", "close"],
        dtype={"code": "string"},
        parse_dates=["date"],
    ):
        chunk = chunk.loc[pd.to_numeric(chunk["close"], errors="coerce") > 0]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=["date"])
    allp = _apply_split_adjustment(pd.concat(parts, ignore_index=True), pd.read_csv(splits_path))
    allp = allp.loc[pd.to_numeric(allp["close"], errors="coerce") > 0]
    allp = allp.sort_values(["code", "date"])
    allp["r"] = allp.groupby("code")["close"].pct_change() * 100.0
    bench = (
        allp.groupby("date")["r"].agg(median_daily_ret_pct="median", n_codes="count").reset_index()
    )
    bench = bench.loc[bench["n_codes"] >= MIN_BENCH_CODES].sort_values("date").reset_index(drop=True)
    one_plus = 1.0 + bench["median_daily_ret_pct"] / 100.0
    for horizon in range(1, MAX_HORIZON + 1):
        compounded = one_plus.rolling(horizon).apply(np.prod, raw=True).shift(-horizon)
        bench[f"bench_t{horizon}"] = (compounded - 1.0) * 100.0
    return bench


def _signal_study(prices: pd.DataFrame, config: dict, bench: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """这批票历史上每一次同类信号的前向表现。

    判据直接取自配置（large_drop_max_pct / other_day_return_max_pct），和主流程
    同源；52 周位置与风险因子都只用截至当日的数据，不含未来函数。
    """
    frame = prices.copy()
    grouped = frame.groupby("code", group_keys=False)
    frame["r"] = grouped["close"].pct_change() * 100.0
    frame["r_1"] = grouped["r"].shift(1)
    frame["r_2"] = grouped["r"].shift(2)
    frame["ret3"] = frame[["r", "r_1", "r_2"]].sum(axis=1, min_count=3)

    large = float(config.get("large_drop_max_pct", -4.5))
    other = float(config.get("other_day_return_max_pct", 1.5))
    window = frame[["r", "r_1", "r_2"]]
    frame["sig"] = window.le(large).any(axis=1) & window.le(other).all(axis=1)

    frame["t1_open"] = grouped["open"].shift(-1)
    frame[f"c{horizon}"] = grouped["close"].shift(-horizon)
    frame["fret"] = (frame[f"c{horizon}"] / frame["t1_open"] - 1.0) * 100.0

    frame["hi"] = grouped["close"].transform(
        lambda v: v.rolling(RANGE_LOOKBACK, min_periods=RANGE_LOOKBACK - 52).max()
    )
    frame["lo"] = grouped["close"].transform(
        lambda v: v.rolling(RANGE_LOOKBACK, min_periods=RANGE_LOOKBACK - 52).min()
    )
    span = (frame["hi"] - frame["lo"]).where(frame["hi"] > frame["lo"])
    frame["range_pos"] = (frame["close"] - frame["lo"]) / span * 100.0
    frame["max20"] = grouped["r"].transform(
        lambda v: v.shift(1).rolling(RISK_LOOKBACK, min_periods=RISK_LOOKBACK).max()
    )
    frame["std20"] = grouped["r"].transform(
        lambda v: v.shift(1).rolling(RISK_LOOKBACK, min_periods=RISK_LOOKBACK).std()
    )
    frame["drop_z"] = (-frame["ret3"]).div(frame["std20"] * np.sqrt(3.0)).where(frame["std20"] > 0)

    frame = frame.merge(bench[["date", f"bench_t{horizon}"]], on="date", how="left")
    frame["excess"] = frame["fret"] - frame[f"bench_t{horizon}"]
    keep = frame["sig"] & frame["fret"].notna() & frame["excess"].notna()
    return frame.loc[keep].copy()


def _summarize(values: pd.Series) -> pd.Series:
    clean = values.dropna()
    if clean.empty:
        return pd.Series({"n": 0, "胜率%": np.nan, "中位%": np.nan, "均值%": np.nan, "t": np.nan})
    deviation = clean.std(ddof=1)
    return pd.Series(
        {
            "n": int(len(clean)),
            "胜率%": (clean > 0).mean() * 100.0,
            "中位%": clean.median(),
            "均值%": clean.mean(),
            "t": clean.mean() / (deviation / np.sqrt(len(clean))) if deviation > 0 else np.nan,
        }
    )


def _render_snapshot(table: pd.DataFrame) -> None:
    view = table.copy()
    for column in (
        "close", "daily_return_pct", "prior_return_pct", "two_days_ago_return_pct",
        "three_day_return_pct", "drop_z", "prior_return_std_pct", "prior_max_return_pct",
        "volume_ratio",
    ):
        if column in view:
            view[column] = pd.to_numeric(view[column], errors="coerce").round(2)
    if "prior_turnover_median" in view:
        view["prior_turnover_median"] = (view["prior_turnover_median"] / 1e6).round(0)
    if "prior_amihud" in view:
        view["prior_amihud"] = (view["prior_amihud"] * 1e8).round(2)
    if "score" in view:
        view["score"] = pd.to_numeric(view["score"], errors="coerce").round(1)
    columns = [
        "code", "close", "two_days_ago_return_pct", "prior_return_pct", "daily_return_pct",
        "three_day_return_pct", "drop_z", "prior_return_std_pct", "prior_max_return_pct",
        "prior_turnover_median", "volume_ratio", "prior_amihud", "passes", "score",
    ]
    view = view[[c for c in columns if c in view.columns]]
    view = view.rename(
        columns={
            "two_days_ago_return_pct": "T-2%", "prior_return_pct": "T-1%",
            "daily_return_pct": "T%", "three_day_return_pct": "三日%", "drop_z": "z",
            "prior_return_std_pct": "σ20", "prior_max_return_pct": "MAX",
            "prior_turnover_median": "中位额$M", "volume_ratio": "量比",
            "prior_amihud": "Amihud×1e8", "score": "评分",
        }
    )
    print(view.to_string(index=False))
    if "passes" in table:
        print(f"\n通过 {int(table['passes'].sum())} / {len(table)}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = _parse_args()
    codes = _read_codes(args)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    # 主流程在拿到拆股事件表后才置这个键；这里直接用权威事件表，复权与主流程一致。
    splits_path = Path(args.splits)
    config["split_events"] = pd.read_csv(splits_path) if splits_path.exists() else pd.DataFrame()
    # 配置里这个键是 null，`setdefault` 不会覆盖它——主流程会联网抓实时汇率，这里
    # 用一个固定值即可：它只参与「单股不超过 HK$30,000」那条展示用的换算，
    # 不参与选股，没必要为它加一次网络往返。
    if not config.get("usd_hkd_rate"):
        config["usd_hkd_rate"] = 7.8
    if not splits_path.exists():
        print(f"⚠️ 缺拆股事件表 {splits_path}：不做复权，含合股的票会出现假断崖")

    prices = _load_prices(codes, Path(args.history), splits_path, args.refresh)
    print(f"行情 {len(prices)} 行 / {prices['code'].nunique()} 只 / 最新 {prices['date'].max().date()}")
    missing = [c for c in codes if c not in set(prices["code"])]
    if missing:
        print(f"⚠️ 没有行情的代码：{','.join(missing)}")
    print()

    print("=== 当前状态（判据与主流程同源）===")
    snapshot = _snapshot(prices, codes, config, Path(args.industry), args.asof)
    if snapshot.empty:
        raise SystemExit("体检结果为空：行情不足（至少需要 3 个交易日）")
    _render_snapshot(snapshot)

    if not args.backtest:
        print("\n（加 --backtest 可验证这个信号在这批票历史上是否成立）")
        return

    print(f"\n=== 历史信号验证（T+1 开盘买 → T+{args.horizon} 收盘卖，扣全市场中位）===")
    bench = _benchmark(Path(args.history), splits_path)
    if bench.empty:
        raise SystemExit("算不出市场基准，跳过后半段")
    study = _signal_study(prices, config, bench, args.horizon)
    if study.empty:
        raise SystemExit("这批票历史上没有出现过同类信号")
    print(
        f"信号 {len(study)} 次  {study['date'].min().date()} ~ {study['date'].max().date()}"
        f"  （{study['code'].nunique()} 只）"
    )
    if len(study) < MIN_STUDY_SIGNALS:
        print("⚠️ 样本太少，下面的数字不能当结论")
    print()
    print(_summarize(study["excess"]).round(2).to_string())
    print()
    print("按 52 周区间位置分档（低位 = 还在跌势里，高位 = 上升趋势中的回调）：")
    buckets = pd.cut(
        study["range_pos"], [-1, 20, 50, 101], labels=["低位<20%", "中位20-50%", "高位>50%"]
    )
    print(study.groupby(buckets, observed=True)["excess"].apply(_summarize).round(2).to_string())
    print()
    print("按 z（归一跌幅）分档：")
    z_buckets = pd.qcut(study["drop_z"], 3, labels=["小z", "中z", "大z"], duplicates="drop")
    print(study.groupby(z_buckets, observed=True)["excess"].apply(_summarize).round(2).to_string())
    print(
        "\n口径与 scripts/ledger_review.py 一致。样本小、多重比较多，分档结果尤其容易"
        "是噪声——要看的是方向能不能在换一批票之后还成立。"
    )


if __name__ == "__main__":
    main()
