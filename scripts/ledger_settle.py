"""给信号台账结算 T+1..T+5 的前向收益。

台账里的信号是「当天看起来像超跌」的候选；这里补上它们后来实际怎么走，
让两三周后能回答「哪条规则、哪个因子真的有效」，而不是继续靠感觉调参。

复权走 `strategy._apply_split_adjustment`，**不走 `_prepare_prices`**：后者会对
code × date 做笛卡尔积 reindex 并 ffill 停牌日，把没有成交的日子填成前一天收盘价
——结算要的恰恰是「没成交就是 NaN」，否则会凭空造出无法执行的收益。复权这一步
两条路径完全等价（同一个函数、同一份拆股事件表）。

用法：
    PYTHONPATH=src python scripts/ledger_settle.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 这两个是下划线开头的模块内部件，但结算必须和主流程共用同一份复权实现，
# 自己照抄一遍迟早会漂移（README 记了两起由此引发的假数事故）。
from hk_rebound_screener.strategy import _apply_split_adjustment, normalize_code

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = PROJECT_ROOT / "data" / "ledger"
DEFAULT_SIGNALS = LEDGER_DIR / "us_signals.csv"
DEFAULT_FORWARD = LEDGER_DIR / "us_forward.csv"
DEFAULT_BENCHMARK = LEDGER_DIR / "us_benchmark.csv"
DEFAULT_HISTORY = PROJECT_ROOT / "data" / "yfinance_cache" / "us_history.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "yfinance_cache" / "us_splits.csv"

PRICE_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume"]
READ_CHUNK = 500_000
MAX_HORIZON = 5
WARMUP_DAYS = 5
LOOKAHEAD_DAYS = 15
KEY_COLUMNS = ("signal_date", "code", "horizon")
DATE_COLUMNS = ("signal_date", "date")
OUTPUT_COLUMNS = [
    "signal_date",
    "code",
    "horizon",
    "date",
    "open",
    "high",
    "low",
    "close",
    "signal_close",
    "t1_open",
    "ret_from_signal_close_pct",
    "ret_from_t1_open_pct",
]
# 台账里的 close 是扫描当天写下的复权价；重算的 signal_close 与它差太多，
# 说明文件名里的日期不是真正的信号日（历史快照的命名并不统一）。
PRICE_CHECK_TOLERANCE = 0.01


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="结算信号台账的 T+1..T+5 前向收益")
    parser.add_argument("--signals", default=str(DEFAULT_SIGNALS))
    parser.add_argument("--forward", default=str(DEFAULT_FORWARD))
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY), help="美股历史行情缓存")
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS), help="拆股事件表")
    return parser.parse_args()


def _read_prices_window(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """分块读 1 GB 行情缓存，只留下窗口内的行。

    全量 read_csv 会把 930 万行一次性铺开；这里只关心信号日前后几十天，
    边读边丢能省下绝大部分内存。
    """
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        encoding="utf-8",
        chunksize=READ_CHUNK,
        usecols=PRICE_COLUMNS,
        dtype={"code": "string"},
        parse_dates=["date"],
    ):
        chunk = chunk.loc[chunk["date"].between(start, end)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    frame = pd.concat(parts, ignore_index=True)
    frame["code"] = frame["code"].map(normalize_code)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _forward_dates(market_dates: pd.DatetimeIndex, signal_date: pd.Timestamp) -> list[pd.Timestamp]:
    position = market_dates.searchsorted(signal_date, side="right")
    return list(market_dates[position : position + MAX_HORIZON])


def _build_observations(
    signals: pd.DataFrame, market_dates: pd.DatetimeIndex, adjusted: pd.DataFrame
) -> pd.DataFrame:
    targets = [
        {"signal_date": signal_date, "date": date, "horizon": horizon}
        for signal_date in sorted(signals["signal_date"].unique())
        for horizon, date in enumerate(_forward_dates(market_dates, signal_date), start=1)
    ]
    if not targets:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    pairs = signals[["signal_date", "code"]].drop_duplicates()
    wanted = pairs.merge(pd.DataFrame(targets), on="signal_date", how="inner")
    observed = wanted.merge(adjusted, on=["code", "date"], how="left")

    signal_close = adjusted[["code", "date", "close"]].rename(
        columns={"date": "signal_date", "close": "signal_close"}
    )
    entry = observed.loc[observed["horizon"] == 1, ["signal_date", "code", "open"]].rename(
        columns={"open": "t1_open"}
    )
    observed = observed.merge(signal_close, on=["code", "signal_date"], how="left")
    observed = observed.merge(entry, on=["signal_date", "code"], how="left")

    close = observed["close"]
    observed["ret_from_signal_close_pct"] = (close / observed["signal_close"] - 1.0) * 100.0
    # 回测口径是 T+1 开盘买；用它做基准，止盈止损也能从 high/low 直接推。
    observed["ret_from_t1_open_pct"] = (close / observed["t1_open"] - 1.0) * 100.0
    return observed[OUTPUT_COLUMNS]


def _upsert(existing_path: Path, fresh: pd.DataFrame, subset: tuple[str, ...]) -> pd.DataFrame:
    if existing_path.exists():
        existing = pd.read_csv(existing_path, encoding="utf-8-sig", dtype={"code": "string"})
    else:
        existing = pd.DataFrame(columns=list(subset))
    combined = pd.concat([existing, fresh], ignore_index=True)
    # 从 CSV 读回来的是 str、刚算出来的是 Timestamp，混在一起不但排序会抛
    # TypeError，drop_duplicates 还会把「同一天」当成两个值、静默堆出重复行。
    # 先统一成 ISO 字符串：CSV 里干净，字典序也正好是时间序。
    for column in DATE_COLUMNS:
        if column in combined.columns:
            combined[column] = pd.to_datetime(combined[column], errors="coerce").dt.strftime("%Y-%m-%d")
    # keep="last"：同一批信号每天都会重算（T+5 要等五个交易日才出现），
    # 用本次结果覆盖旧的，跑多少次结果都一样。
    combined = combined.drop_duplicates(subset=list(subset), keep="last")
    return combined.sort_values(list(subset)).reset_index(drop=True)


def _report_price_check(observed: pd.DataFrame, signals: pd.DataFrame) -> None:
    anchor = observed.loc[observed["horizon"] == 1, ["signal_date", "code", "signal_close"]]
    anchor = anchor.merge(
        signals[["signal_date", "code", "close"]], on=["signal_date", "code"], how="inner"
    )
    anchor = anchor.dropna(subset=["signal_close", "close"])
    if anchor.empty:
        return
    deviation = (anchor["signal_close"] / anchor["close"].where(anchor["close"] != 0) - 1.0).abs()
    median = float(deviation.median())
    print(f"信号日价格校验：复权 close 与台账 close 的中位偏差 {median:.4%}")
    if median > PRICE_CHECK_TOLERANCE:
        worst = anchor.assign(deviation=deviation).nlargest(5, "deviation")
        print("  ⚠️ 偏差偏大，信号日可能不是文件名里的那个日期。偏差最大的几个：")
        for _, row in worst.iterrows():
            print(
                f"    {row['signal_date']} {row['code']}: "
                f"台账 {row['close']:.4g} vs 重算 {row['signal_close']:.4g}"
            )


def main() -> None:
    # Windows 控制台默认 cp936。stderr 也要配：SystemExit 的消息由解释器打到
    # stderr，只配 stdout 的话「缺拆股表」这种关键告警会显示成乱码——看不懂的
    # 告警等于没有告警。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = _parse_args()

    signals_path = Path(args.signals)
    if not signals_path.exists():
        raise SystemExit(f"台账不存在：{signals_path}；先跑 scripts/ledger_append.py")
    signals = pd.read_csv(signals_path, encoding="utf-8-sig", dtype={"code": "string"})
    signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce").dt.normalize()
    signals = signals.dropna(subset=["signal_date", "code"])

    signal_dates = sorted(signals["signal_date"].unique())
    start = pd.Timestamp(signal_dates[0]) - pd.Timedelta(days=WARMUP_DAYS)
    end = pd.Timestamp(signal_dates[-1]) + pd.Timedelta(days=LOOKAHEAD_DAYS)
    print(f"台账 {len(signals)} 行 / {len(signal_dates)} 个信号日：{start.date()} ~ {end.date()}")

    prices = _read_prices_window(Path(args.history), start, end)
    if prices.empty:
        raise SystemExit("行情窗口为空，检查 --history 与信号日期范围")
    splits_path = Path(args.splits)
    # 缺事件表时 _apply_split_adjustment 是「什么都不做」，不是报错——于是反向合股
    # 会留下一道断崖，被当成暴跌选进名单，前瞻收益还会冒出 +4173% 这种不可能成交的
    # 数（README 记了两起）。宁可这天不结算：一份被污染的台账比缺一天更糟，因为它
    # 恰恰毁掉这套东西唯一的用途——让真实胜率可查。
    if not splits_path.exists():
        raise SystemExit(
            f"缺拆股事件表 {splits_path}：没有它就不做复权，结算出的收益会是假数。\n"
            "  该文件由扫描主流程的 fetch_yfinance_splits_bulk 产出；"
            "先跑一次 `python -m hk_rebound_screener.main --mode live --full-market "
            "--config config.us.two_day_drop.json`，或从 CI 缓存恢复 data/yfinance_cache/。"
        )
    splits = pd.read_csv(splits_path, encoding="utf-8")
    adjusted = _apply_split_adjustment(prices, splits)
    # 非正收盘价在质量清单里单独列出，留着会算出无意义的收益率。
    adjusted = adjusted.loc[adjusted["close"] > 0]
    print(f"行情窗口 {len(adjusted)} 行 / {adjusted['code'].nunique()} 只")

    market_dates = pd.DatetimeIndex(sorted(adjusted["date"].unique()))
    observed = _build_observations(signals, market_dates, adjusted)
    if observed.empty:
        raise SystemExit("没有可结算的前向窗口：行情缓存还没走到信号日之后")

    settled = observed.dropna(subset=["date"])
    print(f"结出 {len(settled)} 条前向观测 / {observed['signal_date'].nunique()} 个信号日")
    for horizon in range(1, MAX_HORIZON + 1):
        dates = settled.loc[settled["horizon"] == horizon, "date"]
        status = dates.max().date().isoformat() if not dates.empty else "尚未发生"
        print(f"  T+{horizon}: {len(dates):>5d} 条  最新 {status}")

    _report_price_check(observed, signals)

    forward_path = Path(args.forward)
    forward = _upsert(forward_path, settled, KEY_COLUMNS)
    forward.to_csv(forward_path, index=False, encoding="utf-8-sig")
    print(f"\n前向收益 {forward_path}：{len(forward)} 行")

    ordered = adjusted.sort_values(["code", "date"])
    ordered["daily_ret_pct"] = ordered.groupby("code")["close"].pct_change() * 100.0
    benchmark = (
        ordered.groupby("date")["daily_ret_pct"]
        .agg(median_daily_ret_pct="median", mean_daily_ret_pct="mean", n_codes="count")
        .reset_index()
    )
    # 窗口第一天没有前一日价格，整列是 NaN，别当成「当天市场不涨不跌」写进去。
    benchmark = benchmark.loc[benchmark["n_codes"] > 0]
    benchmark_path = Path(args.benchmark)
    benchmark = _upsert(benchmark_path, benchmark, ("date",))
    benchmark.to_csv(benchmark_path, index=False, encoding="utf-8-sig")
    print(f"市场基准 {benchmark_path}：{len(benchmark)} 个交易日（全市场等权）")


if __name__ == "__main__":
    main()
