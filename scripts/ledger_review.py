"""复盘查询：把信号台账和前向收益拼起来，回答「哪条规则真的有效」。

台账只负责记录，判断留在这里。样本还小的时候这张表的意义不是给出结论，
而是让你看见自己的真实胜率和盈亏比——而不是继续凭印象调参。

用法：
    PYTHONPATH=src python scripts/ledger_review.py                      # 通过 vs 对照组
    PYTHONPATH=src python scripts/ledger_review.py --horizon 2
    PYTHONPATH=src python scripts/ledger_review.py --by drop_strength_pct
    PYTHONPATH=src python scripts/ledger_review.py --by score --bins 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = PROJECT_ROOT / "data" / "ledger"
DEFAULT_SIGNALS = LEDGER_DIR / "us_signals.csv"
DEFAULT_FORWARD = LEDGER_DIR / "us_forward.csv"
DEFAULT_BENCHMARK = LEDGER_DIR / "us_benchmark.csv"

# 回测口径：T+1 开盘买、T+n 收盘卖。台账另存了 high/low，够做止盈止损，
# 但那套规则换一次假设就要重算一遍，留在复盘阶段而不是灌进台账。
RETURN_COLUMN = "ret_from_t1_open_pct"
MAX_GROUP_VALUES = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="信号台账复盘查询")
    parser.add_argument("--signals", default=str(DEFAULT_SIGNALS))
    parser.add_argument("--forward", default=str(DEFAULT_FORWARD))
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--horizon", type=int, default=1, help="持有期 T+n，默认 1")
    parser.add_argument("--by", default="passes", help="分组列；给 none 只看总体")
    parser.add_argument("--bins", type=int, default=5, help="连续因子分几档")
    return parser.parse_args()


def _load(signals_path: Path, forward_path: Path, benchmark_path: Path) -> pd.DataFrame:
    for path in (signals_path, forward_path):
        if not path.exists():
            raise SystemExit(f"缺少 {path}；先跑 scripts/ledger_append.py 与 ledger_settle.py")
    signals = pd.read_csv(signals_path, encoding="utf-8-sig", dtype={"code": "string"})
    forward = pd.read_csv(forward_path, encoding="utf-8-sig", dtype={"code": "string"})
    # forward 和 signals 都有 close（一个是 T+n 的、一个是信号日的），
    # 让 forward 保留原名，台账那侧加后缀，免得把两者搞混。
    merged = forward.merge(
        signals, on=["signal_date", "code"], how="left", suffixes=("", "_signal")
    )

    if not benchmark_path.exists():
        merged["excess_pct"] = np.nan
        return merged
    benchmark = pd.read_csv(benchmark_path, encoding="utf-8-sig", dtype={"code": "string"})
    # 同期基准 = T+1..T+n 每日全市场中位收益的复利累乘。只依赖
    # (signal_date, horizon)，与个股无关，先去重再算，省得每只票重算一遍。
    window = forward[["signal_date", "horizon", "date"]].drop_duplicates()
    window = window.merge(benchmark[["date", "median_daily_ret_pct"]], on="date", how="left")
    cumulative = (
        window.groupby(["signal_date", "horizon"])["median_daily_ret_pct"].apply(
            lambda values: (1.0 + values / 100.0).prod() - 1.0
        )
        * 100.0
    ).rename("bench_pct")
    merged = merged.merge(cumulative.reset_index(), on=["signal_date", "horizon"], how="left")
    merged["excess_pct"] = merged[RETURN_COLUMN] - merged["bench_pct"]
    return merged


def _summarize(group: pd.DataFrame) -> pd.Series:
    returns = group[RETURN_COLUMN].dropna()
    excess = group["excess_pct"].dropna()
    if returns.empty:
        return pd.Series(
            {"n": 0, "胜率%": np.nan, "中位%": np.nan, "均值%": np.nan,
             "超额中位%": np.nan, "超额t": np.nan, "盈亏比": np.nan}
        )
    gains, losses = returns[returns > 0], returns[returns < 0]
    payoff = (
        gains.mean() / abs(losses.mean()) if not gains.empty and not losses.empty else np.nan
    )
    deviation = excess.std(ddof=1) if len(excess) > 1 else np.nan
    t_stat = (
        excess.mean() / (deviation / np.sqrt(len(excess)))
        if deviation and np.isfinite(deviation) and deviation > 0
        else np.nan
    )
    return pd.Series(
        {
            "n": int(len(returns)),
            "胜率%": (returns > 0).mean() * 100.0,
            "中位%": returns.median(),
            "均值%": returns.mean(),
            "超额中位%": excess.median() if not excess.empty else np.nan,
            "超额t": t_stat,
            "盈亏比": payoff,
        }
    )


def _grouped(frame: pd.DataFrame, by: str, bins: int) -> pd.DataFrame:
    values = frame[by]
    if not pd.api.types.is_numeric_dtype(values) or values.nunique() <= MAX_GROUP_VALUES:
        key = values
    else:
        key = pd.qcut(values, bins, duplicates="drop")
    table = frame.groupby(key, observed=True).apply(_summarize, include_groups=False)
    return table.sort_values("n", ascending=False)


def _render(table: pd.DataFrame) -> str:
    formatted = table.copy()
    for column in ("胜率%", "中位%", "均值%", "超额中位%"):
        if column in formatted:
            formatted[column] = formatted[column].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "-")
    for column in ("超额t", "盈亏比"):
        if column in formatted:
            formatted[column] = formatted[column].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    return formatted.to_string()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = _parse_args()
    merged = _load(Path(args.signals), Path(args.forward), Path(args.benchmark))

    subset = merged.loc[merged["horizon"] == args.horizon]
    if subset.empty:
        horizons = sorted(merged["horizon"].dropna().unique())
        raise SystemExit(f"没有 T+{args.horizon} 的观测；当前已有：{horizons}")

    dates = sorted(subset["signal_date"].dropna().unique())
    print(f"T+{args.horizon} ｜ {len(subset)} 条观测 ｜ {len(dates)} 个信号日 "
          f"{dates[0]} ~ {dates[-1]}")
    if len(dates) < 10:
        print("⚠️ 样本不足 10 个信号日，下面的数字只能当管线自检，不能当结论。")
    print()

    if args.by.lower() in {"none", ""}:
        print(_render(_summarize(subset).to_frame().T))
    else:
        if args.by not in subset.columns:
            raise SystemExit(f"台账里没有列 {args.by}")
        print(f"按 {args.by} 分组：")
        print(_render(_grouped(subset, args.by, args.bins)))
    print("\n口径：T+1 开盘买入、T+n 收盘卖出；超额 = 个股收益 − 同期全市场中位（复利累乘）。"
          "未计手续费与滑点，实盘会比这里低。")


if __name__ == "__main__":
    main()
