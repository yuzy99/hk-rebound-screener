"""把美股每日扫描结果追加进信号台账。

台账是纯观测层：只做记录，不重算因子、不参与选股，也不改动现有策略代码。
存在的理由——`outputs/` 被 .gitignore 排除，云端扫描又只把 CSV 传成 Actions
artifact（retention 14 天），于是每天最有价值的那份数据 14 天后就没了，
拼不成面板，也没法回答「哪条规则真的有效」。

用法：
    PYTHONPATH=src python scripts/ledger_append.py            # 回填 outputs 下所有可识别快照
    PYTHONPATH=src python scripts/ledger_append.py --date 2026-09-10
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = PROJECT_ROOT / "data" / "ledger"
DEFAULT_SIGNALS = LEDGER_DIR / "us_signals.csv"
DEFAULT_CSV_DIR = PROJECT_ROOT / "outputs"

KEY_COLUMNS = ("signal_date", "code")

# 同一天会以几种不同命名落盘，候选池口径并不一致，混在一起样本就不可比了，
# 所以每天只取一份。优先级由小到大：规范分档 > 短名分档 > 分档前的全市场文件。
_TIERED = re.compile(r"^scan_us_two_day_drop_(?P<slug>[a-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})\.csv$")
_SHORT = re.compile(r"^us_scan_(?P<date>\d{4}-\d{2}-\d{2})_(?P<tier>[0-9]+M)\.csv$")
_WHOLE_MARKET = re.compile(r"^scan_us_two_day_drop_(?P<date>\d{4}-\d{2}-\d{2})\.csv$")

# 同一天可能同时躺着档位文件和全市场文件，只能取一份，否则同一只票会入账两次。
# 两档切出来的 two_day_drop_ok 池其实是一样的（价格条件跟流动性无关），差异只在
# liquidity_ok/passes 这两个被按档重算的布尔列。所以取哪档都行——台账存的是
# prior_turnover_median 原始值而不是「过没过门槛」的结论，门槛要改再事后过滤，
# 不必回头重跑扫描（README 反复警告：门槛和跌幅权重必须一起验证）。
_TIER_RANK = {"50m": 0, "200m": 1}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把美股扫描快照追加进信号台账")
    parser.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR), help="扫描输出目录")
    parser.add_argument("--signals", default=str(DEFAULT_SIGNALS), help="台账文件路径")
    parser.add_argument("--date", default=None, help="只处理指定信号日 YYYY-MM-DD；默认全部")
    return parser.parse_args()


def _discover(csv_dir: Path, only_date: str | None) -> dict[str, tuple[int, str, Path, bool]]:
    """挑出每个信号日要入账的那一份快照。"""
    chosen: dict[str, tuple[int, str, Path, bool]] = {}
    for path in sorted(csv_dir.glob("*.csv")):
        name = path.name
        if match := _TIERED.match(name):
            slug = match.group("slug")
            date, whole_market = match.group("date"), False
            rank, tier = _TIER_RANK.get(slug, 2), slug.upper()
        elif match := _SHORT.match(name):
            date, tier, whole_market = match.group("date"), match.group("tier"), False
            rank = _TIER_RANK.get(tier.lower(), 2)
        elif match := _WHOLE_MARKET.match(name):
            # 分档之前落的是整份全市场结果，官方门槛就是 200M。
            date, tier, whole_market, rank = match.group("date"), "200M", True, 3
        else:
            continue
        if only_date and date != only_date:
            continue
        if date not in chosen or rank < chosen[date][0]:
            chosen[date] = (rank, tier, path, whole_market)
    return chosen


def _candidates(path: Path) -> pd.DataFrame:
    """只保留「接近信号」的行。

    同一个策略的两种来源行数差 50 倍：`apply_liquidity_tier` 产出的档位 CSV 是
    **全量行**（只把 liquidity_ok/passes 按该档门槛重算，行数不变，5000+ 行），
    而历史快照是另一次运行的产物、已经筛到 105 行。统一按价格条件
    `two_day_drop_ok` 切：它是各档共有的那一步，切完 09-09 得 341 行、
    09-10 得 105 行，两边口径才可比。
    """
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": "string"})
    if "code" not in frame.columns:
        raise SystemExit(f"{path.name} 缺 code 列，无法入账")
    if "two_day_drop_ok" not in frame.columns:
        print(f"  ⚠️ {path.name} 没有 two_day_drop_ok 列，整份入账")
        return frame
    # 留下的里面既有最终通过的，也有过了价格条件但卡在流动性/新闻上的——
    # 后者是现成的对照组，用来验证过滤器本身有没有用。
    return frame.loc[frame["two_day_drop_ok"].astype(bool)]


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(KEY_COLUMNS))
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"code": "string"})


def main() -> None:
    # Windows 控制台默认按 cp936 编码输出，中文在 Git Bash 里会显示成乱码。
    # stderr 一并配置：SystemExit 的消息走的是 stderr。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = _parse_args()
    csv_dir = Path(args.csv_dir)
    signals_path = Path(args.signals)
    signals_path.parent.mkdir(parents=True, exist_ok=True)

    chosen = _discover(csv_dir, args.date)
    if not chosen:
        # 认不出文件名 = 台账停止增长，而这件事本身没有任何症状：脚本退出码 0，
        # CI 那步挂着 continue-on-error，几周后才发现面板是空的。宁可当场报错。
        # 注意「已入账过」不会走到这里——_discover 只看文件在不在，不看是否新增。
        raise SystemExit(
            f"未在 {csv_dir} 找到可入账的美股扫描文件"
            f"{f'（已限定 --date {args.date}）' if args.date else ''}。\n"
            "  认得的命名：scan_us_two_day_drop_<档位>_<日期>.csv、"
            "us_scan_<日期>_<档位>.csv、scan_us_two_day_drop_<日期>.csv。\n"
            "  若扫描确实产出了文件却落到这里，说明命名变了，需要同步改本脚本的正则。"
        )

    fresh: list[pd.DataFrame] = []
    for date in sorted(chosen):
        _rank, tier, path, _whole_market = chosen[date]
        frame = _candidates(path)
        frame.insert(0, "signal_date", date)
        frame["source_tier"] = tier
        frame["source_file"] = path.name
        fresh.append(frame)
        print(f"{date}  {path.name:<42s} {len(frame):>5d} 行  tier={tier}")

    incoming = pd.concat(fresh, ignore_index=True)
    existing = _load_existing(signals_path)
    before = len(existing)
    combined = pd.concat([existing, incoming], ignore_index=True)
    # keep="last"：重跑同一天时用本次读到的覆盖旧的，跑多少次结果都一样。
    combined = combined.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
    combined = combined.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    combined.to_csv(signals_path, index=False, encoding="utf-8-sig")

    print(f"\n台账 {signals_path}")
    print(f"原有 {before} 行 → 新增 {len(combined) - before} 行 → 合计 {len(combined)} 行")


if __name__ == "__main__":
    main()
