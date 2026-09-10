from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

from hk_rebound_screener.notifier import (
    format_report_link_notification,
    send_webhook_notification,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="推送股票代码和 GitHub Pages 报告链接")
    parser.add_argument("--market", required=True, choices=["HK", "US"])
    parser.add_argument("--csv-glob", required=True)
    return parser.parse_args()


def _passed_values(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "通过", "✅"})


def main() -> None:
    args = _parse_args()
    csv_paths = list(Path.cwd().glob(args.csv_glob))
    if not csv_paths:
        raise SystemExit(f"没有找到扫描结果 CSV：{args.csv_glob}")
    csv_path = max(csv_paths, key=lambda path: path.stat().st_mtime)
    result = pd.read_csv(csv_path, dtype={"code": str})
    if "passes" not in result.columns or "code" not in result.columns:
        raise SystemExit(f"扫描结果缺少 code 或 passes 字段：{csv_path}")
    result["passes"] = _passed_values(result["passes"])

    match = re.search(r"_(\d{4}-\d{2}-\d{2})\.csv$", csv_path.name)
    asof = match.group(1) if match else "最新交易日"
    report_url = os.environ.get("REPORT_PAGE_URL", "").strip()
    if not report_url:
        raise SystemExit("缺少 REPORT_PAGE_URL")
    content = format_report_link_notification(result, args.market, asof, report_url)
    title = f"{'美股' if args.market == 'US' else '港股'}市场选股结果（{asof}）"
    if not send_webhook_notification(content, title=title):
        raise SystemExit("PushPlus/通知发送失败")
    print(f"推送完成：{csv_path.name}；命中 {int(result['passes'].sum())} 只；报告链接：{report_url}")


if __name__ == "__main__":
    main()
