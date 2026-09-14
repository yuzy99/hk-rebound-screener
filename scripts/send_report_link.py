from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from hk_rebound_screener.notifier import (
    format_daily_push,
    format_report_link_notification,
    format_tiered_report_link_notification,
    send_webhook_notification,
)
from hk_rebound_screener.strategy import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="推送股票代码和 GitHub Pages 报告链接")
    parser.add_argument("--market", required=True, choices=["HK", "US"])
    parser.add_argument("--csv-glob", default=None, help="单档推送：扫描结果 CSV 的 glob")
    parser.add_argument(
        "--tiers-manifest",
        default=None,
        help="分档推送：main.py 写出的档位清单 JSON；给了它就不再走 glob",
    )
    parser.add_argument(
        "--page-base-url",
        default=None,
        help="GitHub Pages 站点根地址；每档 URL 由它和 slug 拼出，避免 URL 与页面各说各话",
    )
    parser.add_argument(
        "--stable-csv",
        default=None,
        help="只调试用：覆盖清单里「已趋于平稳」那份 CSV 的路径",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="策略配置 JSON；默认按 --market 推断 config.<market>.two_day_drop.json",
    )
    return parser.parse_args()


def _passed_values(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "通过", "✅"})


def _load_scan_frame(csv_path: Path, column: str) -> pd.DataFrame:
    result = pd.read_csv(csv_path, dtype={"code": str})
    missing = [name for name in ("code", column) if name not in result.columns]
    if missing:
        raise SystemExit(f"扫描结果缺少 {' 或 '.join(missing)} 字段：{csv_path}")
    return result


def _count_passed(frame: pd.DataFrame, column: str) -> int:
    return int(_passed_values(frame[column]).sum())


def _load_passed_codes(csv_path: Path) -> list[str]:
    result = _load_scan_frame(csv_path, "passes")
    passed = _passed_values(result["passes"])
    return [str(code).strip() for code in result.loc[passed, "code"].tolist()]


def _stable_section(
    args: argparse.Namespace, manifest: dict, base_url: str
) -> tuple[pd.DataFrame | None, str, str]:
    """读清单里的「已趋于平稳」块，返回 (整帧, 规则文本, 该段报告链接)。"""
    block = manifest.get("stable")
    if not block:
        return None, "", ""
    csv_path = Path(args.stable_csv or str(block["csv"]))
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    if not csv_path.exists():
        raise SystemExit(f"没有找到「已趋于平稳」名单：{csv_path}")
    frame = _load_scan_frame(csv_path, "stable_passes")
    # 和档位名单同一条规矩：数量对不上就吵出来，绝不推一份和页面对不上的名单。
    hits = _count_passed(frame, "stable_passes")
    expected = int(block["count"])
    if hits != expected:
        raise SystemExit(
            f"{block['label']}推荐名单与清单不符：{csv_path.name} 命中 {hits} 只，"
            f"清单记的是 {expected} 只；stable_passes 列可能是陈旧的"
        )
    print(f"{block['label']}推荐：命中 {hits} 只")
    report_url = f"{base_url.rstrip('/')}/{args.market.lower()}_{block['tier_slug']}.html"
    return frame, str(block.get("rules", "")), report_url


def _tier_content(args: argparse.Namespace, manifest_path: Path) -> tuple[str, str]:
    base_url = (args.page_base_url or os.environ.get("REPORT_PAGE_BASE_URL", "")).strip()
    if not base_url:
        raise SystemExit("分档推送缺少 --page-base-url 或 REPORT_PAGE_BASE_URL")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asof = str(manifest.get("asof", "最新交易日"))
    tiers: list[dict[str, object]] = []
    for tier in manifest.get("tiers", []):
        csv_path = Path(tier["csv"])
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        codes = _load_passed_codes(csv_path)
        # 2026-09-10 那份 50M CSV 正是这么坏的：只筛了行、没回写 passes。
        # 数量对不上就吵出来，绝不推一份和页面对不上的名单。
        expected = int(tier["count"])
        if len(codes) != expected:
            raise SystemExit(
                f"档位 {tier['label']} 名单与清单不符：{csv_path.name} 命中 {len(codes)} 只，"
                f"清单记的是 {expected} 只；passes 列可能是陈旧的"
            )
        print(f"档位 {tier['label']}：命中 {len(codes)} 只")
        tiers.append({
            "label": str(tier["label"]),
            "codes": codes,
            "report_url": f"{base_url.rstrip('/')}/{args.market.lower()}_{tier['slug']}.html",
        })

    stable_result, stable_rules, stable_report_url = _stable_section(args, manifest, base_url)
    if stable_result is None:
        return format_tiered_report_link_notification(tiers, args.market, asof), asof

    config_path = Path(args.config or f"config.{args.market.lower()}.two_day_drop.json")
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        raise SystemExit(f"没有找到策略配置：{config_path}")
    content = format_daily_push(
        args.market,
        asof,
        load_config(config_path),
        tiers,
        stable_result=stable_result,
        stable_rules=stable_rules,
        stable_report_url=stable_report_url,
    )
    return content, asof


def main() -> None:
    args = _parse_args()
    if args.tiers_manifest:
        manifest_path = Path(args.tiers_manifest)
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        if not manifest_path.exists():
            raise SystemExit(f"没有找到档位清单：{args.tiers_manifest}")
        content, asof = _tier_content(args, manifest_path)
    else:
        if not args.csv_glob:
            raise SystemExit("需要 --csv-glob 或 --tiers-manifest")
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
    print(f"推送完成：{asof}；内容：\n{content}")


if __name__ == "__main__":
    main()
