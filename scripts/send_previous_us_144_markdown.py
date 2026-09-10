"""One-off PushPlus Markdown delivery for the previous 146-stock result minus WETO/BTCT."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


SOURCE_COMMIT = "caa9fdb"
SOURCE_PATH = "outputs/us_passed_2026-09-09_run-34451320517.csv"
EXCLUDED_CODES = {"WETO", "BTCT"}


def safe_float(value: str | None) -> float | None:
    try:
        number = float(value or "")
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: str | None, digits: int = 2) -> str:
    number = safe_float(value)
    return "—" if number is None else f"{number:.{digits}f}"


def pct(value: str | None) -> str:
    number = safe_float(value)
    return "—" if number is None else f"{number:+.2f}%"


def load_rows() -> list[dict[str, str]]:
    raw = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{SOURCE_PATH}"],
    ).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(raw, newline="")))
    if len(rows) != 146:
        raise SystemExit(f"expected 146 source rows, got {len(rows)}")
    filtered = [row for row in rows if row.get("code") not in EXCLUDED_CODES]
    if len(filtered) != 144:
        raise SystemExit(f"expected 144 rows after exclusion, got {len(filtered)}")
    return filtered


def build_markdown(rows: list[dict[str, str]]) -> str:
    codes = " · ".join(f"`{row['code']}`" for row in rows)
    lines = [
        "# US 美股历史通过名单（144只）",
        "",
        "**数据口径**：复用上次美股全市场扫描的 146 只通过结果，按此前确定的市值门槛剔除 `WETO`、`BTCT`，保留 144 只。",
        "",
        "**重要说明**：本消息只做历史结果的 Markdown 展示，没有重新获取行情、没有重新筛选，也不代表当前实时信号。",
        "",
        "## 全部股票代码",
        "",
        codes,
        "",
        "## 144只核心数据",
        "",
        "格式：**代码**｜今｜昨｜前｜落后行业｜量比｜评分",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}. **{row['code']}**｜今 {pct(row.get('daily_return_pct'))}｜"
            f"昨 {pct(row.get('prior_return_pct'))}｜前 {pct(row.get('two_days_ago_return_pct'))}｜"
            f"落 {pct(row.get('lag_vs_industry_pct'))}｜量 {fmt(row.get('volume_ratio'))}x｜"
            f"分 {fmt(row.get('score'))}"
        )
    lines.extend(
        [
            "",
            "---",
            "",
            "历史扫描结果展示版，不构成投资建议；涨跌幅、行业落后、量比和评分均沿用原始扫描记录。",
        ]
    )
    return "\n".join(lines)


def extract_token(raw: str) -> str:
    value = raw.strip()
    if "token=" in value:
        parsed = urllib.parse.urlparse(value if value.startswith("http") else f"http://{value}")
        return urllib.parse.parse_qs(parsed.query).get("token", [""])[0]
    if "/" in value:
        return value.rstrip("/").split("/")[-1]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_rows()
    content = build_markdown(rows)
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    if args.render_only:
        print(f"markdown_render_ok rows={len(rows)} chars={len(content)} bytes={len(content.encode('utf-8'))}")
        return

    token = extract_token(os.environ.get("NOTIFICATION_WEBHOOK", ""))
    if not token:
        raise SystemExit("NOTIFICATION_WEBHOOK is not configured")
    payload = {
        "token": token,
        "title": "US 美股历史名单（144只）｜Markdown",
        "content": content,
        "template": "markdown",
    }
    request = urllib.request.Request(
        "http://www.pushplus.plus/send",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
            status = response.status
    except Exception as error:  # pragma: no cover - one-off delivery boundary
        raise SystemExit(f"PushPlus request failed: {error}") from error
    print(json.dumps({"http_status": status, "response": body}, ensure_ascii=False))
    if status != 200 or body.get("code") != 200:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
