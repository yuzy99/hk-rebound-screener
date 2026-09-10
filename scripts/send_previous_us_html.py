"""One-off PushPlus HTML delivery for the previous 146-stock US result."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_CSV = Path("outputs/us_passed_2026-09-09_run-34451320517.csv")


def safe_float(value: str | None) -> float | None:
    try:
        number = float(value or "")
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt_number(value: str | None, digits: int = 2) -> str:
    number = safe_float(value)
    return "—" if number is None else f"{number:,.{digits}f}"


def fmt_percent(value: str | None) -> str:
    number = safe_float(value)
    return "—" if number is None else f"{number:+.2f}%"


def fmt_multiple(value: str | None) -> str:
    number = safe_float(value)
    return "—" if number is None else f"{number:.2f}x"


def fmt_market_cap(value: str | None) -> str:
    number = safe_float(value)
    if number is None:
        return "—"
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    return f"${number:,.0f}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 146:
        raise SystemExit(f"expected 146 previous rows, got {len(rows)}")
    if any(str(row.get("passes", "")).lower() != "true" for row in rows):
        raise SystemExit("input contains a row that is not marked passes=True")
    return rows


def scan_date(path: Path) -> str:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    return match.group(1) if match else "历史日期"


def build_html(rows: list[dict[str, str]], source_path: Path) -> str:
    source_date = html.escape(scan_date(source_path))
    code_list = " · ".join(html.escape(row["code"]) for row in rows)
    items: list[str] = []
    for index, row in enumerate(rows, start=1):
        code = html.escape(row.get("code", "—"))
        today = fmt_percent(row.get("daily_return_pct"))
        yesterday = fmt_percent(row.get("prior_return_pct"))
        two_days_ago = fmt_percent(row.get("two_days_ago_return_pct"))
        lag = fmt_percent(row.get("lag_vs_industry_pct"))
        volume = fmt_multiple(row.get("volume_ratio"))
        score = fmt_number(row.get("score"))
        tone = "down" if today.startswith("-") else "up"
        items.append(
            f'<li><b>{code}</b> <i class="{tone}">今 {today}</i>'
            f' <span>昨 {yesterday} · 前 {two_days_ago} · 落 {lag} · 量 {volume} · 分 {score}</span></li>'
        )

    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{margin:0;background:#f3f6fa;color:#172b4d;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif}'
        '.wrap{max-width:720px;margin:auto;padding:12px}.hero{padding:17px 15px;border-radius:15px;background:linear-gradient(135deg,#123b79,#2878c8);color:#fff;box-shadow:0 4px 14px #123b7933}'
        '.hero h1{margin:4px 0;font-size:25px;line-height:31px}.hero p{margin:4px 0 0;opacity:.9;font-size:12px}.note,.codes,.details{margin-top:11px;padding:12px;border:1px solid #dfe7f1;border-radius:12px;background:#fff}.note{background:#f8fbff;color:#49627e;font-size:12px;line-height:19px}.codes{line-height:22px;color:#174ea6;font-weight:700;font-size:12px;word-break:break-word}.title{font-size:17px;font-weight:800;color:#152238;margin-bottom:8px}.details ul{margin:0;padding:0;list-style:none}.details li{padding:7px 0;border-top:1px solid #e8edf3;line-height:20px;font-size:12px}.details li:first-child{border-top:0}.details b{display:inline-block;min-width:47px;color:#152238;font-size:14px}.details i{font-style:normal;font-weight:800}.details .down{color:#b42318}.details .up{color:#067647}.details span{color:#68778d}@media(max-width:430px){.wrap{padding:9px}.hero h1{font-size:23px}.details li{padding:8px 0}}</style>'
        f'<title>US 美股完整通过名单｜{source_date}</title></head><body><div class="wrap">'
        '<header class="hero"><div style="font-size:11px;opacity:.8;letter-spacing:.5px">US MARKET · HISTORICAL RESULT</div>'
        '<h1>美股完整通过名单</h1>'
        f'<p>上次扫描日期：{source_date}　｜　共 {len(rows)} 只</p></header>'
        '<div class="note"><b style="color:#174ea6">说明：</b>以下完全复用上次美股全市场扫描的 146 条通过记录，本次只重新排版为 HTML，未重新获取行情、未重新筛选，也不代表当前实时信号。</div>'
        '<section class="codes"><div class="title">全部股票代码</div>' + code_list + '</section>'
        '<section class="details"><div class="title">完整明细 <span style="float:right;font-size:11px;color:#8996a6;font-weight:400">今/昨/前 · 落后 · 量比 · 评分</span></div><ul>'
        + "".join(items)
        + '</ul></section><div style="padding:12px 2px 0;color:#8996a6;font-size:11px;line-height:16px">历史筛选结果展示版，不构成投资建议；数据均沿用原始扫描记录。</div></div></body></html>'
    )


def extract_token(raw: str) -> str:
    value = raw.strip()
    if "token=" in value:
        parsed = urllib.parse.urlparse(value if value.startswith("http") else f"http://{value}")
        return urllib.parse.parse_qs(parsed.query).get("token", [""])[0]
    if "/" in value:
        return value.rstrip("/").split("/")[-1]
    return value


def send_html(content: str, token: str) -> tuple[int, dict[str, object]]:
    payload = {
        "token": token,
        "title": "US 美股 146 只完整名单｜HTML",
        "content": content,
        "template": "html",
    }
    request = urllib.request.Request(
        "http://www.pushplus.plus/send",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.loads(response.read().decode("utf-8"))
        return response.status, body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    content = build_html(rows, args.csv)
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    if args.render_only:
        print(f"html_render_ok rows={len(rows)} bytes={len(content.encode('utf-8'))}")
        return

    token = extract_token(__import__("os").environ.get("NOTIFICATION_WEBHOOK", ""))
    if not token:
        raise SystemExit("NOTIFICATION_WEBHOOK is not configured")
    try:
        status, body = send_html(content, token)
    except Exception as error:  # pragma: no cover - one-off delivery boundary
        raise SystemExit(f"PushPlus request failed: {error}") from error
    print(json.dumps({"http_status": status, "response": body}, ensure_ascii=False))
    if status != 200 or body.get("code") != 200:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
