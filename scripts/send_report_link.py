"""Send a link to the published static report through PushPlus."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


def token_from(value: str) -> str:
    value = value.strip()
    if "token=" in value:
        parsed = urllib.parse.urlparse(value if value.startswith("http") else f"http://{value}")
        return urllib.parse.parse_qs(parsed.query).get("token", [""])[0]
    if "/" in value:
        return value.rstrip("/").split("/")[-1]
    return value


def main() -> None:
    token = token_from(os.environ.get("NOTIFICATION_WEBHOOK", ""))
    page_url = os.environ.get("REPORT_PAGE_URL", "").rstrip("/")
    if not token or not page_url:
        raise SystemExit("PushPlus token or report URL is missing")
    report_url = f"{page_url}/us_previous_144_template.html"
    content = (
        "# US 美股历史筛选报告\n\n"
        "上次扫描的 144 只完整数据已嵌入手机适配 HTML 模板。\n\n"
        f"[打开完整 144 只股票报告]({report_url})\n\n"
        "本页面仅展示历史结果，没有重新扫描或改变策略。"
    )
    payload = {
        "token": token,
        "title": "US 美股 144 只完整报告",
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
    print(json.dumps({"http_status": status, "response": body, "report_url": report_url}, ensure_ascii=False))
    if status != 200 or body.get("code") != 200:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
