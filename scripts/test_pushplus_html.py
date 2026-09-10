from __future__ import annotations

import json
import os
import sys
from urllib.parse import parse_qs, urlparse

import requests


def extract_token(raw_value: str) -> str:
    value = raw_value.strip()
    if "token=" in value:
        parsed = urlparse(value)
        return parse_qs(parsed.query).get("token", [""])[0]
    if "/" in value:
        return value.rstrip("/").split("/")[-1]
    return value


def main() -> int:
    raw_webhook = os.environ.get("NOTIFICATION_WEBHOOK", "")
    token = extract_token(raw_webhook)
    if not token:
        print("NOTIFICATION_WEBHOOK is not configured", file=sys.stderr)
        return 1

    codes = ["TTAN", "TBBK", "CASY"]
    content = "<div>" + "<br>".join(f"<code>{code}</code>" for code in codes) + "</div>"
    payload = {
        "token": token,
        "title": "HTML 推送测试",
        "content": content,
        "template": "html",
    }
    response = requests.post(
        "http://www.pushplus.plus/send",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:200]}
    print(json.dumps({"http_status": response.status_code, "response": body}, ensure_ascii=False))
    if response.status_code != 200:
        return 1
    if isinstance(body, dict) and body.get("code") not in (None, 200):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
