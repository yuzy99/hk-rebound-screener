from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def format_markdown_report(
    result: pd.DataFrame,
    market: str,
    asof: str,
    config: dict[str, Any],
) -> str:
    """将选股结果格式化为美观、结构清晰的 Markdown 报表。"""
    strategy_mode = str(config.get("strategy_mode", "rebound")).lower()
    strategy_title = "双日下跌+流动性策略" if strategy_mode == "two_day_drop" else "超跌反弹与行业滞涨策略"
    market_upper = market.upper()
    currency = "USD" if market_upper == "US" else "HKD"

    passed = result.loc[result["passes"]].copy() if not result.empty and "passes" in result.columns else pd.DataFrame()
    top_n = int(config.get("top_n", 20))
    passed = passed.head(top_n)

    lines: list[str] = [
        f"## 📊 {market_upper} 市场选股简报 ({asof})",
        "",
        f"- **运行策略**：{strategy_title} (`{strategy_mode}`)",
        f"- **扫描基准日期**：`{asof}`",
        f"- **命中符合条件标的**：**{len(passed)}** 只",
        "",
    ]

    if passed.empty:
        lines.extend([
            "> [!NOTE]",
            "> 今日扫描完毕，**未发现**同时满足跌幅、行业滞涨、流动性及无负面新闻条件的标的。",
            "",
        ])
        return "\n".join(lines)

    lines.append("---")
    lines.append("")

    for rank, (_, row) in enumerate(passed.iterrows(), start=1):
        code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()
        industry = str(row.get("industry", "")).strip()
        close = f"{float(row['close']):.3f}".rstrip("0").rstrip(".") + f" {currency}" if pd.notna(row.get("close")) else "-"
        prior_ret = f"{float(row['prior_return_pct']):+.2f}%" if pd.notna(row.get("prior_return_pct")) else "-"
        daily_ret = f"{float(row['daily_return_pct']):+.2f}%" if pd.notna(row.get("daily_return_pct")) else "-"
        ind_avg = f"{float(row['industry_avg_return_pct']):+.2f}%" if pd.notna(row.get("industry_avg_return_pct")) else "-"
        lag = f"{float(row['lag_vs_industry_pct']):+.2f}%" if pd.notna(row.get("lag_vs_industry_pct")) else "-"
        vol_ratio = f"{float(row['volume_ratio']):.2f}x" if pd.notna(row.get("volume_ratio")) else "-"
        score = f"{float(row['score']):.2f}" if pd.notna(row.get("score")) else "-"

        # 手机端卡片式设计，彻底杜绝竖屏列挤压
        lines.extend([
            f"### {rank:02d}. `{code}` {name}",
            f"> 🏢 **所属行业**：{industry}",
            f"> 💰 **最新收盘**：`{close}`（当日 `{daily_ret}`）",
            f"> 📉 **前序跌幅**：`{prior_ret}` ｜ **行业均值**：`{ind_avg}`",
            f"> 🎯 **相对滞涨**：`{lag}` ｜ **异动量比**：`{vol_ratio}`",
            f"> ⭐ **综合得分**：**{score}**",
            "",
            "---",
            "",
        ])

    lines.append("> 💡 *提示：以上结果基于量化技术面及 48 小时舆情排雷初筛，不构成投资建议。*")
    lines.append("")
    return "\n".join(lines)


def write_step_summary(content: str) -> bool:
    """写入 GitHub Actions 的 $GITHUB_STEP_SUMMARY 环境变量所指向的路径。"""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return False
    try:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(content + "\n\n")
        return True
    except Exception as error:
        print(f"WARN 写入 GITHUB_STEP_SUMMARY 失败: {error}")
        return False


def send_webhook_notification(
    content: str,
    title: str = "选股结果通知",
    webhook_url: str | None = None,
) -> bool:
    """发送 Webhook 通知，自动适配企业微信、钉钉、飞书等常见机器人格式。"""
    url = webhook_url or os.environ.get("NOTIFICATION_WEBHOOK")
    if not url or not url.strip():
        return False

    url = url.strip()
    headers = {"Content-Type": "application/json"}

    # PushPlus (微信直推)
    if "pushplus.plus" in url or (not url.startswith("http://") and not url.startswith("https://")):
        token = url
        if "token=" in url:
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(url)
            token = parse_qs(parsed.query).get("token", [token])[0]
        elif "/" in token:
            token = token.rstrip("/").split("/")[-1]

        target_url = "http://www.pushplus.plus/send"
        payload = {
            "token": token,
            "title": title,
            "content": content,
            "template": "markdown",
        }
        try:
            response = requests.post(target_url, json=payload, headers=headers, timeout=15)
            if response.status_code == 200:
                print("PushPlus 微信通知发送成功")
                return True
            print(f"WARN PushPlus 响应异常: HTTP {response.status_code}, 内容: {response.text[:200]}")
            return False
        except Exception as error:
            print(f"WARN PushPlus 发送失败: {error}")
            return False

    # Bark (iOS 原生通知)
    if "api.day.app" in url:
        payload = {
            "title": title,
            "body": content,
            "group": "StockScreener",
        }
    # 企业微信机器人
    elif "qyapi.weixin.qq.com" in url:
        payload = {
            "msgtype": "markdown",
            "markdown": {"content": content},
        }
    # 钉钉机器人
    elif "oapi.dingtalk.com" in url:
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": content,
            },
        }
    # 飞书机器人
    elif "open.feishu.cn" in url or "open.larksuite.com" in url:
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "blue",
                },
                "elements": [
                    {"tag": "markdown", "content": content}
                ],
            },
        }
    # 默认通用 JSON payload
    else:
        payload = {
            "title": title,
            "text": content,
            "content": content,
        }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            print("Webhook 通知发送成功")
            return True
        print(f"WARN Webhook 响应异常: HTTP {response.status_code}, 内容: {response.text[:200]}")
        return False
    except Exception as error:
        print(f"WARN Webhook 发送失败: {error}")
        return False

