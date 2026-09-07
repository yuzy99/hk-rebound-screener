from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# 行业分类中英文对照字典
INDUSTRY_TRANSLATIONS: dict[str, str] = {
    "Apparel Manufacturing": "服装制造",
    "Apparel Retail": "服装与运动零售",
    "Footwear & Accessories": "鞋类与配饰",
    "Internet Retail": "互联网电商零售",
    "Specialty Retail": "专卖零售",
    "Home Improvement Retail": "家居建材零售",
    "Department Stores": "百货商店",
    "Grocery Stores": "超市便利",
    "Beverages - Non-Alcoholic": "非酒精饮料",
    "Beverages - Brewers": "啤酒酿造",
    "Beverages - Wineries & Distilleries": "白酒与葡萄酒",
    "Packaged Foods": "预包装食品与调味品",
    "Restaurants": "餐饮连锁",
    "Household & Personal Products": "日化与个人护理",
    "Consumer Electronics": "消费电子产品",
    "Auto Manufacturers": "整车制造",
    "Auto Parts": "汽车零部件",
    "Recreational Vehicles": "房车与休闲车",
    "Furnishings, Fixtures & Appliances": "家电与家居用品",
    "Semiconductors": "半导体芯片",
    "Semiconductor Equipment & Materials": "半导体设备与材料",
    "Software - Application": "应用软件与 SaaS",
    "Software - Infrastructure": "基础软件与安全",
    "Information Technology Services": "IT 咨询与服务",
    "Communication Equipment": "通信网络设备",
    "Computer Hardware": "电脑与服务器硬件",
    "Electronic Components": "电子元器件",
    "Internet Content & Information": "互联网信息与社交媒体",
    "Entertainment": "影视娱乐与传媒",
    "Telecom Services": "电信运营服务",
    "Biotechnology": "生物科技与基因",
    "Drug Manufacturers - General": "综合大型制药",
    "Drug Manufacturers - Specialty & Generic": "专科与仿制药",
    "Medical Devices": "医疗器械与耗材",
    "Medical Instruments & Supplies": "医疗仪器与试剂",
    "Diagnostics & Research": "医学诊断与研发服务",
    "Healthcare Plans": "医疗保险与健康服务",
    "Medical Care Facilities": "医疗机构与医院",
    "Waste Management": "废物处理与环保服务",
    "Pollution & Treatment Controls": "污染治理与净化",
    "Aerospace & Defense": "航空航天与国防",
    "Specialty Industrial Machinery": "专用工业机械",
    "Farm & Heavy Construction Machinery": "工程机械与重型装备",
    "Electrical Equipment & Parts": "电气设备与电网部件",
    "Packaging & Containers": "包装与容器制造",
    "Conglomerates": "综合性多元化集团",
    "Building Products & Equipment": "建筑建材与设备",
    "Steel": "钢铁冶炼与加工",
    "Chemicals": "基础与特种化学品",
    "Specialty Chemicals": "精细特种化学品",
    "Other Industrial Metals & Mining": "工业金属采矿",
    "Gold": "黄金与贵金属",
    "Oil & Gas Integrated": "综合油气龙头",
    "Oil & Gas E&P": "石油与天然气勘探开采",
    "Oil & Gas Refining & Marketing": "炼油与成品油销售",
    "Solar": "光伏太阳能",
    "Utilities - Renewable": "清洁与可再生能源",
    "Utilities - Regulated Electric": "电力电网公用事业",
    "Banks - Diversified": "综合商业银行",
    "Banks - Regional": "区域性商业银行",
    "Credit Services": "信贷与支付服务",
    "Capital Markets": "证券与资本市场",
    "Asset Management": "资产管理与财富管理",
    "Insurance - Life": "人寿保险",
    "Insurance - Property & Casualty": "财产与意外保险",
    "Insurance Brokers": "保险经纪与代理",
    "Real Estate - Development": "房地产开发",
    "Real Estate Services": "房地产服务与物业",
    "REIT - Diversified": "综合型不动产信托",
}


def _translate_industry(name: str) -> str:
    cleaned = str(name).strip()
    return INDUSTRY_TRANSLATIONS.get(cleaned, cleaned)


def _format_market_cap(val: Any, currency: str) -> str:
    if pd.isna(val) or val is None:
        return "-"
    try:
        val_num = float(val)
        if val_num <= 0:
            return "-"
        if val_num >= 1e12:
            return f"{val_num / 1e12:.2f} 万亿 {currency}"
        if val_num >= 1e8:
            return f"{val_num / 1e8:.2f} 亿 {currency}"
        if val_num >= 1e4:
            return f"{val_num / 1e4:.0f} 万 {currency}"
        return f"{val_num:.0f} {currency}"
    except Exception:
        return "-"


def _format_pe(val: Any) -> str:
    if pd.isna(val) or val is None:
        return "-"
    try:
        num = float(val)
        if num < 0:
            return "亏损(<0)"
        return f"{num:.1f}x"
    except Exception:
        return "-"


def _format_pb(val: Any) -> str:
    if pd.isna(val) or val is None:
        return "-"
    try:
        num = float(val)
        return f"{num:.2f}x"
    except Exception:
        return "-"


US_COMMON_NAMES: dict[str, str] = {
    "AAPL": "苹果 (Apple)",
    "MSFT": "微软 (Microsoft)",
    "NVDA": "英伟达 (NVIDIA)",
    "AMZN": "亚马逊 (Amazon)",
    "GOOGL": "谷歌A (Alphabet)",
    "GOOG": "谷歌C (Alphabet)",
    "META": "Meta (脸书)",
    "TSLA": "特斯拉 (Tesla)",
    "AMD": "超威半导体 (AMD)",
    "INTC": "英特尔 (Intel)",
    "QCOM": "高通 (Qualcomm)",
    "BABA": "阿里巴巴 (Alibaba)",
    "PDD": "拼多多 (PDD)",
    "JD": "京东 (JD.com)",
    "BIDU": "百度 (Baidu)",
    "NIO": "蔚来 (NIO)",
    "XPEV": "小鹏汽车 (XPeng)",
    "LI": "理想汽车 (Li Auto)",
    "NFLX": "奈飞 (Netflix)",
    "DIS": "迪士尼 (Disney)",
}


def _clean_stock_name(code: str, raw_name: str, market: str) -> str:
    code_upper = code.upper()
    if market.upper() == "US" and code_upper in US_COMMON_NAMES:
        return US_COMMON_NAMES[code_upper]
    name = str(raw_name).strip()
    for suffix in [" - Common Stock", " Common Stock", ", Inc.", " Inc.", " Corp.", " Corporation", " Ltd.", " Limited", " plc"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name


def format_markdown_report(
    result: pd.DataFrame,
    market: str,
    asof: str,
    config: dict[str, Any],
) -> str:
    """将选股结果格式化为美观、结构清晰的 Markdown 报表。"""
    strategy_mode = str(config.get("strategy_mode", "rebound")).lower()
    strategy_title = "双日大跌+未大幅反弹策略" if strategy_mode == "rebound" else "双日下跌+流动性策略"
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
        name = _clean_stock_name(code, row.get("name", ""), market)
        raw_industry = str(row.get("industry", "")).strip()
        industry = _translate_industry(raw_industry)
        close = f"{float(row['close']):.3f}".rstrip("0").rstrip(".") + f" {currency}" if pd.notna(row.get("close")) else "-"
        prior_ret = f"{float(row['prior_return_pct']):+.2f}%" if pd.notna(row.get("prior_return_pct")) else "-"
        daily_ret = f"{float(row['daily_return_pct']):+.2f}%" if pd.notna(row.get("daily_return_pct")) else "-"
        ind_avg = f"{float(row['industry_avg_return_pct']):+.2f}%" if pd.notna(row.get("industry_avg_return_pct")) else "-"
        lag = f"{float(row['lag_vs_industry_pct']):+.2f}%" if pd.notna(row.get("lag_vs_industry_pct")) else "-"
        vol_ratio = f"{float(row['volume_ratio']):.2f}x" if pd.notna(row.get("volume_ratio")) else "-"
        score = f"{float(row['score']):.2f}" if pd.notna(row.get("score")) else "-"

        market_cap_str = _format_market_cap(row.get("market_cap"), currency)
        pe_str = _format_pe(row.get("pe"))
        pb_str = _format_pb(row.get("pb"))

        # 手机端全中文专属卡片视图
        lines.extend([
            f"### {rank:02d}. `{code}` {name}",
            f"> 🏢 **所属行业**：{industry}",
            f"> 💰 **最新现价**：`{close}` ｜ **总市值**：`{market_cap_str}`",
            f"> 📊 **估值指标**：**PE** `{pe_str}` ｜ **PB** `{pb_str}`",
            f"> 📉 **两日涨跌**：**今日(T)** `{daily_ret}` ｜ **昨日(T-1)** `{prior_ret}`",
            f"> 🎯 **行业均值**：`{ind_avg}` ｜ **相对滞涨**：`{lag}`",
            f"> 📈 **异动量比**：`{vol_ratio}` ｜ ⭐ **综合评分**：**{score}**",
            "",
            "---",
            "",
        ])

    lines.extend([
        "> 💡 **PE / PB 极速参考**：",
        "> • **PE(市盈率-回本年限)**：<10 极便宜(黄金坑)；15~25 正常；>50 偏贵；<0 亏损避雷。",
        "> • **PB(市净率-家底折扣)**：<1.0 破净(打折甩卖)；1~3 正常；>5 偏贵(轻资产除外)。",
        "",
        "> ⚠️ *风险提示：以上结果基于量化技术面及 48 小时舆情排雷初筛，不构成投资建议。*",
        "",
    ])
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

