"""并列的第二条规则「已趋于平稳」的单元测试。

线上规则一个字都不动，新规则只是多出一组 `stable_*` 列。这里盯三件事：

1. 语义：大跌必须落在 T-1/T-2/T-3，**T 当天自己是大跌日不算**。
   最后这条是这次改动的核心 —— q 是上限，光有 `R[0] <= q` 的话
   「T 就是大跌日本身」会通过，买点又变回大跌次日早上。
2. 隔离：不配 `stable_filter`（港股）时新列全 False，`passes` 和配了的时候逐行相同。
3. 推送：新规则段落整段排在现规则各档之前，现规则那半截逐字节不变。
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from hk_rebound_screener.notifier import (
    format_daily_push,
    format_markdown_report,
    format_tiered_report_link_notification,
)
from hk_rebound_screener.strategy import (
    apply_liquidity_tier,
    evaluate_signal,
    load_config,
    stable_filter_config,
    stable_rule_text,
)


ROOT = Path(__file__).resolve().parents[1]
US_CONFIG_PATH = ROOT / "config.us.two_day_drop.json"
USD_HKD_RATE = 7.84177017

DATES = pd.bdate_range(end="2026-09-02", periods=30)
VOLUME = 2_500_000.0  # close ≈ 100 → 成交额 ≈ 250M，稳过 200M 档


def _us_config() -> dict:
    config = load_config(US_CONFIG_PATH)
    config["usd_hkd_rate"] = USD_HKD_RATE
    return config


def _closes(returns: list[float]) -> list[float]:
    """把最后 len(returns) 天的涨跌幅（%）展开成收盘价，更早的日子铺平。

    铺平的头部与第一个 tail 值之间的收益正好是 returns[0]，所以最后四个
    交易日的 daily_return_pct 就是传入的这四个数。
    """
    closes = [100.0] * (len(DATES) - len(returns))
    price = 100.0
    for value in returns:
        price *= 1.0 + value / 100.0
        closes.append(price)
    return closes


# code → 最后四日涨跌幅 [T-3, T-2, T-1, T]
#   AAA 大跌在 T-3：现规则看不见（只回看 T-2/T-1/T），新规则看得见
#   BBB 大跌在 T-1：两条规则都命中
#   CCC T 当天自己是大跌日：现规则命中，新规则必须放掉
#   DDD T-1 反弹 +1.2%：超过 q=+1.0%，新规则放掉，现规则（上限 +1.5%）仍命中
#   EEE 大跌在 T-2 且此后安静：两条规则都命中
CASES: dict[str, list[float]] = {
    "AAA": [-6.0, -0.5, -0.3, -0.2],
    "BBB": [-0.2, -0.3, -6.0, -0.1],
    "CCC": [-0.2, -0.3, -0.1, -6.0],
    "DDD": [-0.2, -6.0, 1.2, -0.1],
    "EEE": [-0.2, -6.0, 0.8, -0.4],
}


def _fixture(config: dict | None = None) -> pd.DataFrame:
    config = config or _us_config()
    codes = list(CASES)
    prices = pd.DataFrame(
        [
            {"date": date, "code": code, "close": close, "volume": VOLUME}
            for code, returns in CASES.items()
            for date, close in zip(DATES, _closes(returns))
        ]
    )
    universe = pd.DataFrame(
        {
            "code": codes,
            "name": codes,
            "industry": ["Tech"] * len(codes),
            "lot_size": [1.0] * len(codes),
            "enabled": [True] * len(codes),
        }
    )
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])
    return evaluate_signal(prices, universe, news, config, asof="2026-09-02")


def _flags(frame: pd.DataFrame, column: str) -> dict[str, bool]:
    return frame.set_index("code")[column].astype(bool).to_dict()


# --------------------------------------------------------------------------
# 语义
# --------------------------------------------------------------------------


def test_stable_rule_needs_the_big_drop_to_be_before_today() -> None:
    """T 当天就是大跌日的 CCC：现规则命中，新规则必须放掉。"""
    frame = _fixture()
    passes = _flags(frame, "passes")
    stable = _flags(frame, "stable_passes")

    assert passes["CCC"] is True, "现规则本来就该抓 T 当天的大跌（不许改）"
    assert stable["CCC"] is False, "T 当天是大跌日不算「已趋于平稳」"


def test_stable_rule_sees_the_drop_that_shifted_one_day_back() -> None:
    """大跌在 T-3 的 AAA：现规则盲区，新规则命中。"""
    frame = _fixture()
    passes = _flags(frame, "passes")
    stable = _flags(frame, "stable_passes")

    assert passes["AAA"] is False
    assert stable["AAA"] is True


def test_stable_and_current_rules_overlap_where_they_should() -> None:
    frame = _fixture()
    passes = _flags(frame, "passes")
    stable = _flags(frame, "stable_passes")

    assert passes["BBB"] is True and stable["BBB"] is True
    assert passes["EEE"] is True and stable["EEE"] is True


def test_fourth_day_return_above_q_drops_the_stock() -> None:
    """DDD 的 T-1 是 +1.2%：过了现规则的 +1.5%，但过不了新规则的 +1.0%。"""
    frame = _fixture()
    config = _us_config()
    assert float(config["stable_filter"]["all_days_return_max_pct"]) == 1.0

    assert _flags(frame, "passes")["DDD"] is True
    assert _flags(frame, "stable_passes")["DDD"] is False


def test_stable_drop_strength_covers_all_four_days() -> None:
    """展示用的最大跌幅取四日窗口，不是现规则的三日窗口。"""
    frame = _fixture().set_index("code")

    # AAA 四日里最深的是 T-3 的 -6.0%，三日窗口（T-2/T-1/T）根本没有大跌。
    assert frame.loc["AAA", "stable_drop_strength_pct"] == pytest.approx(6.0)
    assert frame.loc["AAA", "drop_strength_pct"] == pytest.approx(0.5)
    # 四日累计 = 四个日收益之和
    assert frame.loc["AAA", "stable_four_day_return_pct"] == pytest.approx(-7.0)
    # CCC 的 T-3..T 是 -0.2/-0.3/-0.1/-6.0
    assert frame.loc["CCC", "stable_four_day_return_pct"] == pytest.approx(-6.6)


def test_stable_rule_text_names_the_configured_thresholds() -> None:
    config = _us_config()
    text = stable_rule_text(stable_filter_config(config))

    assert "T-1/T-2/T-3" in text
    assert "-4.5%" in text
    assert "+1%" in text
    assert "T 日 > -4.5%" in text


# --------------------------------------------------------------------------
# 隔离：新规则不许碰现规则的任何一行
# --------------------------------------------------------------------------


def test_missing_stable_filter_degrades_to_todays_behaviour() -> None:
    """港股配置没有 stable_filter：新列全 False，passes 逐行不变。"""
    config = _us_config()
    with_stable = _fixture(config)

    without = {key: value for key, value in config.items() if key != "stable_filter"}
    without_stable = _fixture(without)

    assert stable_filter_config(without)["enabled"] is False
    pd.testing.assert_series_equal(without_stable["passes"], with_stable["passes"])
    assert not without_stable["stable_passes"].any()
    assert not without_stable["stable_drop_ok"].any()
    # 没有 T-3 就没有四日窗口，展示列只能是 NaN，不能拿三天冒充四天。
    assert without_stable["stable_four_day_return_pct"].isna().all()


def test_stable_columns_do_not_disturb_the_current_score() -> None:
    config = _us_config()
    with_stable = _fixture(config)
    without = {key: value for key, value in config.items() if key != "stable_filter"}
    without_stable = _fixture(without)

    pd.testing.assert_series_equal(without_stable["score"], with_stable["score"])
    pd.testing.assert_series_equal(
        without_stable["drop_strength_pct"], with_stable["drop_strength_pct"]
    )


def test_tier_recompute_keeps_stable_passes_consistent() -> None:
    """换档后 stable_passes 必须等于用该档门槛重跑 evaluate_signal。"""
    config = _us_config()
    base = _fixture(config)

    loose = apply_liquidity_tier(base, config, 50e6)
    direct = _fixture(
        {
            **config,
            "liquidity_filter": {
                **config["liquidity_filter"],
                "min_prior_median_turnover": 50e6,
            },
        }
    )

    pd.testing.assert_series_equal(
        loose["stable_passes"], direct["stable_passes"], check_names=False
    )
    # 现规则的 passes 也照旧按新档重算
    pd.testing.assert_series_equal(
        loose["passes"], direct["passes"], check_names=False
    )


# --------------------------------------------------------------------------
# 报表
# --------------------------------------------------------------------------


def _report(stable_result: pd.DataFrame | None) -> str:
    config = _us_config()
    frame = _fixture(config)
    return format_markdown_report(
        frame,
        market="US",
        asof="2026-09-02",
        config=config,
        tier_label="200M 流动性门槛",
        stable_result=stable_result,
        stable_rules=stable_rule_text(stable_filter_config(config)),
    )


def _cards_before_and_after(report: str, marker: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """按 marker 把卡片标题切成两段，返回 [(编号, ticker)]。

    只比集合和编号连续性，不比组内先后 —— 同分股票的排序不是稳定排序。
    """
    start = report.index(marker)
    head, tail = report[:start], report[start:]
    pattern = re.compile(r"^### (\d+)\.\s+`?([A-Za-z0-9._-]+)`?")
    split = (
        [
            match.groups()
            for line in part.splitlines()
            if line.startswith("### ") and (match := pattern.search(line))
        ]
        for part in (head, tail)
    )
    return next(split), next(split)


def test_report_without_stable_result_is_untouched() -> None:
    report = _report(None)

    assert "已趋于平稳" not in report
    assert "## 命中股票代码（共 4 只）" in report
    assert "### 01. `BBB`" in report


def test_report_puts_stable_cards_first_with_continuous_numbering() -> None:
    report = _report(_fixture(_us_config()))
    marker = "## 📊 现规则 · 200M 流动性门槛"
    stable_cards, tier_cards = _cards_before_and_after(report, marker)

    assert report.index("## 🧊 已趋于平稳推荐（") < report.index(marker)

    # 新规则抓 AAA / BBB / EEE 三只，编号 01..03；现规则 BBB / CCC / DDD / EEE 从 04 接着排。
    assert [code for _, code in stable_cards] == ["BBB", "EEE", "AAA"]
    assert sorted(code for _, code in tier_cards) == ["BBB", "CCC", "DDD", "EEE"]
    assert [int(number) for number, _ in stable_cards] == [1, 2, 3]
    assert [int(number) for number, _ in tier_cards] == [4, 5, 6, 7]
    # 同一页不能出现第二段 01
    assert report.count("### 01. ") == 1
    # 新规则的卡片用四日窗口，现规则的卡片仍是三日
    assert "**四日信号**" in report
    assert "**四日累计涨跌**" in report
    assert "**三日信号**" in report


def test_stable_card_title_is_parseable_by_the_html_template() -> None:
    """模板的正则必须能从 `### 01. \\`BBB\\` BBB` 里抓出编号和 ticker。"""
    report = _report(_fixture(_us_config()))
    pattern = re.compile(r"(\d+)\.\s+`?([A-Za-z0-9._-]+)`?")
    titles = [line for line in report.splitlines() if line.startswith("### ")]

    assert len(titles) == 7
    for line in titles:
        match = pattern.search(line)
        assert match, line
        assert match.group(2) in CASES


def test_template_knows_about_the_four_day_signal_line() -> None:
    template = (ROOT / "templates" / "stock_report_template.html").read_text(encoding="utf-8")

    assert "四日信号" in template


# --------------------------------------------------------------------------
# 微信推送
# --------------------------------------------------------------------------


def _push(stable_result: pd.DataFrame | None, tiers: list[dict] | None = None) -> str:
    config = _us_config()
    frame = _fixture(config)
    tiers = tiers if tiers is not None else [
        {"label": "200M", "codes": ["BBB", "CCC"], "report_url": "https://x.test/us_200m.html"},
    ]
    return format_daily_push(
        "US",
        "2026-09-02",
        config,
        tiers,
        stable_result=stable_result,
        stable_rules=stable_rule_text(stable_filter_config(config)),
        stable_report_url="https://x.test/us_200m.html",
    )


def test_daily_push_puts_the_new_rule_before_every_tier() -> None:
    content = _push(_fixture(_us_config()))

    marker = "【200M 流动性门槛】"
    stable_cards, tier_cards = _cards_before_and_after(content, marker)

    assert content.index("## 🧊 已趋于平稳推荐（") < content.index(marker)
    # 新规则的三张卡片全部排在现规则段落之前，且现规则那段没有卡片。
    assert [code for _, code in stable_cards] == ["BBB", "EEE", "AAA"]
    assert tier_cards == []


def test_daily_push_keeps_the_existing_tier_block_byte_identical() -> None:
    """「保留线上的筛选」是字面意义上的：现规则那半截一个字符都不变。"""
    tiers = [
        {"label": "200M", "codes": ["BBB", "CCC"], "report_url": "https://x.test/us_200m.html"},
        {"label": "50M", "codes": ["AAA"], "report_url": "https://x.test/us_50m.html"},
    ]
    content = _push(_fixture(_us_config()), tiers=tiers)

    plain = format_tiered_report_link_notification(tiers, "US", "2026-09-02")
    header, _, tier_body = plain.partition("\n")
    assert content.startswith(header)
    # 现规则那半截整段原样接在新规则段落后面，一个字符都没重排。
    assert content.endswith(tier_body)


def test_daily_push_without_stable_result_is_the_plain_tier_message() -> None:
    tiers = [{"label": "200M", "codes": ["BBB"], "report_url": "https://x.test/us_200m.html"}]

    assert _push(None, tiers=tiers) == format_tiered_report_link_notification(
        tiers, "US", "2026-09-02"
    )


def test_daily_push_lists_every_stable_code_after_the_cards() -> None:
    """正文里除卡片外还要有一份完整代码列表，且不跟着 push_max_cards 截断。"""
    config = _us_config()
    frame = _fixture(config)
    config["stable_filter"] = {**config["stable_filter"], "push_max_cards": 2}
    tiers = [{"label": "200M", "codes": [], "report_url": "https://x.test/us_200m.html"}]

    content = format_daily_push(
        "US", "2026-09-02", config, tiers,
        stable_result=frame, stable_rules="规则", stable_report_url="https://x.test/us_200m.html",
    )

    shown, _ = _cards_before_and_after(content, "【已趋于平稳】命中股票代码")
    assert [int(number) for number, _ in shown] == [1, 2]

    stable_codes = set(frame.loc[frame["stable_passes"].astype(bool), "code"].astype(str))
    assert len(stable_codes) == 3

    # 列表在卡片之后、报告链接之前，且列出全部 3 只（卡片只展开 2 张）
    list_start = content.index("【已趋于平稳】命中股票代码（共 3 只）")
    list_end = content.index("🔗 完整报告：https://x.test/us_200m.html")
    assert content.index("### 02. ") < list_start < list_end

    listed = set(re.findall(r"`([A-Za-z0-9._-]+)`", content[list_start:list_end]))
    assert listed == stable_codes
