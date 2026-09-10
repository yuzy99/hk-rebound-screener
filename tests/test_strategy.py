from pathlib import Path

import pandas as pd
import pytest

from hk_rebound_screener.strategy import (
    _compute_industry_benchmark,
    append_forward_observations,
    evaluate_signal,
    load_config,
    load_news,
    load_prices,
    load_universe,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_sample_signal_and_news_gate() -> None:
    config = load_config(ROOT / "config.demo.json")
    universe = load_universe(ROOT / "universe.csv")
    prices = load_prices(ROOT / "data" / "sample" / "prices.csv")
    news = load_news(ROOT / "data" / "sample" / "news.csv")
    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")

    tencent = result.loc[result["code"] == "00700"].iloc[0]
    assert bool(tencent["passes"])
    assert tencent["prior_return_pct"] <= -5.0
    assert tencent["industry_avg_return_pct"] >= 1.0
    assert tencent["lag_vs_industry_pct"] >= 2.0
    assert tencent["lot_value_hkd"] <= 30000

    byd = result.loc[result["code"] == "01211"].iloc[0]
    assert byd["negative_news_score"] >= 5.0
    assert not bool(byd["passes"])


def test_us_sample_signal_and_news_gate() -> None:
    config = load_config(ROOT / "config.us.demo.json")
    universe = load_universe(ROOT / "universe.us.csv", market="US")
    prices = load_prices(ROOT / "data" / "sample" / "us_prices.csv")
    news = load_news(ROOT / "data" / "sample" / "us_news.csv")
    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")

    apple = result.loc[result["code"] == "AAPL"].iloc[0]
    assert bool(apple["passes"])
    assert apple["prior_return_pct"] <= -5.0
    assert apple["industry_avg_return_pct"] >= 1.0
    assert apple["lag_vs_industry_pct"] >= 2.0
    assert apple["lot_value_usd"] <= 30000 / config["usd_hkd_rate"]

    tesla = result.loc[result["code"] == "TSLA"].iloc[0]
    assert tesla["negative_news_score"] >= 5.0
    assert not bool(tesla["passes"])

    trades, summary = run_backtest(prices, universe, news, config)
    assert summary["signals"] >= 1
    assert "AAPL" in set(trades["code"])


def test_two_day_drop_strategy_requires_three_day_pattern_and_liquidity() -> None:
    config = load_config(ROOT / "config.hk.two_day_drop.json")
    dates = pd.bdate_range(end="2026-09-02", periods=24)
    rows = []
    for code, closes, volume in [
        ("00001", [100.0] * 21 + [94.0, 92.0, 90.0], 60000.0),
        ("00002", [100.0] * 21 + [102.0, 104.0, 106.0], 60000.0),
        ("00003", [100.0] * 21 + [94.0, 92.0, 90.0], 1.0),
        ("00004", [100.0] * 21 + [101.0, 103.0, 105.0], 60000.0),
    ]:
        for date, close in zip(dates, closes):
            rows.append({"date": date, "code": code, "close": close, "volume": volume})
    prices = pd.DataFrame(rows)
    universe = pd.DataFrame(
        {
            "code": ["00001", "00002", "00003", "00004"],
            "name": ["Target", "Peer", "Illiquid", "Peer 2"],
            "industry": ["Tech", "Tech", "Other", "Tech"],
            "lot_size": [100.0, 100.0, 100.0, 100.0],
            "enabled": [True, True, True, True],
        }
    )
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])

    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")
    target = result.loc[result["code"] == "00001"].iloc[0]
    illiquid = result.loc[result["code"] == "00003"].iloc[0]

    assert bool(target["large_drop_ok"])
    assert bool(target["consecutive_down_ok"])
    assert bool(target["liquidity_ok"])
    assert bool(target["passes"])
    assert not bool(illiquid["liquidity_ok"])
    assert not bool(illiquid["passes"])


def test_two_day_drop_strategy_uses_three_trading_day_window() -> None:
    config = load_config(ROOT / "config.hk.two_day_drop.json")
    dates = pd.bdate_range(end="2026-09-02", periods=24)
    specs = [
        ("00001", [100.0] * 21 + [94.0, 94.376, 94.752], 60000.0),
        ("00002", [100.0] * 21 + [94.0, 94.564, 94.942], 60000.0),
        ("00003", [100.0] * 21 + [96.0, 96.384, 96.769], 60000.0),
    ]
    rows = [
        {"date": date, "code": code, "close": close, "volume": volume}
        for code, closes, volume in specs
        for date, close in zip(dates, closes)
    ]
    prices = pd.DataFrame(rows)
    universe = pd.DataFrame(
        {
            "code": ["00001", "00002", "00003"],
            "name": ["DropThenSmallUp", "DropThenTooMuchUp", "NoFivePctDrop"],
            "industry": ["Tech", "Tech", "Tech"],
            "lot_size": [100.0, 100.0, 100.0],
            "enabled": [True, True, True],
        }
    )
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])

    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")
    target = result.loc[result["code"] == "00001"].iloc[0]
    too_much_up = result.loc[result["code"] == "00002"].iloc[0]
    no_drop = result.loc[result["code"] == "00003"].iloc[0]

    assert bool(target["large_drop_ok"])
    assert bool(target["other_days_ok"])
    assert bool(target["three_day_drop_ok"])
    assert bool(target["passes"])
    assert not bool(too_much_up["other_days_ok"])
    assert not bool(too_much_up["passes"])
    assert not bool(no_drop["large_drop_ok"])
    assert not bool(no_drop["passes"])


def test_two_day_drop_strategy_does_not_require_industry_filters() -> None:
    config = load_config(ROOT / "config.hk.two_day_drop.json")
    dates = pd.bdate_range(end="2026-09-02", periods=24)
    prices = pd.DataFrame(
        [
            {"date": date, "code": "00001", "close": close, "volume": 60000.0}
            for date, close in zip(dates, [100.0] * 21 + [94.0, 92.0, 90.0])
        ]
    )
    universe = pd.DataFrame(
        {
            "code": ["00001"],
            "name": ["NoIndustryData"],
            "industry": [""],
            "lot_size": [100.0],
            "enabled": [True],
        }
    )
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])

    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")

    assert bool(result.iloc[0]["large_drop_ok"])
    assert bool(result.iloc[0]["consecutive_down_ok"])
    assert bool(result.iloc[0]["liquidity_ok"])
    assert bool(result.iloc[0]["passes"])


def test_two_day_drop_strategy_filters_out_intermittent_trading_days() -> None:
    config = load_config(ROOT / "config.hk.two_day_drop.json")
    dates = pd.bdate_range(end="2026-09-02", periods=24)
    rows = []
    # 00001: 正常交易股票，全周期 24 天均有成交（每天 60000 股）
    # 00002 & 00004: 同行基准股票
    for code, closes, volume in [
        ("00001", [100.0] * 21 + [94.0, 92.0, 90.0], 60000.0),
        ("00002", [100.0] * 21 + [102.0, 104.0, 106.0], 60000.0),
        ("00004", [100.0] * 21 + [101.0, 103.0, 105.0], 60000.0),
    ]:
        for date, close in zip(dates, closes):
            rows.append({"date": date, "code": code, "close": close, "volume": volume})

    # 00005: 间歇停牌/无成交的僵尸股，在过去 24 个市场交易日中仅在 5 天有交易记录（缺失 19 天）
    sparse_dates = [dates[0], dates[5], dates[10], dates[-2], dates[-1]]
    sparse_closes = [100.0, 100.0, 100.0, 92.0, 90.0]
    for date, close in zip(sparse_dates, sparse_closes):
        rows.append({"date": date, "code": "00005", "close": close, "volume": 30000.0})

    prices = pd.DataFrame(rows)
    universe = pd.DataFrame(
        {
            "code": ["00001", "00002", "00004", "00005"],
            "name": ["Target", "Peer", "Peer 2", "SparseTrading"],
            "industry": ["Tech", "Tech", "Tech", "Tech"],
            "lot_size": [100.0, 100.0, 100.0, 100.0],
            "enabled": [True, True, True, True],
        }
    )
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])

    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")
    sparse = result.loc[result["code"] == "00005"].iloc[0]

    # 日历对齐后，过去 20 个交易日实际成交天数仅 4 天（< 15），必须被过滤
    assert sparse["prior_traded_days"] < 15
    assert not bool(sparse["liquidity_ok"])
    assert not bool(sparse["passes"])


def test_notifier_markdown_report_and_step_summary(tmp_path: Path, monkeypatch) -> None:
    from hk_rebound_screener.notifier import format_markdown_report, write_step_summary

    config = load_config(ROOT / "config.demo.json")
    universe = load_universe(ROOT / "universe.csv")
    prices = load_prices(ROOT / "data" / "sample" / "prices.csv")
    news = load_news(ROOT / "data" / "sample" / "news.csv")
    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")

    report = format_markdown_report(result, market="HK", asof="2026-09-02", config=config)
    assert "00700" in report
    assert "市场选股简报" in report
    assert "01." in report
    assert "所属行业" in report

    summary_file = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    assert write_step_summary(report)
    assert summary_file.exists()
    assert "00700" in summary_file.read_text(encoding="utf-8")
    assert "PE / PB 极速参考" in report
    assert "<10" in report
    assert "<1.0" in report


def test_two_day_drop_any_day_logic() -> None:
    """测试新策略：近两天内任一天大跌 <= -5%，另一天涨幅 <= 0.5% 均可通过。"""
    config = load_config(ROOT / "config.demo.json")
    dates = pd.bdate_range(end="2026-09-02", periods=5)

    # 00001 (Tech): 昨天跌 -6%，今天涨 +0.2% -> 满足（大跌+微涨<=0.5%）
    # 00002 (Consumer): 昨天跌 -6%，今天涨 +1.2% -> 过滤（今天涨幅>0.5%）
    # 00003 (Finance): 昨天涨 +0.2%，今天大跌 -6% -> 满足（今天大跌，昨天涨<=0.5%）
    test_specs = [
        ("00001", [100.0, 100.0, 100.0, 94.0, 94.188], "Tech"),
        ("00004", [100.0, 100.0, 100.0, 100.0, 103.0], "Tech"),
        ("00005", [100.0, 100.0, 100.0, 100.0, 104.0], "Tech"),

        ("00002", [100.0, 100.0, 100.0, 94.0, 95.128], "Consumer"),
        ("00006", [100.0, 100.0, 100.0, 100.0, 103.0], "Consumer"),
        ("00007", [100.0, 100.0, 100.0, 100.0, 104.0], "Consumer"),

        ("00003", [100.0, 100.0, 100.0, 100.2, 94.188], "Finance"),
        ("00008", [100.0, 100.0, 100.0, 100.0, 103.0], "Finance"),
        ("00009", [100.0, 100.0, 100.0, 100.0, 104.0], "Finance"),
    ]
    rows = []
    for code, closes, _ in test_specs:
        for d, c in zip(dates, closes):
            rows.append({"date": d, "code": code, "close": c, "volume": 10000.0})
    prices = pd.DataFrame(rows)
    universe = pd.DataFrame({
        "code": [s[0] for s in test_specs],
        "name": [f"Stock_{s[0]}" for s in test_specs],
        "industry": [s[2] for s in test_specs],
        "lot_size": [100.0] * len(test_specs),
        "enabled": [True] * len(test_specs),
    })
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])

    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")
    p1 = result.loc[result["code"] == "00001"].iloc[0]
    p2 = result.loc[result["code"] == "00002"].iloc[0]
    p3 = result.loc[result["code"] == "00003"].iloc[0]

    assert bool(p1["two_day_drop_ok"])
    assert bool(p1["passes"])
    assert not bool(p2["two_day_drop_ok"])
    assert not bool(p2["passes"])
    assert bool(p3["two_day_drop_ok"])
    assert bool(p3["passes"])


def test_industry_lag_rebound_falls_back_to_today_rule_without_official_data() -> None:
    config = load_config(ROOT / "config.hk.industry_lag_rebound.json")
    universe = load_universe(ROOT / "universe.csv")
    prices = load_prices(ROOT / "data" / "sample" / "prices.csv")
    news = load_news(ROOT / "data" / "sample" / "news.csv")

    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")

    tencent = result.loc[result["code"] == "00700"].iloc[0]
    assert tencent["industry_return_source"] == "unavailable"
    assert pd.isna(tencent["industry_return_pct"])
    assert tencent["yesterday_return_pct"] == pytest.approx(-6.0)
    assert tencent["day_before_yesterday_return_pct"] == pytest.approx(0.990099, abs=1e-6)
    assert tencent["today_return_pct"] == pytest.approx(-0.500625, abs=1e-6)
    assert pd.isna(tencent["lag_vs_industry_pct"])
    assert tencent["signal_day_volume"] == pytest.approx(4000000.0)
    assert bool(tencent["signal_volume_ok"])
    assert not bool(tencent["prior_drop_ok"])
    assert not bool(tencent["passes"])
    meituan = result.loc[result["code"] == "03690"].iloc[0]
    assert not bool(meituan["industry_data_ok"])
    assert not bool(meituan["industry_relationship_checked"])
    assert bool(meituan["today_not_up_ok"])
    assert bool(meituan["prior_drop_ok"])
    assert bool(meituan["passes"])
    alibaba = result.loc[result["code"] == "09988"].iloc[0]
    assert not bool(alibaba["today_not_up_ok"])
    assert not bool(alibaba["passes"])

    from hk_rebound_screener.notifier import format_markdown_report

    report = format_markdown_report(result, market="HK", asof="2026-09-02", config=config)
    assert "行业涨幅" in report
    assert "无行业数据，不判断行业关系" in report
    assert "落后行业" in report
    assert "前天(T-2)涨跌" in report
    assert "仅生成筛选结果，不下单" in report
    assert "48 小时舆情" not in report


def test_industry_lag_rebound_uses_official_index_when_available() -> None:
    config = load_config(ROOT / "config.hk.industry_lag_rebound.json")
    universe = load_universe(ROOT / "universe.csv")
    universe["official_industry"] = "Information Technology"
    prices = load_prices(ROOT / "data" / "sample" / "prices.csv")
    news = load_news(ROOT / "data" / "sample" / "news.csv")
    industry_returns = pd.DataFrame({
        "date": [pd.Timestamp("2026-09-02")],
        "official_industry": ["Information Technology"],
        "index_code": ["HSCIIT"],
        "daily_return_pct": [0.5],
        "source": ["hang_seng_composite_industry_index"],
    })

    result = evaluate_signal(
        prices,
        universe,
        news,
        config,
        asof="2026-09-02",
        industry_returns=industry_returns,
    )

    meituan = result.loc[result["code"] == "03690"].iloc[0]
    assert bool(meituan["industry_data_ok"])
    assert bool(meituan["industry_relationship_checked"])
    assert meituan["industry_index_code"] == "HSCIIT"
    assert meituan["industry_return_pct"] == pytest.approx(0.5)
    assert meituan["industry_return_source"] == "hang_seng_composite_industry_index"
    assert bool(meituan["passes"])


def test_forward_observations_are_appended_after_screening() -> None:
    from hk_rebound_screener.notifier import format_markdown_report

    config = load_config(ROOT / "config.hk.industry_lag_rebound.json")
    universe = load_universe(ROOT / "universe.csv")
    prices = load_prices(ROOT / "data" / "sample" / "prices.csv")
    news = load_news(ROOT / "data" / "sample" / "news.csv")

    result = evaluate_signal(prices, universe, news, config, asof="2026-09-01")
    original_passes = result.set_index("code")["passes"].copy()
    enriched = append_forward_observations(result, prices, asof="2026-09-01", trading_days=2)

    pd.testing.assert_series_equal(enriched.set_index("code")["passes"], original_passes)
    tencent = enriched.loc[enriched["code"] == "00700"].iloc[0]
    assert tencent["forward_t1_date"] == "2026-09-02"
    assert tencent["forward_t1_close"] == pytest.approx(95.4)
    assert tencent["forward_t2_date"] == "2026-09-03"
    assert tencent["forward_t2_close"] == pytest.approx(97.0)

    report_input = enriched.copy()
    report_input.loc[report_input["code"] == "00700", "passes"] = True
    report = format_markdown_report(report_input, market="HK", asof="2026-09-01", config=config)
    assert "T+1 (2026-09-02)" in report
    assert "T+2 (2026-09-03)" in report


def test_industry_lag_rebound_uses_only_requested_conditions() -> None:
    config = load_config(ROOT / "config.hk.industry_lag_rebound.json")
    dates = pd.bdate_range(end="2026-09-02", periods=4)
    specs = [
        ("00001", "YesterdayDrop", "A", [100.0, 100.0, 95.0, 95.0]),
        ("00002", "PeerA1", "A", [100.0, 100.0, 100.0, 102.0]),
        ("00003", "PeerA2", "A", [100.0, 100.0, 100.0, 101.0]),
        ("00004", "EarlierDrop", "B", [100.0, 94.0, 93.0, 92.5]),
        ("00005", "PeerB", "B", [100.0, 100.0, 100.0, 103.0]),
        ("00006", "PriorTooSmall", "C", [100.0, 100.0, 95.1, 94.0]),
        ("00007", "IndustryCold", "D", [100.0, 100.0, 94.0, 93.0]),
        ("00008", "PeerD", "D", [100.0, 100.0, 100.0, 100.5]),
        ("00009", "MissingIndustry", "", [100.0, 100.0, 94.0, 93.0]),
    ]
    rows = []
    for code, _, _, closes in specs:
        for date, close in zip(dates, closes):
            rows.append({"date": date, "code": code, "close": close, "volume": 600001.0})
    prices = pd.DataFrame(rows)
    universe = pd.DataFrame(
        {
            "code": [spec[0] for spec in specs],
            "name": [spec[1] for spec in specs],
            "industry": [spec[2] for spec in specs],
            "lot_size": [None] * len(specs),
            "enabled": [True] * len(specs),
        }
    )
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])

    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")

    assert bool(result.loc[result["code"] == "00001", "passes"].iloc[0])
    low_volume = prices.copy()
    low_volume.loc[
        (low_volume["code"] == "00001") & (low_volume["date"] == low_volume["date"].max()),
        "volume",
    ] = 30000.0
    low_volume_result = evaluate_signal(low_volume, universe, news, config, asof="2026-09-02")
    low_volume_row = low_volume_result.loc[low_volume_result["code"] == "00001"].iloc[0]
    assert not bool(low_volume_row["signal_volume_ok"])
    assert not bool(low_volume_row["passes"])
    assert bool(result.loc[result["code"] == "00004", "passes"].iloc[0])
    assert not bool(result.loc[result["code"] == "00006", "passes"].iloc[0])
    assert bool(result.loc[result["code"] == "00007", "passes"].iloc[0])
    missing = result.loc[result["code"] == "00009"].iloc[0]
    assert not bool(missing["industry_data_ok"])
    assert bool(missing["today_not_up_ok"])
    assert bool(missing["passes"])


def test_industry_mean_and_lag_or_positive_rule() -> None:
    """验证新规则：行业平均涨幅必须 > 0，且（个股显著落后行业 >= 1% 或 当日涨幅 > 0）"""
    config = load_config(ROOT / "config.demo.json")
    dates = pd.bdate_range(end="2026-09-02", periods=5)

    # 行业 A：同行平均上涨 +0.2% (> 0)
    # 00001: 昨天 -6%，今天 +0.1% (> 0)，落后行业仅 0.1% (< 1.0) -> 因今日微涨为正，通过！
    # 00002: 同行 A1，今天 +0.2%
    # 00003: 同行 A2，今天 +0.2%
    #
    # 行业 B：同行平均上涨 +0.5% (> 0)
    # 00004: 昨天 -6%，今天 -1.0% (<= 0)，落后行业 1.5% (>= 1.0) -> 因落后行业大于 1%，通过！
    # 00005: 同行 B1，今天 +0.5%
    # 00006: 同行 B2，今天 +0.5%
    #
    # 行业 C：同行平均上涨 +0.2% (> 0)
    # 00007: 昨天 -6%，今天 -0.2% (<= 0)，落后行业 0.4% (< 1.0) -> 涨幅非正且落后未超 1%，淘汰！
    # 00008: 同行 C1，今天 +0.2%
    # 00009: 同行 C2，今天 +0.2%
    #
    # 行业 D：同行平均 0.0% (<= 0)
    # 00010: 昨天 -6%，今天 +0.2% (> 0) -> 行业均值未大于0，淘汰！
    # 00011: 同行 D1，今天 0.0%
    # 00012: 同行 D2，今天 0.0%
    specs = [
        ("00001", [100.0, 100.0, 100.0, 94.0, 94.094], "IndA"),
        ("00002", [100.0, 100.0, 100.0, 100.0, 100.2], "IndA"),
        ("00003", [100.0, 100.0, 100.0, 100.0, 100.2], "IndA"),

        ("00004", [100.0, 100.0, 100.0, 94.0, 93.06], "IndB"),
        ("00005", [100.0, 100.0, 100.0, 100.0, 100.5], "IndB"),
        ("00006", [100.0, 100.0, 100.0, 100.0, 100.5], "IndB"),

        ("00007", [100.0, 100.0, 100.0, 94.0, 93.812], "IndC"),
        ("00008", [100.0, 100.0, 100.0, 100.0, 100.2], "IndC"),
        ("00009", [100.0, 100.0, 100.0, 100.0, 100.2], "IndC"),

        ("00010", [100.0, 100.0, 100.0, 94.0, 94.188], "IndD"),
        ("00011", [100.0, 100.0, 100.0, 100.0, 100.0], "IndD"),
        ("00012", [100.0, 100.0, 100.0, 100.0, 100.0], "IndD"),
    ]
    rows = []
    for code, closes, _ in specs:
        for d, c in zip(dates, closes):
            rows.append({"date": d, "code": code, "close": c, "volume": 10000.0})
    prices = pd.DataFrame(rows)
    universe = pd.DataFrame({
        "code": [s[0] for s in specs],
        "name": [f"Stock_{s[0]}" for s in specs],
        "industry": [s[2] for s in specs],
        "lot_size": [100.0] * len(specs),
        "enabled": [True] * len(specs),
    })
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])

    result = evaluate_signal(prices, universe, news, config, asof="2026-09-02")
    r1 = result.loc[result["code"] == "00001"].iloc[0]
    r4 = result.loc[result["code"] == "00004"].iloc[0]
    r7 = result.loc[result["code"] == "00007"].iloc[0]
    r10 = result.loc[result["code"] == "00010"].iloc[0]

    # 00001 满足条件（微涨转正）
    assert r1["industry_avg_return_pct"] > 0
    assert r1["daily_return_pct"] > 0
    assert bool(r1["passes"])

    # 00004 满足条件（滞涨落后超过 1%）
    assert r4["industry_avg_return_pct"] > 0
    assert r4["daily_return_pct"] <= 0
    assert r4["lag_vs_industry_pct"] >= 1.0
    assert bool(r4["passes"])

    # 00007 淘汰（虽行业微涨，但跌幅未止且落后不足 1%）
    assert r7["industry_avg_return_pct"] > 0
    assert r7["daily_return_pct"] <= 0
    assert r7["lag_vs_industry_pct"] < 1.0
    assert not bool(r7["passes"])

    # 00010 淘汰（行业均值等于 0，不满足大于 0）
    assert r10["industry_avg_return_pct"] == pytest.approx(0.0)
    assert not bool(r10["passes"])


def test_industry_benchmark_methods_and_outlier_resilience() -> None:
    """验证行业基准计算对仙股/妖股极值的免疫能力（中位数与截断均值）"""
    # 模拟一个 10 只股票的行业，其中 9 只股票微涨微跌在 0%~0.6%，1 只妖股暴涨 +800%
    data = pd.DataFrame({
        "industry": ["SpecialtyMachinery"] * 10,
        "daily_return_pct": [0.2, 0.3, 0.4, 0.1, 0.5, 0.2, 0.3, -0.1, 0.4, 800.0],
        "turnover": [1e6] * 9 + [1e3],  # 妖股成交额很小
    })

    # 1. 传统 mean 模式：均值被暴拉至 80% 以上
    mean_bench, peer_count = _compute_industry_benchmark(data, method="mean")
    assert peer_count.iloc[0] == 9
    assert mean_bench.iloc[0] > 80.0  # 严重失真

    # 2. median 模式（默认）：完全免疫暴涨 800% 的极值
    median_bench, _ = _compute_industry_benchmark(data, method="median")
    assert median_bench.iloc[0] == pytest.approx(0.3)  # 中位数稳定在 0.3%
    assert median_bench.iloc[1] == pytest.approx(0.3)

    # 3. trimmed_mean 模式：剔除 10% 极值后均值回归正常
    trimmed_bench, _ = _compute_industry_benchmark(data, method="trimmed_mean")
    assert trimmed_bench.iloc[0] < 1.0  # 极值被剔除，均值在正常区间

    # 4. turnover_weighted 模式：资金加权后妖股影响微乎其微
    weighted_bench, _ = _compute_industry_benchmark(data, method="turnover_weighted")
    assert weighted_bench.iloc[0] < 1.0
