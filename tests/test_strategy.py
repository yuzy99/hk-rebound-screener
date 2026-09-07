from pathlib import Path

import pandas as pd

from hk_rebound_screener.strategy import (
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


def test_two_day_drop_strategy_requires_consecutive_down_days_and_liquidity() -> None:
    config = load_config(ROOT / "config.hk.two_day_drop.json")
    dates = pd.bdate_range(end="2026-09-02", periods=24)
    rows = []
    for code, closes, volume in [
        ("00001", [100.0] * 21 + [94.0, 92.0, 90.0], 25000.0),
        ("00002", [100.0] * 21 + [102.0, 104.0, 106.0], 25000.0),
        ("00003", [100.0] * 21 + [94.0, 92.0, 90.0], 1.0),
        ("00004", [100.0] * 21 + [101.0, 103.0, 105.0], 25000.0),
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


def test_two_day_drop_strategy_filters_out_intermittent_trading_days() -> None:
    config = load_config(ROOT / "config.hk.two_day_drop.json")
    dates = pd.bdate_range(end="2026-09-02", periods=24)
    rows = []
    # 00001: 正常交易股票，全周期 24 天均有成交（每天 25000 股）
    # 00002 & 00004: 同行基准股票
    for code, closes, volume in [
        ("00001", [100.0] * 21 + [94.0, 92.0, 90.0], 25000.0),
        ("00002", [100.0] * 21 + [102.0, 104.0, 106.0], 25000.0),
        ("00004", [100.0] * 21 + [101.0, 103.0, 105.0], 25000.0),
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

