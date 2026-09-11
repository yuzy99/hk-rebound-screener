"""按流动性门槛分档（200M / 50M）发布两份美股名单的回归测试。

最强的一条是 test_apply_liquidity_tier_matches_evaluate_signal_frame：把
evaluate_signal 里那段门槛逻辑抽成 apply_liquidity_tier 之后，用官方那档门槛
重算必须逐列逐行等于原结果。它过了就说明抽取没有改变任何行为。

离线回归（outputs/ 与 data/yfinance_cache/ 都在 .gitignore 里）只在本地跑，
缺文件时整组跳过。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hk_rebound_screener.notifier import format_markdown_report, render_html_report
from hk_rebound_screener.strategy import (
    apply_liquidity_tier,
    evaluate_signal,
    load_config,
    load_prices,
)


ROOT = Path(__file__).resolve().parents[1]
US_CONFIG_PATH = ROOT / "config.us.two_day_drop.json"
HK_CONFIG_PATH = ROOT / "config.hk.two_day_drop.json"

# 取自官方 base CSV 里的实际汇率；美股策略没有它连 lot_value 都算不出来。
USD_HKD_RATE = 7.84177017

US_BASE_CSV = ROOT / "outputs" / "scan_us_two_day_drop_2026-09-10.csv"
US_50M_CSV = ROOT / "outputs" / "us_scan_2026-09-10_50M.csv"
US_200M_CSV = ROOT / "outputs" / "us_scan_2026-09-10_200M.csv"
US_HISTORY_CSV = ROOT / "data" / "yfinance_cache" / "us_history.csv"
US_SPLITS_CSV = ROOT / "data" / "yfinance_cache" / "us_splits.csv"
US_INDUSTRY_CSV = ROOT / "data" / "cache" / "us_industry.csv"

# 官方 5,120 行结果与 2026-09-10 两档参考名单
OFFICIAL_ROW_COUNT = 5120
REFERENCE_50M_COUNT = 105


def _us_config() -> dict:
    config = load_config(US_CONFIG_PATH)
    config["usd_hkd_rate"] = USD_HKD_RATE
    return config


def _tier_fixture() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """三只票：AAA 过 200M 档，BBB 只过 50M 档，CCC 两档都不过。

    成交额 = close × volume：AAA 250M / BBB 60M / CCC 6M。
    """
    config = _us_config()
    dates = pd.bdate_range(end="2026-09-02", periods=24)
    rows = [
        {"date": date, "code": code, "close": close, "volume": volume}
        for code, volume in [("AAA", 2500000.0), ("BBB", 600000.0), ("CCC", 60000.0)]
        for date, close in zip(dates, [100.0] * 21 + [94.0, 92.0, 90.0])
    ]
    prices = pd.DataFrame(rows)
    universe = pd.DataFrame(
        {
            "code": ["AAA", "BBB", "CCC"],
            "name": ["Large", "Mid", "Tiny"],
            "industry": ["Tech", "Tech", "Tech"],
            "lot_size": [1.0, 1.0, 1.0],
            "enabled": [True, True, True],
        }
    )
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])
    return config, prices, universe, news


def _passed_codes(frame: pd.DataFrame) -> set[str]:
    return set(frame.loc[frame["passes"], "code"].astype(str))


def _flag_mismatches(left: pd.Series, right: pd.Series) -> int:
    return int((left.astype(bool) != right.astype(bool)).sum())


# --------------------------------------------------------------------------
# 单元测试（无网络，CI 可跑）
# --------------------------------------------------------------------------


def test_apply_liquidity_tier_matches_evaluate_signal_frame() -> None:
    """最强守卫：同档重算必须整帧相等，含列顺序、dtype 和行序。"""
    config, prices, universe, news = _tier_fixture()
    base = evaluate_signal(prices, universe, news, config, asof="2026-09-02")

    rebuilt = apply_liquidity_tier(
        base, config, float(config["liquidity_filter"]["min_prior_median_turnover"])
    )

    pd.testing.assert_frame_equal(rebuilt, base)


def test_looser_tier_only_adds_and_tighter_tier_only_removes() -> None:
    config, prices, universe, news = _tier_fixture()
    base = evaluate_signal(prices, universe, news, config, asof="2026-09-02")

    loose = apply_liquidity_tier(base, config, 50e6)
    tight = apply_liquidity_tier(base, config, 300e6)

    # score 不按档重算：同一只票在两个页面上分数必须相同。
    pd.testing.assert_series_equal(loose["score"], base["score"])
    pd.testing.assert_series_equal(tight["score"], base["score"])
    # 也绝不重排：入参已按 score 降序，行序必须原样保留。
    assert loose["code"].tolist() == base["code"].tolist()
    assert tight["code"].tolist() == base["code"].tolist()

    assert _passed_codes(base) == {"AAA"}
    assert _passed_codes(loose) == {"AAA", "BBB"}
    assert _passed_codes(tight) == set()
    assert _passed_codes(base) < _passed_codes(loose)
    assert _passed_codes(tight) < _passed_codes(base)
    assert int(loose["passes"].sum()) > int(base["passes"].sum())
    assert int(tight["passes"].sum()) < int(base["passes"].sum())


def test_non_two_day_drop_mode_returns_frame_unchanged() -> None:
    frame = pd.DataFrame({"code": ["AAA", "BBB"], "passes": [True, False]})

    out = apply_liquidity_tier(frame, {"strategy_mode": "rebound"}, 50e6)

    pd.testing.assert_frame_equal(out, frame)


def test_loosened_preliminary_pass_set_is_a_superset() -> None:
    """放宽初筛只能让候选集变大 —— 这是「50M 那批票也抓得到新闻」的前提。"""
    config, prices, universe, news = _tier_fixture()
    strict = evaluate_signal(prices, universe, news, config, asof="2026-09-02")

    loosest = min(float(tier["min_prior_median_turnover"]) for tier in config["liquidity_tiers"])
    loosened = evaluate_signal(
        prices,
        universe,
        news,
        {
            **config,
            "liquidity_filter": {
                **config["liquidity_filter"],
                "min_prior_median_turnover": loosest,
            },
        },
        asof="2026-09-02",
    )

    assert _passed_codes(strict) <= _passed_codes(loosened)
    assert _passed_codes(loosened) == {"AAA", "BBB"}


def test_tier_scoped_config_reports_its_own_threshold() -> None:
    """档位作用域 config 下，门槛条读的是这一档，公式串一个字不变。"""
    config = _us_config()
    tier_config = {
        **config,
        "liquidity_filter": {
            **config["liquidity_filter"],
            "min_prior_median_turnover": 50e6,
        },
    }

    report = format_markdown_report(
        pd.DataFrame(columns=["passes"]),
        market="US",
        asof="2026-09-10",
        config=tier_config,
        tier_label="50M 流动性门槛",
    )

    assert "US$50.00m" in report
    assert "US$200.00m" not in report
    assert "S = min(D, 15) + 2 × log₂(min(max(R, 1), 8)) + 2.5 × log₁₀(min(max(L, 1), 1000)) − N" in report
    assert "· 50M 流动性门槛" in report
    # 档位只出现在标题行，不额外塞一条门槛说明。
    assert report.count("50M 流动性门槛") == 1


def test_render_html_report_market_label_lands_in_title_and_h1(tmp_path: Path) -> None:
    template_path = tmp_path / "template.html"
    template_path.write_text(
        "<title>{{MARKET_NAME}}</title><h1>{{MARKET_NAME}}</h1>"
        '<script type="text/markdown" id="raw-data">old data</script>',
        encoding="utf-8",
    )

    plain_path = tmp_path / "plain.html"
    render_html_report(
        "x",
        market="US",
        asof="2026-09-10",
        template_path=template_path,
        output_path=plain_path,
    )
    plain = plain_path.read_text(encoding="utf-8")
    assert "<title>美股</title>" in plain and "<h1>美股</h1>" in plain

    labeled_path = tmp_path / "labeled.html"
    render_html_report(
        "x",
        market="US",
        asof="2026-09-10",
        template_path=template_path,
        output_path=labeled_path,
        market_label="美股（50M 流动性门槛）",
    )
    labeled = labeled_path.read_text(encoding="utf-8")
    assert "<title>美股（50M 流动性门槛）</title>" in labeled
    assert "<h1>美股（50M 流动性门槛）</h1>" in labeled
    # 不传 market_label 时，输出必须和带标签的那份只在标签处不同。
    assert labeled == plain.replace("美股", "美股（50M 流动性门槛）")


def test_us_config_declares_two_tiers_and_hk_declares_none() -> None:
    us = load_config(US_CONFIG_PATH)
    tiers = us["liquidity_tiers"]

    assert [tier["slug"] for tier in tiers] == ["200m", "50m"]
    assert [tier["label"] for tier in tiers] == ["200M", "50M"]
    thresholds = [float(tier["min_prior_median_turnover"]) for tier in tiers]
    assert min(thresholds) == 50e6
    # 官方基准仍是 200M，liquidity_tiers 只是额外多出的一档。
    assert float(us["liquidity_filter"]["min_prior_median_turnover"]) == 200e6
    assert min(thresholds) <= float(us["liquidity_filter"]["min_prior_median_turnover"])
    # slug 不带市场前缀：CSV 名读作 scan_us_two_day_drop_200m_<date>.csv，
    # 页面名由 market + slug 拼成，send_report_link.py 的日期正则可以原样复用。
    assert all(not tier["slug"].startswith("us") for tier in tiers)

    assert not load_config(HK_CONFIG_PATH).get("liquidity_tiers")


# --------------------------------------------------------------------------
# 离线回归（只在本地跑）
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (US_BASE_CSV.exists() and US_50M_CSV.exists() and US_200M_CSV.exists()),
    reason="outputs/ 在 .gitignore 里，离线回归只在本地跑",
)
def test_offline_regression_reruns_official_base_threshold() -> None:
    """A：从官方 5,120 行结果出发，重算两档，和已归档的参考名单对上。

    参考文件 us_scan_2026-09-10_50M.csv 的 passes/liquidity_ok 是陈旧的
    （105 行里只有 36 个 True —— 当时只筛了行、没回写标志位），所以只比
    行集、行序和 score，绝不拿它做 assert_frame_equal。
    """
    base = pd.read_csv(US_BASE_CSV, dtype={"code": str})
    config = _us_config()
    official = float(config["liquidity_filter"]["min_prior_median_turnover"])
    assert official == 200e6
    assert len(base) == OFFICIAL_ROW_COUNT

    rebuilt = apply_liquidity_tier(base, config, official)
    assert rebuilt["code"].tolist() == base["code"].tolist()
    assert _flag_mismatches(rebuilt["passes"], base["passes"]) == 0
    assert _flag_mismatches(rebuilt["liquidity_ok"], base["liquidity_ok"]) == 0

    ref_200m = pd.read_csv(US_200M_CSV, dtype={"code": str})
    assert base.loc[base["passes"].astype(bool), "code"].tolist() == ref_200m["code"].tolist()

    loose = apply_liquidity_tier(base, config, 50e6)
    ref_50m = pd.read_csv(US_50M_CSV, dtype={"code": str})
    picked = loose.loc[loose["passes"]]
    assert picked["code"].tolist() == ref_50m["code"].tolist()
    assert int(loose["passes"].sum()) == REFERENCE_50M_COUNT
    # score 不按档重算：50M 那份的分数必须和官方结果逐值相等。
    pd.testing.assert_series_equal(
        picked["score"].reset_index(drop=True),
        ref_50m["score"].reset_index(drop=True),
    )


@pytest.mark.skipif(
    not (US_HISTORY_CSV.exists() and US_SPLITS_CSV.exists() and US_INDUSTRY_CSV.exists()),
    reason="本地行情缓存缺失，离线回归跳过",
)
def test_offline_regression_rebuilds_official_frame_from_cached_prices() -> None:
    """B：不联网重建整条行情管线，在真实 5,120 行上再证一次抽取无副作用。

    唯一无法离线复现的是 name（来自 Nasdaq 主数据，实测 0/5120 匹配），
    它只用于展示，排除在严格比较之外。新闻相关列按构造就不同（真实那次抓了
    实时新闻），改为断言「每一行 passes 不符的记录在基准里都有新闻风险」。
    """
    base = pd.read_csv(US_BASE_CSV, dtype={"code": str}).set_index("code")
    config = _us_config()
    config["split_events"] = pd.read_csv(US_SPLITS_CSV)
    prices = load_prices(US_HISTORY_CSV)
    industry = pd.read_csv(US_INDUSTRY_CSV, dtype={"code": str})
    universe = pd.DataFrame(
        {
            "code": industry["code"],
            "name": industry["name"],
            "industry": industry["industry"],
            "lot_size": 1.0,
            "enabled": True,
        }
    )
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])

    out = evaluate_signal(prices, universe, news, config, asof="2026-09-10")
    rebuilt = apply_liquidity_tier(out, config, 200e6)

    # 抽取本身不改变任何东西
    pd.testing.assert_frame_equal(rebuilt, out)

    assert len(out) == OFFICIAL_ROW_COUNT
    rebuilt = rebuilt.set_index("code")
    shared = rebuilt.index.intersection(base.index)
    assert len(shared) == OFFICIAL_ROW_COUNT

    # 与新闻无关的列，逐行严格相等
    for column in (
        "liquidity_ok",
        "turnover",
        "prior_turnover_median",
        "prior_traded_days",
        "drop_strength_pct",
        "three_day_drop_ok",
        "volume_ratio",
        "lot_value_usd",
    ):
        left = rebuilt.loc[shared, column]
        right = base.loc[shared, column]
        if left.dtype == bool or right.dtype == bool:
            assert _flag_mismatches(left, right) == 0, column
        else:
            assert np.allclose(
                left.astype(float).to_numpy(),
                right.astype(float).to_numpy(),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            ), column

    # passes 的差异只可能来自离线跑没有新闻这件事
    differing = rebuilt.loc[shared, "passes"].astype(bool) != base.loc[shared, "passes"].astype(bool)
    if bool(differing.any()):
        offenders = base.loc[shared[differing.to_numpy()]]
        assert (
            (offenders["negative_news_score"] > 0) | (~offenders["news_fetch_ok"].astype(bool))
        ).all()
