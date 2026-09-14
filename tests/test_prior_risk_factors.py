"""#6.1 MAX / #6.2 波动率 / #6.3 Amihud 三个 20 日风险因子的窗口口径回归。

重点不是「有没有这一列」，而是 **shift(1)**：窗口必须停在信号日之前。信号日那根
深跌本身就是被预测对象，混进窗口等于拿答案当特征 —— 而这三个因子恰恰是用来
判断「这根跌是错杀还是回吐」的，把跌本身算进去就自我循环了。

顺带守住 52 周高/低的取值逻辑：yfinance 1.7 的键是 yearHigh/yearLow，别的版本
写法不同，取不到要退化成 NaN 而不是抛异常。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hk_rebound_screener.adapters import _fast_info_value, _finite_float
from hk_rebound_screener.strategy import _prepare_prices, evaluate_signal


LOOKBACK = 20


def _frame_from_returns(returns_pct: list[float], code: str = "00001") -> pd.DataFrame:
    """returns_pct[i] 是第 i 天的日涨跌（%）；第 0 天只作为起点，收益记 NaN。"""
    dates = pd.bdate_range(end="2026-09-11", periods=len(returns_pct))
    closes = [100.0]
    for pct in returns_pct[1:]:
        closes.append(closes[-1] * (1.0 + pct / 100.0))
    return pd.DataFrame(
        {
            "date": dates,
            "code": code,
            "close": closes,
            "volume": 1_000_000.0,
        }
    )


# 全为正收益的窗口，最后一天才是深跌：一旦 shift(1) 写漏，MAX 会立刻跳到 30。
_SPIKE = 30.0
_RETURNS = [
    0.0,
    1.0, -2.0, 3.5, 0.5, -1.5, 4.2, -0.3, 2.2, -3.1, 0.8,
    -4.0, 1.2, -0.7, 2.9, -1.1, 0.4, -2.5, 1.7, -0.9, 5.0,
    -6.0, 0.3, -1.8, 2.4, -0.2, 1.1, -3.3, 0.6, _SPIKE,
]


def test_windows_stop_before_the_signal_day() -> None:
    prepared = _prepare_prices(
        _frame_from_returns(_RETURNS),
        {"volume_lookback": LOOKBACK, "split_adjust": False},
    )
    row = prepared.iloc[-1]

    # 起点那天的 NaN 和信号日本身都不该进窗口
    window = _RETURNS[1:-1][-LOOKBACK:]
    assert _SPIKE not in window, "信号日自己滑进了窗口"

    assert row["prior_max_return_pct"] == pytest.approx(max(window))
    assert row["prior_return_std_pct"] == pytest.approx(np.std(window, ddof=1))
    # σ 是「暴跌之前」的波动率，不是含暴跌的窗口 —— 否则崩盘撑大 σ 会自我抵消
    assert row["prior_return_std_pct"] != pytest.approx(np.std(_RETURNS[1:][-LOOKBACK:], ddof=1))


def test_windows_are_nan_until_enough_history() -> None:
    prepared = _prepare_prices(
        _frame_from_returns(_RETURNS),
        {"volume_lookback": LOOKBACK, "split_adjust": False},
    )
    for column in ("prior_max_return_pct", "prior_return_std_pct", "prior_amihud"):
        # 第 i 天要看到 i-20..i-1，所以前 21 行（含起点 NaN）必然是空的
        assert prepared[column].iloc[: LOOKBACK + 1].isna().all(), column
        assert prepared[column].iloc[-1] == prepared[column].iloc[-1], column


def test_amihud_averages_absolute_return_over_turnover() -> None:
    prepared = _prepare_prices(
        _frame_from_returns(_RETURNS),
        {"volume_lookback": LOOKBACK, "split_adjust": False},
    )
    tail = prepared.iloc[-(LOOKBACK + 1) : -1]
    expected = (tail["daily_return_pct"].abs() / tail["turnover"]).mean()
    assert prepared["prior_amihud"].iloc[-1] == pytest.approx(expected)


def _two_day_drop_config() -> dict:
    return {
        "market": "HK",
        "currency": "HKD",
        "strategy_mode": "two_day_drop",
        "negative_news_threshold": 10.0,
        "strict_news_fetch": False,
        "strict_lot_size": True,
        "max_lot_value_hkd": 30000.0,
        "liquidity_filter": {
            "min_current_turnover": 0.0,
            "min_prior_median_turnover": 0.0,
            "min_traded_days": 0,
        },
        "score_parameters": {},
        "volume_lookback": LOOKBACK,
        "split_adjust": False,
        "large_drop_max_pct": -4.5,
        "other_day_return_max_pct": 1.5,
    }


def _evaluate_one(returns_pct: list[float]) -> pd.Series:
    universe = pd.DataFrame(
        {
            "code": ["00001"],
            "name": ["Test"],
            "industry": ["Tech"],
            "lot_size": [100.0],
            "enabled": [True],
        }
    )
    news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])
    result = evaluate_signal(
        _frame_from_returns(returns_pct),
        universe,
        news,
        _two_day_drop_config(),
        asof="2026-09-11",
    )
    assert len(result) == 1
    return result.iloc[0]


# 末三天为 0 / 0 / −8，于是三日累计跌幅恒为 −8，只有波动率在变。
_CALM_PATTERN = [
    1.0, -2.0, 3.5, 0.5, -1.5, 4.2, -0.3, 2.2, -3.1, 0.8,
    -4.0, 1.2, -0.7, 2.9, -1.1, 0.4, -2.5, 1.7, 0.0, 0.0,
]


def test_drop_z_normalises_by_pre_crash_volatility() -> None:
    row = _evaluate_one([0.0, *_CALM_PATTERN, -8.0])

    # 末两天是 0，三日累计跌幅就等于信号日那根
    assert row["three_day_return_pct"] == pytest.approx(-8.0)
    sigma = row["prior_return_std_pct"]
    # σ 取的是信号日之前那 20 天，窗口里全是 _CALM_PATTERN
    assert sigma == pytest.approx(np.std(_CALM_PATTERN, ddof=1))
    assert row["drop_z"] == pytest.approx(8.0 / (sigma * np.sqrt(3.0)))
    assert row["drop_z"] > 0


def test_drop_z_halves_when_volatility_doubles() -> None:
    """同样是跌 8 个点，日常波动小一半的票 z 翻倍 —— 这就是 #6.2 的全部意义。

    现在的评分没有这层归一化：3% 日波动的票跌 8%（2.7σ）和 8% 波动的票跌 8%
    （1σ）拿一样的分。
    """
    calm = _evaluate_one([0.0, *_CALM_PATTERN, -8.0])
    wild = _evaluate_one([0.0, *[value * 2.0 for value in _CALM_PATTERN], -8.0])

    assert calm["three_day_return_pct"] == pytest.approx(wild["three_day_return_pct"])
    assert wild["prior_return_std_pct"] == pytest.approx(2.0 * calm["prior_return_std_pct"])
    assert calm["drop_z"] == pytest.approx(2.0 * wild["drop_z"])
    # 未归一化的 D 在两只票上完全相同，区分不出「温和回调」和「家常便饭」
    assert calm["drop_strength_pct"] == pytest.approx(wild["drop_strength_pct"])


class _FakeFastInfo:
    """复刻 FastInfo 的两种访问方式：下标取键名、属性取蛇形名。"""

    def __init__(self, **values: object) -> None:
        self._values = values

    def __getitem__(self, key: str) -> object:
        if key not in self._values:
            raise KeyError(key)
        return self._values[key]

    def __getattr__(self, item: str) -> object:
        return self._values.get(item)


def test_fast_info_value_accepts_both_key_styles() -> None:
    camel = _FakeFastInfo(yearHigh=683.0, **{"yearLow": 411.0})
    assert _fast_info_value(camel, "yearHigh", "year_high", "fiftyTwoWeekHigh") == 683.0
    assert _fast_info_value(camel, "yearLow", "year_low", "fiftyTwoWeekLow") == 411.0

    snake = _FakeFastInfo(year_high=12.5, year_low=3.25)
    assert _fast_info_value(snake, "yearHigh", "year_high") == 12.5

    # 完全没有这几个键时必须给 None，让上层退化成 NaN 而不是炸掉整轮扫描
    assert _fast_info_value(_FakeFastInfo(), "yearHigh", "year_high") is None


def test_finite_float_never_leaks_inf_or_junk() -> None:
    assert _finite_float(12.5) == 12.5
    assert _finite_float("12.5") == 12.5
    for junk in (None, "abc", float("inf"), float("-inf"), float("nan"), object()):
        assert np.isnan(_finite_float(junk)), junk
