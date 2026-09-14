"""行情快照的名字覆盖。

美股 build_full_market_prices 把 spot["name"] 填成代码本身，而 main() 原本无条件
用快照名覆盖 universe —— 结果是 us_industry.csv 里带出来的真实公司名（"Agilent
Technologies"）被整列冲成 Ticker。2026-09-10 那 5,174 只全军覆没，报告上三个字母
重复两遍读作 "CASY CASY"。港股不受影响：AKShare 快照给的是中文名。
"""

from __future__ import annotations

import pandas as pd

from hk_rebound_screener.main import _apply_snapshot_names


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["AAPL", "CASY", "09880"],
            "name": ["Apple", "Caseys General Stores", "优必选"],
            "industry": ["Tech", "Retail", "Machinery"],
        }
    )


def test_ticker_as_name_does_not_overwrite_real_names() -> None:
    """美股那条路径：快照名就是代码，必须整列保原样。"""
    spot = pd.DataFrame({"code": ["AAPL", "CASY"], "name": ["AAPL", "CASY"]})

    out = _apply_snapshot_names(_universe(), spot)

    assert out["name"].tolist() == ["Apple", "Caseys General Stores", "优必选"]


def test_real_names_still_win() -> None:
    """港股那条路径：快照给的是真名字（中文），照旧覆盖。

    strip 只用来判断「这一格算不算名字」，写回去的仍是快照原值 —— 重构前就是
    这么写的，这里锁住同样的行为，免得重构顺手改了输出。
    """
    spot = pd.DataFrame({"code": ["09880"], "name": ["  优必选  "]})

    out = _apply_snapshot_names(_universe(), spot)

    assert out.loc[out["code"] == "09880", "name"].iloc[0] == "  优必选  "


def test_blank_and_missing_snapshot_names_are_ignored() -> None:
    spot = pd.DataFrame(
        {"code": ["AAPL", "CASY", "09880"], "name": ["Apple Inc.", "", None]}
    )

    out = _apply_snapshot_names(_universe(), spot)

    # 只有 AAPL 被覆盖；空串和 None 都不能把名字抹掉
    assert out["name"].tolist() == ["Apple Inc.", "Caseys General Stores", "优必选"]


def test_missing_spot_columns_leave_universe_alone() -> None:
    universe = _universe()
    for spot in (pd.DataFrame(), pd.DataFrame({"code": ["AAPL"]})):
        pd.testing.assert_frame_equal(_apply_snapshot_names(universe, spot), universe)


def test_mixed_snapshot_only_overwrites_where_it_has_a_name() -> None:
    """同一份快照里既有真名又有代码名时，逐行判断而不是整批取舍。"""
    spot = pd.DataFrame(
        {"code": ["AAPL", "CASY"], "name": ["Apple Inc.", "CASY"]}
    )

    out = _apply_snapshot_names(_universe(), spot)

    assert out["name"].tolist() == ["Apple Inc.", "Caseys General Stores", "优必选"]
