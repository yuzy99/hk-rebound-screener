"""拉行情前的「最新交易日是否已落地」探测与等待。

2026-10-13 美股那次：Yahoo 还没汇总完当天日线，给的是占位行（日期是当天、收盘价沿用
前一交易日、成交量 0）。策略的流动性门槛要求当日成交量 > 0，于是 5,124 只里 4,989 只
在第一道门槛就被刷掉，三档名单全是 0 只，推送发的是「无命中股票」。这里锁住两件事：

1. 占位行必须被识别成「没就绪」（用真实 yfinance 返回结构喂进去）；
2. 没就绪时确实按「等一轮再测」推进，推迟次数用尽就抛错 —— 不许带着空数据往下跑。
"""

from __future__ import annotations

import pandas as pd
import pytest

from hk_rebound_screener.adapters import (
    DATA_PROBE_READY_SHARE,
    probe_latest_daily_session,
    wait_for_daily_data,
)


def _download_frame(volumes: dict[str, float], session: str = "2026-10-13") -> pd.DataFrame:
    """造一份 yf.download(group_by="ticker") 形状的数据：MultiIndex 列 + 纽约时区索引。"""
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-10-09 00:00", tz="America/New_York"),
         pd.Timestamp("2026-10-12 00:00", tz="America/New_York"),
         pd.Timestamp(f"{session} 00:00", tz="America/New_York")],
        name="Date",
    )
    columns = pd.MultiIndex.from_product(
        [sorted(volumes), ["Close", "Volume"]], names=["Ticker", "Price"]
    )
    frame = pd.DataFrame(index=index, columns=columns, dtype=float)
    for code, volume in volumes.items():
        frame[(code, "Close")] = [100.0, 101.0, 101.0 if volume == 0 else 103.0]
        frame[(code, "Volume")] = [1_000_000.0, 1_100_000.0, volume]
    return frame


@pytest.fixture()
def fake_download(monkeypatch: pytest.MonkeyPatch):
    def install(volumes: dict[str, float], session: str = "2026-10-13") -> None:
        def download(**kwargs):
            return _download_frame(volumes, session=session)

        monkeypatch.setattr("yfinance.download", download)

    return install


def test_placeholder_session_is_not_ready(fake_download) -> None:
    """全市场占位行的样子：最新日期在，但成交量是 0 —— 必须判成没就绪。"""
    fake_download({symbol: 0.0 for symbol in ("AAPL", "MSFT", "NVDA", "SPY")})

    session, share, total = probe_latest_daily_session("US", symbols=("AAPL", "MSFT", "NVDA", "SPY"))

    assert session is not None and session.date().isoformat() == "2026-10-13"
    assert (share, total) == (0.0, 4)
    assert share < DATA_PROBE_READY_SHARE


def test_real_session_is_ready(fake_download) -> None:
    """数据已落地：最新日期上参考标的都有真实成交量。"""
    fake_download({symbol: 5_000_000.0 for symbol in ("AAPL", "MSFT", "NVDA", "SPY")})

    session, share, total = probe_latest_daily_session("US", symbols=("AAPL", "MSFT", "NVDA", "SPY"))

    assert session is not None and session.date().isoformat() == "2026-10-13"
    assert (share, total) == (1.0, 4)


def test_partial_placeholder_waits_but_tolerates_single_lagging_symbol(fake_download) -> None:
    """3/4 有真实成交（有只票停牌/取不到）仍算就绪；4 只全是占位行就不算。"""
    fake_download({"AAPL": 5_000_000.0, "MSFT": 5_000_000.0, "NVDA": 5_000_000.0, "SPY": 0.0})
    _, share, _ = probe_latest_daily_session("US", symbols=("AAPL", "MSFT", "NVDA", "SPY"))
    assert share == pytest.approx(0.75) and share >= DATA_PROBE_READY_SHARE

    fake_download({"AAPL": 0.0, "MSFT": 0.0, "NVDA": 0.0, "SPY": 0.0})
    _, share, _ = probe_latest_daily_session("US", symbols=("AAPL", "MSFT", "NVDA", "SPY"))
    assert share == 0.0


def test_probe_failure_is_treated_as_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """请求整个失败时返回「未知日期 + 0 就绪」，而不是抛出去中断任务。"""

    def boom(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("yfinance.download", boom)

    session, share, total = probe_latest_daily_session("US", symbols=("AAPL", "MSFT"))
    assert session is None and (share, total) == (0.0, 2)


def test_wait_retries_then_succeeds() -> None:
    """第一次没就绪 → 等一轮 → 第二次就绪：正好睡一次，并返回就绪的那一天。"""
    results = [
        (pd.Timestamp("2026-10-13"), 0.0, 10),
        (pd.Timestamp("2026-10-13"), 0.9, 10),
    ]
    sleeps: list[float] = []

    session = wait_for_daily_data(
        market="US",
        max_delays=4,
        delay_seconds=3600.0,
        probe=lambda: results.pop(0),
        sleep=sleeps.append,
    )

    assert session == pd.Timestamp("2026-10-13")
    assert sleeps == [3600.0]


def test_wait_gives_up_after_max_delays() -> None:
    """推迟次数用尽仍没就绪 → 抛 SystemExit，绝不带着空数据继续扫描。"""
    probes: list[int] = []
    sleeps: list[float] = []

    def always_placeholder():
        probes.append(1)
        return pd.Timestamp("2026-10-13"), 0.0, 10

    with pytest.raises(SystemExit) as error:
        wait_for_daily_data(
            market="US",
            max_delays=4,
            delay_seconds=3600.0,
            probe=always_placeholder,
            sleep=sleeps.append,
        )

    assert len(probes) == 5          # 初次探测 + 4 次重测
    assert sleeps == [3600.0] * 4    # 一共推迟 4 小时
    assert "4 小时" in str(error.value)


def test_wait_can_be_disabled() -> None:
    """max_delays=0 即「只测一次，不推迟」：没就绪直接抛错，不再睡。"""
    sleeps: list[float] = []
    with pytest.raises(SystemExit):
        wait_for_daily_data(
            market="US",
            max_delays=0,
            delay_seconds=3600.0,
            probe=lambda: (None, 0.0, 10),
            sleep=sleeps.append,
        )
    assert sleeps == []
