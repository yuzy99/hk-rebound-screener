import sys
import types
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from hk_rebound_screener import adapters
from hk_rebound_screener.main import _validated_live_asof
from hk_rebound_screener.strategy import _compute_industry_benchmark, load_news, _parse_datetime_series


def test_news_datetime_parsing_treats_naive_values_as_hong_kong_time(tmp_path: Path) -> None:
    values = pd.Series([
        "2026-09-02 09:00:00",
        "2026-09-02T01:00:00Z",
        "2026-09-02T09:00:00+08:00",
    ])
    parsed = _parse_datetime_series(values)

    assert parsed.tolist() == [
        pd.Timestamp("2026-09-02 09:00:00"),
        pd.Timestamp("2026-09-02 09:00:00"),
        pd.Timestamp("2026-09-02 09:00:00"),
    ]
    assert adapters._yfinance_news_datetime("2026-09-02 09:00:00") == pd.Timestamp(
        "2026-09-02 09:00:00"
    )

    news_path = tmp_path / "news.csv"
    pd.DataFrame({
        "code": ["00001"],
        "published_at": ["2026-09-02 09:00:00"],
        "title": ["公告"],
        "body": ["正文"],
        "url": ["https://example.test/news"],
    }).to_csv(news_path, index=False)
    loaded = load_news(news_path)
    assert loaded.loc[0, "published_at"] == pd.Timestamp("2026-09-02 09:00:00")


def test_akshare_news_datetime_parsing_uses_hong_kong_for_naive_values(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({
        "关键词": ["00001"],
        "新闻标题": ["公告"],
        "新闻内容": ["正文"],
        "发布时间": ["2026-09-02 09:00:00"],
        "新闻链接": ["https://example.test/news"],
    })
    fake_akshare = types.SimpleNamespace(stock_news_em=lambda symbol: frame)
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)

    news, statuses = adapters.fetch_akshare_news(["00001"])

    assert news.loc[0, "published_at"] == pd.Timestamp("2026-09-02 09:00:00")
    assert bool(statuses.loc[0, "news_fetch_ok"])


def test_hk_live_entry_keeps_complete_daily_bar_when_snapshot_is_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    stale_date = pd.Timestamp.now(tz="Asia/Hong_Kong").normalize() - pd.Timedelta(days=1)
    history = pd.DataFrame([{
        "date": stale_date,
        "code": "00001",
        "close": 100.0,
        "volume": 1000.0,
    }])
    spot = pd.DataFrame([{
        "timestamp": stale_date + pd.Timedelta(hours=15),
        "date": stale_date,
        "code": "00001",
        "name": "测试",
        "close": 90.0,
        "volume": 10.0,
    }])
    monkeypatch.setattr(adapters, "fetch_akshare_history", lambda *args, **kwargs: history)
    monkeypatch.setattr(adapters, "fetch_akshare_spot", lambda: spot)

    prices, _ = adapters.build_live_prices(["00001"], history_days=5, pause_seconds=0)

    assert prices.loc[prices["code"].eq("00001"), "close"].iloc[0] == pytest.approx(100.0)


def test_hk_full_market_entry_accepts_current_day_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    today = pd.Timestamp.now(tz="Asia/Hong_Kong").normalize().tz_localize(None)
    history = pd.DataFrame([{
        "date": today,
        "code": "00001",
        "close": 100.0,
        "volume": 1000.0,
    }])
    spot = pd.DataFrame([{
        "timestamp": today + pd.Timedelta(hours=15),
        "date": today,
        "code": "00001",
        "name": "测试",
        "close": 90.0,
        "volume": 10.0,
    }])
    monkeypatch.setattr(adapters, "fetch_yfinance_history_bulk", lambda *args, **kwargs: history)
    monkeypatch.setattr(adapters, "fetch_akshare_spot", lambda: spot)

    prices, _ = adapters.build_full_market_prices(["00001"], "HK", history_days=5)

    assert prices.loc[prices["code"].eq("00001"), "close"].iloc[0] == pytest.approx(90.0)
    assert _validated_live_asof(spot, prices) == spot.loc[0, "timestamp"]


def test_stale_snapshot_does_not_set_live_asof_over_newer_daily_coverage() -> None:
    prices = pd.DataFrame({"date": [pd.Timestamp("2026-09-10")]})
    spot = pd.DataFrame({
        "date": [pd.Timestamp("2026-09-09")],
        "timestamp": [pd.Timestamp("2026-09-09 15:15")],
    })

    assert _validated_live_asof(spot, prices) is None


def test_us_spot_accumulates_latest_intraday_session_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTicker:
        def history(self, **kwargs: object) -> pd.DataFrame:
            index = pd.date_range(
                "2026-09-10 09:30",
                periods=3,
                freq="min",
                tz="America/New_York",
            )
            return pd.DataFrame(
                {"Close": [10.0, 10.1, 10.2], "Volume": [1.0, 2.0, 3.0]},
                index=index,
            )

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=lambda symbol: FakeTicker()))

    spot = adapters.fetch_yfinance_spot(["AAPL"], pause_seconds=0)

    assert spot.loc[0, "close"] == pytest.approx(10.2)
    assert spot.loc[0, "volume"] == pytest.approx(6.0)


def _fake_daily_download(calls: list[dict[str, object]], factor: float = 1.0):
    def download(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        tickers = list(kwargs["tickers"])
        start = pd.Timestamp(kwargs["start"])
        end = pd.Timestamp(kwargs["end"])
        dates = pd.date_range(start, end - pd.Timedelta(days=1), freq="B")
        columns = pd.MultiIndex.from_product([tickers, ["Open", "Close", "Volume", "Adj Close"]])
        values: list[list[float]] = []
        for _ in tickers:
            values.extend([
                [100.0] * len(dates),
                [100.0] * len(dates),
                [1000.0] * len(dates),
                [100.0 * factor] * len(dates),
            ])
        return pd.DataFrame(dict(zip(columns, values)), index=dates)

    return download


def _write_history_cache(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path / "us_history.csv", index=False)


def test_yfinance_cache_requests_each_symbol_from_its_own_coverage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    start = date(2026, 1, 1)
    end = date(2026, 9, 10)
    _write_history_cache(tmp_path, [
        {"date": start, "code": "AAPL", "open": 1, "close": 1, "volume": 1, "auto_adjust": True, "adjustment_factor": 1.0},
        {"date": end, "code": "AAPL", "open": 1, "close": 1, "volume": 1, "auto_adjust": True, "adjustment_factor": 1.0},
        {"date": "2026-08-01", "code": "MSFT", "open": 1, "close": 1, "volume": 1, "auto_adjust": True, "adjustment_factor": 1.0},
        {"date": end, "code": "MSFT", "open": 1, "close": 1, "volume": 1, "auto_adjust": True, "adjustment_factor": 1.0},
    ])
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(adapters, "YFINANCE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(adapters, "YFINANCE_RETRY_DELAYS", (0.0,))
    monkeypatch.setattr(adapters.random, "uniform", lambda *args: 0.0)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=_fake_daily_download(calls)))

    result = adapters.fetch_yfinance_history_bulk(
        ["AAPL", "MSFT"], "US", start, end, auto_adjust=True, chunk_size=1
    )

    assert [call["start"] for call in calls] == [end - timedelta(days=6), start]
    assert result.loc[result["code"].eq("MSFT"), "date"].min() == pd.Timestamp(start)


def test_hk_full_market_uses_separate_yfinance_history_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    start = date(2026, 1, 1)
    end = date(2026, 9, 10)
    pd.DataFrame([
        {"date": start, "code": "00001", "open": 1, "close": 1, "volume": 1, "auto_adjust": False, "adjustment_factor": 1.0},
        {"date": end, "code": "00001", "open": 1, "close": 1, "volume": 1, "auto_adjust": False, "adjustment_factor": 1.0},
    ]).to_csv(tmp_path / "hk_history.csv", index=False)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(adapters, "YFINANCE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(adapters, "YFINANCE_RETRY_DELAYS", (0.0,))
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=_fake_daily_download(calls)))

    result = adapters.fetch_yfinance_history_bulk(["00001"], "HK", start, end, auto_adjust=False)

    assert calls[0]["start"] == end - timedelta(days=6)
    assert (tmp_path / "hk_history.csv").exists()
    assert result["code"].eq("00001").all()


def test_akshare_history_cache_fills_only_symbols_without_requested_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    start = date(2026, 1, 1)
    end = date(2026, 9, 10)
    pd.DataFrame([
        {"date": start, "code": "00001", "close": 1, "volume": 1, "adjust": ""},
        {"date": end, "code": "00001", "close": 1, "volume": 1, "adjust": ""},
    ]).to_csv(tmp_path / "hk_history.csv", index=False)
    calls: list[str] = []

    def stock_hk_daily(symbol: str, adjust: str) -> pd.DataFrame:
        calls.append(symbol)
        return pd.DataFrame({
            "date": [start.isoformat(), end.isoformat()],
            "close": [2.0, 2.0],
            "volume": [100.0, 100.0],
        })

    monkeypatch.setattr(adapters, "AKSHARE_HISTORY_CACHE_FILE", tmp_path / "hk_history.csv")
    monkeypatch.setitem(sys.modules, "akshare", types.SimpleNamespace(stock_hk_daily=stock_hk_daily))

    result = adapters.fetch_akshare_history(["00001", "00002"], start, end, adjust="", pause_seconds=0)

    assert calls == ["00002"]
    assert set(result["code"]) == {"00001", "00002"}
    assert result.loc[result["code"].eq("00002"), "date"].min() == pd.Timestamp(start)


def test_yfinance_cache_mode_mismatch_forces_full_resync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    start = date(2026, 1, 1)
    end = date(2026, 9, 10)
    _write_history_cache(tmp_path, [
        {"date": start, "code": "AAPL", "open": 1, "close": 1, "volume": 1, "auto_adjust": False, "adjustment_factor": 0.9},
        {"date": end, "code": "AAPL", "open": 1, "close": 1, "volume": 1, "auto_adjust": False, "adjustment_factor": 0.9},
    ])
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(adapters, "YFINANCE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(adapters, "YFINANCE_RETRY_DELAYS", (0.0,))
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=_fake_daily_download(calls)))

    adapters.fetch_yfinance_history_bulk(["AAPL"], "US", start, end, auto_adjust=True)

    assert calls[0]["start"] == start
    saved = pd.read_csv(tmp_path / "us_history.csv")
    assert set(saved["auto_adjust"].astype(str).str.lower()) == {"true"}


def test_yfinance_cache_factor_change_resyncs_full_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    start = date(2026, 1, 1)
    end = date(2026, 9, 10)
    _write_history_cache(tmp_path, [
        {"date": start, "code": "AAPL", "open": 1, "close": 1, "volume": 1, "auto_adjust": False, "adjustment_factor": 1.0},
        {"date": end, "code": "AAPL", "open": 1, "close": 1, "volume": 1, "auto_adjust": False, "adjustment_factor": 1.0},
    ])
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(adapters, "YFINANCE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(adapters, "YFINANCE_RETRY_DELAYS", (0.0,))
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=_fake_daily_download(calls, factor=0.8)))

    result = adapters.fetch_yfinance_history_bulk(["AAPL"], "US", start, end, auto_adjust=False)

    assert [call["start"] for call in calls] == [end - timedelta(days=6), start]
    assert result["date"].min() == pd.Timestamp(start)
    saved = pd.read_csv(tmp_path / "us_history.csv")
    assert saved["adjustment_factor"].eq(0.8).all()


def test_industry_benchmark_counts_only_finite_peers_and_uses_valid_mean_denominator() -> None:
    data = pd.DataFrame({
        "industry": ["Tech"] * 4,
        "daily_return_pct": [1.0, float("nan"), float("inf"), 3.0],
    })

    benchmark, peer_count = _compute_industry_benchmark(data, method="mean")

    assert benchmark.tolist() == [3.0, 2.0, 2.0, 1.0]
    assert peer_count.tolist() == [1, 2, 2, 1]
