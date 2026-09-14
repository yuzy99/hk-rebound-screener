from __future__ import annotations

import json
from datetime import date

import pandas as pd

from scripts.download_hkex_sources import (
    HKEXNEWS_COLUMNS,
    HKEX_QUOTE_COLUMNS,
    _flatten_hkexnews_record,
    _month_ranges,
    _parse_daily_page,
)


def test_month_ranges_are_clipped_to_requested_dates() -> None:
    assert _month_ranges(date(2022, 9, 14), date(2022, 11, 3)) == [
        (date(2022, 9, 14), date(2022, 9, 30), "202209"),
        (date(2022, 10, 1), date(2022, 10, 31), "202210"),
        (date(2022, 11, 1), date(2022, 11, 3), "202211"),
    ]


def test_parse_daily_page_keeps_only_ordinary_equity_and_maps_ohlc() -> None:
    html = """
    <html><body><pre>
    DATE : 01 SEP 2026 (TUESDAY)
    QUOTATIONS
     CODE  NAME OF STOCK    CUR PRV.CLO./    ASK/    HIGH/      SHARES TRADED/
                                CLOSING      BID     LOW        TURNOVER ($)
        1 CKH HOLDINGS     HKD   70.50    69.10    70.35            7,261,139
                                  69.05    69.05    68.60          502,493,854
      700 TENCENT          HKD  600.00   590.00   605.00           1,000,000
                                 595.00   594.50   588.00          595,000,000
      99999 WARRANT        HKD    1.00     1.00     1.00                 100
                                   1.00     1.00     1.00                  100
    SALES RECORDS FOR ALL STOCKS
    </pre></body></html>
    """.encode("iso-8859-1")
    frame = _parse_daily_page(html, "https://example.test/d260901e.htm", "Main Board", {"00001", "00700"})
    assert list(frame.columns) == HKEX_QUOTE_COLUMNS
    assert frame["code"].tolist() == ["00001", "00700"]
    assert frame.loc[0, "high"] == 70.35
    assert frame.loc[0, "low"] == 68.60
    assert frame.loc[1, "turnover"] == 595000000


def test_flatten_hkexnews_explodes_multiple_stock_codes_without_losing_link() -> None:
    record = {
        "DATE_TIME": "14/09/2026 15:22",
        "NEWS_ID": "123",
        "STOCK_CODE": "00001<br/>00700",
        "STOCK_NAME": "CKH HOLDINGS<br/>TENCENT",
        "TITLE": "Announcement &amp; Notice",
        "LONG_TEXT": "Announcements and Notices - [Inside Information]",
        "FILE_TYPE": "PDF",
        "FILE_INFO": "100KB",
        "FILE_LINK": "/listedco/listconews/sehk/2026/0914/2026091400001.pdf",
    }
    rows = _flatten_hkexnews_record(record, "0", "https://example.test/query", "2026-09-14T20:00:00+08:00")
    frame = pd.DataFrame(rows, columns=HKEXNEWS_COLUMNS)
    assert frame["code"].tolist() == ["00001", "00700"]
    assert frame["published_date"].tolist() == ["2026-09-14", "2026-09-14"]
    assert frame["category"].tolist() == ["Announcements and Notices"] * 2
    assert frame["file_url"].iloc[0].endswith("2026091400001.pdf")
