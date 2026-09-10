from pathlib import Path

import pandas as pd

from hk_rebound_screener.notifier import (
    format_report_link_notification,
    render_html_report,
)


def test_report_link_notification_keeps_all_passed_codes() -> None:
    codes = [f"TICKER{index:03d}" for index in range(1, 147)]
    result = pd.DataFrame({"code": codes, "passes": [True] * len(codes)})

    content = format_report_link_notification(
        result,
        market="US",
        asof="2026-09-10",
        report_url="https://example.test/us_latest.html",
    )

    assert "命中股票代码（共 146 只）" in content
    assert all(code in content for code in codes)
    assert "https://example.test/us_latest.html" in content


def test_render_html_report_replaces_only_raw_data_and_market_placeholders(tmp_path: Path) -> None:
    template_path = tmp_path / "template.html"
    output_path = tmp_path / "reports" / "hk_latest.html"
    template_path.write_text(
        '<title>{{MARKET_NAME}}</title><style>.keep{color:red}</style>'
        '<script type="text/markdown" id="raw-data">old data</script>',
        encoding="utf-8",
    )

    rendered = render_html_report(
        "### 001. 00001\n- 所属行业：测试",
        market="HK",
        asof="2026-09-10",
        template_path=template_path,
        output_path=output_path,
    )

    content = output_path.read_text(encoding="utf-8")
    assert rendered == output_path
    assert "<title>港股</title>" in content
    assert "old data" not in content
    assert "### 001. 00001" in content
    assert '<style>.keep{color:red}</style>' in content
