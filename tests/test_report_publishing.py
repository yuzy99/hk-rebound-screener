import re
from pathlib import Path

import pandas as pd

from hk_rebound_screener.notifier import (
    format_report_link_notification,
    format_markdown_report,
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


def test_current_score_formula_is_in_markdown_and_html_report(tmp_path: Path) -> None:
    from hk_rebound_screener.strategy import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "config.hk.two_day_drop.json")
    report = format_markdown_report(
        pd.DataFrame(columns=["passes"]),
        market="HK",
        asof="2026-09-10",
        config=config,
    )
    # drop_weight = +1.0：跌幅是正向因子，跌得越深得分越高，不留负号
    expected_formula = (
        "S = min(D, 15) + 2 × log₂(min(max(R, 1), 8))"
        " + 2.5 × log₁₀(min(max(L, 1), 1000)) − N"
    )
    assert expected_formula in report
    assert "按风险类别去重" in report

    output_path = tmp_path / "hk_latest.html"
    render_html_report(
        report,
        market="HK",
        asof="2026-09-10",
        template_path=Path(__file__).resolve().parents[1] / "templates" / "stock_report_template.html",
        output_path=output_path,
    )
    html = output_path.read_text(encoding="utf-8")
    assert "当前评分公式" in html
    assert expected_formula in html

    # 模板里的「当前评分公式」是写死的一块，必须和配置推导出的公式一致。
    # 上面那行断言会撞上内嵌的 markdown 原文而恒真，所以这里先把 raw-data 挖掉，
    # 只检查用户真正看到的那部分。
    visible = re.sub(
        r'<script type="text/markdown" id="raw-data">.*?</script>',
        "",
        html,
        flags=re.S,
    )
    assert visible != html, "raw-data 未被挖掉，下面的断言会失去意义"
    assert expected_formula in visible
    assert "1000 倍封顶" in visible


def test_html_template_parses_backtick_quoted_tickers() -> None:
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "stock_report_template.html"
    ).read_text(encoding="utf-8")

    assert "(\\d+)\\.\\s+`?([A-Za-z0-9._-]+)`?" in template
