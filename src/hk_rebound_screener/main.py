from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .adapters import (
    build_full_market_prices,
    build_full_universe,
    build_live_prices,
    build_us_live_prices,
    enrich_candidate_fundamentals,
    fetch_akshare_news,
    fetch_hkex_security_master,
    fetch_usd_hkd_rate,
    fetch_yfinance_news,
    fetch_yfinance_splits_bulk,
)
from .notifier import (
    format_markdown_report,
    render_html_report,
    send_webhook_notification,
    write_step_summary,
)
from .strategy import (
    append_forward_observations,
    apply_liquidity_tier,
    evaluate_signal,
    load_config,
    load_industry_returns,
    load_news,
    load_prices,
    load_universe,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]


def _validated_live_asof(spot: pd.DataFrame, prices: pd.DataFrame) -> object | None:
    """Use a snapshot timestamp only when it is not older than daily coverage."""
    if spot.empty:
        return None
    spot_date = pd.to_datetime(spot["date"], errors="coerce").dt.normalize().max()
    price_date = pd.to_datetime(prices["date"], errors="coerce").dt.normalize().max()
    if pd.isna(spot_date) or (pd.notna(price_date) and spot_date < price_date):
        return None
    if "timestamp" in spot:
        return spot["timestamp"].max()
    return spot_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="港股/美股反弹与相对行业滞涨筛选")
    parser.add_argument("--mode", choices=["sample", "backtest", "live"], default="sample")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--universe", default=None, help="代码池；默认按 market 选择 HK/US 示例文件")
    parser.add_argument("--prices", default=None, help="样例/回测价格文件；默认按 market 选择")
    parser.add_argument("--news", default=None, help="样例/回测新闻文件；默认按 market 选择")
    parser.add_argument("--asof", default=None, help="YYYY-MM-DD；默认使用输入数据的最后交易日")
    parser.add_argument("--limit-codes", type=int, default=0, help="live 模式仅抓取前 N 个启用代码，0 表示全部")
    parser.add_argument("--full-market", action="store_true", help="live 模式自动生成普通股全市场代码池")
    parser.add_argument("--refresh-metadata", action="store_true", help="重新抓取全市场行业元数据")
    parser.add_argument("--metadata-cache", default=None, help="全市场行业缓存文件")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    market = str(config.get("market", "HK")).upper()
    strategy_mode = str(config.get("strategy_mode", "rebound")).lower()
    # 同一份扫描结果按不同流动性门槛切档，每档一个 CSV + 一个页面。
    # 只有配了 liquidity_tiers 的市场（目前仅美股）才走这条路径，港股逐字节不变。
    tiers = config.get("liquidity_tiers") or []
    tiered = bool(tiers) and strategy_mode == "two_day_drop"
    if args.mode == "backtest" and strategy_mode == "industry_lag_rebound":
        raise SystemExit("industry_lag_rebound 当前只生成筛选结果，不支持回测模式")
    default_files = {
        "HK": {
            "universe": ROOT / "universe.csv",
            "prices": ROOT / "data" / "sample" / "prices.csv",
            "news": ROOT / "data" / "sample" / "news.csv",
        },
        "US": {
            "universe": ROOT / "universe.us.csv",
            "prices": ROOT / "data" / "sample" / "us_prices.csv",
            "news": ROOT / "data" / "sample" / "us_news.csv",
        },
    }
    if market not in default_files:
        raise SystemExit(f"暂不支持 market={market}，可选 HK 或 US")
    universe_path = Path(args.universe) if args.universe else default_files[market]["universe"]
    prices_path = Path(args.prices) if args.prices else default_files[market]["prices"]
    news_path = Path(args.news) if args.news else default_files[market]["news"]
    if args.full_market and args.mode != "live":
        raise SystemExit("--full-market 只适用于 --mode live")
    if args.full_market and args.limit_codes:
        raise SystemExit("--full-market 不能与 --limit-codes 同时使用")
    if args.full_market:
        metadata_cache = Path(args.metadata_cache) if args.metadata_cache else ROOT / "data" / "cache" / f"{market.lower()}_industry.csv"
        universe = build_full_universe(
            market=market,
            seed_path=str(universe_path),
            metadata_cache=str(metadata_cache),
            refresh_metadata=args.refresh_metadata,
        )
        print(f"普通股全市场代码池: {len(universe)} 条；行业缓存: {metadata_cache}")
    else:
        universe = load_universe(universe_path, market=market)
    if market == "US" and config.get("max_lot_value_hkd") is not None and not config.get("usd_hkd_rate"):
        config["usd_hkd_rate"] = fetch_usd_hkd_rate()
        print(f"USD/HKD 汇率: {config['usd_hkd_rate']:.6f}；美股单股上限: HK$30,000 等值")
    news_status = None
    live_asof = None
    full_market_scan = False

    if args.mode in {"sample", "backtest"}:
        prices = load_prices(prices_path)
        news = load_news(news_path)
    else:
        full_market_scan = args.full_market
        selected = universe
        if args.limit_codes:
            selected = selected.head(args.limit_codes)
        codes = selected["code"].tolist()
        if args.full_market:
            prices, spot = build_full_market_prices(
                codes=codes,
                market=market,
                history_days=int(config["history_days"]),
                auto_adjust=bool(config.get("auto_adjust", market == "US")),
            )
            news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])
            if not spot.empty:
                live_asof = _validated_live_asof(spot, prices)
                print(f"批量行情快照: {spot['date'].max().date()}，代码数={len(spot)}")
        elif market == "HK":
            prices, spot = build_live_prices(
                codes=codes,
                history_days=int(config["history_days"]),
                adjust=str(config.get("price_adjust", "")),
            )
            if strategy_mode == "industry_lag_rebound":
                news = pd.DataFrame(columns=["code", "published_at", "title", "body", "url"])
            else:
                news, news_status = fetch_akshare_news(codes)
            try:
                hkex_master = fetch_hkex_security_master()
                universe = universe.drop(columns=["lot_size"]).merge(hkex_master, on="code", how="left")
                print(f"HKEX 手数主数据: {len(hkex_master)} 条证券记录")
            except Exception as error:  # noqa: BLE001 - CSV metadata remains a safe fallback
                print(f"WARN HKEX 手数主数据: {error}; 使用 universe.csv 中的缓存值")
            if not spot.empty:
                live_asof = _validated_live_asof(spot, prices)
                print(f"AKShare 实时快照: {spot['date'].max().date()}，代码数={len(spot)}（接口为延时行情）")
        elif market == "US":
            prices, spot = build_us_live_prices(
                codes=codes,
                history_days=int(config["history_days"]),
                auto_adjust=bool(config.get("auto_adjust", True)),
            )
            news, news_status = fetch_yfinance_news(codes)
            if not spot.empty:
                live_asof = _validated_live_asof(spot, prices)
                print(f"yfinance 美股快照: {spot['date'].max().date()}，代码数={len(spot)}（免费源可能延时/限流）")

    if prices.empty:
        raise SystemExit("没有可用价格数据，请检查数据源、代码和日期")

    # 拆股复权必须走权威事件表：yfinance 的 auto_adjust=True 并不真正复权拆股，
    # 留着断崖会让合股被当成暴跌选进名单、也会让前瞻收益出现 +4173% 这种假数。
    # 抓取失败时不做复权（而不是退化成按价格跳变猜），宁可少修也不能把真实行情改坏。
    if bool(config.get("split_adjust", True)):
        try:
            split_events = fetch_yfinance_splits_bulk(
                codes=prices["code"].dropna().unique().tolist(),
                market=market,
                start_date=pd.Timestamp(prices["date"].min()).date(),
                end_date=pd.Timestamp(prices["date"].max()).date(),
            )
            config["split_events"] = split_events
            affected = split_events["code"].nunique() if len(split_events) else 0
            print(f"拆股事件表: {len(split_events)} 条，涉及 {affected} 只股票")
        except Exception as error:  # noqa: BLE001 - 复权失败不该中断选股
            print(f"WARN 拆股事件表抓取失败: {error}; 本轮不做拆股复权")

    if strategy_mode == "industry_lag_rebound" and not args.asof and live_asof is not None:
        spot_date = pd.Timestamp(live_asof).normalize()
        completed_dates = sorted(
            pd.Timestamp(value) for value in prices["date"].dropna().unique()
            if pd.Timestamp(value) < spot_date
        )
        if completed_dates:
            live_asof = completed_dates[-1]
            print(f"行业回暖策略使用最近完整收盘日: {live_asof.date()}")
    configured_test_asof = config.get("test_asof_date") if args.mode != "backtest" else None
    asof = args.asof or configured_test_asof or live_asof or str(prices["date"].max().date())
    if configured_test_asof and not args.asof:
        print(f"历史测试使用配置的信号日: {configured_test_asof}")
    industry_returns = None
    industry_returns_setting = config.get("industry_returns_path")
    if strategy_mode == "industry_lag_rebound" and industry_returns_setting:
        industry_returns_path = Path(industry_returns_setting)
        if not industry_returns_path.is_absolute():
            industry_returns_path = ROOT / industry_returns_path
        if industry_returns_path.exists():
            industry_returns = load_industry_returns(industry_returns_path)
            print(f"官方行业指数数据: {industry_returns_path}")
        else:
            print(f"WARN 无官方行业指数数据: {industry_returns_path}; 仅判断今日涨跌")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "backtest":
        trades, summary = run_backtest(prices, universe, news, config)
        strategy_mode = str(config.get("strategy_mode", "rebound")).lower()
        strategy_suffix = "" if strategy_mode == "rebound" else f"_{strategy_mode}"
        output_path = output_dir / f"backtest_trades_{market.lower()}{strategy_suffix}.csv"
        trades.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"回测交易记录: {output_path}")
        print(f"回测摘要: {summary}")
        return

    fundamentals = pd.DataFrame(columns=["code", "market_cap", "pe", "pb"])
    if full_market_scan and strategy_mode != "industry_lag_rebound":
        if not spot.empty and "name" in spot:
            cn_names = spot.loc[spot["name"].fillna("").astype(str).str.strip().ne(""), ["code", "name"]].drop_duplicates("code")
            if not cn_names.empty:
                name_map = dict(zip(cn_names["code"], cn_names["name"]))
                universe["name"] = universe["code"].map(name_map).fillna(universe["name"])

        # 新闻与估值是第二阶段：先筛价格/行业条件，再对候选股抓新闻和PE/PB/市值。
        # 初筛必须用最松的那一档门槛：若按 200M 初筛，50–200M 那批票根本不会被送进
        # 新闻抓取，negative_news_score 会被 fillna 成 0，在 50M 页上等同于「没有新闻
        # 风险」。放宽初筛只会让候选集变大 —— 最终那次 evaluate_signal 仍用原 config，
        # 官方 CSV 和 200M 名单因此完全不受影响。
        preliminary_config = config
        if tiered:
            loosest = min(float(tier["min_prior_median_turnover"]) for tier in tiers)
            preliminary_config = {
                **config,
                "liquidity_filter": {
                    **config.get("liquidity_filter", {}),
                    "min_prior_median_turnover": loosest,
                },
            }
        preliminary_status = pd.DataFrame({"code": universe["code"], "news_fetch_ok": True})
        preliminary = evaluate_signal(
            prices,
            universe,
            news,
            preliminary_config,
            asof=asof,
            news_status=preliminary_status,
        )
        candidate_codes = preliminary.loc[preliminary["passes"], "code"].tolist()
        print(f"价格/行业初筛候选: {len(candidate_codes)} 条；正在获取候选股新闻与估值指标...")
        if candidate_codes:
            if market == "HK":
                news, candidate_status = fetch_akshare_news(candidate_codes)
            else:
                news, candidate_status = fetch_yfinance_news(candidate_codes)
            status_map = dict(zip(candidate_status["code"], candidate_status["news_fetch_ok"]))
            news_status = pd.DataFrame({"code": universe["code"]})
            news_status["news_fetch_ok"] = news_status["code"].map(status_map).fillna(False)
            fundamentals = enrich_candidate_fundamentals(candidate_codes, market=market)
        else:
            news_status = preliminary_status
    result = evaluate_signal(
        prices,
        universe,
        news,
        config,
        asof=asof,
        news_status=news_status,
        industry_returns=industry_returns,
    )
    forward_days = int(config.get("append_forward_trading_days", 0))
    if forward_days:
        result = append_forward_observations(
            result, prices, asof=asof, trading_days=forward_days, config=config
        )
    if not fundamentals.empty:
        result = result.merge(fundamentals, on="code", how="left")
    strategy_suffix = "" if strategy_mode == "rebound" else f"_{strategy_mode}"
    output_path = output_dir / f"scan_{market.lower()}{strategy_suffix}_{pd.Timestamp(asof).date().isoformat()}.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    passed = result.loc[result["passes"]]
    print(f"扫描日期: {asof}; 输出: {output_path}")
    if passed.empty:
        print("没有同时满足全部条件的标的。")
    else:
        print(passed.to_string(index=False))

    report_md = format_markdown_report(result, market=market, asof=str(pd.Timestamp(asof).date()), config=config)
    write_step_summary(report_md)
    template_path = os.environ.get(
        "REPORT_TEMPLATE_PATH",
        str(ROOT / "templates" / "stock_report_template.html"),
    )
    html_output_path = os.environ.get("REPORT_HTML_PATH")
    html_dir = os.environ.get("REPORT_HTML_DIR")
    asof_text = str(pd.Timestamp(asof).date())
    market_label = "美股" if market == "US" else "港股"
    if html_dir:
        if not tiered:
            raise SystemExit("REPORT_HTML_DIR 需要配置 liquidity_tiers；单页市场请继续用 REPORT_HTML_PATH")
        # 按档发布：每档一个 CSV、一个页面，外加一份清单。
        # 推送脚本只读清单 —— 两个 CSV 相隔几秒写完，glob + mtime 排序不可靠。
        tiers_dir = Path(html_dir)
        tiers_dir.mkdir(parents=True, exist_ok=True)
        manifest_tiers: list[dict[str, object]] = []
        for tier in tiers:
            threshold = float(tier["min_prior_median_turnover"])
            label = str(tier["label"])
            slug = str(tier["slug"])
            tier_result = apply_liquidity_tier(result, config, threshold)
            tier_csv = output_dir / f"scan_{market.lower()}{strategy_suffix}_{slug}_{asof_text}.csv"
            tier_result.to_csv(tier_csv, index=False, encoding="utf-8-sig")
            tier_config = {
                **config,
                "liquidity_filter": {
                    **config.get("liquidity_filter", {}),
                    "min_prior_median_turnover": threshold,
                },
            }
            tier_md = format_markdown_report(
                tier_result,
                market=market,
                asof=asof_text,
                config=tier_config,
                tier_label=f"{label} 流动性门槛",
            )
            write_step_summary(tier_md)
            render_html_report(
                tier_md,
                market=market,
                asof=asof_text,
                template_path=template_path,
                output_path=tiers_dir / f"{market.lower()}_{slug}.html",
                market_label=f"{market_label}（{label} 流动性门槛）",
            )
            count = int(tier_result["passes"].sum())
            print(f"档位 {label}: 命中 {count} 只；CSV: {tier_csv}")
            manifest_tiers.append({
                "label": label,
                "slug": slug,
                "csv": Path(os.path.relpath(tier_csv, ROOT)).as_posix(),
                "count": count,
            })
        manifest_path = tiers_dir / f"{market.lower()}_tiers.json"
        manifest_path.write_text(
            json.dumps(
                {"market": market, "asof": asof_text, "tiers": manifest_tiers},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"档位清单: {manifest_path}")
    elif html_output_path:
        render_html_report(
            report_md,
            market=market,
            asof=asof_text,
            template_path=template_path,
            output_path=html_output_path,
        )
        print(f"HTML 报告: {html_output_path}")
    # 设了 REPORT_HTML_DIR 就由按档推送独挑通知，这里绝不能补发第二条。
    if not html_dir and not html_output_path:
        send_webhook_notification(report_md, title=f"{market} 市场选股简报 ({asof_text})")


if __name__ == "__main__":
    main()
