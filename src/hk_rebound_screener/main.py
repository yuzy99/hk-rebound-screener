from __future__ import annotations

import argparse
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
)
from .notifier import format_markdown_report, send_webhook_notification, write_step_summary
from .strategy import (
    append_forward_observations,
    evaluate_signal,
    load_config,
    load_news,
    load_prices,
    load_universe,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]


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
                live_asof = spot["timestamp"].max() if "timestamp" in spot else spot["date"].max()
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
                live_asof = spot["timestamp"].max()
                print(f"AKShare 实时快照: {spot['date'].max().date()}，代码数={len(spot)}（接口为延时行情）")
        elif market == "US":
            prices, spot = build_us_live_prices(
                codes=codes,
                history_days=int(config["history_days"]),
                auto_adjust=bool(config.get("auto_adjust", True)),
            )
            news, news_status = fetch_yfinance_news(codes)
            if not spot.empty:
                live_asof = spot["timestamp"].max()
                print(f"yfinance 美股快照: {spot['date'].max().date()}，代码数={len(spot)}（免费源可能延时/限流）")

    if prices.empty:
        raise SystemExit("没有可用价格数据，请检查数据源、代码和日期")
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

        # 新闻与估值是第二阶段：先筛价格/行业条件，再对候选股抓新闻和PE/PB/市值
        preliminary_status = pd.DataFrame({"code": universe["code"], "news_fetch_ok": True})
        preliminary = evaluate_signal(
            prices,
            universe,
            news,
            config,
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
    result = evaluate_signal(prices, universe, news, config, asof=asof, news_status=news_status)
    forward_days = int(config.get("append_forward_trading_days", 0))
    if forward_days:
        result = append_forward_observations(result, prices, asof=asof, trading_days=forward_days)
    if not fundamentals.empty:
        result = result.merge(fundamentals, on="code", how="left")
    strategy_suffix = "" if strategy_mode == "rebound" else f"_{strategy_mode}"
    output_path = output_dir / f"scan_{market.lower()}{strategy_suffix}_{pd.Timestamp(asof).date().isoformat()}.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    passed = result.loc[result["passes"]].head(int(config["top_n"]))
    print(f"扫描日期: {asof}; 输出: {output_path}")
    if passed.empty:
        print("没有同时满足全部条件的标的。")
    else:
        print(passed.to_string(index=False))

    report_md = format_markdown_report(result, market=market, asof=str(pd.Timestamp(asof).date()), config=config)
    write_step_summary(report_md)
    send_webhook_notification(report_md, title=f"{market} 市场选股简报 ({pd.Timestamp(asof).date()})")


if __name__ == "__main__":
    main()
