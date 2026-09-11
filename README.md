# 港股 / 美股三日下跌与流动性筛选器

本仓库当前以“三日下跌 + 流动性”策略作为港股和美股的唯一正式扫描策略。规则引擎使用 pandas，行业数据仅作参考，不参与该策略的通过条件和评分。

## 当前策略规则

该策略不设最低股价，但会单独过滤低流动性股票，且不把“无成交时显示 0%”当作平盘信号：

1. 最近三个交易日中至少有一天跌幅 `<= -4.5%`；
2. 另外两天收跌或涨幅 `<= 1.5%`；
3. 当日必须有成交，当前成交额 `>= HK$2,000,000`、过去 20 日成交额中位数 `>= HK$200,000,000`，且过去 20 日至少有 12 个交易日成交；
4. 继续执行普通股、每手金额 `<= HK$30,000` 过滤；48 小时新闻风险为分级扣分——累计到 10 分才淘汰，单类命中仅扣分，抓取失败只标注“未核实”而不剔除；
5. 按 `min(最大单日跌幅, 15) + 2 × log₂(量比封顶 8) + 2.5 × log₁₀(20 日中位成交额相对门槛的倍数，封顶 1000) − 新闻风险分` 排序；行业数据仅作参考，不参与通过条件和评分。

   **跌幅是显著的正向因子**（`score_parameters.drop_weight: 1.0`）：跌得越深，T+3 超额越高。实测平均 IC 两个市场都是 **+0.108**；按跌幅分档近乎完美单调（美股 4.5~6% 档超额中位 +0.77%，>15% 档 +4.17%）；把样本期切成前后两半，符号一致（美股 +0.112 / +0.103，两半均显著）。分数只影响展示顺序，不影响「选哪些票」——`passes` 完全由过滤器决定。

   > ⚠️ **这个结论只在 200M 流动性门槛下成立。** 门槛为 1M 时跌幅的 IC 是 **−0.056（显著为负）**，因为那时深跌样本以仙股为主，跌了会继续跌；提到 200M 后深跌样本变成大票的恐慌性抛售，会反弹。**改流动性门槛和改跌幅权重必须放在一起验证**——分开做会把符号搞反（本项目就踩过：用 1M 门槛下的统计去定 200M 门槛下的权重，方向正好相反）。

港股价格按权威拆股事件表做后向复权（`split_adjust: true`），事件表由 `adapters.fetch_yfinance_splits_bulk` 抓取并缓存在 `data/yfinance_cache/{market}_splits.csv`。

> ⚠️ **不要退回 `auto_adjust: true` 了事**：实测 yfinance 的 `auto_adjust=True` 并不真正复权拆股 —— ASBP 反向合股（1 并 40）后单日仍留 +2883% 的断崖，MGN 留 −93.4%。后果是合股被当成暴跌选进名单，前瞻收益还会出现 +4173% 这种根本不可能成交的数值。
>
> 也**不要**改用「单日跳变超过某阈值就当作拆股」的启发式：港股仙股一天翻倍是家常便饭（实测 00117 连跳三次却没有任何拆股事件），而真正的合股未必在价格序列里留下跳变（实测 00022 的 1 并 50）。港股 2757 只里权威事件只有 66 条（2.4%），阈值法却会报出 217 条，约七成是误判。`find_split_adjustments` 仅用于审计线索，不参与修改价格。

手动执行港股全市场扫描（复用已有行业缓存）：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m hk_rebound_screener.main --mode live --full-market --config .\config.hk.two_day_drop.json --asof 2026-09-04
```

输出会额外包含 `two_days_ago_return_pct`、`turnover`、`prior_turnover_median`、`prior_traded_days` 和 `liquidity_ok`，便于复核为什么某只股票被剔除。

美股对应配置为 `config.us.two_day_drop.json`，官方流动性门槛同样是 `2,000,000 / 200,000,000`，但单位为美元，筛选窗口和跌幅条件相同。GitHub Actions 的定时运行和手动运行均固定使用这两个最新配置；旧配置文件仅保留作本地兼容和历史参考，不再作为 GitHub 扫描入口。

## 先跑确定性示例

```powershell
cd D:\CodexWorkSpace\hk-rebound-screener
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "$PWD\src"
python -m hk_rebound_screener.main --mode sample --config .\config.demo.json --asof 2026-09-02
python -m pytest -q
```

示例数据中 `00700` 应通过，`01211` 因“重大诉讼”新闻被拦截。示例行情是测试夹具，不是投资建议。

## 美股示例

美股使用 [yfinance](https://github.com/ranaroussi/yfinance) 读取 Yahoo Finance 的免费历史行情、最新可用报价和新闻；行业分类仍由 `universe.us.csv` 维护。规则与港股相同，但金额单位为 USD，`lot_size=1` 表示 1 股。

```powershell
cd D:\CodexWorkSpace\hk-rebound-screener
.\.venv\Scripts\python.exe -m hk_rebound_screener.main --mode sample --config .\config.us.demo.json --universe .\universe.us.csv --prices .\data\sample\us_prices.csv --news .\data\sample\us_news.csv --asof 2026-09-02
.\.venv\Scripts\python.exe -m hk_rebound_screener.main --mode live --config .\config.us.two_day_drop.json --universe .\universe.us.csv --limit-codes 4
```

美股 live 模式按代码抓取日线、最新可用分钟线和新闻；免费源可能延时、限流或缺少新闻，因此新闻抓取失败不再阻断候选，而是在报告里标注“⚠️ 未核实”，提示人工核对公告。美股没有港股 board lot，不能套用 HKEX 手数主数据。

## 手动扫描全市场普通股

`--full-market` 会在手动执行时自动生成普通股代码池、补齐并缓存行业元数据，然后批量读取日线。价格/行业条件先初筛，新闻只抓初筛候选，避免免费源对数千只股票逐只重复请求。

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m hk_rebound_screener.main --mode live --full-market --config .\config.hk.two_day_drop.json
python -m hk_rebound_screener.main --mode live --full-market --config .\config.us.two_day_drop.json
```

首次运行需要建立行业缓存，可能较慢；以后手动运行会复用 `data/cache/hk_industry.csv` 或 `data/cache/us_industry.csv`。需要重建时再加 `--refresh-metadata`。该模式不会创建定时任务。

## 港股盘中/日频扫描

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m hk_rebound_screener.main --mode live --limit-codes 8
```

live 模式会调用 AKShare 的港股全市场延时快照，再按 `universe.csv` 中启用的代码抓取单股历史日线和新闻。全市场逐代码历史接口可能触发上游限流，所以示例保留了 `--limit-codes` 和逐代码间隔；扩大到全市场前建议先建立本地日线缓存。

## GitHub Actions 云端定时扫描

仓库内的 `.github/workflows/scan-hk.yml` 和 `.github/workflows/scan-us.yml` 直接调用上面的 `--mode live --full-market` 入口，并固定使用最新港股和美股配置，不再提供旧策略选择。两个 workflow 都支持 Actions 页面上的 **Run workflow** 手动触发，并可选重建行业元数据缓存。

- 港股：工作日 `08:30 UTC`，即北京时间/香港时间 `16:30`，用于港股收市后的全市场扫描。
- 美股：工作日北京时间 `08:00`（GitHub Actions `00:00 UTC`），周六北京时间 `14:00`（`06:00 UTC`）；workflow 使用 `TZ=Asia/Hong_Kong` 处理运行日志和日期。
- 依赖从 `requirements.txt` 安装；扫描产生的 `outputs/*.csv` 会作为 Actions artifact 保存 14 天。
- 基础行业元数据 `data/cache/*_industry.csv` 已随仓库纳入版本控制，避免首次运行时冷启动爬取导致超时；云端通过 Actions cache 继续持久化增量更新。
- **免下载直观简报**：工作流会自动将入选标的或无标的提示写入 GitHub Actions 运行详情页的 **Summary** 区块，无需下载解压 CSV。
- **机器人消息推送（可选）**：在 GitHub 仓库的 **Settings -> Secrets and variables -> Actions** 中添加名为 `NOTIFICATION_WEBHOOK` 的 Secret（填入 PushPlus Token 或其他支持的 Webhook 地址）。工作流会先把完整中文报告发布到 GitHub Pages，再只推送全部股票代码和对应的 GitHub Pages 链接；港股与美股共用 `templates/stock_report_template.html` 的手机模板。

本仓库的 GitHub 远端为 `https://github.com/yuzy99/hk-rebound-screener.git`；推送后可在仓库的 **Actions** 页面启用或查看 workflow。定时任务只会使用最新策略配置，扫描产生的结果仍按现有 workflow 规则保存为 Actions artifact。

## 必须维护的元数据

`universe.csv` 中的 `industry` 需要从券商/交易所或人工复核后的证券主数据维护；live 模式会优先用港交所公开的 `ListOfSecurities.xlsx` 更新 `lot_size`，失败时才回退到 CSV 缓存。港交所每只证券的 board lot 由发行人决定，不能把所有港股统一当作 100 股一手。代码对缺失 `lot_size` 默认 fail-closed，不会把未知手数当作满足 HK$30,000。

## 已知边界

- AKShare 文档列明港股实时接口约 15 分钟延时；它适合日频/低频扫描，不是秒级行情。
- yfinance 免费数据适合研究和低频扫描，不等同于券商实时行情；美股盘中结果必须人工核对成交时间与数据延迟。
- 港股普通股由 HKEX 主数据筛选；美股使用 Nasdaq Trader 的 ETF/测试证券字段及证券名称排除规则，少数特殊证券仍建议人工抽查。
- 新闻关键词是第一道排雷，不等于法律或基本面判断；建议把命中的公告链接人工复核后再交易。
- 当前回测只用于验证信号逻辑，没有加入交易费用、印花税、滑点、停牌无法成交、公司行动和组合资金分配。
- 若要更快的批量回测，可把 `prices.csv` 换成缓存后的全市场面板，再接 VectorBT；这不是当前最小示例的必要依赖。
