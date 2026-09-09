# 港股 / 美股反弹与相对行业滞涨筛选示例

这个示例保留港股策略，并新增美股策略。规则引擎使用 pandas，实现：

1. 上一交易日跌幅 `<= -5%`；
2. 当日同行平均涨幅 `>= +1%`，并且按剔除自身的 leave-one-out 平均计算；
3. 个股当日涨幅 `<= 0%`，或落后同行平均 `>= 2` 个百分点；
4. 近 48 小时命中重大利空关键词则排除；
5. 只纳入普通股；港股为最新价 × 每手股数 `<= HK$30,000`，美股按 1 股交易单位计算并使用 `HK$30,000 ÷ USD/HKD` 等值上限；
6. 以“昨日跌幅绝对值 + 相对行业滞涨 + 成交量异常 - 负面新闻分”排序；
7. 用同一信号函数提供一个“下一交易日开盘买入、收盘卖出”的简化研究回测。

## 新增：双日下跌 + 流动性策略

该策略不设最低股价，但会单独过滤低流动性股票，且不把“无成交时显示 0%”当作平盘信号：

1. 最新完整交易日与前一交易日连续下跌；
2. 前一交易日或前两交易日至少有一天跌幅 `<= -5%`；
3. 当日必须有成交，当前成交额和过去 20 日成交额中位数均 `>= HK$5,000,000`，过去 20 日至少有 15 个交易日成交；
4. 继续执行普通股、每手金额 `<= HK$30,000`、48 小时重大利空过滤；
5. 按最大单日跌幅 + 成交量异常 - 负面新闻分排序；行业数据仅作参考，不参与通过条件和评分。

手动执行港股全市场扫描（复用已有行业缓存）：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m hk_rebound_screener.main --mode live --full-market --config .\config.hk.two_day_drop.json --asof 2026-09-04
```

输出会额外包含 `two_days_ago_return_pct`、`turnover`、`prior_turnover_median`、`prior_traded_days` 和 `liquidity_ok`，便于复核为什么某只股票被剔除。

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
.\.venv\Scripts\python.exe -m hk_rebound_screener.main --mode live --config .\config.us.json --universe .\universe.us.csv --limit-codes 4
```

美股 live 模式按代码抓取日线、最新可用分钟线和新闻；免费源可能延时、限流或缺少新闻，因此新闻抓取失败仍按严格模式阻断候选。美股没有港股 board lot，不能套用 HKEX 手数主数据。

## 手动扫描全市场普通股

`--full-market` 会在手动执行时自动生成普通股代码池、补齐并缓存行业元数据，然后批量读取日线。价格/行业条件先初筛，新闻只抓初筛候选，避免免费源对数千只股票逐只重复请求。

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m hk_rebound_screener.main --mode live --full-market --config .\config.json
python -m hk_rebound_screener.main --mode live --full-market --config .\config.us.json
```

首次运行需要建立行业缓存，可能较慢；以后手动运行会复用 `data/cache/hk_industry.csv` 或 `data/cache/us_industry.csv`。需要重建时再加 `--refresh-metadata`。该模式不会创建定时任务。

## 港股盘中/日频扫描

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m hk_rebound_screener.main --mode live --limit-codes 8
```

live 模式会调用 AKShare 的港股全市场延时快照，再按 `universe.csv` 中启用的代码抓取单股历史日线和新闻。全市场逐代码历史接口可能触发上游限流，所以示例保留了 `--limit-codes` 和逐代码间隔；扩大到全市场前建议先建立本地日线缓存。

## GitHub Actions 云端定时扫描

仓库内的 `.github/workflows/scan-hk.yml` 和 `.github/workflows/scan-us.yml` 直接调用上面的 `--mode live --full-market` 入口，不会改写筛选规则或参数。两个 workflow 都支持 Actions 页面上的 **Run workflow** 手动触发，并可选重建行业元数据缓存。

- 港股：工作日 `08:30 UTC`，即 `Asia/Taipei` `16:30`，用于港股收市后的全市场扫描。
- 美股：工作日 `21:30 UTC`，即台北时间次日 `05:30`；这个时间在美国夏令时和冬令时都位于收市之后。GitHub Actions 的 cron 只接受 UTC，workflow 同时设置 `TZ=Asia/Taipei` 供运行日志和日期处理使用。
- 依赖从 `requirements.txt` 安装；扫描产生的 `outputs/*.csv` 会作为 Actions artifact 保存 14 天。
- 基础行业元数据 `data/cache/*_industry.csv` 已随仓库纳入版本控制，避免首次运行时冷启动爬取导致超时；云端通过 Actions cache 继续持久化增量更新。
- **免下载直观简报**：工作流会自动将入选标的或无标的提示写入 GitHub Actions 运行详情页的 **Summary** 区块，无需下载解压 CSV。
- **机器人消息推送（可选）**：在 GitHub 仓库的 **Settings -> Secrets and variables -> Actions** 中添加名为 `NOTIFICATION_WEBHOOK` 的 Secret（填入企业微信、飞书或钉钉机器人的 Webhook 地址），工作流执行完毕后会自动将简报推送到群聊。

如果本地目录还没有 GitHub 远端，需要先将本目录初始化为 Git 仓库并推送到一个 GitHub repository；推送后在仓库的 **Actions** 页面启用 workflow。当前工作区未检测到 Git 远端，且本机 `gh` 登录令牌已失效，因此不会自动创建或推送到未知的 GitHub repository。

## 必须维护的元数据

`universe.csv` 中的 `industry` 需要从券商/交易所或人工复核后的证券主数据维护；live 模式会优先用港交所公开的 `ListOfSecurities.xlsx` 更新 `lot_size`，失败时才回退到 CSV 缓存。港交所每只证券的 board lot 由发行人决定，不能把所有港股统一当作 100 股一手。代码对缺失 `lot_size` 默认 fail-closed，不会把未知手数当作满足 HK$30,000。

## 已知边界

- AKShare 文档列明港股实时接口约 15 分钟延时；它适合日频/低频扫描，不是秒级行情。
- yfinance 免费数据适合研究和低频扫描，不等同于券商实时行情；美股盘中结果必须人工核对成交时间与数据延迟。
- 港股普通股由 HKEX 主数据筛选；美股使用 Nasdaq Trader 的 ETF/测试证券字段及证券名称排除规则，少数特殊证券仍建议人工抽查。
- 新闻关键词是第一道排雷，不等于法律或基本面判断；建议把命中的公告链接人工复核后再交易。
- 当前回测只用于验证信号逻辑，没有加入交易费用、印花税、滑点、停牌无法成交、公司行动和组合资金分配。
- 若要更快的批量回测，可把 `prices.csv` 换成缓存后的全市场面板，再接 VectorBT；这不是当前最小示例的必要依赖。
