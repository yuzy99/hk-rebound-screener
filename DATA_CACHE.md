# 本地数据缓存入口

港股十年历史行情位于 `data/yfinance_cache/hk_history.csv`，当前为 5,408,322 行、2,778 只代码，时间范围为 2016-09-14 至 2026-09-14。该文件按 `code + date` 一行一条日线记录，编码为 UTF-8 无 BOM、逗号分隔，字段包含 `date`、`code`、`open`、`high`、`low`、`close`、`volume`、`auto_adjust` 和 `adjustment_factor`。读取前请先查看 `data/yfinance_cache/README.md` 和 `data/yfinance_cache/hk_history_quality.json`，其中包含字段定义、读取示例、覆盖率和质量边界。

目前已验证 Python 标准库 `csv.DictReader` 和 Pandas 均能直接读取港股行情、拆股事件、行业和市场指数文件。行情的基本完整性检查通过：必需字段无空值、代码格式正确、日期可解析、`code + date` 无重复、成交量无负数。另有 1,056 行非正收盘价（全部来自 00831）以及 124,583 行约 2.30% 的 OHLC 结构异常，涉及最高价/最低价与开收盘价不自洽；这些值保留原样，进行价格筛选或最高价、最低价止损回测前必须使用质量清单过滤或标记。

拆股事件位于 `data/yfinance_cache/hk_splits.csv`，检查进度位于 `data/yfinance_cache/hk_splits_checked.csv`，行业和市场指数文件分别位于 `data/cache/hk_industry.csv` 与 `data/cache/hk_market_index.csv`。这些文件属于本机缓存，并被 `.gitignore` 排除，不会随着普通 Git 分支或新 worktree 自动复制。当前工作区内的其他 AI 可以按上述相对路径读取；如果要在另一台电脑、云端任务或新建 worktree 中使用，必须显式复制、上传或挂载 `data/yfinance_cache/`，不能只依赖 Git 仓库本身。

美股十年历史行情位于 `data/yfinance_cache/us_history.csv`，当前为 9,321,625 行、5,126 只代码，时间范围为 2016-09-14 至 2026-09-11。美股代码必须按字符串处理，文件包含完整的 `open`、`high`、`low`、`close`、`volume` 字段；50 只普通股主列表中的特殊代码没有取得可用历史，不能把它们当作零值补入。详细字段、覆盖率和异常规则见 `data/yfinance_cache/us_history_quality.json`，原四年文件已备份为 `data/yfinance_cache/us_history_4y_backup_20260914.csv`。

美股十年源文件同时保留在 `data/yfinance_cache_10y/us_history.csv`，当前与活动文件哈希一致。该文件由 `scripts/extend_us_history_10y.py` 增量维护，质量清单是 `data/yfinance_cache_10y/us_history_quality.json`，其中记录 50 个无历史代码、4,116 行非正收盘价和 8,667 行 OHLC 结构异常；原四年文件则单独保存在 `data/yfinance_cache/us_history_4y_backup_20260914.csv`。

官方源分开缓存：`data/hkex_cache/securities_master.csv` 是 HKEX 当前证券主表快照，`data/hkex_cache/daily_quotations.csv` 是公开 HKEX Daily Quotations 页面中当前可见窗口的主板/GEM 日报，质量与覆盖范围见 `data/hkex_cache/quality.json`。`data/hkexnews_cache/announcements.csv` 是 HKEXnews 按月份检索的公告索引，包含发布日期、证券代码、公告类别、标题和官方 PDF 链接；它不把公告正文 PDF 批量复制到本地，查询结果与完整性状态见 `data/hkexnews_cache/quality.json`。

港股十年历史下载入口为 `scripts/extend_hk_history.py`；官方源下载入口为 `scripts/download_hkex_sources.py`，默认请求最近约四年；HKEX 公开日报页面只提供当前网页窗口，脚本会把更早日期记为不可用，不会用 yfinance 数据冒充 HKEX 官方数据。HKEXnews 的当前/已除牌证券范围分别记录在 `security_scope` 字段，回测使用时必须按 `published_at` 做点时过滤，不能把后来发布的公告用于更早的信号。
