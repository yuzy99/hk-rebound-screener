$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $project 'src'
python -m hk_rebound_screener.main --mode sample --config (Join-Path $project 'config.us.demo.json') --universe (Join-Path $project 'universe.us.csv') --prices (Join-Path $project 'data' 'sample' 'us_prices.csv') --news (Join-Path $project 'data' 'sample' 'us_news.csv') --asof 2026-09-02
