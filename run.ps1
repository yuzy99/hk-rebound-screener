$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $project 'src'
python -m hk_rebound_screener.main --mode sample --config (Join-Path $project 'config.demo.json') --asof 2026-09-02
