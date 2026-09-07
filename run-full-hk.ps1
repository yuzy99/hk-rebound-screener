$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $project 'src'
python -m hk_rebound_screener.main --mode live --full-market --config (Join-Path $project 'config.json')
