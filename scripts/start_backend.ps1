$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:TEMP = Join-Path $ProjectRoot "tmp"
$env:TMP = Join-Path $ProjectRoot "tmp"
Set-Location $ProjectRoot
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

