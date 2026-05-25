$ProjectRoot = Split-Path -Parent $PSScriptRoot
$NodeDir = Join-Path $ProjectRoot "tools\node-v24.16.0-win-x64"
$env:TEMP = Join-Path $ProjectRoot "tmp"
$env:TMP = Join-Path $ProjectRoot "tmp"
Set-Location (Join-Path $ProjectRoot "frontend")
if (Test-Path (Join-Path $NodeDir "npm.cmd")) {
    $env:PATH = "$NodeDir;$env:PATH"
    & (Join-Path $NodeDir "npm.cmd") run dev
} else {
    npm run dev
}
