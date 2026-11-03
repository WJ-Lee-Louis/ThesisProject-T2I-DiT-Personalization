param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Command,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ClientArgs
)
$ErrorActionPreference = 'Stop'
$clientPython = Join-Path $PSScriptRoot '.venv-modal311/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $clientPython)) {
    throw 'Install the Python 3.11 Modal client environment from MODAL_GUIDE.md first.'
}
$previousUtf8 = $env:PYTHONUTF8
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $env:PYTHONUTF8 = '1'
    & $clientPython -m freecus.modal_client $Command @ClientArgs
    if ($LASTEXITCODE -ne 0) { throw "FreeCus Modal command failed: $Command" }
} finally {
    Pop-Location
    if ($null -eq $previousUtf8) { Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue }
    else { $env:PYTHONUTF8 = $previousUtf8 }
}
