param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Command,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ClientArgs
)
$ErrorActionPreference = 'Stop'
$previousProfile = $env:FREECUS_MODAL_PROFILE
$previousGpu = $env:FREECUS_MODAL_GPU
try {
    $env:FREECUS_MODAL_PROFILE = 'method-comparison-dbpp'
    $env:FREECUS_MODAL_GPU = 'H100'
    & (Join-Path $PSScriptRoot 'modal.ps1') $Command @ClientArgs
} finally {
    if ($null -eq $previousProfile) { Remove-Item Env:FREECUS_MODAL_PROFILE -ErrorAction SilentlyContinue }
    else { $env:FREECUS_MODAL_PROFILE = $previousProfile }
    if ($null -eq $previousGpu) { Remove-Item Env:FREECUS_MODAL_GPU -ErrorAction SilentlyContinue }
    else { $env:FREECUS_MODAL_GPU = $previousGpu }
}
