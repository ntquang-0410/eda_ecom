# Run the judge with a hidden API key prompt (ASCII only so Windows PowerShell 5.1 parses it).
# The key lives only in this process' environment variable: never written to a file, never echoed.
# Usage (from the repo root or from this folder):
#   .\scripts\judge\chay_judge.ps1 --kiem-tra
#   .\scripts\judge\chay_judge.ps1 --thu 5 --ten thu_5
#   .\scripts\judge\chay_judge.ps1 --thu 300
#   .\scripts\judge\chay_judge.ps1 --tat-ca --tran-usd 20
# NVIDIA free trial (asks for NVIDIA_API_KEY, NOT the OpenCode key):
#   .\scripts\judge\chay_judge.ps1 --nha-cung-cap nvidia --kiem-tra
param([Parameter(ValueFromRemainingArguments = $true)] $Doi)

$py = "D:\nam4_ky1\e-commerce - crawl\ecom_crawler\.venv\Scripts\python.exe"
$kich_ban = Join-Path $PSScriptRoot "chay_judge.py"
# --so-cap: chay bo SO CAP MU (so_cap_chon.py) thay cho cham diem don
if ($Doi -contains "--so-cap") {
    $kich_ban = Join-Path $PSScriptRoot "so_cap_chon.py"
    $Doi = @($Doi | Where-Object { $_ -ne "--so-cap" })
}

if ($Doi -contains "--chi-luat" -or $Doi -contains "--gia" -or $Doi -contains "--bao-cao") {
    & $py -I -X utf8 $kich_ban @Doi
    exit $LASTEXITCODE
}

$ten_bien = "OPENCODE_API_KEY"
if ($Doi -contains "nvidia") { $ten_bien = "NVIDIA_API_KEY" }

if (-not [Environment]::GetEnvironmentVariable($ten_bien, "Process")) {
    $an = Read-Host "Paste $ten_bien (hidden input)" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($an)
    try { [Environment]::SetEnvironmentVariable($ten_bien, [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr), "Process") }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}
$ma = 1
try {
    $env:OPENBLAS_NUM_THREADS = "1"
    & $py -I -X utf8 $kich_ban @Doi
    $ma = $LASTEXITCODE
}
finally {
    [Environment]::SetEnvironmentVariable($ten_bien, $null, "Process")
}
exit $ma
