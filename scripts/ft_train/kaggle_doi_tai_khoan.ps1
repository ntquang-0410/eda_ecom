# Doi token Kaggle tren may nay sang tai khoan khac (vd. tai khoan 2). KHONG in token, KHONG ghi token vao repo.
#
# Cach dung (chay trong PowerShell, tu chinh nguoi dung chay, token go vao se bi an):
#   powershell -ExecutionPolicy Bypass -File .\scripts\ft_train\kaggle_doi_tai_khoan.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\ft_train\kaggle_doi_tai_khoan.ps1 -KhoiPhuc   # quay lai tai khoan cu
#
# Token duoc ghi o ~/.kaggle/access_token (ngoai repo). Token cu duoc sao luu o ~/.kaggle/access_token.tk_cu.bak
# (chi sao luu lan dau, khong ghi de ban sao luu da co).
param([switch]$KhoiPhuc)
$ErrorActionPreference = "Stop"

$thuMuc = Join-Path $env:USERPROFILE ".kaggle"
$tep    = Join-Path $thuMuc "access_token"
$sao    = Join-Path $thuMuc "access_token.tk_cu.bak"
$kaggle = "D:\nam4_ky1\e-commerce - EDA\.venv\Scripts\kaggle.exe"

function Hien-TaiKhoan {
    if (Test-Path $kaggle) {
        Write-Host "Tai khoan Kaggle hien tai (khong hien token):"
        & $kaggle config view
    } else {
        Write-Host "Khong thay kaggle.exe o $kaggle, hay chay: kaggle config view"
    }
}

if ($KhoiPhuc) {
    if (-not (Test-Path $sao)) { throw "Khong co ban sao luu $sao" }
    Copy-Item $sao $tep -Force
    Write-Host "Da khoi phuc token cu."
    Hien-TaiKhoan
    return
}

if (-not (Test-Path $thuMuc)) { New-Item -ItemType Directory -Force $thuMuc | Out-Null }
if ((Test-Path $tep) -and -not (Test-Path $sao)) {
    Copy-Item $tep $sao
    Write-Host "Da sao luu token cu sang $sao"
}

$an = Read-Host "Dan token Kaggle cua tai khoan moi (go/dan se KHONG hien ra), roi Enter" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($an)
try { $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim() }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
if ([string]::IsNullOrWhiteSpace($token)) { throw "Token rong, khong ghi gi ca." }
if (-not $token.StartsWith("KGAT_")) {
    Write-Host "Luu y: token thuong bat dau bang KGAT_. Neu day la file kaggle.json (username + key) thi bao lai de doi cach."
}
[IO.File]::WriteAllText($tep, $token, (New-Object Text.UTF8Encoding $false))
$token = $null
Write-Host "Da ghi token moi vao $tep (khong in ra)."
Hien-TaiKhoan
Write-Host "Kiem tra dong 'username' o tren: phai la tai khoan 2. Neu van la tai khoan cu thi bao lai."
