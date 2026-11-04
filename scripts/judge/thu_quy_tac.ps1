# DO BO QUY TAC v2 tren CUNG 300 muc (100 title, 100 cum, 100 pair) da co ket qua v1; chay du HAI thu tu de do do lech, roi so voi v1
# va mo phong che do MOT LUOT. Chi la phep do: KHONG ghep bo, KHONG day HF, KHONG sua du lieu.
# Dung (tu thu muc goc repo):
#   .\scripts\judge\thu_quy_tac.ps1 -NhaCungCap alibaba -SongSong 8
#   .\scripts\judge\thu_quy_tac.ps1 -Gia                 # may moc bang judge gia, KHONG can khoa, KHONG goi API
# Ket qua + LOG toan bo man hinh nam o data\processed\so_cap\ (log_thu_quy_tac_<gio>.txt, ket_qua_qt_*.jsonl, so_sanh_quy_tac.json):
# chay xong chi can bao "chay xong", Claude doc thang tu file. Khoa chi nam trong bien moi truong cua tien trinh nay.
param([switch]$Gia, [ValidateSet("opencode", "nvidia", "alibaba")][string]$NhaCungCap = "opencode", [int]$SongSong = 4, [int]$N = 100)

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$py = "D:\nam4_ky1\e-commerce - crawl\ecom_crawler\.venv\Scripts\python.exe"
$chon = Join-Path $PSScriptRoot "so_cap_chon.py"
$cj = Join-Path $PSScriptRoot "chay_judge.py"
$sosanh = Join-Path $PSScriptRoot "so_sanh_quy_tac.py"
$bienKhoa = @{ opencode = "OPENCODE_API_KEY"; nvidia = "NVIDIA_API_KEY"; alibaba = "DASHSCOPE_API_KEY" }[$NhaCungCap]
$thuMucThat = Join-Path $PSScriptRoot "..\..\data\processed\so_cap"
New-Item -ItemType Directory -Force $thuMucThat | Out-Null
foreach ($l in "title", "cum", "pair") {
    if (-not (Test-Path (Join-Path $thuMucThat "ket_qua_${l}_$N.jsonl"))) { Write-Host "Thieu ket qua v1: ket_qua_${l}_$N.jsonl (chay thu_so_cap.ps1 -N $N truoc)."; exit 2 }
}
$pre = ""
$ra = @()
if ($Gia) {
    $tmp = Join-Path $env:TEMP "thu_quy_tac_gia"
    New-Item -ItemType Directory -Force $tmp | Out-Null
    foreach ($l in "title", "cum", "pair") { Copy-Item (Join-Path $thuMucThat "ket_qua_${l}_$N.jsonl") $tmp -Force }     # ban sao v1 de so sanh
    $ra = @("--ra", $tmp, "--gia")
    $pre = "gia_"
    $thuMucLog = $tmp
}
else {
    $thuMucLog = $thuMucThat
    $ra = @("--nha-cung-cap", $NhaCungCap)
    if ($NhaCungCap -eq "alibaba" -and -not $env:ALIBABA_BASE_URL) {
        $env:ALIBABA_BASE_URL = Read-Host "Dan dia chi goc (https://<WorkspaceId>.<vung>.maas.aliyuncs.com/compatible-mode/v1 hoac token-plan...)"
    }
    $an = Read-Host "Dan $bienKhoa (go an, khong hien)" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($an)
    try { [Environment]::SetEnvironmentVariable($bienKhoa, [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr), "Process") }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}
$log = Join-Path $thuMucLog ("log_thu_quy_tac_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
function Ghi { process { Add-Content -LiteralPath $script:log -Value $_ -Encoding utf8; $_ } }
"thu_quy_tac | N=$N | nha cung cap=$NhaCungCap | song song=$SongSong | Gia=$Gia" | Ghi
$ma = 0
try {
    $env:OPENBLAS_NUM_THREADS = "1"
    if (-not $Gia) {
        "=== Buoc 0: kiem tra khoa va dia chi ($NhaCungCap) ===" | Ghi
        & $py -I -X utf8 $cj --kiem-tra --nha-cung-cap $NhaCungCap | Ghi
        if ($LASTEXITCODE -ne 0) { "Khoa hoac dia chi loi: dung lai." | Ghi; $ma = 2; exit 2 }
    }
    foreach ($l in "title", "cum", "pair") {
        "=== Quy tac v2, hai thu tu, $N muc loai '$l' ===" | Ghi
        & $py -I -X utf8 $chon --loai $l --thu $N --rubric v2 --che-do hai_luot --tat-suy-luan --song-song $SongSong --tran-usd 0.5 --ten "qt_${l}_$N" @ra | Ghi
        if ($LASTEXITCODE -ne 0) { "Loi nang o loai '$l': dung lai." | Ghi; $ma = 2; exit 2 }
    }
    "=== So v2 voi v1, mo phong MOT LUOT ===" | Ghi
    $raSS = @(); if ($Gia) { $raSS = @("--ra", $tmp) }
    & $py -I -X utf8 $sosanh --v1 "title_$N,cum_$N,pair_$N" --v2 "${pre}qt_title_$N,${pre}qt_cum_$N,${pre}qt_pair_$N" @raSS | Ghi
}
finally {
    [Environment]::SetEnvironmentVariable($bienKhoa, $null, "Process")
    "=== KET THUC | ma thoat $ma | log: $log ===" | Ghi
}
exit $ma
