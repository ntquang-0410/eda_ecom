# THI DIEM CHE DO GOM: chay MOT luot, gom nhieu muc/loi goi (rubric v2) tren CUNG 450 muc (150 title, 150 cum_ctx, 150 pair) da co ket qua
# hai thu tu (m2_v2_*), roi so voi ket qua do. Muc dich: kiem tra gom co lam judge kem di khong va tiet kiem bao nhieu token.
# Chi la phep do: KHONG ghep bo, KHONG day HF, KHONG sua du lieu. Uoc tinh chi phi: khoang 0,15-0,3 trieu token (moi loai co tran 0,5 trieu).
# Dung (tu thu muc goc repo):
#   .\scripts\judge\thu_gom.ps1 -NhaCungCap alibaba -SongSong 8            # gom 15 muc/loi goi
#   .\scripts\judge\thu_gom.ps1 -NhaCungCap alibaba -Gom 20
#   .\scripts\judge\thu_gom.ps1 -Gia                                       # may moc bang judge gia, KHONG can khoa, KHONG goi API
# Ket qua + LOG toan bo man hinh nam o data\processed\so_cap\ (log_thu_gom_<gio>.txt, ket_qua_b*_150.jsonl, so_sanh_gom.json):
# chay xong chi can bao "chay xong", Claude doc thang tu file. Khoa chi nam trong bien moi truong cua tien trinh nay.
param([switch]$Gia, [ValidateSet("opencode", "nvidia", "alibaba")][string]$NhaCungCap = "opencode", [int]$SongSong = 4, [int]$N = 150, [int]$Gom = 15)

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$py = "D:\nam4_ky1\e-commerce - crawl\ecom_crawler\.venv\Scripts\python.exe"
$chon = Join-Path $PSScriptRoot "so_cap_chon.py"
$cj = Join-Path $PSScriptRoot "chay_judge.py"
$sosanh = Join-Path $PSScriptRoot "so_sanh_gom.py"
$bienKhoa = @{ opencode = "OPENCODE_API_KEY"; nvidia = "NVIDIA_API_KEY"; alibaba = "DASHSCOPE_API_KEY" }[$NhaCungCap]
$thuMucThat = Join-Path $PSScriptRoot "..\..\data\processed\so_cap"
New-Item -ItemType Directory -Force $thuMucThat | Out-Null
foreach ($l in "title", "cum_ctx", "pair") {
    if (-not (Test-Path (Join-Path $thuMucThat "ket_qua_m2_v2_${l}_$N.jsonl"))) { Write-Host "Thieu ket qua hai thu tu: ket_qua_m2_v2_${l}_$N.jsonl (chay do_lai_quy_tac.ps1 -N $N truoc)."; exit 2 }
}
$pre = ""
$ra = @()
if ($Gia) {
    $tmp = Join-Path $env:TEMP "thu_gom_gia"
    New-Item -ItemType Directory -Force $tmp | Out-Null
    foreach ($l in "title", "cum_ctx", "pair") { Copy-Item (Join-Path $thuMucThat "ket_qua_m2_v2_${l}_$N.jsonl") $tmp -Force }
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
$log = Join-Path $thuMucLog ("log_thu_gom_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
function Ghi { process { Add-Content -LiteralPath $script:log -Value $_ -Encoding utf8; $_ } }
"thu_gom | N=$N | gom=$Gom | nha cung cap=$NhaCungCap | song song=$SongSong | Gia=$Gia" | Ghi
$ma = 0
try {
    $env:OPENBLAS_NUM_THREADS = "1"
    if (-not $Gia) {
        "=== Buoc 0: kiem tra khoa va dia chi ($NhaCungCap) ===" | Ghi
        & $py -I -X utf8 $cj --kiem-tra --nha-cung-cap $NhaCungCap | Ghi
        if ($LASTEXITCODE -ne 0) { "Khoa hoac dia chi loi: dung lai." | Ghi; $ma = 2; exit 2 }
    }
    foreach ($l in "title", "cum_ctx", "pair") {
        "=== Che do GOM $Gom, mot luot, $N muc loai '$l' ===" | Ghi
        & $py -I -X utf8 $chon --loai $l --thu $N --rubric v2 --che-do mot_luot --gom $Gom --tat-suy-luan --song-song $SongSong --tran-token 500000 --ten "b_${l}_$N" @ra | Ghi
        if ($LASTEXITCODE -ne 0) { "Loi nang o loai '$l': dung lai." | Ghi; $ma = 2; exit 2 }
    }
    "=== So gom voi hai thu tu khong gom ===" | Ghi
    $raSS = @(); if ($Gia) { $raSS = @("--ra", $tmp) }
    & $py -I -X utf8 $sosanh --hai-luot "m2_v2_title_$N,m2_v2_cum_ctx_$N,m2_v2_pair_$N" --gom "${pre}b_title_$N,${pre}b_cum_ctx_$N,${pre}b_pair_$N" @raSS | Ghi
}
finally {
    [Environment]::SetEnvironmentVariable($bienKhoa, $null, "Process")
    "=== KET THUC | ma thoat $ma | log: $log ===" | Ghi
}
exit $ma
