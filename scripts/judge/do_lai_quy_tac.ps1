# DO LAI v1 va v2 (HAI thu tu) tren bang muc MOI (da bo cac ung vien rong), CUNG mau cho ca hai lan, roi so sanh + mo phong MOT LUOT.
# Loai 'cum' dung cum_ctx (CO ngu canh) vi do la giao thuc cua lan chay chinh. Chi la phep do: KHONG ghep bo, KHONG day HF, KHONG sua du lieu.
# Dung (tu thu muc goc repo):
#   .\scripts\judge\do_lai_quy_tac.ps1 -NhaCungCap alibaba -SongSong 8            # N = 150 muc moi loai (~1.800 loi goi)
#   .\scripts\judge\do_lai_quy_tac.ps1 -Gia                                       # may moc bang judge gia, KHONG can khoa, KHONG goi API
# Ket qua + LOG toan bo man hinh nam o data\processed\so_cap\ (log_do_lai_quy_tac_<gio>.txt, ket_qua_m2_*.jsonl, so_sanh_quy_tac.json):
# chay xong chi can bao "chay xong", Claude doc thang tu file. Khoa chi nam trong bien moi truong cua tien trinh nay.
param([switch]$Gia, [ValidateSet("opencode", "nvidia", "alibaba")][string]$NhaCungCap = "opencode", [int]$SongSong = 4, [int]$N = 150)

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$py = "D:\nam4_ky1\e-commerce - crawl\ecom_crawler\.venv\Scripts\python.exe"
$chon = Join-Path $PSScriptRoot "so_cap_chon.py"
$cj = Join-Path $PSScriptRoot "chay_judge.py"
$sosanh = Join-Path $PSScriptRoot "so_sanh_quy_tac.py"
$bienKhoa = @{ opencode = "OPENCODE_API_KEY"; nvidia = "NVIDIA_API_KEY"; alibaba = "DASHSCOPE_API_KEY" }[$NhaCungCap]
$thuMucThat = Join-Path $PSScriptRoot "..\..\data\processed\so_cap"
New-Item -ItemType Directory -Force $thuMucThat | Out-Null
$pre = ""
$ra = @()
if ($Gia) {
    $tmp = Join-Path $env:TEMP "do_lai_quy_tac_gia"
    New-Item -ItemType Directory -Force $tmp | Out-Null
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
$log = Join-Path $thuMucLog ("log_do_lai_quy_tac_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
function Ghi { process { Add-Content -LiteralPath $script:log -Value $_ -Encoding utf8; $_ } }
"do_lai_quy_tac | N=$N | nha cung cap=$NhaCungCap | song song=$SongSong | Gia=$Gia" | Ghi
$ma = 0
try {
    $env:OPENBLAS_NUM_THREADS = "1"
    if (-not $Gia) {
        "=== Buoc 0: kiem tra khoa va dia chi ($NhaCungCap) ===" | Ghi
        & $py -I -X utf8 $cj --kiem-tra --nha-cung-cap $NhaCungCap | Ghi
        if ($LASTEXITCODE -ne 0) { "Khoa hoac dia chi loi: dung lai." | Ghi; $ma = 2; exit 2 }
    }
    foreach ($l in "title", "cum_ctx", "pair") {
        foreach ($v in "v1", "v2") {
            "=== Rubric $v, hai thu tu, $N muc loai '$l' ===" | Ghi
            & $py -I -X utf8 $chon --loai $l --thu $N --rubric $v --che-do hai_luot --tat-suy-luan --song-song $SongSong --tran-usd 0.5 --ten "m2_${v}_${l}_$N" @ra | Ghi
            if ($LASTEXITCODE -ne 0) { "Loi nang o '$v' loai '$l': dung lai." | Ghi; $ma = 2; exit 2 }
        }
    }
    "=== So v2 voi v1 tren bang muc moi, mo phong MOT LUOT ===" | Ghi
    $raSS = @(); if ($Gia) { $raSS = @("--ra", $tmp) }
    & $py -I -X utf8 $sosanh --v1 "${pre}m2_v1_title_$N,${pre}m2_v1_cum_ctx_$N,${pre}m2_v1_pair_$N" --v2 "${pre}m2_v2_title_$N,${pre}m2_v2_cum_ctx_$N,${pre}m2_v2_pair_$N" @raSS | Ghi
}
finally {
    [Environment]::SetEnvironmentVariable($bienKhoa, $null, "Process")
    "=== KET THUC | ma thoat $ma | log: $log ===" | Ghi
}
exit $ma
