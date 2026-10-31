# THI DIEM bo CUM co ngu canh: chay N cap (san pham, cap) CO ngu canh, roi chay CUNG cac cap do KHONG ngu canh, roi so hai cach.
# Do chi phi (token, giay) va xem ngu canh co doi ket qua khong. Chi la thi diem: KHONG ghep bo, KHONG day HF, KHONG sua du lieu.
# Dung (tu thu muc goc repo):
#   .\scripts\judge\thu_cum_ngu_canh.ps1 -NhaCungCap alibaba            # N = 40 (20 cap 1688 + 20 cap Tiki)
#   .\scripts\judge\thu_cum_ngu_canh.ps1 -NhaCungCap alibaba -N 50 -SongSong 8
#   .\scripts\judge\thu_cum_ngu_canh.ps1 -NhaCungCap alibaba -LapKhong  # sau khi da chay xong: chay LAI chi phan KHONG ngu canh de do do nhieu cua judge
#   .\scripts\judge\thu_cum_ngu_canh.ps1 -Gia                           # may moc bang judge gia, KHONG can khoa, KHONG goi API
# Ket qua + LOG toan bo man hinh nam o data\processed\so_cap\ (log_thu_cum_ngu_canh_<gio>.txt, ket_qua_*.jsonl, so_sanh_*.json):
# chay xong chi can bao "chay xong", Claude doc thang tu file, khong can dan tu terminal. Khoa chi nam trong bien moi truong cua tien trinh nay.
param([int]$N = 40, [switch]$Gia, [switch]$LapKhong, [ValidateSet("opencode", "nvidia", "alibaba")][string]$NhaCungCap = "opencode", [int]$SongSong = 4)

[Console]::OutputEncoding = [Text.Encoding]::UTF8     # de log doc dung tieng Viet
$py = "D:\nam4_ky1\e-commerce - crawl\ecom_crawler\.venv\Scripts\python.exe"
$chon = Join-Path $PSScriptRoot "so_cap_chon.py"
$cj = Join-Path $PSScriptRoot "chay_judge.py"
$sosanh = Join-Path $PSScriptRoot "so_sanh_ngu_canh.py"
$bienKhoa = @{ opencode = "OPENCODE_API_KEY"; nvidia = "NVIDIA_API_KEY"; alibaba = "DASHSCOPE_API_KEY" }[$NhaCungCap]
$pre = ""
$ra = @()
if ($Gia) {
    $tmp = Join-Path $env:TEMP "thu_cum_ctx_gia"
    New-Item -ItemType Directory -Force $tmp | Out-Null
    $ra = @("--ra", $tmp, "--gia")
    $pre = "gia_"
    $thuMucLog = $tmp
}
else {
    $thuMucLog = Join-Path $PSScriptRoot "..\..\data\processed\so_cap"
    New-Item -ItemType Directory -Force $thuMucLog | Out-Null
    $ra = @("--nha-cung-cap", $NhaCungCap)
    if ($NhaCungCap -eq "alibaba" -and -not $env:ALIBABA_BASE_URL) {
        $env:ALIBABA_BASE_URL = Read-Host "Dan dia chi goc (https://<WorkspaceId>.<vung>.maas.aliyuncs.com/compatible-mode/v1 hoac token-plan...)"
    }
    $an = Read-Host "Dan $bienKhoa (go an, khong hien)" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($an)
    try { [Environment]::SetEnvironmentVariable($bienKhoa, [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr), "Process") }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}
$log = Join-Path $thuMucLog ("log_thu_cum_ngu_canh_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
function Ghi { process { Add-Content -LiteralPath $script:log -Value $_ -Encoding utf8; $_ } }     # in ra man hinh VA ghi vao log
"thu_cum_ngu_canh | N=$N | nha cung cap=$NhaCungCap | song song=$SongSong | LapKhong=$LapKhong | Gia=$Gia" | Ghi
$ma = 0
try {
    $env:OPENBLAS_NUM_THREADS = "1"
    if (-not $Gia) {
        "=== Buoc 0: kiem tra khoa va dia chi ($NhaCungCap) ===" | Ghi
        & $py -I -X utf8 $cj --kiem-tra --nha-cung-cap $NhaCungCap | Ghi
        if ($LASTEXITCODE -ne 0) { "Khoa hoac dia chi loi: dung lai." | Ghi; exit 2 }
    }
    $raSS = @(); if ($Gia) { $raSS = @("--ra", $tmp) }
    if ($LapKhong) {
        # chay lai CHI phan khong ngu canh (cung cac cap, cung cau hinh) de do do nhieu cua chinh judge
        "=== Chay LAI phan KHONG ngu canh (do do nhieu) ===" | Ghi
        & $py -I -X utf8 $chon --loai cum --theo-ctx "${pre}cumctx_$N" --tat-ca --tat-suy-luan --song-song $SongSong --tran-usd 0.5 --ten "cumkhong_${N}_lap" @ra | Ghi
        if ($LASTEXITCODE -ne 0) { "Loi nang: dung lai." | Ghi; exit 2 }
        & $py -I -X utf8 $sosanh --ctx "${pre}cumctx_$N" --khong "${pre}cumkhong_$N" --khong2 "${pre}cumkhong_${N}_lap" @raSS | Ghi
        exit 0
    }
    "=== Buoc 1: $N cap CO ngu canh (moi cap hoi 2 thu tu) ===" | Ghi
    & $py -I -X utf8 $chon --loai cum_ctx --thu $N --tat-suy-luan --song-song $SongSong --tran-usd 0.5 --ten "cumctx_$N" @ra | Ghi
    if ($LASTEXITCODE -ne 0) { "Loi nang o buoc 1: dung lai." | Ghi; exit 2 }
    "=== Buoc 2: CUNG cac cap do, KHONG ngu canh ===" | Ghi
    & $py -I -X utf8 $chon --loai cum --theo-ctx "${pre}cumctx_$N" --tat-ca --tat-suy-luan --song-song $SongSong --tran-usd 0.5 --ten "cumkhong_$N" @ra | Ghi
    if ($LASTEXITCODE -ne 0) { "Loi nang o buoc 2: dung lai." | Ghi; exit 2 }
    "=== Buoc 3: so sanh hai cach (token, ket qua, ca dao nguoc) ===" | Ghi
    & $py -I -X utf8 $sosanh --ctx "${pre}cumctx_$N" --khong "${pre}cumkhong_$N" --in-mau 8 @raSS | Ghi
}
finally {
    [Environment]::SetEnvironmentVariable($bienKhoa, $null, "Process")
    "=== KET THUC | ma thoat $ma | log: $log ===" | Ghi
}
exit $ma
