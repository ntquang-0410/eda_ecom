# Chay THU vai mau cho bo SO CAP (DeepSeek chon Gemma/Qwen) - dan khoa MOT LAN, chay het cac buoc, xoa khoa khi xong.
# Dung (tu thu muc goc repo):
#   .\scripts\judge\thu_so_cap.ps1                          # OpenCode Go, 10 muc moi loai (title, cum, pair)
#   .\scripts\judge\thu_so_cap.ps1 -NhaCungCap alibaba -N 20  # Alibaba Cloud Model Studio (hoi khoa DASHSCOPE_API_KEY va dia chi co WorkspaceId)
#   .\scripts\judge\thu_so_cap.ps1 -NhaCungCap alibaba -N 100 -SongSong 8   # do toc do voi nhieu luong hon (mac dinh 4)
#   .\scripts\judge\thu_so_cap.ps1 -Gia                     # chay may moc bang judge gia, KHONG can khoa, KHONG goi API
# Ket qua + LOG toan bo man hinh nam o data\processed\so_cap\ (log_thu_so_cap_<gio>.txt, ket_qua_*.jsonl, bao_cao_*.json):
# chay xong chi can bao "chay xong", Claude doc thang tu file, khong can dan tu terminal. Khoa chi nam trong bien moi truong cua tien trinh nay.
param([int]$N = 10, [switch]$Gia, [ValidateSet("opencode", "nvidia", "alibaba")][string]$NhaCungCap = "opencode", [int]$SongSong = 4)

[Console]::OutputEncoding = [Text.Encoding]::UTF8     # de log doc dung tieng Viet
$py = "D:\nam4_ky1\e-commerce - crawl\ecom_crawler\.venv\Scripts\python.exe"
$chon = Join-Path $PSScriptRoot "so_cap_chon.py"
$cj = Join-Path $PSScriptRoot "chay_judge.py"
$bienKhoa = @{ opencode = "OPENCODE_API_KEY"; nvidia = "NVIDIA_API_KEY"; alibaba = "DASHSCOPE_API_KEY" }[$NhaCungCap]
$pre = ""
$ra = @()
if ($Gia) {
    $tmp = Join-Path $env:TEMP "so_cap_thu_gia"
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
        # dia chi KHONG phai bi mat (chi chua WorkspaceId), nen hien khi go
        $env:ALIBABA_BASE_URL = Read-Host "Dan dia chi goc (https://<WorkspaceId>.<vung>.maas.aliyuncs.com/compatible-mode/v1)"
    }
    $an = Read-Host "Dan $bienKhoa (go an, khong hien)" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($an)
    try { [Environment]::SetEnvironmentVariable($bienKhoa, [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr), "Process") }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}
$log = Join-Path $thuMucLog ("log_thu_so_cap_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
function Ghi { process { Add-Content -LiteralPath $script:log -Value $_ -Encoding utf8; $_ } }     # in ra man hinh VA ghi vao log
"thu_so_cap | N=$N | nha cung cap=$NhaCungCap | song song=$SongSong | Gia=$Gia" | Ghi
$ma = 0
try {
    $env:OPENBLAS_NUM_THREADS = "1"
    if (-not $Gia) {
        "=== Buoc 0: kiem tra khoa va dia chi ($NhaCungCap) ===" | Ghi
        & $py -I -X utf8 $cj --kiem-tra --nha-cung-cap $NhaCungCap | Ghi
        if ($LASTEXITCODE -ne 0) { "Khoa hoac dia chi loi: dung lai." | Ghi; exit 2 }
    }
    foreach ($l in "title", "cum", "pair") {
        "=== Chay thu $N muc loai '$l' ===" | Ghi
        & $py -I -X utf8 $chon --loai $l --thu $N --tat-suy-luan --song-song $SongSong --tran-usd 0.5 --ten "${l}_$N" @ra | Ghi
        if ($LASTEXITCODE -ne 0) { "Loi nang o loai '$l': dung lai." | Ghi; exit 2 }
    }
    foreach ($l in "title", "cum", "pair") {
        "=== Bao cao loai '$l' ===" | Ghi
        & $py -I -X utf8 $chon --bao-cao "${pre}${l}_$N" @(if ($Gia) { "--ra"; $tmp }) | Ghi
    }
}
finally {
    [Environment]::SetEnvironmentVariable($bienKhoa, $null, "Process")
    "=== KET THUC | ma thoat $ma | log: $log ===" | Ghi
}
exit $ma
