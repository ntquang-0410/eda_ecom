# THI DIEM TANG 2 cua title va pair: chay lai CO ngu canh cac muc 'khong nhat quan' cua tang 1, roi so voi tang 1.
# Can co ket qua tang 1 san (mac dinh dung title_100 va pair_100 cua lan thu 8 luong: .\scripts\judge\thu_so_cap.ps1 -N 100 ...).
# Chi la thi diem: KHONG ghep bo, KHONG day HF, KHONG sua du lieu.
# Dung (tu thu muc goc repo):
#   .\scripts\judge\thu_tang2.ps1 -NhaCungCap alibaba -SongSong 8
#   .\scripts\judge\thu_tang2.ps1 -NhaCungCap alibaba -Tang1Title title_100 -Tang1Pair pair_100
# Ket qua + LOG toan bo man hinh nam o data\processed\so_cap\ (log_thu_tang2_<gio>.txt, ket_qua_*_t2.jsonl, so_sanh_tang_*.json):
# chay xong chi can bao "chay xong", Claude doc thang tu file. Khoa chi nam trong bien moi truong cua tien trinh nay.
param([ValidateSet("opencode", "nvidia", "alibaba")][string]$NhaCungCap = "opencode", [int]$SongSong = 4,
      [string]$Tang1Title = "title_100", [string]$Tang1Pair = "pair_100")

[Console]::OutputEncoding = [Text.Encoding]::UTF8     # de log doc dung tieng Viet
$py = "D:\nam4_ky1\e-commerce - crawl\ecom_crawler\.venv\Scripts\python.exe"
$chon = Join-Path $PSScriptRoot "so_cap_chon.py"
$cj = Join-Path $PSScriptRoot "chay_judge.py"
$sosanh = Join-Path $PSScriptRoot "so_sanh_tang.py"
$bienKhoa = @{ opencode = "OPENCODE_API_KEY"; nvidia = "NVIDIA_API_KEY"; alibaba = "DASHSCOPE_API_KEY" }[$NhaCungCap]
$thuMuc = Join-Path $PSScriptRoot "..\..\data\processed\so_cap"
New-Item -ItemType Directory -Force $thuMuc | Out-Null
foreach ($t in $Tang1Title, $Tang1Pair) {
    if (-not (Test-Path (Join-Path $thuMuc "ket_qua_$t.jsonl"))) { Write-Host "Thieu ket qua tang 1: ket_qua_$t.jsonl (chay thu_so_cap.ps1 truoc)."; exit 2 }
}
if ($NhaCungCap -eq "alibaba" -and -not $env:ALIBABA_BASE_URL) {
    $env:ALIBABA_BASE_URL = Read-Host "Dan dia chi goc (https://<WorkspaceId>.<vung>.maas.aliyuncs.com/compatible-mode/v1 hoac token-plan...)"
}
$an = Read-Host "Dan $bienKhoa (go an, khong hien)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($an)
try { [Environment]::SetEnvironmentVariable($bienKhoa, [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr), "Process") }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
$log = Join-Path $thuMuc ("log_thu_tang2_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
function Ghi { process { Add-Content -LiteralPath $script:log -Value $_ -Encoding utf8; $_ } }     # in ra man hinh VA ghi vao log
"thu_tang2 | nha cung cap=$NhaCungCap | song song=$SongSong | tang1 title=$Tang1Title pair=$Tang1Pair" | Ghi
$ma = 0
try {
    $env:OPENBLAS_NUM_THREADS = "1"
    "=== Buoc 0: kiem tra khoa va dia chi ($NhaCungCap) ===" | Ghi
    & $py -I -X utf8 $cj --kiem-tra --nha-cung-cap $NhaCungCap | Ghi
    if ($LASTEXITCODE -ne 0) { "Khoa hoac dia chi loi: dung lai." | Ghi; $ma = 2; exit 2 }
    foreach ($m in @(@("title", $Tang1Title), @("pair", $Tang1Pair))) {
        $l = $m[0]; $t1 = $m[1]
        "=== Tang 2 loai '$l': chay lai CO ngu canh cac muc khong nhat quan cua $t1 ===" | Ghi
        & $py -I -X utf8 $chon --loai "${l}_ctx" --chi-khong-nhat-quan $t1 --tat-ca --tat-suy-luan --song-song $SongSong --tran-usd 0.5 --nha-cung-cap $NhaCungCap --ten "${t1}_t2" | Ghi
        if ($LASTEXITCODE -ne 0) { "Loi nang o loai '$l': dung lai." | Ghi; $ma = 2; exit 2 }
        "=== So tang 2 voi tang 1 (loai '$l') ===" | Ghi
        & $py -I -X utf8 $sosanh --loai $l --tang1 $t1 --tang2 "${t1}_t2" --in-mau 6 | Ghi
    }
}
finally {
    [Environment]::SetEnvironmentVariable($bienKhoa, $null, "Process")
    "=== KET THUC | ma thoat $ma | log: $log ===" | Ghi
}
exit $ma
