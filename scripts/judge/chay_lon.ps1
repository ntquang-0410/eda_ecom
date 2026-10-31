# CHAY LON (DeepSeek chon Gemma/Qwen cho Title + Mo ta, ca ban CUM va ban PAIR), che do GOM + MOT LUOT, rubric v2.
# Thu tu uu tien: pair (top N don vi dung nhieu nhat) -> cum LAI (4.446 muc) -> title (tat ca). Moi giai doan co TRAN TOKEN rieng; chay tiep duoc
# (ket qua da xong khong hoi lai). Chay het 3 giai doan thi tu dong GHEP hai bo cuoi (CHI luu tren may, KHONG day HF).
# Dung (tu thu muc goc repo):
#   .\scripts\judge\chay_lon.ps1 -NhaCungCap alibaba -SongSong 8                    # mac dinh: gom 15, pair top 5000
#   .\scripts\judge\chay_lon.ps1 -NhaCungCap alibaba -ChiGiaiDoan pair              # chi chay mot giai doan: pair | cum | title | ghep
#   .\scripts\judge\chay_lon.ps1 -Gia                                               # CA QUY TRINH bang judge gia tren du lieu that, KHONG can khoa, KHONG goi API
# Ket qua + LOG nam o data\processed\so_cap\ (log_chay_lon_<gio>.txt, ket_qua_lon_*.jsonl, tom_tat_lon_*.json); bo cuoi o data\processed\bo_chon\.
# Chay xong chi can bao "chay xong", Claude doc thang tu file. Khoa chi nam trong bien moi truong cua tien trinh nay.
param([switch]$Gia, [ValidateSet("opencode", "nvidia", "alibaba")][string]$NhaCungCap = "opencode", [int]$SongSong = 8, [int]$Gom = 15,
      [int]$TopPair = 5000, [int]$TranPair = 1400000, [int]$TranCum = 3300000, [int]$TranTitle = 7500000,
      [ValidateSet("tat_ca", "pair", "cum", "title", "ghep")][string]$ChiGiaiDoan = "tat_ca",
      [ValidateSet("giu_hien_tai", "theo_judge")][string]$KhiTuongDuong = "giu_hien_tai")

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$py = "D:\nam4_ky1\e-commerce - crawl\ecom_crawler\.venv\Scripts\python.exe"
$chon = Join-Path $PSScriptRoot "so_cap_chon.py"
$cj = Join-Path $PSScriptRoot "chay_judge.py"
$ghep = Join-Path $PSScriptRoot "ghep_bo.py"
$bienKhoa = @{ opencode = "OPENCODE_API_KEY"; nvidia = "NVIDIA_API_KEY"; alibaba = "DASHSCOPE_API_KEY" }[$NhaCungCap]
$thuMucThat = Join-Path $PSScriptRoot "..\..\data\processed\so_cap"
New-Item -ItemType Directory -Force $thuMucThat | Out-Null
$pre = ""
$ra = @()
$raGhep = @()
if ($Gia) {
    $tmp = Join-Path $env:TEMP "chay_lon_gia"
    $tmpBo = Join-Path $env:TEMP "chay_lon_gia_bo"
    New-Item -ItemType Directory -Force $tmp | Out-Null
    $ra = @("--ra", $tmp, "--gia")
    $raGhep = @("--thu-muc-qd", $tmp, "--ra", $tmpBo)
    $pre = "gia_"
    $thuMucLog = $tmp
}
else {
    $thuMucLog = $thuMucThat
    $ra = @("--nha-cung-cap", $NhaCungCap)
    if ($ChiGiaiDoan -ne "ghep") {
        if ($NhaCungCap -eq "alibaba" -and -not $env:ALIBABA_BASE_URL) {
            $env:ALIBABA_BASE_URL = Read-Host "Dan dia chi goc (https://<WorkspaceId>.<vung>.maas.aliyuncs.com/compatible-mode/v1 hoac token-plan...)"
        }
        $an = Read-Host "Dan $bienKhoa (go an, khong hien)" -AsSecureString
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($an)
        try { [Environment]::SetEnvironmentVariable($bienKhoa, [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr), "Process") }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    }
}
$log = Join-Path $thuMucLog ("log_chay_lon_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
function Ghi { process { Add-Content -LiteralPath $script:log -Value $_ -Encoding utf8; Write-Host $_ } }     # in ra man hinh + ghi log, KHONG dua vao pipeline (de ham tra ve dung ma)
"chay_lon | gom=$Gom | top pair=$TopPair | tran (pair/cum/title)=$TranPair/$TranCum/$TranTitle | nha cung cap=$NhaCungCap | song song=$SongSong | giai doan=$ChiGiaiDoan | Gia=$Gia | khi tuong duong=$KhiTuongDuong" | Ghi
$ma = 0
$tongToken = 0
function Chay-GiaiDoan($ten, $loai, $tran, $them) {
    "=== Giai doan '$ten' (loai $loai, tran $tran token) ===" | Ghi
    $o = $null
    & $py -I -X utf8 $chon --loai $loai --tat-ca --rubric v2 --che-do mot_luot --gom $Gom --tat-suy-luan --song-song $SongSong --tran-token $tran --ten "lon_$ten" @them @ra | Tee-Object -Variable o | Ghi
    if ($LASTEXITCODE -ne 0) { "Loi nang o giai doan '$ten': dung lai (chay lai lenh nay de tiep tuc)." | Ghi; return 2 }
    if (($o -join "`n") -match "chạm trần") { "Giai doan '$ten' CHAM TRAN token, CHUA XONG: dung lai, khong ghep. Chay lai de tiep tuc (hoac tang tran)." | Ghi; return 3 }
    $tt = Join-Path $thuMucLog "tom_tat_${pre}lon_$ten.json"
    if (Test-Path $tt) { $j = Get-Content $tt -Raw -Encoding utf8 | ConvertFrom-Json; $script:tongToken += [double]$j.tokens_vao + [double]$j.tokens_ra; "Giai doan '$ten': xong, token lan nay $([math]::Round($j.tokens_vao + $j.tokens_ra)); cong don trong lan chay nay $([math]::Round($script:tongToken))" | Ghi }
    return 0
}
try {
    $env:OPENBLAS_NUM_THREADS = "1"
    if (-not $Gia -and $ChiGiaiDoan -ne "ghep") {
        "=== Buoc 0: kiem tra khoa va dia chi ($NhaCungCap) ===" | Ghi
        & $py -I -X utf8 $cj --kiem-tra --nha-cung-cap $NhaCungCap | Ghi
        if ($LASTEXITCODE -ne 0) { "Khoa hoac dia chi loi: dung lai." | Ghi; $ma = 2; exit 2 }
    }
    if ($ChiGiaiDoan -in "tat_ca", "pair") { $r = Chay-GiaiDoan "pair" "pair" $TranPair @("--top", $TopPair); if ($r -ne 0) { $ma = $r; exit $r } }
    if ($ChiGiaiDoan -in "tat_ca", "cum") { $r = Chay-GiaiDoan "cum" "cum_lai" $TranCum @(); if ($r -ne 0) { $ma = $r; exit $r } }
    if ($ChiGiaiDoan -in "tat_ca", "title") { $r = Chay-GiaiDoan "title" "title" $TranTitle @(); if ($r -ne 0) { $ma = $r; exit $r } }
    if ($ChiGiaiDoan -in "tat_ca", "ghep") {
        "=== Ghep hai bo cuoi (chi luu tren may) ===" | Ghi
        & $py -I -X utf8 $ghep --title "${pre}lon_title" --cum "${pre}lon_cum" --pair "${pre}lon_pair" --cum-che-do lai --khi-tuong-duong $KhiTuongDuong @raGhep | Ghi
        if ($LASTEXITCODE -ne 0) { "Loi o buoc ghep." | Ghi; $ma = 2; exit 2 }
    }
}
finally {
    if ($bienKhoa) { [Environment]::SetEnvironmentVariable($bienKhoa, $null, "Process") }
    "=== KET THUC | ma thoat $ma | token cong don lan chay nay $([math]::Round($tongToken)) | log: $log ===" | Ghi
}
exit $ma
