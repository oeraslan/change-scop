<#
.SYNOPSIS
    ChangeScope'u TEK DOSYA exe olarak derler: dist\ChangeScope.exe

.DESCRIPTION
    1. CPU PyTorch'lu derleme ortamını (.venv-cpu) yoksa kurar. GPU (CUDA) PyTorch ~4 GB olduğu için tek exe
       CPU sürümüyle derlenir; exe her Windows 10/11 x64 bilgisayarda GPU gerektirmeden çalışır.
    2. İkon ve açılış ekranını üretir (scripts\make_icon.py).
    3. PyInstaller'ı çalıştırır (packaging\changescope.spec). models\ (SuperPoint, LightGlue, DINOv2) exe'nin
       içine gömülür.
    4. Paketlenmiş exe'yi doğrular: pencere açılıyor mu (duman testi) ve gerçek pipeline LightGlue ile
       çalışıyor mu (öz test). -SkipSmokeTest ile atlanır.

.PARAMETER Clean
    Önceki build\ klasörünü ve dist\ChangeScope.exe'yi sil.

.PARAMETER SkipSmokeTest
    Derleme sonrası doğrulamaları atla.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -Clean
#>
param(
    [switch]$Clean,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv-cpu"
$Python = Join-Path $Venv "Scripts\python.exe"
$Exe = Join-Path $Root "dist\ChangeScope.exe"

# Sürümler geliştirme ortamıyla (.venv) aynı tutulur; torch/torchvision CPU build'i.
$TorchIndex = "https://download.pytorch.org/whl/cpu"
$TorchPkgs = @("torch==2.14.1", "torchvision==0.29.1")
$Pkgs = @(
    "numpy==2.5.2", "opencv-python==5.0.0.93", "scikit-image==0.26.0", "pillow==12.3.0", "kornia==0.8.3",
    "lightglue @ git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8",
    "PySide6-Essentials==6.12.0", "PySide6-Fluent-Widgets==1.11.3",
    "pyinstaller==6.22.3", "pyinstaller-hooks-contrib==2026.9"
)

function Invoke-Checked([string]$What, [scriptblock]$Block) {
    # Yerel araçlar (pip, PyInstaller) bilgi satırlarını stderr'e yazar; PS 5.1 bunları çıktı yönlendirilince
    # hata sayar. Başarı yalnızca çıkış koduyla belirlenir.
    $ErrorActionPreference = "Continue"
    & $Block 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$What başarısız (çıkış kodu $LASTEXITCODE)." }
}

# --- 1. Derleme ortamı -------------------------------------------------------------------------------
if (-not (Test-Path $Python)) {
    Write-Host "CPU derleme ortamı kuruluyor (.venv-cpu, bir kez; ~1 GB indirme)..."
    Invoke-Checked "venv oluşturma" { py -3.12 -m venv $Venv }
    Invoke-Checked "pip güncelleme" { & $Python -m pip install -q --upgrade pip }
    Invoke-Checked "PyTorch (CPU) kurulumu" { & $Python -m pip install -q @TorchPkgs --index-url $TorchIndex }
    Invoke-Checked "bağımlılık kurulumu" { & $Python -m pip install -q @Pkgs }
}
$cuda = & $Python -c "import torch; print(torch.version.cuda or '')"
if ($cuda) { throw ".venv-cpu içindeki PyTorch CUDA build'i ($cuda); tek exe için CPU build'i gerekli." }

foreach ($m in @("superpoint\superpoint_v1.pth", "lightglue\superpoint_lightglue.pth", "dinov2\dinov2_vits14_pretrain.pth")) {
    if (-not (Test-Path (Join-Path $Root "models\$m"))) {
        throw "models\$m yok: önce '.venv\Scripts\python.exe scripts\download_models.py' çalıştırın."
    }
}

if ($Clean) {
    Write-Host "Temizleniyor: build\, dist\ChangeScope.exe"
    Remove-Item -Recurse -Force (Join-Path $Root "build") -ErrorAction SilentlyContinue
    Remove-Item -Force $Exe -ErrorAction SilentlyContinue
}

# --- 2. İkon + açılış ekranı --------------------------------------------------------------------------
Invoke-Checked "ikon üretimi" { & $Python (Join-Path $Root "scripts\make_icon.py") }

# --- 3. PyInstaller -----------------------------------------------------------------------------------
Write-Host "PyInstaller çalışıyor (birkaç dakika sürer)..."
Invoke-Checked "PyInstaller" {
    & $Python -m PyInstaller --noconfirm --distpath (Join-Path $Root "dist") --workpath (Join-Path $Root "build") `
        (Join-Path $Root "packaging\changescope.spec")
}

# --- 4. Doğrulama -------------------------------------------------------------------------------------
function Read-LogFrom([string]$Log, [long]$Start) {
    if (-not (Test-Path $Log)) { return "" }
    $fs = [System.IO.File]::Open($Log, 'Open', 'Read', 'ReadWrite')
    try {
        [void]$fs.Seek($Start, 'Begin')
        return (New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)).ReadToEnd()
    } finally { $fs.Close() }
}

if (-not $SkipSmokeTest) {
    $Log = Join-Path $env:LOCALAPPDATA "ChangeScope\logs\app.log"

    Write-Host "Duman testi: paket açılıyor mu? (tek exe ilk açılışta kendini geçici klasöre açar)"
    # Pencereli exe'de çökme bir hata penceresi açıp süreci canlı tutar; bu yüzden yalnızca
    # "süreç yaşıyor mu" yetmez, günlükte bu açılışa ait Traceback olmamalı.
    $logStart = if (Test-Path $Log) { (Get-Item $Log).Length } else { 0 }
    $p = Start-Process -FilePath $Exe -ArgumentList "--demo" -PassThru
    Start-Sleep -Seconds 40
    $newLog = Read-LogFrom $Log $logStart
    $exited = $p.HasExited
    if (-not $exited) { Stop-Process -Id $p.Id -Force }
    if ($exited -or $newLog -match "Traceback") {
        Write-Host $newLog
        throw "Duman testi başarısız (çıktı yukarıda). Günlük: $Log"
    }
    Write-Host "Duman testi geçti (pencere açıldı, 40 sn çalıştı, günlükte hata yok)."

    # Gerçek pipeline (torch/LightGlue) paketlenmiş exe içinde, gömülü modellerle çalışıyor mu?
    $Pair = Join-Path $Root "tests\data\real\piknik"
    if (Test-Path $Pair) {
        Write-Host "Öz test: gerçek pipeline piknik çiftinde..."
        $logStart = if (Test-Path $Log) { (Get-Item $Log).Length } else { 0 }
        $st = Start-Process -FilePath $Exe -ArgumentList "--selftest", "`"$Pair`"" -PassThru -Wait
        $stLog = Read-LogFrom $Log $logStart
        $line = ($stLog -split "`n" | Where-Object { $_ -match "^SELFTEST" }) -join ""
        if ($st.ExitCode -ne 0) {
            Write-Host $stLog
            throw "Öz test başarısız (çıkış kodu $($st.ExitCode))."
        }
        if ($line -match "LightGlue yüklenemedi|geri dönüşü kullanıldı") {
            Write-Host $line
            throw "Öz test: paketlenmiş exe'de LightGlue yüklenemedi (SIFT'e düşüldü). Eksik ikili dosya olabilir."
        }
        Write-Host "Öz test geçti: $line"
    }
}

$sizeMB = [math]::Round((Get-Item $Exe).Length / 1MB)
Write-Host ""
Write-Host "Hazır: $Exe  ($sizeMB MB, tek dosya)"
Write-Host "Başka bir bilgisayara yalnızca bu exe'yi kopyalamanız yeterli."
