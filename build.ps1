<#
.SYNOPSIS
    ChangeScope'u PyInstaller onedir paketi olarak derler: dist\ChangeScope\ChangeScope.exe

.DESCRIPTION
    1. .venv Python'u ile PyInstaller'ı çalıştırır (packaging\changescope.spec).
    2. models\ ve bin\ klasörlerini exe'nin yanına kopyalar (pakete gömülmez).
    3. Paketin açılıp açılmadığını kısa bir duman testiyle doğrular (-SkipSmokeTest ile atlanır).

.PARAMETER SkipModels
    models\ klasörünü kopyalama (Gemma ~6.7 GB; sonra elle kopyalanabilir).

.PARAMETER Clean
    Önceki build\ ve dist\ChangeScope\ klasörlerini sil.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -SkipModels -Clean
#>
param(
    [switch]$SkipModels,
    [switch]$Clean,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Dist = Join-Path $Root "dist\ChangeScope"

if (-not (Test-Path $Python)) {
    throw ".venv bulunamadı: önce 'py -3.12 -m venv .venv' ve bağımlılık kurulumu yapın (BASLA.md)."
}

if ($Clean) {
    Write-Host "Temizleniyor: build\, dist\ChangeScope\"
    Remove-Item -Recurse -Force (Join-Path $Root "build") -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $Dist -ErrorAction SilentlyContinue
}

Write-Host "PyInstaller çalışıyor (birkaç dakika sürer)..."
& $Python -m PyInstaller --noconfirm --distpath (Join-Path $Root "dist") --workpath (Join-Path $Root "build") `
    (Join-Path $Root "packaging\changescope.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller başarısız (çıkış kodu $LASTEXITCODE)." }

function Copy-Folder([string]$Name) {
    $src = Join-Path $Root $Name
    $dst = Join-Path $Dist $Name
    if (-not (Test-Path $src)) { Write-Warning "$Name\ yok, atlanıyor."; return }
    Write-Host "Kopyalanıyor: $Name\ -> dist\ChangeScope\$Name\"
    # robocopy: yalnızca değişen dosyalar; .part ve indirme önbelleği hariç. 0-7 başarı kodlarıdır.
    robocopy $src $dst /E /NFL /NDL /NJH /NJS /NP /XF *.part .gitkeep /XD _downloads | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy $Name başarısız (kod $LASTEXITCODE)." }
    $global:LASTEXITCODE = 0
}

Copy-Folder "bin"
if ($SkipModels) {
    Write-Warning "models\ kopyalanmadı (-SkipModels). Çalıştırmadan önce dist\ChangeScope\models\ altına kopyalayın."
} else {
    Copy-Folder "models"
}

$Exe = Join-Path $Dist "ChangeScope.exe"
if (-not $SkipSmokeTest) {
    Write-Host "Duman testi: paket açılıyor mu?"
    # Pencereli exe'de çökme bir hata penceresi açıp süreci canlı tutar; bu yüzden yalnızca
    # "süreç yaşıyor mu" yetmez, günlükte bu açılışa ait Traceback olmamalı.
    $Log = Join-Path $env:LOCALAPPDATA "ChangeScope\logs\app.log"
    $logStart = if (Test-Path $Log) { (Get-Item $Log).Length } else { 0 }
    $p = Start-Process -FilePath $Exe -ArgumentList "--demo" -PassThru
    Start-Sleep -Seconds 25
    $newLog = ""
    if (Test-Path $Log) {
        $fs = [System.IO.File]::Open($Log, 'Open', 'Read', 'ReadWrite')
        try {
            [void]$fs.Seek($logStart, 'Begin')
            $newLog = (New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)).ReadToEnd()
        } finally { $fs.Close() }
    }
    $exited = $p.HasExited
    if (-not $exited) { Stop-Process -Id $p.Id -Force }
    if ($exited -or $newLog -match "Traceback") {
        Write-Host $newLog
        throw "Duman testi başarısız (çıktı yukarıda). Günlük: $Log"
    }
    Write-Host "Duman testi geçti (pencere açıldı, 25 sn çalıştı, günlükte hata yok)."

    # Gerçek pipeline (torch/LightGlue/DINOv2) paketlenmiş exe içinde çalışıyor mu?
    $Pair = Join-Path $Root "tests\data\real\piknik"
    if (Test-Path $Pair) {
        Write-Host "Öz test: gerçek pipeline piknik çiftinde..."
        $logStart = if (Test-Path $Log) { (Get-Item $Log).Length } else { 0 }
        $oldModels = $env:CHANGESCOPE_MODELS_DIR
        if ($SkipModels) { $env:CHANGESCOPE_MODELS_DIR = Join-Path $Root "models" }   # kopyalanmadıysa kaynaktan
        try {
            $st = Start-Process -FilePath $Exe -ArgumentList "--selftest", "`"$Pair`"" -PassThru -Wait
        } finally { $env:CHANGESCOPE_MODELS_DIR = $oldModels }
        $fs = [System.IO.File]::Open($Log, 'Open', 'Read', 'ReadWrite')
        try {
            [void]$fs.Seek($logStart, 'Begin')
            $stLog = (New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)).ReadToEnd()
        } finally { $fs.Close() }
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

$sizeGB = [math]::Round(((Get-ChildItem $Dist -Recurse -File | Measure-Object Length -Sum).Sum / 1GB), 2)
Write-Host ""
Write-Host "Hazır: $Exe  (toplam $sizeGB GB)"
Write-Host "Başka bir bilgisayara taşımak için dist\ChangeScope\ klasörünün tamamını kopyalayın."
