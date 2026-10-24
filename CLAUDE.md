# ChangeScope — Offline Görüntü Değişiklik Tespiti

Aynı bölgenin farklı zamanlarda (ör. 1 hafta arayla), farklı açılardan ve farklı kaynaklardan
(drone, İHA, uydu) çekilmiş PNG/JPG görüntülerini karşılaştırıp değişen bölgeleri işaretleyen
**tamamen offline** bir **Windows masaüstü** uygulaması. Kişisel kullanım içindir (ticari değil).

Bu dalda (`dev`) yerel yapay zekâ (Gemma / llama-server) **yoktur**: uygulama yalnızca görüntü karşılaştırır
ve görüntülerin **hizalama eşleşme yüzdesini** gösterir. Dağıtım **tek dosya exe** (CPU) olarak yapılır.

## Temel kurallar
- **Offline zorunlu.** Çalışma zamanında hiçbir ağ çağrısı yok. Tüm model ağırlıkları `models/` klasöründen yüklenir.
  İnternetten indirme yapan kod (`from_pretrained(...)`, `torch.hub.load` vb.) çalışma zamanında YASAK; sadece
  `scripts/download_models.py` geliştirme sırasında model indirebilir.
- **Python 3.12**, hedef platform **Windows 10/11 x64**. Yol işlemleri için her zaman `pathlib`.
  Sistemde varsayılan Python 3.14'tür; bu proje SADECE `.venv` (3.12) ile çalışır. Tüm komutlar
  `.venv\Scripts\python.exe -m ...` ile çalıştırılır, sistemdeki `python`/`pip` asla kullanılmaz.
  İstisna: tek exe derlemesi CPU PyTorch'lu `.venv-cpu` ile yapılır (`build.ps1` kurar ve kullanır).
- Modüller birbirine **sadece `core/contracts.py`** üzerinden bağlanır. Bir modül başka bir modülün iç
  fonksiyonlarını import etmez. Ayarlar için `core/config.py` (salt okunur) import edilebilir. Sözleşmede değişiklik gerekiyorsa değiştirme; gerekçesiyle raporla,
  karar ana oturumda verilir.
- Her public fonksiyon type hint'li ve docstring'li olur. Her modülün `tests/<modül>/` altında pytest testleri olur.
- Ağır işlemler (model yükleme, eşleştirme) UI thread'inde ÇALIŞMAZ.

## Donanım
- Geliştirme: Windows 11, **NVIDIA RTX 2060 Laptop, 6 GB VRAM**, PyTorch CUDA 12.6 build'i (`cu126`, `.venv`).
- **CUDA yoksa her şey CPU'da çalışabilmeli.** Cihaz seçimi `core/config.py` üzerinden
  (`"auto"` → CUDA varsa `cuda`, yoksa `cpu`). Dağıtılan tek exe CPU build'idir (CUDA PyTorch ~4 GB).
- CPU'da varsayılan fark yönteminde DINOv2 hız için kapalıdır (yalnızca SSIM); kullanıcı ayarlardan açıkça
  seçerse çalışır. SuperPoint NMS'i CPU'da `cv2.dilate` ile hızlandırılır (`core/registration/matching.py`).
- Karşılaştırma bitince CV modelleri bellekten bırakılır ve `torch.cuda.empty_cache()` ile VRAM boşaltılır.

## Mimari (pipeline)
```
PNG/JPG yükleme → Ön işleme → Eşleştirme (LightGlue) → Hizalama (homografi/warp)
→ Değişiklik tespiti (SSIM + DINOv2) → Son işleme (maske, bbox) → UI (eşleşme % + bölgeler)
```
**Eşleşme yüzdesi** = `RegistrationResult.match_percent` = homografiyle tutarlı (inlier) eşleşme / tüm eşleşme.

## Klasör yapısı
```
core/
  contracts.py        # Modüller arası veri sözleşmesi (TEK doğruluk kaynağı)
  config.py           # Tüm ayarlar (yollar, eşikler, cihaz); modüller SADECE okur
  registration/       # image-registration ajanı
  detection/          # change-detector ajanı
  pipeline.py         # Ana oturum: modülleri birleştiren orkestratör
app/
  ui/                 # desktop-ui ajanı (PySide6 + Fluent Widgets)
  storage/            # desktop-ui ajanı: klasör ağacı + görüntü kütüphanesi (SQLite)
  assets/             # Uygulama ikonu (.ico/.png) ve açılış ekranı (scripts/make_icon.py üretir)
  main.py
models/               # Offline model dosyaları (git'e girmez; tek exe'ye gömülür)
  lightglue/  superpoint/  dinov2/
packaging/            # PyInstaller tek dosya spec'i + launcher
scripts/
  download_models.py  # Geliştirme sırasında modelleri indirir (--only, --list, devam ettirme)
  make_icon.py        # app/assets/ ikon + splash üretir
  make_synthetic_pair.py  # tests/data/synthetic/ için önce/sonra çifti + doğruluk verisi üretir
tests/
  data/               # Örnek görüntü çiftleri
```

## Teknoloji yığını
| Alan | Seçim |
|---|---|
| UI | PySide6 (Qt 6) + PySide6-Fluent-Widgets (modern Windows 11 görünümü, koyu tema) |
| Görüntü işleme | OpenCV (`opencv-python`), scikit-image, NumPy |
| Eşleştirme | LightGlue + SuperPoint (PyTorch; kişisel kullanım olduğu için SuperPoint lisansı sorun değil) |
| Fark tespiti | SSIM (scikit-image) + DINOv2-small patch öznitelikleri (PyTorch) |
| Veri | SQLite (`sqlite3`, standart kütüphane) |
| Paketleme | PyInstaller **onefile**: `dist\ChangeScope.exe`, modeller içine gömülü, açılış ekranlı |
| Test | pytest |

## Önemli tasarım kararları
1. **Görüntüler sohbet uygulamalarından PNG/JPG olarak gelir:** coğrafi koordinat yok, sıkıştırma artefaktı olabilir.
   Hizalama tamamen görseldir. Ön işlemede hafif denoise + artefakt bastırma yapılır; fark eşikleri
   sıkıştırma gürültüsüne toleranslı olmalıdır.
2. **Ölçek farkı:** Drone ve uydu görüntüleri farklı çözünürlükte olabilir. Eşleşmelerden ölçek oranı
   tahmin edilip görüntüler ortak ölçeğe getirilir.
3. **Büyük görüntüler** tile'lara bölünür (varsayılan 1024 px, %10 örtüşme).
4. **Örtüşmeyen alanlar maskelenir.** Warp sonrası iki görüntüde de bulunmayan pikseller asla "değişiklik" sayılmaz.
5. **Tek dosya exe:** `core.config.app_root()` paketlenmişken `sys._MEIPASS` (exe'nin açıldığı geçici klasör)
   döndürür; `models/` oradan okunur. Kullanıcı verisi `%LOCALAPPDATA%/ChangeScope` altındadır.
6. **Kullanıcı deneyimi:**
   - Görüntüler, kullanıcının oluşturduğu **iç içe klasörlerde** saklanır (ör. "X Bölgesi / Y Dağı / Kuzey Yamaç").
     Sınırsız alt klasör derinliği. Görüntüler uygulamanın kütüphanesine kopyalanır.
   - Kullanıcı galeriden **istediği iki görüntüyü** seçip karşılaştırır (farklı klasörlerden de olabilir).
   - **Karşılaştırma ekranı:** iki görüntü **yan yana** (sağdaki hizalanmış), senkron zoom/pan,
     **"Farkları göster" anahtarı** (kapalıyken hiçbir işaret görünmez), opsiyonel ısı haritası,
     başlıkta **"Eşleşme %N"** rozeti, **altta sonuç paneli** (eşleşme halka göstergesi + istatistikler +
     bölge kartları).
   - Arayüz modern olmalı: varsayılan gri Qt görünümü kabul edilmez. Detaylar `desktop-ui` ajanında.

## Ajanlar (`.claude/agents/`)
| Ajan | Sahip olduğu klasör |
|---|---|
| image-registration | `core/registration/`, `tests/registration/` |
| change-detector | `core/detection/`, `tests/detection/` |
| desktop-ui | `app/` (ui + storage), `tests/ui/` |
| test-reviewer | Salt okunur inceleme + `tests/integration/` |

Ajanlar sadece kendi klasörlerinde yazar. `core/contracts.py` ve `core/pipeline.py` ana oturuma aittir.

## Komutlar (her zaman venv python'u)
- Testler: `.venv\Scripts\python.exe -m pytest -q` (~15 dk)
- Tek exe: `.\build.ps1` (→ `dist\ChangeScope.exe`; `.venv-cpu` yoksa kurar; `-Clean`, `-SkipSmokeTest`)
- Uygulama: `.venv\Scripts\python.exe -m app.main`
- Modelleri indir (sadece geliştirmede): `.venv\Scripts\python.exe scripts/download_models.py`
- İkonu yeniden üret: `.venv\Scripts\python.exe scripts/make_icon.py`
- PyTorch kurulumu: `.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
