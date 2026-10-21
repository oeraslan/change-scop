# ChangeScope — Offline Görüntü Değişiklik Tespiti

Aynı bölgenin farklı zamanlarda (ör. 1 hafta arayla), farklı açılardan ve farklı kaynaklardan
(drone, İHA, uydu) çekilmiş PNG/JPG görüntülerini karşılaştırıp değişen bölgeleri işaretleyen
**tamamen offline** bir **Windows masaüstü** uygulaması. Kişisel kullanım içindir (ticari değil).

## Temel kurallar
- **Offline zorunlu.** Çalışma zamanında hiçbir ağ çağrısı yok. Tüm model ağırlıkları `models/` klasöründen yüklenir.
  İnternetten indirme yapan kod (`from_pretrained(...)`, `torch.hub.load` vb.) çalışma zamanında YASAK; sadece
  `scripts/download_models.py` geliştirme sırasında model indirebilir.
- **Python 3.12**, hedef platform **Windows 10/11 x64**. Yol işlemleri için her zaman `pathlib`.
  Sistemde varsayılan Python 3.14'tür; bu proje SADECE `.venv` (3.12) ile çalışır. Tüm komutlar
  `.venv\Scripts\python.exe -m ...` ile çalıştırılır, sistemdeki `python`/`pip` asla kullanılmaz.
- Modüller birbirine **sadece `core/contracts.py`** üzerinden bağlanır. Bir modül başka bir modülün iç
  fonksiyonlarını import etmez. Ayarlar için `core/config.py` (salt okunur) import edilebilir. Sözleşmede değişiklik gerekiyorsa değiştirme; gerekçesiyle raporla,
  karar ana oturumda verilir.
- Her public fonksiyon type hint'li ve docstring'li olur. Her modülün `tests/<modül>/` altında pytest testleri olur.
- Ağır işlemler (model yükleme, eşleştirme, LLM çağrısı) UI thread'inde ÇALIŞMAZ.

## Donanım
- Windows 11, **NVIDIA RTX 2060 Laptop, 6 GB VRAM**. PyTorch CUDA 12.6 build'i (`cu126`).
- **CUDA yoksa her şey CPU'da çalışabilmeli.** Cihaz seçimi `core/config.py` üzerinden
  (`"auto"` → CUDA varsa `cuda`, yoksa `cpu`).
- **LLM model seçimi** (config, `llm.model`):
  - `"gemma-4-12b"` (varsayılan, daha iyi): QAT Q4_0 ~7 GB → 6 GB VRAM'e tam sığmaz, **kısmi GPU offload**.
  - `"gemma-4-e4b"`: tamamen GPU'ya sığar, hızlı.
  - İkisinin de resmi QAT Q4_0 GGUF + mmproj dosyaları `scripts/download_models.py` ile indirilebilir.
- **`-ngl` (GPU katman sayısı):** config'de `llm.n_gpu_layers = "auto"` ise başlatma anındaki boş
  VRAM'e göre hesaplanır (model boyutu / katman sayısı, mmproj + KV cache + güvenlik payı düşülerek);
  config'de tam sayı verilirse o değer kullanılır. CUDA yoksa `-ngl 0`.
- **VRAM kuralı:** CV modelleri (SuperPoint / LightGlue / DINOv2) ile llama-server **aynı anda VRAM'de
  olmaz.** Pipeline önce tüm CV adımlarını yapar, CV modellerini serbest bırakır ve
  `torch.cuda.empty_cache()` ile VRAM'i boşaltır, ancak ondan sonra LLM adımına geçer.
- **llama-server lazy başlar:** uygulama açılışında DEĞİL, ilk LLM ihtiyacında başlatılır.

## Mimari (pipeline)
```
PNG/JPG yükleme → Ön işleme → Eşleştirme (LightGlue) → Hizalama (homografi/warp)
→ Değişiklik tespiti (SSIM + DINOv2) → Son işleme (maske, bbox) → Gemma 4 12B yorumu → UI
```

## Klasör yapısı
```
core/
  contracts.py        # Modüller arası veri sözleşmesi (TEK doğruluk kaynağı)
  config.py           # Tüm ayarlar (yollar, eşikler, cihaz, LLM model tablosu); modüller SADECE okur
  registration/       # image-registration ajanı
  detection/          # change-detector ajanı
  llm/                # llm-analyst ajanı
  pipeline.py         # Ana oturum: modülleri birleştiren orkestratör
app/
  ui/                 # desktop-ui ajanı (PySide6 + Fluent Widgets)
  storage/            # desktop-ui ajanı: klasör ağacı + görüntü kütüphanesi (SQLite)
  main.py
models/               # Offline model dosyaları (git'e girmez)
  lightglue/  superpoint/  dinov2/  gemma/
bin/llama/            # llama-server.exe ve DLL'leri (git'e girmez)
scripts/
  download_models.py  # Geliştirme sırasında modelleri + llama.cpp'yi indirir (--only, --list, devam ettirme)
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
| LLM | Gemma 4 12B, resmi QAT Q4_0 GGUF + mmproj, gömülü `llama-server.exe` (llama.cpp) üzerinden |
| Veri | SQLite (`sqlite3`, standart kütüphane) |
| Paketleme | PyInstaller **onedir** modu; `models/` ve `bin/` exe'nin yanında ayrı klasör |
| Test | pytest |

## Önemli tasarım kararları
1. **Görüntüler sohbet uygulamalarından PNG/JPG olarak gelir:** coğrafi koordinat yok, sıkıştırma artefaktı olabilir.
   Hizalama tamamen görseldir. Ön işlemede hafif denoise + artefakt bastırma yapılır; fark eşikleri
   sıkıştırma gürültüsüne toleranslı olmalıdır.
2. **Ölçek farkı:** Drone ve uydu görüntüleri farklı çözünürlükte olabilir. Eşleşmelerden ölçek oranı
   tahmin edilip görüntüler ortak ölçeğe getirilir.
3. **Büyük görüntüler** tile'lara bölünür (varsayılan 1024 px, %10 örtüşme).
4. **Örtüşmeyen alanlar maskelenir.** Warp sonrası iki görüntüde de bulunmayan pikseller asla "değişiklik" sayılmaz.
5. **LLM bulmaz, yorumlar.** Değişiklikleri CV pipeline bulur. Gemma 4 sadece her aday bölgenin önce/sonra
   crop'larına bakarak (a) değişikliği tanımlar, (b) gölge/ışık/sıkıştırma kaynaklı yanlış alarmları işaretler.
   LLM devre dışıyken de uygulama tam çalışmalıdır.
6. **llama-server yaşam döngüsü:** İlk LLM ihtiyacında (lazy) `bin/llama/llama-server.exe` alt süreç olarak başlatılır
   (`127.0.0.1`, boş bir port), `/health` hazır olana kadar beklenir, uygulama kapanınca süreç sonlandırılır.
   OpenAI uyumlu `/v1/chat/completions` endpoint'i kullanılır.

7. **Kullanıcı deneyimi:**
   - Görüntüler, kullanıcının oluşturduğu **iç içe klasörlerde** saklanır (ör. "X Bölgesi / Y Dağı / Kuzey Yamaç").
     Sınırsız alt klasör derinliği. Görüntüler uygulamanın kütüphanesine kopyalanır.
   - Kullanıcı galeriden **istediği iki görüntüyü** seçip karşılaştırır (farklı klasörlerden de olabilir).
   - **Karşılaştırma ekranı:** iki görüntü **yan yana** (sağdaki hizalanmış), senkron zoom/pan,
     **"Farkları göster" anahtarı** (kapalıyken hiçbir işaret görünmez), opsiyonel ısı haritası,
     **altta yapay zekâ yorumu paneli** (genel özet + bölge kartları).
   - Arayüz modern olmalı: varsayılan gri Qt görünümü kabul edilmez. Detaylar `desktop-ui` ajanında.

## Ajanlar (`.claude/agents/`)
| Ajan | Sahip olduğu klasör |
|---|---|
| image-registration | `core/registration/`, `tests/registration/` |
| change-detector | `core/detection/`, `tests/detection/` |
| llm-analyst | `core/llm/`, `tests/llm/` |
| desktop-ui | `app/` (ui + storage), `tests/ui/` |
| test-reviewer | Salt okunur inceleme + `tests/integration/` |

Ajanlar sadece kendi klasörlerinde yazar. `core/contracts.py` ve `core/pipeline.py` ana oturuma aittir.

## Yol haritası
- **Faz 1 (MVP):** Ön işleme, LightGlue + homografi, SSIM fark haritası; modern UI: klasör ağacı + galeri +
  iki görüntü seçme + yan yana karşılaştırma ekranı + "Farkları göster" anahtarı (AI paneli boş durumda)
- **Faz 2:** DINOv2 tabanlı fark, ignore maskeleri, ışık normalizasyonu, karşılaştırma geçmişi
- **Faz 3:** Gömülü Gemma 4 12B ile bölge yorumu + genel özet (alt panel dolar), PyInstaller paketi

## Komutlar (her zaman venv python'u)
- Testler: `.venv\Scripts\python.exe -m pytest -q`
- Uygulama: `.venv\Scripts\python.exe -m app.main`
- Modelleri indir (sadece geliştirmede): `.venv\Scripts\python.exe scripts/download_models.py`
- PyTorch kurulumu: `.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
