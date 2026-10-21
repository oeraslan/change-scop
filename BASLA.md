# Başlangıç Rehberi

## 1. Kurulum (bir kez)
```powershell
cd change-detect
git init
py -3.12 -m venv .venv
.venv\Scripts\activate
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install -r requirements.txt
claude
```

## 2. Claude Code'a sırayla vereceğin prompt'lar

Her fazı bitirip test-reviewer raporunu gördükten sonra bir sonrakine geç.

### Faz 0 — Temel (ana oturum)
```
CLAUDE.md ve core/contracts.py'yi oku. Şunları sen yaz (ajan kullanma):
1. scripts/download_models.py: SuperPoint ve LightGlue ağırlıklarını models/superpoint ve
   models/lightglue'ya, DINOv2-small'u models/dinov2'ye, Gemma 4 12B resmi QAT Q4_0 GGUF ve
   mmproj dosyasını models/gemma'ya indirsin. Her model için ayrı flag olsun (--only gemma gibi).
   Ayrıca llama.cpp'nin güncel Windows CUDA release'ini indirip bin/llama/'ya açsın.
2. core/config.py: model yolları, eşikler, tile boyutu gibi tüm ayarlar tek yerde.
3. core/pipeline.py: run_comparison(before, after, progress, use_llm) -> ComparisonResult.
   Henüz modüller yok; Protocol'leri kullanarak bağımlılık enjeksiyonuyla yaz.
4. tests/data/ için sentetik görüntü çifti üreten bir script (döndürme + ölçek + eklenen nesne + JPEG).
Bitince ne yaptığını özetle.
```

### Faz 1 — MVP (paralel ajanlar)
```
Şu üç ajanı paralel çalıştır, her biri kendi worktree'sinde:
- image-registration: ön işleme + SuperPoint/LightGlue + homografi (SIFT fallback dahil)
- change-detector: sadece SSIM yöntemi + ışık normalizasyonu + son işleme
- desktop-ui: modern Fluent arayüz; app/storage (iç içe klasörler + görüntü kütüphanesi, SQLite),
  sol klasör ağacı, galeri, iki görüntü seçip karşılaştırma, yan yana senkron zoom'lu karşılaştırma
  ekranı, "Farkları göster" anahtarı, alttaki AI yorum paneli (şimdilik boş durum + sahte veriyle test)
Hepsi bitince worktree'leri birleştir, core/pipeline.py'de gerçek modülleri bağla,
sonra test-reviewer ajanını çalıştır ve raporunu bana göster.
Ekran görüntülerini de göster; arayüz sıkıcı/varsayılan Qt gibi görünüyorsa desktop-ui ile düzelt.
```

### Faz 2 — Doğruluk
```
change-detector ajanıyla DINOv2 yöntemini ve "ssim+dinov2" birleştirmesini ekle.
desktop-ui ajanıyla ignore maskesi çizme aracını (fırça ile alan boyama) ve
"Geçmiş karşılaştırmalar" sekmesini ekle. Sonra test-reviewer'ı çalıştır.
```

### Faz 3 — Gemma ve paketleme
```
llm-analyst ajanıyla gömülü llama-server yönetimini, bölge analizini ve genel özeti (summarize) yaz.
desktop-ui ajanıyla alttaki AI yorum panelini gerçek verilere bağla (özet + bölge kartları,
yükleme animasyonu, "Yorumla" butonu) ve durum çubuğuna Gemma göstergesini ekle.
Sonra sen PyInstaller onedir paketini hazırla (models/ ve bin/ exe'nin yanına kopyalansın)
ve build.ps1 scripti yaz. En son test-reviewer'ı çalıştır.
```

## 3. Faydalı ipuçları
- Bir ajanı açıkça çağırmak için: `@"change-detector (agent)" eşikleri biraz düşür`
- Yanlış alarm çoksa: tests/data/ altına kendi gerçek görüntü çiftlerini koy ve
  "test-reviewer ile gerçek verideki yanlış alarmları analiz et" de.
- Görüntüleri sohbet uygulamasından "dosya/belge" olarak al, "fotoğraf" olarak değil
  (sıkıştırmayı önler).
