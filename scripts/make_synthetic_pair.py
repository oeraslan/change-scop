"""Testler için sentetik "önce/sonra" hava görüntüsü çifti üretir (internet gerekmez).

Üretilen sahne: arazi dokusu, tarlalar, yollar, binalar ve ağaçlar. "Sonra" görüntüsünde:
    1. yeni nesneler eklenir (bina + araç) → gerçek değişiklik,
    2. hafif parlaklık/kontrast farkı uygulanır → değişiklik SAYILMAMALI,
    3. bilinen bir homografi ile döndürülür + ölçeklenir + kaydırılır (farklı açı/çözünürlük),
    4. JPEG ile sıkıştırılır (sohbet uygulaması artefaktı).

Çıktı (`--out`, varsayılan `tests/data/synthetic/<ad>/`):
    before.png          referans görüntü
    after.jpg           dönüştürülmüş + değişmiş + sıkıştırılmış görüntü
    truth_mask.png      before koordinatlarında eklenen nesnelerin maskesi (255 = değişiklik)
    truth.json          homografi (after → before), eklenen nesnelerin bbox'ları, parametreler

Kullanım:
    .venv\\Scripts\\python.exe scripts/make_synthetic_pair.py
    .venv\\Scripts\\python.exe scripts/make_synthetic_pair.py --seed 7 --rotate 12 --scale 0.6 --jpeg 60
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "tests" / "data" / "synthetic"


@dataclass
class PairParams:
    """Sentetik çift parametreleri (truth.json'a yazılır)."""

    seed: int = 42
    width: int = 1600
    height: int = 1200
    rotate_deg: float = 8.0
    scale: float = 0.75             # after çözünürlüğü / before çözünürlüğü
    shift_px: tuple[float, float] = (40.0, -25.0)
    perspective: float = 0.00004    # küçük perspektif bozulması (eğik çekim)
    brightness: float = 18.0        # after'a eklenen parlaklık (değişiklik sayılmamalı)
    contrast: float = 1.08
    jpeg_quality: int = 70
    n_new_objects: int = 2


@dataclass
class SyntheticPair:
    """Bellekte üretilmiş çift (testler doğrudan kullanabilir). Görüntüler RGB uint8."""

    before: np.ndarray
    after: np.ndarray
    truth_mask: np.ndarray                      # before boyutunda uint8 (0/255)
    homography_after_to_before: np.ndarray      # 3x3
    added_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)
    params: PairParams = field(default_factory=PairParams)


# ---------------------------------------------------------------------------
# Sahne üretimi
# ---------------------------------------------------------------------------

def _terrain(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Çok ölçekli gürültüden toprak/çim dokusu."""
    base = np.zeros((h, w), np.float32)
    for cells, amp in ((4, 1.0), (16, 0.5), (64, 0.25), (256, 0.12)):
        small = rng.random((max(2, cells * h // w), cells)).astype(np.float32)
        base += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    green = np.array([70, 110, 55], np.float32)
    soil = np.array([150, 125, 90], np.float32)
    img = soil * (1 - base[..., None]) + green * base[..., None]
    img += rng.normal(0, 6, img.shape)
    return img


def _draw_scene(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    img = _terrain(rng, h, w)

    # Tarlalar (dönük dikdörtgenler)
    for _ in range(14):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sw, sh = rng.uniform(120, 380), rng.uniform(90, 300)
        box = cv2.boxPoints(((cx, cy), (sw, sh), rng.uniform(0, 90))).astype(np.int32)
        color = rng.choice([[110, 140, 60], [180, 160, 100], [90, 120, 50], [160, 150, 70]])
        cv2.fillPoly(img, [box], color.tolist())
    img += rng.normal(0, 4, img.shape)

    # Yollar
    for _ in range(4):
        pts = np.cumsum(rng.normal(0, 60, (8, 2)), axis=0) + [rng.uniform(0, w), rng.uniform(0, h)]
        pts = np.vstack([[rng.uniform(0, w), 0], pts, [rng.uniform(0, w), h]]).astype(np.int32)
        cv2.polylines(img, [pts], False, (95, 95, 95), int(rng.uniform(10, 22)), cv2.LINE_AA)
        cv2.polylines(img, [pts], False, (200, 200, 190), 1, cv2.LINE_AA)

    # Binalar (çatı + gölge) ve ağaçlar
    for _ in range(40):
        _building(img, rng, rng.uniform(30, w - 60), rng.uniform(30, h - 60))
    for _ in range(160):
        c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        r = int(rng.uniform(5, 14))
        cv2.circle(img, (c[0] + 3, c[1] + 3), r, (40, 50, 35), -1, cv2.LINE_AA)
        cv2.circle(img, c, r, (35, int(rng.uniform(80, 120)), 40), -1, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8)


def _building(img: np.ndarray, rng: np.random.Generator, x: float, y: float,
              size: tuple[float, float] | None = None) -> np.ndarray:
    """Gölgeli bir bina çizer ve kapladığı alanın köşe noktalarını döndürür."""
    bw, bh = size or (rng.uniform(18, 45), rng.uniform(18, 45))
    ang = rng.uniform(0, 90)
    box = cv2.boxPoints(((x, y), (bw, bh), ang)).astype(np.int32)
    cv2.fillPoly(img, [box + [5, 5]], (45, 45, 45))
    roof = rng.choice([[170, 70, 55], [190, 190, 185], [120, 120, 130], [150, 90, 60]])
    cv2.fillPoly(img, [box], roof.tolist())
    cv2.polylines(img, [box], True, (60, 60, 60), 1, cv2.LINE_AA)
    return np.vstack([box, box + [5, 5]])


def _random_homography(p: PairParams, rng: np.random.Generator) -> np.ndarray:
    """before → after homografisi: merkez etrafında dönme + ölçek + kaydırma + hafif perspektif."""
    w, h = p.width, p.height
    c = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], np.float64)
    a = np.deg2rad(p.rotate_deg)
    rs = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) * \
        np.array([[p.scale], [p.scale], [1]])
    persp = np.array([[1, 0, 0], [0, 1, 0], [p.perspective, -p.perspective * 0.5, 1]], np.float64)
    out_w, out_h = int(w * p.scale), int(h * p.scale)
    back = np.array([[1, 0, out_w / 2 + p.shift_px[0]], [0, 1, out_h / 2 + p.shift_px[1]], [0, 0, 1]])
    H = back @ rs @ persp @ c
    return H / H[2, 2]


def make_pair(params: PairParams | None = None) -> SyntheticPair:
    """Sentetik önce/sonra çifti üretir (deterministik: aynı seed → aynı çıktı)."""
    p = params or PairParams()
    rng = np.random.default_rng(p.seed)
    h, w = p.height, p.width

    before = _draw_scene(rng, h, w)
    changed = before.copy()
    truth = np.zeros((h, w), np.uint8)
    bboxes: list[tuple[int, int, int, int]] = []

    # Yeni nesneler: büyük bir bina ve bir araç grubu (kenarlardan uzak, örtüşme alanında)
    margin_x, margin_y = int(w * 0.25), int(h * 0.25)
    for i in range(p.n_new_objects):
        x = rng.uniform(margin_x, w - margin_x)
        y = rng.uniform(margin_y, h - margin_y)
        if i % 2 == 0:
            pts = _building(changed, rng, x, y, size=(rng.uniform(70, 110), rng.uniform(50, 90)))
        else:
            pts_list = []
            for k in range(4):
                vx, vy = x + k * 16, y + rng.uniform(-3, 3)
                box = cv2.boxPoints(((vx, vy), (12, 24), 0)).astype(np.int32)
                cv2.fillPoly(changed, [box], rng.choice([[220, 30, 30], [240, 240, 240], [30, 60, 200]]).tolist())
                pts_list.append(box)
            pts = np.vstack(pts_list)
        m = np.zeros_like(truth)
        cv2.fillPoly(m, [cv2.convexHull(pts.astype(np.int32))], 255)
        truth |= m
        bx, by, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        bboxes.append((int(bx), int(by), int(bw), int(bh)))

    # Işık farkı (gerçek değişiklik değil)
    lit = np.clip(changed.astype(np.float32) * p.contrast + p.brightness, 0, 255).astype(np.uint8)

    H = _random_homography(p, rng)                       # before → after
    out_size = (int(w * p.scale), int(h * p.scale))
    after = cv2.warpPerspective(lit, H, out_size, flags=cv2.INTER_AREA, borderValue=(0, 0, 0))

    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(after, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
    assert ok
    after = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    H_inv = np.linalg.inv(H)
    return SyntheticPair(
        before=before, after=after, truth_mask=truth,
        homography_after_to_before=H_inv / H_inv[2, 2], added_bboxes=bboxes, params=p,
    )


def save_pair(pair: SyntheticPair, out_dir: Path) -> None:
    """Çifti diske yazar (`before.png`, `after.jpg`, `truth_mask.png`, `truth.json`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "before.png"), cv2.cvtColor(pair.before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "after.jpg"), cv2.cvtColor(pair.after, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 100])  # zaten sıkıştırılmış; tekrar bozma
    cv2.imwrite(str(out_dir / "truth_mask.png"), pair.truth_mask)
    truth = {
        "homography_after_to_before": pair.homography_after_to_before.tolist(),
        "added_bboxes_xywh": pair.added_bboxes,
        "before_size_wh": [pair.before.shape[1], pair.before.shape[0]],
        "after_size_wh": [pair.after.shape[1], pair.after.shape[0]],
        "params": asdict(pair.params),
    }
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="Çıktı klasörü")
    parser.add_argument("--name", default="default", help="Alt klasör adı (varsayılan: default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotate", type=float, default=8.0, help="Dönme açısı (derece)")
    parser.add_argument("--scale", type=float, default=0.75, help="after/before ölçek oranı")
    parser.add_argument("--jpeg", type=int, default=70, help="JPEG kalitesi")
    parser.add_argument("--brightness", type=float, default=18.0)
    args = parser.parse_args(argv)

    params = PairParams(seed=args.seed, rotate_deg=args.rotate, scale=args.scale,
                        jpeg_quality=args.jpeg, brightness=args.brightness)
    out = args.out or DEFAULT_OUT / args.name
    pair = make_pair(params)
    save_pair(pair, out)
    print(f"Yazıldı: {out}  (before {pair.before.shape[1]}x{pair.before.shape[0]}, "
          f"after {pair.after.shape[1]}x{pair.after.shape[0]}, {len(pair.added_bboxes)} yeni nesne)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
