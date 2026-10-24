"""Uygulama ikonunu üretir: `app/assets/changescope.ico` (16-256 px), `changescope.png` (512 px), `splash.png`.

Tasarım: koyu lacivert yuvarlatılmış kare zemin; arkada "önce" fotoğraf çerçevesi (çizgi), önde "sonra"
fotoğrafı (camgöbeği-yeşil manzara) ve üzerinde kırmızı değişiklik kutusu. Ağdan hiçbir şey indirmez.

Kullanım:
    .venv\\Scripts\\python.exe scripts/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "app" / "assets"
ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
S = 1024                                  # çizim çözünürlüğü (sonra küçültülür)

BG_TOP, BG_BOTTOM = (30, 41, 59), (10, 15, 28)
ACCENT = (34, 211, 238)
CHANGE = (244, 63, 94)


def _lerp(a: tuple[int, ...], b: tuple[int, ...], t: float) -> tuple[int, ...]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _vertical_gradient(size: tuple[int, int], top: tuple[int, ...], bottom: tuple[int, ...]) -> Image.Image:
    w, h = size
    grad = Image.new("RGBA", (1, h))
    for y in range(h):
        grad.putpixel((0, y), _lerp(top, bottom, y / max(1, h - 1)) + (255,))
    return grad.resize((w, h))


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def _landscape(size: tuple[int, int]) -> Image.Image:
    """Ön fotoğrafın içeriği: gökyüzü gradyanı + iki tepe + güneş."""
    w, h = size
    img = _vertical_gradient(size, (103, 232, 249), (14, 116, 144))
    d = ImageDraw.Draw(img)
    d.ellipse((w * 0.62, h * 0.14, w * 0.80, h * 0.14 + w * 0.18), fill=(254, 240, 138, 255))
    d.polygon([(0, h * 0.78), (w * 0.30, h * 0.40), (w * 0.58, h * 0.80), (0, h)], fill=(16, 185, 129, 255))
    d.polygon([(w * 0.30, h), (w * 0.68, h * 0.50), (w, h * 0.82), (w, h)], fill=(5, 150, 105, 255))
    d.rectangle((0, h * 0.90, w, h), fill=(4, 120, 87, 255))
    return img


def render(size: int = S) -> Image.Image:
    """İkonu `size` px kare RGBA olarak çizer."""
    k = size / S
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # Zemin: yuvarlatılmış kare + ince camgöbeği kenar parıltısı
    bg = _vertical_gradient((size, size), BG_TOP, BG_BOTTOM)
    icon.paste(bg, (0, 0), _rounded_mask((size, size), round(220 * k)))
    d = ImageDraw.Draw(icon)
    d.rounded_rectangle((round(6 * k), round(6 * k), size - round(6 * k), size - round(6 * k)),
                        radius=round(214 * k), outline=ACCENT + (70,), width=max(1, round(10 * k)))

    # Arka çerçeve ("önce"): sola-yukarı kaydırılmış, sadece kenar
    back = (round(150 * k), round(170 * k), round(690 * k), round(640 * k))
    d.rounded_rectangle(back, radius=round(56 * k), outline=ACCENT + (150,), width=max(2, round(30 * k)))

    # Ön fotoğraf ("sonra"): gölge + manzara + beyaz kenar
    fx0, fy0, fx1, fy1 = round(330 * k), round(370 * k), round(874 * k), round(844 * k)
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((fx0 + round(18 * k), fy0 + round(30 * k), fx1 + round(18 * k),
                                              fy1 + round(30 * k)), radius=round(60 * k), fill=(0, 0, 0, 150))
    icon.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(max(1, 28 * k))))
    photo = _landscape((fx1 - fx0, fy1 - fy0))
    icon.paste(photo, (fx0, fy0), _rounded_mask(photo.size, round(56 * k)))
    d = ImageDraw.Draw(icon)
    d.rounded_rectangle((fx0, fy0, fx1, fy1), radius=round(56 * k), outline=(241, 245, 249, 255),
                        width=max(2, round(22 * k)))

    # Değişiklik kutusu: yarı saydam kırmızı dolgu + kalın kenar + köşe rozeti
    cx0, cy0, cx1, cy1 = round(560 * k), round(520 * k), round(790 * k), round(740 * k)
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle((cx0, cy0, cx1, cy1), radius=round(26 * k), fill=CHANGE + (90,),
                         outline=CHANGE + (255,), width=max(2, round(26 * k)))
    r = round(44 * k)
    od.ellipse((cx1 - r, cy0 - r, cx1 + r, cy0 + r), fill=CHANGE + (255,))
    icon.alpha_composite(overlay)
    return icon


def _small(size: int) -> Image.Image:
    """Küçük boyutlar (≤32 px): ayrıntı azaltılmış, keskin çizim."""
    big = render(S)
    return big.resize((size, size), Image.Resampling.LANCZOS)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_splash(icon: Image.Image, width: int = 520, height: int = 300) -> Image.Image:
    """Tek dosya exe açılırken (paket geçici klasöre açılırken) gösterilen açılış ekranı."""
    img = _vertical_gradient((width, height), BG_TOP, BG_BOTTOM).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, width - 1, height - 1), outline=ACCENT, width=2)
    img.paste(icon.resize((128, 128), Image.Resampling.LANCZOS), (40, 70), icon.resize((128, 128)))
    d.text((192, 92), "ChangeScope", font=_font(40, bold=True), fill=(241, 245, 249))
    d.text((194, 146), "Görüntü değişiklik tespiti", font=_font(18), fill=(148, 163, 184))
    d.rectangle((194, 186, 194 + 140, 189), fill=ACCENT)
    return img


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master = render(S)
    master.resize((512, 512), Image.Resampling.LANCZOS).save(OUT_DIR / "changescope.png")
    frames = [master.resize((n, n), Image.Resampling.LANCZOS) if n > 32 else _small(n) for n in ICO_SIZES]
    frames[-1].save(OUT_DIR / "changescope.ico", format="ICO", sizes=[(n, n) for n in ICO_SIZES],
                    append_images=frames[:-1])
    render_splash(master).save(OUT_DIR / "splash.png")
    print(f"yazıldı: {OUT_DIR / 'changescope.ico'}, changescope.png, splash.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
