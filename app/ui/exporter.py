"""Karşılaştırma sonucunu dışa aktarır: kutulu yan yana PNG + bölge listesi CSV + özet metni."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.ui.theme import VERDICT_COLORS, VERDICT_LABELS
from core.contracts import ChangeVerdict, ComparisonResult


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("segoeuib.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_side_by_side(
    result: ComparisonResult, show_boxes: bool = True, hide_false_alarms: bool = False, gap: int = 16
) -> Image.Image:
    """Önce | Sonra (hizalanmış) yan yana görüntüsü; `show_boxes` ise kutular ve numaralar çizilir."""
    reg = result.registration
    panes = [Image.fromarray(reg.before.pixels).convert("RGBA"), Image.fromarray(reg.after_warped).convert("RGBA")]
    if show_boxes and result.detection is not None:
        h, w = reg.before.pixels.shape[:2]
        line = max(2, round(max(h, w) / 500))
        font = _font(max(14, round(max(h, w) / 55)))
        for pane in panes:
            overlay = Image.new("RGBA", pane.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for r in result.detection.regions:
                if hide_false_alarms and r.verdict is ChangeVerdict.FALSE_ALARM:
                    continue
                color = VERDICT_COLORS[r.verdict].lstrip("#")
                rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
                b = r.bbox
                box = (b.x, b.y, b.x + b.width, b.y + b.height)
                draw.rectangle(box, fill=rgb + (52,), outline=rgb + (255,), width=line)
                label = str(r.region_id)
                tw = draw.textlength(label, font=font)
                th = font.size if hasattr(font, "size") else 14
                draw.rounded_rectangle(
                    (b.x, b.y - th - 8, b.x + tw + 12, b.y), radius=4, fill=rgb + (255,))
                draw.text((b.x + 6, b.y - th - 5), label, fill=(255, 255, 255, 255), font=font)
            pane.alpha_composite(overlay)
    ph, pw = panes[0].height, panes[0].width
    canvas = Image.new("RGB", (pw * 2 + gap, ph), (20, 22, 27))
    canvas.paste(panes[0].convert("RGB"), (0, 0))
    canvas.paste(panes[1].resize((pw, ph)).convert("RGB"), (pw + gap, 0))
    return canvas


def export_comparison(
    result: ComparisonResult, png_path: Path, show_boxes: bool = True, hide_false_alarms: bool = False,
    title: str = "",
) -> list[Path]:
    """PNG + `<ad>_bolgeler.csv` + `<ad>_ozet.txt` dosyalarını yazar ve yollarını döndürür."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    render_side_by_side(result, show_boxes, hide_false_alarms).save(png_path, "PNG")

    csv_path = png_path.with_name(png_path.stem + "_bolgeler.csv")
    regions = result.detection.regions if result.detection else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["No", "X", "Y", "Genişlik", "Yükseklik", "Alan_px", "Skor", "Karar",
                         "Güven", "Açıklama"])
        for r in regions:
            writer.writerow([
                r.region_id, r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height, r.area_px,
                f"{r.score:.3f}", VERDICT_LABELS[r.verdict],
                "" if r.llm_confidence is None else f"{r.llm_confidence:.2f}", r.description or "",
            ])

    txt_path = png_path.with_name(png_path.stem + "_ozet.txt")
    lines = [title or "ChangeScope karşılaştırma özeti", "=" * 40,
             f"Hizalama: {result.registration.quality.value} "
             f"({result.registration.num_inliers}/{result.registration.num_matches} eşleşme)",
             f"Bölge sayısı: {len(regions)}", ""]
    lines.append(result.summary or "Yapay zekâ özeti yok.")
    for r in regions:
        lines.append(f"  #{r.region_id} [{VERDICT_LABELS[r.verdict]}] {r.description or ''}".rstrip())
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [png_path, csv_path, txt_path]

