"""Render the TinyCD demo masks and comparison PNGs."""

from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PREDICTION_COLOR = (56, 189, 248)
COMPARISON_COLORS = (
    ("Doğru yakalanan", (34, 197, 94)),
    ("Kaçırılan", (168, 85, 247)),
    ("Fazladan işaretlenen", (249, 115, 22)),
)
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def png_bytes(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def mask_png(mask):
    return png_bytes(Image.fromarray(mask.astype(np.uint8) * 255))


def color_overlay(after, layers, alpha=0.55):
    image = Image.fromarray(after)
    result = image.copy()
    for mask, color in layers:
        tint = Image.blend(image, Image.new("RGB", image.size, color), alpha)
        result.paste(tint, mask=Image.fromarray(mask))
    return result


def prediction_overlay(after, prediction, alpha=0.55):
    return color_overlay(after, [(prediction, PREDICTION_COLOR)], alpha=alpha)


def error_overlay(after, prediction, reference, alpha=0.55):
    if after.shape[:2] != prediction.shape or prediction.shape != reference.shape:
        raise ValueError("Görüntü, tahmin ve referans boyutları eşleşmiyor.")
    regions = (prediction & reference, ~prediction & reference, prediction & ~reference)
    colors = [color for _, color in COMPARISON_COLORS]
    return color_overlay(after, zip(regions, colors, strict=True), alpha=alpha)


def comparison_image(after, prediction, reference, scene, alpha=0.55):
    overlay = error_overlay(after, prediction, reference, alpha=alpha)
    width, height = overlay.size
    canvas = Image.new("RGB", (width, height + 110), (15, 23, 42))
    canvas.paste(overlay, (0, 0))
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.truetype(str(FONT), 27)
    legend = ImageFont.truetype(str(FONT), 21)
    draw.text((24, height + 14), f"{scene} · Bina değişimi", font=heading, fill="white")
    for x, (label, color) in zip((24, 350, 655), COMPARISON_COLORS, strict=True):
        draw.rounded_rectangle((x, height + 65, x + 21, height + 86), radius=4, fill=color)
        draw.text((x + 32, height + 61), label, font=legend, fill=(226, 232, 240))
    return canvas


def probability_heatmap(after, probabilities, threshold=0.5, alpha=0.75):
    if after.shape[:2] != probabilities.shape:
        raise ValueError("Görüntü ve olasılık haritası boyutları eşleşmiyor.")
    probs = np.clip(probabilities, 0.0, 1.0)
    cmap = matplotlib.colormaps["inferno"]
    heat_rgb = (cmap(probs)[..., :3] * 255.0).astype(np.float32)
    base = after.astype(np.float32)
    # Keep near-zero background crisp while highlighting uncertain and changed regions.
    blend = np.clip((probs - 0.05) / 0.45, 0.0, 1.0)[..., None] * float(alpha)
    blended = np.rint(base * (1.0 - blend) + heat_rgb * blend).clip(0, 255).astype(np.uint8)
    overlay = Image.fromarray(blended)

    width, height = overlay.size
    bar_height = 104
    canvas = Image.new("RGB", (width, height + bar_height), (15, 23, 42))
    canvas.paste(overlay, (0, 0))
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.truetype(str(FONT), 24)
    label_font = ImageFont.truetype(str(FONT), 19)
    threshold_text = f"{threshold:.2f}".replace(".", ",")
    draw.text((24, height + 12), f"Değişim Olasılığı (p) · Eşik {threshold_text}", font=heading, fill="white")

    bar_x0, bar_x1 = 24, width - 24
    bar_y0, bar_y1 = height + 50, height + 72
    gradient = np.linspace(0.0, 1.0, bar_x1 - bar_x0, dtype=np.float32)
    grad_rgb = (cmap(gradient)[:, :3] * 255.0).astype(np.uint8)[None, :, :]
    grad_img = Image.fromarray(np.repeat(grad_rgb, bar_y1 - bar_y0, axis=0))
    canvas.paste(grad_img, (bar_x0, bar_y0))

    marker_x = int(round(bar_x0 + np.clip(threshold, 0.0, 1.0) * (bar_x1 - bar_x0)))
    draw.line((marker_x, bar_y0 - 4, marker_x, bar_y1 + 4), fill=(56, 189, 248), width=3)
    draw.text((bar_x0, bar_y1 + 6), "0,00", font=label_font, fill=(203, 213, 225))
    draw.text((max(bar_x0 + 60, min(bar_x1 - 110, marker_x - 24)), bar_y1 + 6),
              f"eşik {threshold_text}", font=label_font, fill=(56, 189, 248))
    draw.text((bar_x1 - 46, bar_y1 + 6), "1,00", font=label_font, fill=(203, 213, 225))
    return canvas
