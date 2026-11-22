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


def color_overlay(after, layers):
    image = Image.fromarray(after)
    result = image.copy()
    for mask, color in layers:
        tint = Image.blend(image, Image.new("RGB", image.size, color), 0.55)
        result.paste(tint, mask=Image.fromarray(mask))
    return result


def prediction_overlay(after, prediction):
    return color_overlay(after, [(prediction, PREDICTION_COLOR)])


def comparison_image(after, prediction, reference, scene):
    if after.shape[:2] != prediction.shape or prediction.shape != reference.shape:
        raise ValueError("Görüntü, tahmin ve referans boyutları eşleşmiyor.")
    regions = (prediction & reference, ~prediction & reference, prediction & ~reference)
    colors = [color for _, color in COMPARISON_COLORS]
    overlay = color_overlay(after, zip(regions, colors, strict=True))
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
