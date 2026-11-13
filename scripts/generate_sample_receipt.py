"""
Synthetic Receipt Generator (Application Testing & Portfolio Demo Fixture Only)
------------------------------------------------------------------------------
Generates a clean, non-sensitive synthetic receipt PNG for browser E2E testing
and reproducible portfolio demonstrations. Never used as ML benchmark evidence.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def generate_synthetic_receipt(
    output_path: Path,
    vendor: str = "SUNRISE ARTISAN BAKERY",
    address: str = "742 Evergreen Terrace, Austin, TX 78701",
    date_str: str = "2026-10-18",
    subtotal: str = "24.00",
    tax: str = "1.92",
    total: str = "25.92",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (760, 1020), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 30)
        body_font = ImageFont.truetype("arial.ttf", 22)
        mono_font = ImageFont.truetype("cour.ttf", 21)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        mono_font = ImageFont.load_default()

    # Crisp receipt border
    draw.rectangle([(28, 28), (732, 992)], outline=(215, 220, 228), width=2)

    y = 70
    draw.text((70, y), vendor, fill=(20, 25, 35), font=title_font)
    y += 50
    draw.text((70, y), address, fill=(70, 80, 95), font=body_font)
    y += 45
    draw.line([(60, y), (700, y)], fill=(180, 185, 195), width=2)
    y += 35

    draw.text((70, y), f"RECEIPT #: DM-2026-0418", fill=(35, 45, 58), font=mono_font)
    y += 34
    draw.text((70, y), f"DATE: {date_str}", fill=(35, 45, 58), font=mono_font)
    y += 45
    draw.line([(60, y), (700, y)], fill=(210, 215, 222), width=1)
    y += 30

    items = [
        ("Sourdough Loaf x2", "USD 12.00"),
        ("Espresso Roast 500g", "USD  8.50"),
        ("Almond Croissant", "USD  3.50"),
    ]
    for name, price in items:
        draw.text((70, y), name, fill=(30, 35, 45), font=mono_font)
        draw.text((510, y), price, fill=(30, 35, 45), font=mono_font)
        y += 42

    y += 20
    draw.line([(60, y), (700, y)], fill=(180, 185, 195), width=2)
    y += 34

    draw.text((70, y), "SUBTOTAL", fill=(55, 65, 80), font=mono_font)
    draw.text((510, y), f"USD {subtotal}", fill=(55, 65, 80), font=mono_font)
    y += 38
    draw.text((70, y), "TAX (8%)", fill=(55, 65, 80), font=mono_font)
    draw.text((510, y), f"USD  {tax}", fill=(55, 65, 80), font=mono_font)
    y += 46
    draw.text((70, y), "TOTAL PAID", fill=(15, 23, 42), font=title_font)
    draw.text((485, y), f"USD {total}", fill=(15, 23, 42), font=title_font)
    y += 90

    draw.text((70, y), "SYNTHETIC DEMO RECEIPT - NON-SENSITIVE TEST DATA", fill=(110, 120, 135), font=body_font)

    img.save(output_path, format="PNG")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate non-sensitive synthetic receipt image")
    parser.add_argument("--output", default="docs/demo_sample_receipt.png", help="Output PNG path")
    args = parser.parse_args()
    out = generate_synthetic_receipt(Path(args.output))
    print(f"Generated synthetic sample receipt at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

