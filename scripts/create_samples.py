from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIRS = [
    ROOT / "samples",
    ROOT / "frontend" / "public" / "samples",
]


def ensure_dirs() -> None:
    for directory in SAMPLE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def make_room() -> Image.Image:
    width, height = 960, 640
    image = Image.new("RGB", (width, height), "#f3f1e8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 410), fill="#dbe5df")
    draw.rectangle((0, 410, width, height), fill="#c6aa79")
    for y in range(420, height, 34):
        draw.line((0, y, width, y - 50), fill="#b68f5f", width=2)
    draw.rectangle((80, 86, 310, 250), fill="#b7d2df", outline="#6e7f83", width=5)
    draw.rectangle((608, 122, 830, 324), fill="#d7ddd7", outline="#8b9189", width=4)
    draw.rectangle((565, 372, 875, 432), fill="#8d674e")
    draw.rectangle((590, 332, 850, 384), fill="#a77f5d")
    draw.ellipse((650, 235, 780, 360), fill="#4c765f")
    draw.rectangle((706, 330, 726, 372), fill="#725235")
    draw.rectangle((90, 350, 365, 392), fill="#6d8f9f")
    draw.rectangle((116, 392, 138, 510), fill="#4e6974")
    draw.rectangle((320, 392, 342, 510), fill="#4e6974")
    return image.filter(ImageFilter.SMOOTH_MORE)


def make_plant() -> Image.Image:
    image = Image.new("RGBA", (360, 430), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((120, 250, 242, 396), fill="#bd7852", outline="#8b4b35", width=5)
    draw.rectangle((151, 190, 212, 310), fill="#5d6a46")
    leaves = [
        (80, 110, 190, 240, "#3f7f55"),
        (165, 88, 300, 225, "#2f6f49"),
        (56, 170, 160, 288, "#4c8b5d"),
        (205, 162, 326, 292, "#3e8056"),
        (114, 45, 238, 184, "#5e9b67"),
    ]
    for box, color in [(leaf[:4], leaf[4]) for leaf in leaves]:
        draw.ellipse(box, fill=color, outline="#24583c", width=3)
    draw.line((182, 300, 178, 95), fill="#314d30", width=8)
    return image.filter(ImageFilter.SMOOTH_MORE)


def main() -> None:
    ensure_dirs()
    room = make_room()
    plant = make_plant()
    for directory in SAMPLE_DIRS:
        room.save(directory / "room_background.png")
        plant.save(directory / "plant_foreground.png")
    print("sample images written")


if __name__ == "__main__":
    main()

