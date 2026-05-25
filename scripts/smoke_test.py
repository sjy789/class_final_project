from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.pipeline import analyze_auto, evaluate_manual  # noqa: E402


def main() -> None:
    background = (ROOT / "samples" / "room_background.png").read_bytes()
    foreground = (ROOT / "samples" / "plant_foreground.png").read_bytes()
    auto = analyze_auto(background, foreground, top_k=3, scale=0.22, explain=True)
    manual = evaluate_manual(background, foreground, 0.35, 0.52, 0.22, True, False)
    print("auto_top", auto["top"][0]["score"], auto["top"][0]["label"])
    print("manual", manual["candidate"]["score"], manual["candidate"]["label"])
    print("output", auto["output_dir"])


if __name__ == "__main__":
    main()
