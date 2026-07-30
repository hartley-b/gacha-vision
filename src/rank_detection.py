"""Identify which rank's banner (if any) sits in a region of a screenshot.

Banners ("ランクA", "ランクS", ...) are fixed game graphics, so this is
template matching, not learned classification: crop each rank's banner once
from a real screenshot (ref/banner_templates/<RANK>.png) and match new
regions against that small set.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "ref" / "banner_templates"

# below this, "best match" isn't a match — it's just whichever template
# happened to score highest against plain background/text.
_MATCH_THRESHOLD = 0.6

# Templates get cropped from screenshots of whatever resolution they happened
# to come from, so there's no single fixed scale factor between a template
# and a new screenshot. Rather than track each template's source resolution
# (fragile — breaks the moment a template is dropped in without that
# bookkeeping), try a spread of scales and keep whichever wins; one real
# banner match tends to score far above every wrong-rank/no-banner case
# (seen 0.8-1.0 for correct matches vs <0.6 for everything else), so a wrong
# scale just means a weak score, not a false match.
_SCALE_FACTORS = np.linspace(0.5, 2.2, 18)

# matchTemplate cost scales with pixel count; screenshots run ~1000-1640px
# wide but that precision buys nothing for "which of 14 banners is this" —
# downscale before matching for a large speedup with no accuracy loss we've
# been able to measure.
_MATCH_WIDTH = 480


@dataclass
class RankMatch:
    rank: str
    score: float


def _load_templates() -> dict[str, np.ndarray]:
    templates = {}
    for path in TEMPLATES_DIR.glob("*.png"):
        templates[path.stem] = cv2.imread(str(path))
    return templates


_TEMPLATES = _load_templates()


def detect_rank(img: np.ndarray, y0: int, y1: int, margin: int = 50) -> RankMatch | None:
    """Check whether a rank banner sits within img[y0:y1]."""
    h, w = img.shape[:2]
    y0 = max(0, y0 - margin)
    y1 = min(h, y1 + margin)
    band = img[y0:y1]
    if band.shape[0] < 10:
        return None

    band_scale = min(1.0, _MATCH_WIDTH / w)
    if band_scale < 1.0:
        band = cv2.resize(band, (int(w * band_scale), int(band.shape[0] * band_scale)))

    best: RankMatch | None = None
    for rank, template in _TEMPLATES.items():
        t_h, t_w = template.shape[:2]
        for scale in _SCALE_FACTORS:
            eff_scale = scale * band_scale
            s_w, s_h = int(t_w * eff_scale), int(t_h * eff_scale)
            if s_h < 8 or s_w < 8 or s_h > band.shape[0] or s_w > band.shape[1]:
                continue

            scaled = cv2.resize(template, (s_w, s_h))
            result = cv2.matchTemplate(band, scaled, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if best is None or score > best.score:
                best = RankMatch(rank=rank, score=score)

    if best is None or best.score < _MATCH_THRESHOLD:
        return None
    return best
