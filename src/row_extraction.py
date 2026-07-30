"""Detect icon rows in a gacha rate-list screenshot.

Icons are left-aligned, roughly square, dark-outlined blobs of consistent
size within a single screenshot. Detection is structural (contour-based),
not tied to any fixed resolution or pixel offsets, so it generalizes across
different devices/crops.
"""

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

# icon width/height, as a fraction of image width, that counts as icon-sized
_MIN_SIZE_FRAC = 0.05
_MAX_SIZE_FRAC = 0.15
_MIN_ASPECT = 0.7
_MAX_ASPECT = 1.4
_DARK_THRESHOLD = 80
# candidates must share a left-x within this many pixels to count as one column
_X_CLUSTER_TOLERANCE = 20


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y : self.y + self.h, self.x : self.x + self.w]


@dataclass
class Gap:
    """A larger-than-expected vertical gap between two consecutive rows.

    Could mean a rank banner sits here, or that icon detection missed a
    row (e.g. a broken outline). Row extraction can't tell which — that's
    resolved by the caller (banner template match, or interpolate).
    """

    y_start: int
    y_end: int
    n_missing: int


def find_icon_boxes(img: np.ndarray) -> list[Box]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, _DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if not (w * _MIN_SIZE_FRAC <= cw <= w * _MAX_SIZE_FRAC):
            continue
        if not (w * _MIN_SIZE_FRAC <= ch <= w * _MAX_SIZE_FRAC):
            continue
        aspect = cw / ch
        if not (_MIN_ASPECT < aspect < _MAX_ASPECT):
            continue
        candidates.append(Box(x, y, cw, ch))

    if not candidates:
        return []

    # icons in the same list are left-aligned at one consistent x; anything
    # else (banner letters, decorations) sits at a different x and gets
    # dropped here.
    bucketed = Counter(round(box.x / 10) * 10 for box in candidates)
    mode_bucket = bucketed.most_common(1)[0][0]
    filtered = [box for box in candidates if abs(box.x - mode_bucket) < _X_CLUSTER_TOLERANCE]
    filtered.sort(key=lambda box: box.y)
    return filtered


def find_gaps(boxes: list[Box]) -> list[Gap]:
    """Flag consecutive-row spacing larger than the typical row pitch.

    Pitch is estimated as the *minimum* observed spacing, not the median.
    A screenshot can have more banner-transition gaps than normal single-row
    gaps (e.g. several ranks with only 1-2 rows each) — in that case the
    median lands on a gap-sized value instead of the true row pitch, and
    every real gap looks "normal" relative to it. The smallest spacing is
    always a real single-row gap, since a gap is by definition >= one pitch.
    """
    if len(boxes) < 3:
        return []

    diffs = [boxes[i + 1].y - boxes[i].y for i in range(len(boxes) - 1)]
    pitch = float(np.min(diffs))

    gaps = []
    for i, diff in enumerate(diffs):
        n_missing = round(diff / pitch) - 1
        if n_missing > 0:
            gaps.append(Gap(y_start=boxes[i].bottom, y_end=boxes[i + 1].y, n_missing=n_missing))
    return gaps


_BG_COLOR_TOLERANCE = 60


def _background_color(img: np.ndarray, box: Box) -> np.ndarray:
    """Sample the list background just left of a box (outside the icon)."""
    y = box.y + box.h // 2
    x = max(0, box.x - 15)
    return img[y, x].astype(float)


def _is_plausible_icon(img: np.ndarray, box: Box, reference_bg: np.ndarray) -> bool:
    """Reject candidates whose surroundings don't match the list background.

    Icons sit in a uniform pale-blue list row; other UI elements (buttons,
    banners) sit on a different-colored background. A same-sized,
    same-aspect blob of button text can otherwise pass every geometric
    filter (this happened: the "まえへ" pagination button got picked up as
    a recovered row before this check existed).
    """
    bg = _background_color(img, box)
    distance = float(np.linalg.norm(bg - reference_bg))
    return distance < _BG_COLOR_TOLERANCE


_SIZE_MATCH_TOLERANCE = 0.35


def _search_strip(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, expected_w: int, expected_h: int
) -> Box | None:
    """Re-run detection on a small crop with a local morphological close.

    A whole-image close bridges gaps between unrelated dense text/icons and
    over-merges everything (tried it — one giant blob). Scoped to a narrow
    strip around a single expected icon, the same close just bridges a
    broken outline without touching anything else.

    Candidates are filtered against the *actual* icon size already measured
    in this screenshot (expected_w/h), not a size range derived from the
    strip's own width — that fraction-of-strip-width approach had no floor
    on height at all, and let a single stray kanji character (65x57, no
    relation to the ~214x195 icons around it) through as a "recovered" row.
    """
    y0, x0 = max(0, y0), max(0, x0)
    strip = img[y0:y1, x0:x1]
    if strip.size == 0:
        return None

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, _DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    w_lo, w_hi = expected_w * (1 - _SIZE_MATCH_TOLERANCE), expected_w * (1 + _SIZE_MATCH_TOLERANCE)
    h_lo, h_hi = expected_h * (1 - _SIZE_MATCH_TOLERANCE), expected_h * (1 + _SIZE_MATCH_TOLERANCE)

    best = None
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if not (w_lo <= cw <= w_hi and h_lo <= ch <= h_hi):
            continue
        aspect = cw / ch if ch else 0
        if not (_MIN_ASPECT < aspect < _MAX_ASPECT):
            continue
        if best is None or cw * ch > best[2] * best[3]:
            best = (cx, cy, cw, ch)

    if best is None:
        return None
    cx, cy, cw, ch = best
    return Box(x0 + cx, y0 + cy, cw, ch)


def recover_edge_rows(img: np.ndarray, boxes: list[Box]) -> list[Box]:
    """Look one row-pitch above the first / below the last confident box.

    Catches rows missed only because a compression artifact broke their
    outline (so no contour ever closed for them) — a case `find_gaps` can't
    flag, since it only compares *between* already-detected rows and there's
    nothing on the far side of an edge row to compare against.
    """
    if len(boxes) < 2:
        return boxes

    pitch = float(np.median([boxes[i + 1].y - boxes[i].y for i in range(len(boxes) - 1)]))
    margin = 10
    reference_bg = _background_color(img, boxes[len(boxes) // 2])
    exp_w = int(np.median([b.w for b in boxes]))
    exp_h = int(np.median([b.h for b in boxes]))
    result = list(boxes)

    first = boxes[0]
    above = _search_strip(
        img,
        first.x - margin,
        int(first.y - pitch * 1.3),
        first.x + first.w + margin,
        first.y,
        exp_w,
        exp_h,
    )
    if above is not None and _is_plausible_icon(img, above, reference_bg):
        result.insert(0, above)

    last = boxes[-1]
    below = _search_strip(
        img,
        last.x - margin,
        last.bottom,
        last.x + last.w + margin,
        int(last.bottom + pitch * 1.3),
        exp_w,
        exp_h,
    )
    if below is not None and _is_plausible_icon(img, below, reference_bg):
        result.append(below)

    return result


def resolve_internal_gaps(
    img: np.ndarray, boxes: list[Box], gaps: list[Gap]
) -> tuple[list[Box], list[Gap]]:
    """Try to explain each flagged gap as a missed icon via local search.

    A gap can mean: a rank banner sits here (real, leave it flagged for the
    caller to confirm against banner templates), a row was missed (recover
    it here), or neither — just a taller-than-usual row from a wrapped
    two-line name, which is not an icon and not a banner. Local icon search
    correctly finds nothing for that last case too, same as a real banner
    gap; distinguishing "banner" from "just tall text" needs rank_detection,
    not this module — this only peels off the "actually a missed icon" case.
    """
    if not gaps:
        return boxes, []

    reference_bg = _background_color(img, boxes[len(boxes) // 2])
    margin = 10
    exp_w = int(np.median([b.w for b in boxes]))
    exp_h = int(np.median([b.h for b in boxes]))
    result = list(boxes)
    unresolved = []

    for gap in gaps:
        candidate = _search_strip(
            img,
            boxes[0].x - margin,
            gap.y_start,
            boxes[0].x + boxes[0].w + margin,
            gap.y_end,
            exp_w,
            exp_h,
        )
        if candidate is not None and _is_plausible_icon(img, candidate, reference_bg):
            result.append(candidate)
        else:
            unresolved.append(gap)

    result.sort(key=lambda box: box.y)
    return result, unresolved
