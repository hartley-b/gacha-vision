"""Assign each detected icon row to a rank, and summarize counts per rank.

Ties together row_extraction (where are the icons) and rank_detection
(where are the banners) into the actual thing we want out of a screenshot:
how many icons belong to each rank, with anything we can't determine
labeled "unknown" rather than guessed.
"""

from dataclasses import dataclass, field

import numpy as np

import rank_detection as rd
import row_extraction as rx

UNKNOWN = "unknown"


@dataclass
class RankSection:
    rank: str  # a rank string, or UNKNOWN
    boxes: list[rx.Box] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.boxes)


def _band_height(boxes: list[rx.Box], img_width: int) -> float:
    """Estimate how tall a banner-sized search window should be.

    Same reasoning as row_extraction's edge recovery: a banner occupies
    roughly one row-pitch of vertical space. With fewer than two rows there's
    no pitch to measure, so fall back to a proportion of image width (banners
    across our templates run ~11-13% of their source screenshot's width).
    """
    if len(boxes) >= 2:
        diffs = [boxes[i + 1].y - boxes[i].y for i in range(len(boxes) - 1)]
        return float(np.min(diffs)) * 1.4
    return img_width * 0.15


def analyze_screenshot(img: np.ndarray) -> list[RankSection]:
    """Run the full pipeline and return rank-labeled row sections, in order.

    A row's rank comes only from a banner confirmed *above* it — never from
    a banner that appears further down, even if one immediately follows.
    Rows before the first confirmed banner (or when none is found at all,
    e.g. a screenshot that's a pure mid-scroll continuation) are UNKNOWN,
    not guessed from context.
    """
    boxes = rx.find_icon_boxes(img)
    boxes = rx.recover_edge_rows(img, boxes)
    if not boxes:
        return []

    gaps = rx.find_gaps(boxes)
    boxes, unresolved_gaps = rx.resolve_internal_gaps(img, boxes, gaps)

    band_h = _band_height(boxes, img.shape[1])

    # leading edge: is a banner visible above the very first row?
    first = boxes[0]
    leading_match = rd.detect_rank(img, int(first.y - band_h), first.y)

    # each unresolved gap is either a real rank transition (banner found) or
    # just leftover spacing from a wrapped two-line name (no match) — only
    # real transitions become section boundaries.
    boundaries: list[tuple[int, str]] = []
    for gap in unresolved_gaps:
        match = rd.detect_rank(img, gap.y_start, gap.y_end)
        if match:
            boundaries.append((gap.y_end, match.rank))
    boundaries.sort(key=lambda b: b[0])

    sections: list[RankSection] = []
    current_rank = leading_match.rank if leading_match else UNKNOWN
    current = RankSection(rank=current_rank)
    b_idx = 0

    for box in boxes:
        while b_idx < len(boundaries) and box.y >= boundaries[b_idx][0]:
            sections.append(current)
            current = RankSection(rank=boundaries[b_idx][1])
            b_idx += 1
        current.boxes.append(box)
    sections.append(current)

    return sections


def summarize(sections: list[RankSection]) -> dict[str, int]:
    """Total icon count per rank label (merges same-rank sections)."""
    counts: dict[str, int] = {}
    for s in sections:
        counts[s.rank] = counts.get(s.rank, 0) + s.count
    return counts


def format_summary(sections: list[RankSection]) -> str:
    lines = [f"{count} {rank.lower()}" for rank, count in summarize(sections).items()]
    return "\n".join(lines)
