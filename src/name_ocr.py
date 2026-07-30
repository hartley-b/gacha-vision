"""Read the character name next to an icon and cross-check it against a
visual match — an independent second signal that catches exactly the case
visual matching structurally can't: two icons so similar the embedding
can't confidently tell them apart, but their names are completely
different (confirmed on the real 2263/4399 confusable pair).

Historically a confirm/flag mechanism only. After the multi-line region
detection and Levenshtein/NFKC lookup fixes, the text signal measures
180/180 on the hand-audited labeled set (read_name + rank-scoped
lookup_name, checked against tests/labels.json), while the visual match
runs 175/180 — so it now also gets real authority, in two calibrated tiers
(see identification.py): a near-tie tiebreak among the visual top-k, and a
high-confidence override (near-exact reads, sim >= 0.85) that can overrule
the visual pick outright. Measured on the audited set the override fixed
all 6 visual-signal errors and broke none; any threshold change must be
re-judged by evaluate.py's BROKE/MISSED counts, not aggregate accuracy.

Known failure modes, worth knowing before trusting a read too far:
  - A trailing percentage sometimes leaks into the crop — stripped before
    matching, but not always cleanly.
  - Tesseract can be simply unable to recognize a rare glyph in this font
    (measured: 傀 in 傀軍座敷童子 reads as 便 at every scale tried). The
    Levenshtein lookup absorbs a single such misread, but two or more in a
    short name could still flip the match to a wrong-but-similar sibling —
    a confidently wrong lookup a similarity threshold can't catch. This is
    why the near-tie tiebreak's margin stays tight: at looser margins,
    confidently-wrong lookups measurably made accuracy worse, not better.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

import row_extraction as rx

ROOT = Path(__file__).resolve().parent.parent
NAMES_PATH = ROOT / "data" / "character_names.json"

_GAP_THRESHOLD = 15  # px between dark-pixel column runs that separates glyphs/words
_NAME_END_GAP = 150  # px gap that marks "name text ended, percentage column starts"
_MARGIN = 10

# The name text block is vertically centered on the row, NOT aligned to the
# icon: a name that wraps to two lines starts *above* the icon's top edge
# and ends *below* its bottom edge, with the percentage forming its own
# vertical band in between (all measured on the labeled screenshots — a
# two-line first line sits at roughly box-relative -0.5h..0.05h). So line
# detection scans a strip taller than the icon on both sides.
_STRIP_ABOVE_FRAC = 0.7  # strip extends this fraction of icon height above box top
_STRIP_BELOW_FRAC = 1.7  # ...and to this fraction below box top
# rows of the strip with at most this many dark pixels count as blank space
# between text lines
_BAND_EMPTY_MAX_PX = 2
# a text line can split into two bands when a thin glyph row dips under the
# blank threshold (seen: a 1px gap mid-glyph) — re-merge across tiny gaps.
# Must stay below the real gap between a name line and the percentage band
# (measured 5-8px), though an accidental merge there is harmless: the
# percentage is excluded again per-band by the column-gap logic.
_BAND_MERGE_GAP = 3
# bands of one row's text block sit within ~0.15h of each other; the nearest
# text of an adjacent row is several times further (and two-line rows get
# extra row pitch from the game), so accretion stops well before it.
_BLOCK_GAP_FRAC = 0.25
# a column dark for nearly the whole strip height is the screen border /
# scrollbar at the right edge, not text
_BORDER_COL_DARK_FRAC = 0.8
_BAND_PAD = 4  # vertical padding around a detected line, px

# a trailing percentage ("1.800%") sometimes ends up inside the crop —
# strip it before matching rather than let it drag the name text down.
_PERCENT_SUFFIX = re.compile(r"[\s]*[\d０-９.]+[%％]\s*$")
# characters that show up when OCR reads background decoration/noise instead
# of real text
_GARBAGE_CHARS = set("ー。、.- 　")


@dataclass
class NameLookup:
    ids: list[str]  # normally one; can be >1 for the rare same-name-same-rank case
    name: str
    similarity: float  # normalized Levenshtein similarity to this name, 0-1


class NameIndex:
    def __init__(self, entries: list[dict]):
        self.id_to_name = {e["id"]: e["jpn_name"] for e in entries}
        self.id_to_eng_name = {e["id"]: e.get("eng_name", "?") for e in entries}
        self.by_rank: dict[str, list[str]] = {}
        self.name_to_ids: dict[tuple[str, str], list[str]] = {}
        for e in entries:
            self.by_rank.setdefault(e["rank"], []).append(e["jpn_name"])
            key = (e["jpn_name"], e["rank"])
            self.name_to_ids.setdefault(key, []).append(e["id"])


def load_name_index() -> NameIndex:
    entries = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    return NameIndex(entries)


def _column_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous dark-pixel column spans (glyphs/words) within a band."""
    nonzero = np.where(mask.sum(axis=0) > 0)[0]
    if len(nonzero) == 0:
        return []
    runs = []
    start = prev = nonzero[0]
    for x in nonzero[1:]:
        if x - prev > _GAP_THRESHOLD:
            runs.append((int(start), int(prev)))
            start = x
        prev = x
    runs.append((int(start), int(prev)))
    return runs


def _text_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    """Vertical spans of text lines in a strip: runs of non-blank rows,
    re-merged across gaps small enough to be a dip inside one glyph row."""
    ys = np.where(mask.sum(axis=1) > _BAND_EMPTY_MAX_PX)[0]
    if len(ys) == 0:
        return []
    bands = []
    start = prev = ys[0]
    for y in ys[1:]:
        if y - prev - 1 > _BAND_MERGE_GAP:
            bands.append((int(start), int(prev) + 1))
            start = y
        prev = y
    bands.append((int(start), int(prev) + 1))
    return bands


def _name_span(runs: list[tuple[int, int]]) -> tuple[int, int]:
    """Take runs from the first while the gap stays small — the gap to the
    percentage column is reliably much bigger than the gap between
    characters in a name."""
    start, end = runs[0]
    for run_start, run_end in runs[1:]:
        if run_start - end > _NAME_END_GAP:
            break
        end = run_end
    return start, end


def find_name_line_regions(img: np.ndarray, box: rx.Box) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the name text lines to the right of an icon, top to
    bottom — one region for a single-line name, two for a wrapped one.

    Structural, not hardcoded:
      - Text lines are horizontal bands of dark pixels in a strip taller
        than the icon (the name block is centered on the row, so a wrapped
        name overhangs the icon on both sides — see the strip constants).
      - The band nearest the icon's vertical center always holds the small
        type-symbol icon (heart/claw/etc.), which is row-centered; that
        anchor band plus any tightly-packed bands above/below it form this
        row's text block, and anything further away belongs to a neighboring
        row or a banner.
      - Within each band, the name span is the tight cluster of column runs
        (skipping the type icon in the anchor band); the percentage column
        sits behind a much bigger gap and is excluded. In the two-line case
        the anchor band holds *only* the type icon and the percentage —
        recognizable by that same oversized gap — and contributes no region.
    """
    img_h, img_w = img.shape[:2]
    y0 = max(0, box.y - int(box.h * _STRIP_ABOVE_FRAC))
    y1 = min(img_h, box.y + int(box.h * _STRIP_BELOW_FRAC))
    x0 = box.x + box.w
    strip = img[y0:y1, x0:img_w]
    if strip.size == 0:
        return []

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    mask = thresh > 0
    mask[:, mask.mean(axis=0) > _BORDER_COL_DARK_FRAC] = False

    bands = _text_bands(mask)
    if not bands:
        return []

    box_center = box.y + box.h / 2 - y0
    anchor_i = min(
        range(len(bands)), key=lambda i: abs((bands[i][0] + bands[i][1]) / 2 - box_center)
    )
    max_gap = box.h * _BLOCK_GAP_FRAC
    lo = anchor_i
    while lo > 0 and bands[lo][0] - bands[lo - 1][1] <= max_gap:
        lo -= 1
    hi = anchor_i
    while hi < len(bands) - 1 and bands[hi + 1][0] - bands[hi][1] <= max_gap:
        hi += 1

    anchor_runs = _column_runs(mask[bands[anchor_i][0] : bands[anchor_i][1]])
    if not anchor_runs:
        return []
    type_icon_end = anchor_runs[0][1]

    regions = []
    for i in range(lo, hi + 1):
        band_y0, band_y1 = bands[i]
        if i == anchor_i:
            runs = anchor_runs[1:]  # skip the type icon
            if not runs or runs[0][0] - type_icon_end > _NAME_END_GAP:
                continue  # nothing but the percentage here (two-line case)
        else:
            runs = _column_runs(mask[band_y0:band_y1])
            if not runs:
                continue
        name_start, name_end = _name_span(runs)
        regions.append(
            (
                x0 + max(0, name_start - _MARGIN),
                y0 + max(0, band_y0 - _BAND_PAD),
                x0 + name_end + _MARGIN,
                y0 + min(mask.shape[0], band_y1 + _BAND_PAD),
            )
        )
    return regions


def find_name_region(img: np.ndarray, box: rx.Box) -> tuple[int, int, int, int] | None:
    """Union bounding box of the name's text lines (for visualization)."""
    regions = find_name_line_regions(img, box)
    if not regions:
        return None
    return (
        min(r[0] for r in regions),
        min(r[1] for r in regions),
        max(r[2] for r in regions),
        max(r[3] for r in regions),
    )


def _clean(text: str) -> str:
    """Strip a trailing percentage that leaked into the crop."""
    return _PERCENT_SUFFIX.sub("", text).strip()


def _is_garbage(text: str) -> bool:
    """True if OCR read noise (background decoration) rather than real text."""
    return len(text) < 2 or all(c in _GARBAGE_CHARS for c in text)


def _ocr_region(img: np.ndarray, region: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = region
    crop = img[y0:y1, x0:x1]
    # psm 8 (treat as a single word) reads short game-font names much more
    # reliably than psm 6/7 here — measured, not assumed: psm 7 misread a
    # clean crop of "轟獅子" as "田名子" at every scale tried, psm 8 got it
    # right every time.
    return pytesseract.image_to_string(crop, lang="jpn", config="--psm 8").strip()


def read_name(img: np.ndarray, box: rx.Box) -> str | None:
    """OCR the name text next to an icon. None if nothing usable was found.

    Each detected text line is OCR'd separately (psm 8 works per line, not
    per block) and the results concatenated in reading order — wrapped names
    have no separator character at the break, so plain concatenation
    reconstructs the original name. A line is dropped only when it contains
    nothing but noise characters; length alone isn't a garbage signal here,
    since a wrapped second line can legitimately be a single character.
    """
    parts = []
    for region in find_name_line_regions(img, box):
        text = _clean(_ocr_region(img, region))
        if text and not all(c in _GARBAGE_CHARS for c in text):
            parts.append(text)
    joined = "".join(parts)
    if _is_garbage(joined):
        return None
    return joined


def _normalize(text: str) -> str:
    """Fold away differences OCR introduces that carry no identity signal:
    NFKC unifies full-width/half-width forms (tesseract emits ASCII "()"
    where the game and database use full-width "（）" — enough by itself to
    make the wrong, bracket-less name win a ratio match), and spaces go
    entirely (tesseract inserts them mid-name at glyph gaps)."""
    return "".join(unicodedata.normalize("NFKC", text).split())


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    """Levenshtein-based similarity on normalized text, 0-1.

    Not difflib's ratio, deliberately: ratio counts common characters
    regardless of position, so a single misread character can produce a
    dead tie between the right name and a wrong one (real case: 傀軍座敷童子
    misread as 便軍座敷童子 scored identically against 傀軍座敷童子 and
    軍呪座敷童子 — the shifted 軍 got full credit). Edit distance charges
    that shift properly: 1 substitution vs 2.
    """
    if not a and not b:
        return 1.0
    return 1.0 - _edit_distance(a, b) / max(len(a), len(b))


def lookup_name(ocr_text: str, rank: str, index: NameIndex) -> NameLookup | None:
    """Fuzzy-match OCR'd text against the known names for this rank.

    Exact string match isn't the bar — OCR on small stylized game text gets
    individual characters wrong often enough (measured: 41% exact match)
    that fuzzy matching against the small, known, rank-scoped candidate set
    is what actually makes this usable.
    """
    if not ocr_text:
        return None
    candidates = index.by_rank.get(rank)
    if not candidates:
        return None
    normalized = _normalize(ocr_text)
    name = max(candidates, key=lambda c: _similarity(normalized, _normalize(c)))
    similarity = _similarity(normalized, _normalize(name))
    return NameLookup(ids=index.name_to_ids[(name, rank)], name=name, similarity=similarity)
