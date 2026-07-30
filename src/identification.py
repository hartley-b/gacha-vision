"""End-to-end: screenshot -> identified character IDs, per row.

Ties rank_grouping (which rows belong to which rank) and matching (which
reference icon a crop is) together. For a row whose rank is known from a
banner, matching is scoped to just that rank's reference icons. For an
`unknown` section, every row in it belongs to the *same* rank — it's one
continuous run of the list, just with no banner in frame — so rank is
decided once for the whole section (majority vote across a full-index match
of every row in it), then every row in that section is rematched scoped to
the winning rank for a more precise ID. Matching each row's rank
independently would let different rows in one unbroken section drift to
different guessed ranks whenever their individual top match happened to
land in different folders.

Within a section, character IDs are usually ascending in row order (how the
game lists them) — but not an unconditional rule (verified counter-example:
two adjacent, independently-confirmed-correct matches that decrease). So a
broken row only gets overridden when a same-rank candidate both restores the
ascending trend AND is independently high-confidence, never just "whatever
fits the order."

The name (OCR) cross-check gets authority in two tiers, both empirically
calibrated against scripts/evaluate.py's MISSED count (never aggregate
accuracy alone — a blanket margin widen was tried once at ~89% OCR accuracy
and made things silently worse):

  1. Near-tie tiebreak: when the visual top candidate barely beats the
     runner-up (the exact situation that let a wrong answer through on the
     real 2263/4399 confusable pair), the visual signal isn't actually
     confident regardless of its raw score, so a moderately confident OCR
     read may pick between candidates the visual match already considered
     plausible. It can't invent an answer here — only choose among the
     visual top-k.
  2. High-confidence override: when the OCR read resolves to a name with
     very high Levenshtein similarity (>= _TEXT_OVERRIDE_MIN_SIM) and that
     name's id disagrees with the visual pick, the text wins outright —
     even if its id sits outside the visual top-k. Rationale: the text
     signal now measures 180/180 standalone on the fully audited labeled
     set vs. the visual match's 175/180, and every disagreement it raised
     was a real visual error (5/5, 0 false alarms). The similarity bar is
     high because tesseract has hard glyph blind spots in this font (傀
     reads as 便 at every scale) — one misread in a short name can produce
     a confidently-wrong lookup, so only near-exact reads get to overrule.
     The visual pick is preserved in visual_id/visual_score so evaluate.py
     can report every override as fixed/broke, not just net accuracy.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

import matching as mt
import name_ocr
import rank_grouping as rg
import row_extraction as rx

_ORDER_MIN_SCORE = 0.90
_ORDER_MIN_MARGIN = 0.03
_ORDER_TOP_K = 10

# how close the top two visual candidates have to be before the visual
# match no longer counts as confident on its own — measured from the real
# case that motivated this: 2263 vs 4399 scored 0.9008 vs 0.9007.
_NEAR_TIE_MARGIN = 0.0005
_TEXT_TIEBREAK_MIN_SIM = 0.85

# minimum OCR lookup similarity for the text signal to overrule the visual
# match outright (tier 2 in the module docstring). Calibrated on the fully
# audited 180-row labeled set: every real visual error the text signal
# caught resolved at sim >= 0.86, and no correct visual match had a
# disagreeing text read at any similarity (0 false alarms) — so 0.85 covers
# all measured real errors while still refusing garbled reads.
_TEXT_OVERRIDE_MIN_SIM = 0.85


@dataclass
class IdentifiedIcon:
    box: rx.Box
    id: str
    rank: str
    score: float
    rank_was_guessed: bool
    resolved_by: str = "visual"  # "visual", "text-tiebreak", or "text-override"
    visual_id: str | None = None  # the visual match's own pick, kept for audit
    visual_score: float | None = None
    name_ocr_text: str | None = None
    name_match_ids: list[str] | None = None
    name_match_name: str | None = None  # the jpn_name OCR resolved to, for display
    name_match_score: float | None = None  # Levenshtein similarity, 0-1 — see module docstring
    name_agrees: bool | None = None  # None = no OCR result to compare against


def _correct_ascending_order(
    results: list[IdentifiedIcon], embeddings: list[np.ndarray], rank: str, index: mt.ReferenceIndex
) -> list[IdentifiedIcon]:
    running_max = -1
    corrected = []
    for result, embedding in zip(results, embeddings):
        current_id = int(result.id)
        if result.resolved_by == "text-override":
            # this row was decided by a near-exact OCR read, a stronger
            # signal than the ascending heuristic — don't second-guess it,
            # and don't let it drag running_max down for later rows either.
            running_max = max(running_max, current_id)
            corrected.append(result)
            continue
        if current_id > running_max:
            running_max = current_id
            corrected.append(result)
            continue

        # this row breaks the ascending trend — only override with a
        # candidate we'd trust standing alone: high absolute score, a clear
        # margin over the next-best order-restoring option (so we're not
        # picking between two similarly-plausible guesses), AND at least as
        # confident as the original match itself. That last check is the
        # one that was missing: a real screenshot proved order isn't a
        # strict rule (two independently-correct, adjacent, descending
        # matches), so overriding a *stronger* original match just because
        # a *weaker* candidate happens to satisfy the sequence is trading
        # the model's own best judgment for a worse guess — caught this
        # exact case, a correct 0.9496 match downgraded to a wrong 0.9226
        # one purely because the wrong one fit the expected order.
        candidates = mt.top_k_matches(embedding, index, rank=rank, k=_ORDER_TOP_K)
        valid = [c for c in candidates if int(c.id) > running_max]
        if not valid:
            corrected.append(result)
            continue

        best = valid[0]
        runner_up_score = valid[1].score if len(valid) > 1 else 0.0
        if (
            best.score >= _ORDER_MIN_SCORE
            and (best.score - runner_up_score) >= _ORDER_MIN_MARGIN
            and best.score >= result.score
        ):
            running_max = int(best.id)
            corrected.append(
                IdentifiedIcon(
                    box=result.box,
                    id=best.id,
                    rank=rank,
                    score=best.score,
                    rank_was_guessed=result.rank_was_guessed,
                    resolved_by="visual",  # this correction is itself a visual-only decision
                    visual_id=best.id,
                    visual_score=best.score,
                    name_ocr_text=result.name_ocr_text,
                    name_match_ids=result.name_match_ids,
                    name_match_name=result.name_match_name,
                    name_match_score=result.name_match_score,
                    name_agrees=(
                        best.id in result.name_match_ids
                        if result.name_match_ids is not None
                        else None
                    ),
                )
            )
        else:
            corrected.append(result)
    return corrected


def _resolve_rows(
    img: np.ndarray,
    boxes: list[rx.Box],
    embeddings: list[np.ndarray],
    rank: str,
    rank_was_guessed: bool,
    index: mt.ReferenceIndex,
    name_index: name_ocr.NameIndex | None,
) -> list[IdentifiedIcon]:
    """Pick each row's id: the visual top match, unless it's a near-tie with
    the runner-up and a confident OCR read points at one of the candidates
    instead. See module docstring for why this is narrower than "trust
    whichever signal is more confident" — it's a tiebreak, not a vote.
    """
    results = []
    for box, embedding in zip(boxes, embeddings):
        candidates = mt.top_k_matches(embedding, index, rank=rank, k=_ORDER_TOP_K)
        top = candidates[0]
        runner_up_score = candidates[1].score if len(candidates) > 1 else 0.0
        is_near_tie = (top.score - runner_up_score) < _NEAR_TIE_MARGIN

        name_ocr_text = name_match_ids = name_match_name = name_match_score = None
        if name_index is not None:
            name_ocr_text = name_ocr.read_name(img, box)
            lookup = name_ocr.lookup_name(name_ocr_text, rank, name_index) if name_ocr_text else None
            if lookup is not None:
                name_match_ids = lookup.ids
                name_match_name = lookup.name
                name_match_score = lookup.similarity

        chosen = top
        resolved_by = "visual"
        text_disagrees = name_match_ids is not None and top.id not in name_match_ids
        if text_disagrees and name_match_score >= _TEXT_OVERRIDE_MIN_SIM:
            # tier 2: a near-exact OCR read overrules the visual pick, even
            # when its id sits outside the visual top-k. Among same-name ids
            # (rare same-name-same-rank case) take the visually best one.
            override_id = max(
                name_match_ids,
                key=lambda i: mt.score_for_id(embedding, index, i) or -1.0,
            )
            override_score = mt.score_for_id(embedding, index, override_id)
            if override_score is not None:
                chosen = mt.Match(id=override_id, rank=rank, score=override_score)
                resolved_by = "text-override"
        elif (
            is_near_tie
            and name_match_ids is not None
            and name_match_score >= _TEXT_TIEBREAK_MIN_SIM
        ):
            tiebreak_candidate = next(
                (c for c in candidates if c.id in name_match_ids), None
            )
            if tiebreak_candidate is not None and tiebreak_candidate.id != top.id:
                chosen = tiebreak_candidate
                resolved_by = "text-tiebreak"

        results.append(
            IdentifiedIcon(
                box=box,
                id=chosen.id,
                rank=rank,
                score=chosen.score,
                rank_was_guessed=rank_was_guessed,
                resolved_by=resolved_by,
                visual_id=top.id,
                visual_score=top.score,
                name_ocr_text=name_ocr_text,
                name_match_ids=name_match_ids,
                name_match_name=name_match_name,
                name_match_score=name_match_score,
                name_agrees=(chosen.id in name_match_ids) if name_match_ids is not None else None,
            )
        )
    return results


def _identify_known_section(
    img: np.ndarray,
    section: rg.RankSection,
    index: mt.ReferenceIndex,
    name_index: name_ocr.NameIndex | None,
) -> list[IdentifiedIcon]:
    embeddings = mt.embed_images([box.crop(img) for box in section.boxes])
    results = _resolve_rows(img, section.boxes, embeddings, section.rank, False, index, name_index)
    return _correct_ascending_order(results, embeddings, section.rank, index)


def _identify_unknown_section(
    img: np.ndarray,
    section: rg.RankSection,
    index: mt.ReferenceIndex,
    name_index: name_ocr.NameIndex | None,
) -> list[IdentifiedIcon]:
    if not section.boxes:
        return []

    embeddings = mt.embed_images([box.crop(img) for box in section.boxes])

    # first pass: full-index match per row, just to vote on the section's rank
    first_pass = [mt.match_embedding(e, index, rank=None) for e in embeddings]
    votes = Counter(m.rank for m in first_pass if m.rank)
    section_rank = votes.most_common(1)[0][0] if votes else None
    if not section_rank:
        return [
            IdentifiedIcon(box=b, id=m.id, rank=rg.UNKNOWN, score=m.score, rank_was_guessed=True)
            for b, m in zip(section.boxes, first_pass)
        ]

    # second pass: resolve every row scoped to the winning rank, so the
    # final ID benefits from the same narrower/more-accurate search a known
    # section gets, instead of staying with each row's independent guess.
    results = _resolve_rows(
        img, section.boxes, embeddings, section_rank, True, index, name_index
    )
    return _correct_ascending_order(results, embeddings, section_rank, index)


def identify_screenshot(
    img: np.ndarray, index: mt.ReferenceIndex, name_index: name_ocr.NameIndex | None = None
) -> list[IdentifiedIcon]:
    results = []
    for section in rg.analyze_screenshot(img):
        if section.rank == rg.UNKNOWN:
            results.extend(_identify_unknown_section(img, section, index, name_index))
        else:
            results.extend(_identify_known_section(img, section, index, name_index))
    return results


def format_result_row(r: IdentifiedIcon, name_index: name_ocr.NameIndex) -> str:
    """One line: rank, id, jpn/eng name, visual score — and, only when the
    OCR text cross-check actually disagrees with the visual match, what the
    text itself thinks the answer is instead. Shared by scripts/evaluate.py
    and scripts/explore.py so both report identification results the same way.
    """
    jpn = name_index.id_to_name.get(r.id, "?")
    eng = name_index.id_to_eng_name.get(r.id, "?")
    row = f"{r.rank} | {r.id} | {jpn} | {eng} | {r.score:.3f}"
    if r.name_agrees is False:
        text_id = r.name_match_ids[0]
        row += f" | text thinks: {r.name_match_name} ({text_id}, sim={r.name_match_score:.2f})"
    if r.resolved_by == "text-override" and r.visual_id is not None:
        vis_jpn = name_index.id_to_name.get(r.visual_id, "?")
        row += (
            f" | overrode visual: {vis_jpn} ({r.visual_id}, score={r.visual_score:.3f})"
        )
    return row
