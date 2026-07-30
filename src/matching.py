"""Match a cropped icon to its character ID via embedding nearest-neighbor.

final_sorted/ALL holds every reference icon once (filename = character ID).
Each rank subfolder holds the same icons split by rank, so an icon's rank is
just "which subfolder contains this filename" — no separate rank classifier
needed once we know the ID; reference embeddings are computed once and
cached to disk (cache/), since embedding ~2700 icons is slow but the icon
set itself rarely changes.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FINAL_SORTED_DIR = ROOT / "final_sorted"
ALL_DIR = FINAL_SORTED_DIR / "ALL"
CACHE_PATH = ROOT / "cache" / "reference_embeddings.npz"

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "openai"
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Match:
    id: str
    rank: str | None  # None if this ID isn't found in any rank subfolder
    score: float


class ReferenceIndex:
    def __init__(self, ids: list[str], embeddings: np.ndarray, id_to_rank: dict[str, str]):
        self.ids = ids
        self.embeddings = embeddings  # (N, D), L2-normalized
        self.id_to_rank = id_to_rank

    def subset_for_rank(self, rank: str) -> "ReferenceIndex":
        idxs = [i for i, id_ in enumerate(self.ids) if self.id_to_rank.get(id_) == rank]
        return ReferenceIndex(
            ids=[self.ids[i] for i in idxs],
            embeddings=self.embeddings[idxs],
            id_to_rank=self.id_to_rank,
        )


_model = None
_preprocess = None


def _get_model():
    global _model, _preprocess
    if _model is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        model.eval().to(_DEVICE)
        _model = model
        _preprocess = preprocess
    return _model, _preprocess


def embed_images(images: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
    """Embed a list of BGR (OpenCV-convention) images, L2-normalized."""
    model, preprocess = _get_model()
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        tensors = []
        for img in batch:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(preprocess(Image.fromarray(rgb)))
        batch_tensor = torch.stack(tensors).to(_DEVICE)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def _build_id_to_rank() -> dict[str, str]:
    id_to_rank = {}
    for rank_dir in FINAL_SORTED_DIR.iterdir():
        if not rank_dir.is_dir() or rank_dir.name == "ALL":
            continue
        for f in rank_dir.glob("*.png"):
            id_to_rank[f.stem] = rank_dir.name
    return id_to_rank


def _fingerprint(files: list[Path]) -> np.ndarray:
    """Cheap per-file change signal: (mtime, size), no image reads needed.

    Catches a file being replaced in place (same id, new content) — id-list
    comparison alone misses that, since the filename didn't change. Bit us
    once already: an icon's reference image got swapped and the cache kept
    serving the stale embedding until someone remembered to force-rebuild.
    """
    return np.array([[f.stat().st_mtime, f.stat().st_size] for f in files])


def build_reference_index(force_rebuild: bool = False) -> ReferenceIndex:
    id_to_rank = _build_id_to_rank()
    current_ids = sorted(f.stem for f in ALL_DIR.glob("*.png"))
    files = [ALL_DIR / f"{id_}.png" for id_ in current_ids]
    fingerprint = _fingerprint(files)

    if not force_rebuild and CACHE_PATH.exists():
        data = np.load(CACHE_PATH, allow_pickle=True)
        cached_ids = list(data["ids"])
        cached_fingerprint = data.get("fingerprint")
        unchanged = cached_ids == current_ids and (
            cached_fingerprint is not None
            and cached_fingerprint.shape == fingerprint.shape
            and np.array_equal(cached_fingerprint, fingerprint)
        )
        if unchanged:
            return ReferenceIndex(
                ids=cached_ids, embeddings=data["embeddings"], id_to_rank=id_to_rank
            )
        print("final_sorted/ALL has changed since the cache was built — rebuilding.")

    print(f"embedding {len(current_ids)} reference icons (one-time, cached after this)...")
    images = [cv2.imread(str(f)) for f in files]
    embeddings = embed_images(images)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, ids=np.array(current_ids), embeddings=embeddings, fingerprint=fingerprint)

    return ReferenceIndex(ids=current_ids, embeddings=embeddings, id_to_rank=id_to_rank)


def match_embedding(
    embedding: np.ndarray, index: ReferenceIndex, rank: str | None = None
) -> Match:
    """Find the closest reference icon to an already-computed embedding.

    If `rank` is given and has reference icons, search is scoped to just
    that rank — fewer, more visually similar candidates than searching all
    ~2700, so more accurate. Otherwise (unknown rank, or a rank string with
    no matching folder) search the full index and read the rank off the
    matched ID's actual folder, rather than guessing rank from the crop
    directly.
    """
    search_index = index.subset_for_rank(rank) if rank else index
    if len(search_index.ids) == 0:
        search_index = index

    sims = search_index.embeddings @ embedding
    best_i = int(np.argmax(sims))
    best_id = search_index.ids[best_i]
    return Match(id=best_id, rank=index.id_to_rank.get(best_id), score=float(sims[best_i]))


def score_for_id(embedding: np.ndarray, index: ReferenceIndex, id_: str) -> float | None:
    """Cosine similarity between an embedding and one specific reference id.

    Used when a non-visual signal proposes an id that may sit outside the
    visual top-k — the caller still wants the visual score of that proposal
    for display/audit, even though the score didn't drive the choice.
    """
    try:
        i = index.ids.index(id_)
    except ValueError:
        return None
    return float(index.embeddings[i] @ embedding)


def match_icon(crop: np.ndarray, index: ReferenceIndex, rank: str | None = None) -> Match:
    """Find the closest reference icon to `crop`. See match_embedding."""
    [embedding] = embed_images([crop])
    return match_embedding(embedding, index, rank)


def top_k_matches(
    embedding: np.ndarray, index: ReferenceIndex, rank: str | None = None, k: int = 10
) -> list[Match]:
    """Like match_embedding, but returns the k best candidates, best first."""
    search_index = index.subset_for_rank(rank) if rank else index
    if len(search_index.ids) == 0:
        search_index = index

    sims = search_index.embeddings @ embedding
    k = min(k, len(sims))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [
        Match(id=search_index.ids[i], rank=index.id_to_rank.get(search_index.ids[i]), score=float(sims[i]))
        for i in top_idx
    ]
