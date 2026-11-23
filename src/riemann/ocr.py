from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from rapidfuzz import fuzz

from riemann.models import PublicLocale, RequestLocale, ResolveAugmentResponse, ResolveCandidate

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_ocr_text(text: str) -> str:
    """Deterministic normalization in strict order:
    1. Unicode NFKD accent strip
    2. Confusable folding BEFORE lowercasing: I -> l, 1 -> l, 0 -> o
    3. Lowercase
    4. Strip non-alphanumeric characters and collapse whitespace
    """
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    folded = stripped.replace("I", "l").replace("1", "l").replace("0", "o")
    lowered = folded.lower()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def resolve_augment_text(
    *,
    text: str,
    locale: RequestLocale,
    entries: Iterable[tuple[str, PublicLocale, str, str]],
    confidence_threshold: float = 0.80,
    margin_threshold: float = 0.08,
    max_candidates: int = 5,
) -> ResolveAugmentResponse:
    """Resolve raw OCR text against catalog entries (canonical_id, locale, display_name, normalized_name)."""
    norm_query = normalize_ocr_text(text)
    if not norm_query:
        return ResolveAugmentResponse(
            resolved=False,
            canonical_id=None,
            display_name=None,
            detected_locale=None,
            confidence=0.0,
            reason="below_confidence_threshold",
            candidates=[],
        )

    best_per_id: dict[str, ResolveCandidate] = {}
    locale_rank: dict[PublicLocale, int] = {"en_us": 0, "pt_br": 1}

    for cid, entry_locale, display_name, normalized_name in entries:
        if locale != "auto" and entry_locale != locale:
            continue
        target_norm = normalized_name if normalized_name else normalize_ocr_text(display_name)
        score = float(fuzz.WRatio(norm_query, target_norm)) / 100.0
        cand = ResolveCandidate(
            canonical_id=cid,
            display_name=display_name,
            locale=entry_locale,
            confidence=round(score, 6),
        )
        prev = best_per_id.get(cid)
        if prev is None:
            best_per_id[cid] = cand
        else:
            if (cand.confidence > prev.confidence) or (
                cand.confidence == prev.confidence
                and locale_rank[cand.locale] < locale_rank[prev.locale]
            ):
                best_per_id[cid] = cand

    ranked = sorted(
        best_per_id.values(),
        key=lambda c: (-c.confidence, locale_rank[c.locale], c.canonical_id),
    )
    top_candidates = ranked[:max_candidates]
    if not ranked:
        return ResolveAugmentResponse(
            resolved=False,
            canonical_id=None,
            display_name=None,
            detected_locale=None,
            confidence=0.0,
            reason="below_confidence_threshold",
            candidates=[],
        )

    top1 = ranked[0]
    if top1.confidence < confidence_threshold:
        return ResolveAugmentResponse(
            resolved=False,
            canonical_id=None,
            display_name=None,
            detected_locale=None,
            confidence=top1.confidence,
            reason="below_confidence_threshold",
            candidates=top_candidates,
        )

    if len(ranked) > 1 and (top1.confidence - ranked[1].confidence) < margin_threshold:
        return ResolveAugmentResponse(
            resolved=False,
            canonical_id=None,
            display_name=None,
            detected_locale=None,
            confidence=top1.confidence,
            reason="ambiguous_match",
            candidates=top_candidates,
        )

    return ResolveAugmentResponse(
        resolved=True,
        canonical_id=top1.canonical_id,
        display_name=top1.display_name,
        detected_locale=top1.locale,
        confidence=top1.confidence,
        reason="matched",
        candidates=top_candidates,
    )
