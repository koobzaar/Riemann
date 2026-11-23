from __future__ import annotations

import re
from typing import Any

import httpx

from riemann.models import (
    CatalogAugment,
    CatalogBundle,
    CatalogChampion,
    CatalogItem,
    GameMode,
    PublicLocale,
)

CDRAGON_LOCALE_MAP: dict[PublicLocale, str] = {
    "en_us": "default",
    "pt_br": "pt_br",
}

CDRAGON_BASE_URL = (
    "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global"
)
CDRAGON_METADATA_URL = "https://raw.communitydragon.org/latest/content-metadata.json"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_PATCH_RE = re.compile(r"^(\d+)\.(\d+)")


def canonicalize_slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.strip().lower()).strip("_")


def build_patch_calendar_from_version(version_str: str, depth: int = 4) -> tuple[str, list[str]]:
    """Extract major.minor patch (e.g. '16.23') and construct descending patch calendar."""
    m = _PATCH_RE.match(version_str.strip())
    if not m:
        raise ValueError(f"Unparseable CommunityDragon version string: {version_str!r}")
    major, minor = int(m.group(1)), int(m.group(2))
    cal = [f"{major}.{minor - i}" for i in range(depth) if minor - i >= 1]
    return cal[0], cal


def _infer_augment_modes(raw: dict[str, Any]) -> list[GameMode]:
    modes_raw = raw.get("modes")
    if modes_raw:
        return sorted(set(modes_raw))  # type: ignore[return-value]
    aug_name_id = str(raw.get("augmentNameId") or raw.get("apiName") or "")
    mode_tag = str(raw.get("mode") or "").lower()
    if mode_tag == "both":
        return ["aram_mayhem", "arena"]
    if (
        aug_name_id.upper().startswith(("ARAM_", "KIWI_", "MAYHEM_"))
        or "mayhem" in mode_tag
        or "kiwi" in mode_tag
    ):
        return ["aram_mayhem"]
    return ["arena"]


def parse_cdragon_payloads(
    *,
    default_payload: dict[str, Any],
    pt_br_payload: dict[str, Any],
) -> CatalogBundle:
    """Parse raw CommunityDragon payloads (`default` -> `en_us`, `pt_br` -> `pt_br`).

    Strictly validates required non-empty sections and deterministically deduplicates
    entities that share a canonical_id across mode variants (unioning their modes).
    """
    if not isinstance(default_payload, dict) or not isinstance(pt_br_payload, dict):
        raise ValueError("corrupt_catalog_payload: payloads must be JSON objects")

    for req_key in ("augments", "items", "champions"):
        val = default_payload.get(req_key)
        if not isinstance(val, list) or len(val) == 0:
            raise ValueError(
                f"corrupt_catalog_payload: missing or empty required section {req_key!r}"
            )

    patch = str(default_payload.get("patch") or default_payload.get("version") or "").strip()
    if not patch:
        raise ValueError("corrupt_catalog_payload: missing patch/version")
    patch_calendar = list(default_payload.get("patch_calendar") or [patch])
    mode_rework = default_payload.get("mode_rework_patches") or {
        "aram_mayhem": None,
        "arena": None,
    }

    def _aug_lookup_key(row: dict[str, Any]) -> str:
        return str(row.get("augmentNameId") or row.get("apiName") or row.get("id"))

    pt_aug_by_id = {_aug_lookup_key(a): a for a in pt_br_payload.get("augments", [])}
    pt_item_by_id = {str(i.get("id")): i for i in pt_br_payload.get("items", [])}
    pt_champ_by_id = {str(c.get("id")): c for c in pt_br_payload.get("champions", [])}

    # Deduplicate augments by canonical_id, merging modes across Arena + ARAM Mayhem variants
    aug_by_cid: dict[str, CatalogAugment] = {}
    for raw in default_payload["augments"]:
        raw_key = _aug_lookup_key(raw)
        en_name = str(raw.get("nameTRA") or raw.get("name") or raw_key).strip()
        if not en_name:
            continue
        pt_row = pt_aug_by_id.get(raw_key, {})
        pt_name = str(pt_row.get("nameTRA") or pt_row.get("name") or en_name).strip() or en_name
        cid = str(raw.get("canonical_id") or canonicalize_slug(en_name))
        if not cid:
            continue
        modes = _infer_augment_modes(raw)
        rarity = str(raw.get("rarity", "unknown")).lower()
        desc = str(
            raw.get("desc")
            or raw.get("description")
            or raw.get("summaryDescription")
            or ""
        ).strip()

        prev = aug_by_cid.get(cid)
        if prev is None:
            aug_by_cid[cid] = CatalogAugment(
                canonical_id=cid,
                modes=modes,
                rarity=rarity,
                names={"en_us": en_name, "pt_br": pt_name},
                description=desc,
                rework_patch=raw.get("rework_patch"),
                mechanic_tags=list(raw.get("mechanic_tags", [])),
                crit_chance=float(raw.get("crit_chance", 0.0)),
                item_haste=float(raw.get("item_haste", 0.0)),
            )
        else:
            merged_modes: list[GameMode] = sorted(set(prev.modes) | set(modes))  # type: ignore[assignment]
            merged_desc = prev.description if len(prev.description) >= len(desc) else desc
            merged_tags = sorted(set(prev.mechanic_tags) | set(raw.get("mechanic_tags", [])))
            aug_by_cid[cid] = prev.model_copy(
                update={
                    "modes": merged_modes,
                    "description": merged_desc,
                    "mechanic_tags": merged_tags,
                    "crit_chance": max(prev.crit_chance, float(raw.get("crit_chance", 0.0))),
                    "item_haste": max(prev.item_haste, float(raw.get("item_haste", 0.0))),
                }
            )

    # Deduplicate items by canonical_id, merging modes & preferring active/legendary definitions
    item_by_cid: dict[str, CatalogItem] = {}
    for raw in default_payload["items"]:
        raw_key = str(raw.get("id"))
        en_name = str(raw.get("name") or raw_key).strip()
        if not en_name:
            continue
        pt_row = pt_item_by_id.get(raw_key, {})
        pt_name = str(pt_row.get("name") or en_name).strip() or en_name
        cid = str(raw.get("canonical_id") or canonicalize_slug(en_name))
        if not cid:
            continue
        modes = sorted(set(raw.get("modes") or ["aram_mayhem", "arena"]))
        cand = CatalogItem(
            canonical_id=cid,
            modes=modes,  # type: ignore[arg-type]
            names={"en_us": en_name, "pt_br": pt_name},
            is_legendary=bool(raw.get("is_legendary", False)),
            is_unique=bool(raw.get("is_unique", True)),
            unique_group=raw.get("unique_group"),
            has_item_cooldown=bool(raw.get("has_item_cooldown", False)),
            base_cooldown_s=raw.get("base_cooldown_s"),
            cooldown_leverage_weight=float(raw.get("cooldown_leverage_weight", 0.0)),
            crit_chance=float(raw.get("crit_chance", 0.0)),
            stat_scaling_tier=float(raw.get("stat_scaling_tier", 0.5)),
            transformed_from=raw.get("transformed_from"),
            rework_patch=raw.get("rework_patch"),
            mechanic_tags=list(raw.get("mechanic_tags", [])),
        )
        prev_i = item_by_cid.get(cid)
        if prev_i is None:
            item_by_cid[cid] = cand
        else:
            merged_modes_i: list[GameMode] = sorted(set(prev_i.modes) | set(cand.modes))  # type: ignore[assignment]
            item_by_cid[cid] = prev_i.model_copy(
                update={
                    "modes": merged_modes_i,
                    "is_legendary": prev_i.is_legendary or cand.is_legendary,
                    "has_item_cooldown": prev_i.has_item_cooldown or cand.has_item_cooldown,
                    "base_cooldown_s": prev_i.base_cooldown_s or cand.base_cooldown_s,
                    "cooldown_leverage_weight": max(
                        prev_i.cooldown_leverage_weight, cand.cooldown_leverage_weight
                    ),
                    "crit_chance": max(prev_i.crit_chance, cand.crit_chance),
                    "mechanic_tags": sorted(set(prev_i.mechanic_tags) | set(cand.mechanic_tags)),
                }
            )

    champ_by_cid: dict[str, CatalogChampion] = {}
    for raw in default_payload["champions"]:
        raw_key = str(raw.get("id"))
        en_name = str(raw.get("name") or raw_key).strip()
        if not en_name:
            continue
        pt_row = pt_champ_by_id.get(raw_key, {})
        pt_name = str(pt_row.get("name") or en_name).strip() or en_name
        cid = str(raw.get("canonical_id") or canonicalize_slug(str(raw.get("alias") or en_name)))
        if not cid or cid in champ_by_cid:
            continue
        modes = sorted(set(raw.get("modes") or ["aram_mayhem", "arena"]))
        champ_by_cid[cid] = CatalogChampion(
            canonical_id=cid,
            modes=modes,  # type: ignore[arg-type]
            names={"en_us": en_name, "pt_br": pt_name},
            rework_patch=raw.get("rework_patch"),
            mechanic_tags=list(raw.get("mechanic_tags", [])),
        )

    return CatalogBundle(
        catalog_patch=patch,
        patch_calendar=patch_calendar,
        mode_rework_patches=mode_rework,
        augments=list(aug_by_cid.values()),
        items=list(item_by_cid.values()),
        champions=list(champ_by_cid.values()),
    )


def fetch_live_cdragon_catalog(timeout_s: float = 15.0) -> CatalogBundle:
    """Fetch live CommunityDragon JSON endpoints for `default` (en_us) and `pt_br`."""
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        meta_resp = client.get(CDRAGON_METADATA_URL)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        patch, patch_calendar = build_patch_calendar_from_version(str(meta["version"]))

        def _fetch_locale(public_locale: PublicLocale) -> dict[str, Any]:
            cd_loc = CDRAGON_LOCALE_MAP[public_locale]
            base = f"{CDRAGON_BASE_URL}/{cd_loc}/v1"
            aug_resp = client.get(f"{base}/cherry-augments.json")
            aug_resp.raise_for_status()
            item_resp = client.get(f"{base}/items.json")
            item_resp.raise_for_status()
            champ_resp = client.get(f"{base}/champion-summary.json")
            champ_resp.raise_for_status()
            augments = aug_resp.json()
            items = [i for i in item_resp.json() if i.get("inStore", True)]
            champions = [c for c in champ_resp.json() if c.get("id", -1) > 0]
            return {
                "patch": patch,
                "patch_calendar": patch_calendar,
                "augments": augments,
                "items": items,
                "champions": champions,
            }

        return parse_cdragon_payloads(
            default_payload=_fetch_locale("en_us"),
            pt_br_payload=_fetch_locale("pt_br"),
        )
