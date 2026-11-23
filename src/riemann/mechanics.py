from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field, model_validator

from riemann.models import (
    AugmentBindings,
    CatalogAugment,
    CatalogChampion,
    CatalogItem,
    ConverterSpec,
    GameMode,
    ReasonItem,
)

EdgeType = Literal[
    "augment_pair_synergy",
    "context_to_augment",
    "augment_item_synergy",
    "item_pair_synergy",
    "amplifies_locked_legendary",
]


class GraphEdge(BaseModel):
    edge_id: str
    edge_type: EdgeType
    source_tag: str | None = None
    target_tag: str | None = None
    source_entity_id: str | None = None
    target_entity_id: str | None = None
    weight: float
    symmetric: bool = False
    modes: list[GameMode] = Field(default_factory=lambda: ["aram_mayhem", "arena"])
    reason: str = ""

    @model_validator(mode="after")
    def validate_endpoints(self) -> GraphEdge:
        if self.source_tag is None and self.source_entity_id is None:
            raise ValueError(f"Edge {self.edge_id} requires source_tag or source_entity_id")
        if self.target_tag is None and self.target_entity_id is None:
            raise ValueError(f"Edge {self.edge_id} requires target_tag or target_entity_id")
        return self


class AugmentGraphMeta(BaseModel):
    canonical_id: str
    modes: list[GameMode] = Field(default_factory=lambda: ["aram_mayhem", "arena"])
    mechanic_tags: list[str] = Field(default_factory=list)
    crit_chance: float = Field(default=0.0, ge=0.0)
    item_haste: float = Field(default=0.0, ge=0.0)
    converter_specs: list[ConverterSpec] = Field(default_factory=list)
    reference_prior_logit: float = 0.0


class ItemGraphMeta(BaseModel):
    canonical_id: str
    modes: list[GameMode] = Field(default_factory=lambda: ["aram_mayhem", "arena"])
    is_legendary: bool = True
    is_unique: bool = True
    unique_group: str | None = None
    has_item_cooldown: bool = False
    base_cooldown_s: float | None = Field(default=None, gt=0.0)
    cooldown_leverage_weight: float = Field(default=0.0, ge=0.0)
    crit_chance: float = Field(default=0.0, ge=0.0)
    stat_scaling_tier: float = Field(default=0.5, ge=0.0, le=1.0)
    transformed_from: str | None = None
    mechanic_tags: list[str] = Field(default_factory=list)
    reference_prior_logit: float = 0.0


class ChampionGraphMeta(BaseModel):
    canonical_id: str
    mechanic_tags: list[str] = Field(default_factory=list)
    converter_specs: list[ConverterSpec] = Field(default_factory=list)
    stat_scaling_overrides: dict[str, float] = Field(default_factory=dict)


class RelationGraph(BaseModel):
    graph_version: str
    augments: dict[str, AugmentGraphMeta] = Field(default_factory=dict)
    items: dict[str, ItemGraphMeta] = Field(default_factory=dict)
    champions: dict[str, ChampionGraphMeta] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> RelationGraph:
        # Validate acyclic transformed_from DAG
        for item_id, item in self.items.items():
            visited = {item_id}
            curr = item.transformed_from
            while curr is not None:
                if curr in visited:
                    raise ValueError(f"Cycle detected in transformed_from DAG involving {item_id}")
                visited.add(curr)
                parent = self.items.get(curr)
                curr = parent.transformed_from if parent is not None else None

        # Validate explicit entity references on edges
        all_entities = set(self.augments) | set(self.items) | set(self.champions)
        for edge in self.edges:
            if edge.source_entity_id is not None and edge.source_entity_id not in all_entities:
                raise ValueError(
                    f"Edge {edge.edge_id} references unknown source_entity_id {edge.source_entity_id!r}"
                )
            if edge.target_entity_id is not None and edge.target_entity_id not in all_entities:
                raise ValueError(
                    f"Edge {edge.edge_id} references unknown target_entity_id {edge.target_entity_id!r}"
                )
        return self


def load_relation_graph(path: Path) -> RelationGraph:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RelationGraph.model_validate(raw)


# --- Deterministic CommunityDragon Tag + Numeric Value Extraction Fallback ---

_ITEM_HASTE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*item\s*haste", re.IGNORECASE)
_CRIT_CHANCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(?:critical\s*strike\s*chance|crit\s*chance)", re.IGNORECASE
)
_ABILITY_HASTE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ability\s*haste", re.IGNORECASE)
_ON_HIT_SPELL_RE = re.compile(r"abilities\s+apply\s+on[\-\s]?hit", re.IGNORECASE)
_ON_HIT_RE = re.compile(r"on[\-\s]?hit", re.IGNORECASE)
_SCALES_CRIT_RE = re.compile(
    r"(?:abilities|damage\s+over\s+time|heals?)\s+can\s+critically\s+strike", re.IGNORECASE
)
_BURN_RE = re.compile(r"\b(?:burn|burns|damage\s+over\s+time|ignite)\b", re.IGNORECASE)
_HEAL_SHIELD_RE = re.compile(r"\b(?:heal\s+and\s+shield|healing|shielding)\b", re.IGNORECASE)
_FIRST_LEGENDARY_RE = re.compile(r"first\s+legendary\s+item", re.IGNORECASE)
_HP_STACK_RE = re.compile(r"(?:max(?:imum)?\s+health|bonus\s+health)", re.IGNORECASE)


def extract_fallback_augment_mechanics(aug: CatalogAugment) -> AugmentGraphMeta:
    """Deterministic tag + numeric value extraction from CommunityDragon descriptions."""
    desc = aug.description or ""
    tags = set(aug.mechanic_tags)
    item_haste = aug.item_haste
    crit_chance = aug.crit_chance

    m_haste = _ITEM_HASTE_RE.search(desc)
    if m_haste:
        tags.add("grants_item_haste")
        item_haste = max(item_haste, float(m_haste.group(1)))

    m_crit = _CRIT_CHANCE_RE.search(desc)
    if m_crit:
        tags.add("grants_crit_chance")
        crit_chance = max(crit_chance, float(m_crit.group(1)) / 100.0)

    if _SCALES_CRIT_RE.search(desc):
        tags.add("scales_with_crit")
    if _ON_HIT_SPELL_RE.search(desc):
        tags.add("applies_on_hit_on_spell")
    elif _ON_HIT_RE.search(desc):
        tags.add("grants_on_hit")
    if _ABILITY_HASTE_RE.search(desc):
        tags.add("ability_haste")
    if _BURN_RE.search(desc):
        tags.add("burn_dot")
    if _HEAL_SHIELD_RE.search(desc):
        tags.add("heal_shield")
    if _FIRST_LEGENDARY_RE.search(desc):
        tags.add("empowers_first_legendary")
    if _HP_STACK_RE.search(desc):
        tags.add("hp_stacking")

    return AugmentGraphMeta(
        canonical_id=aug.canonical_id,
        modes=list(aug.modes),
        mechanic_tags=sorted(tags),
        crit_chance=crit_chance,
        item_haste=item_haste,
        converter_specs=list(aug.converter_specs),
        reference_prior_logit=0.0,
    )


def effective_augment_meta(
    aug: CatalogAugment, graph: RelationGraph
) -> AugmentGraphMeta:
    gmeta = graph.augments.get(aug.canonical_id)
    if gmeta is not None:
        tags = sorted(set(gmeta.mechanic_tags) | set(aug.mechanic_tags))
        return AugmentGraphMeta(
            canonical_id=aug.canonical_id,
            modes=list(aug.modes),
            mechanic_tags=tags,
            crit_chance=max(gmeta.crit_chance, aug.crit_chance),
            item_haste=max(gmeta.item_haste, aug.item_haste),
            converter_specs=list(gmeta.converter_specs) or list(aug.converter_specs),
            reference_prior_logit=gmeta.reference_prior_logit,
        )
    return extract_fallback_augment_mechanics(aug)


def effective_item_meta(item: CatalogItem, graph: RelationGraph) -> ItemGraphMeta:
    gmeta = graph.items.get(item.canonical_id)
    if gmeta is not None:
        tags = sorted(set(gmeta.mechanic_tags) | set(item.mechanic_tags))
        return ItemGraphMeta(
            canonical_id=item.canonical_id,
            modes=list(item.modes),
            is_legendary=gmeta.is_legendary or item.is_legendary,
            is_unique=item.is_unique and gmeta.is_unique,
            unique_group=item.unique_group or gmeta.unique_group,
            has_item_cooldown=gmeta.has_item_cooldown or item.has_item_cooldown,
            base_cooldown_s=item.base_cooldown_s or gmeta.base_cooldown_s,
            cooldown_leverage_weight=(
                item.cooldown_leverage_weight
                if item.cooldown_leverage_weight > 0.0
                else gmeta.cooldown_leverage_weight
            ),
            crit_chance=max(gmeta.crit_chance, item.crit_chance),
            stat_scaling_tier=(
                item.stat_scaling_tier if item.stat_scaling_tier != 0.5 else gmeta.stat_scaling_tier
            ),
            transformed_from=item.transformed_from or gmeta.transformed_from,
            mechanic_tags=tags,
            reference_prior_logit=gmeta.reference_prior_logit,
        )
    return ItemGraphMeta(
        canonical_id=item.canonical_id,
        modes=list(item.modes),
        is_legendary=item.is_legendary,
        is_unique=item.is_unique,
        unique_group=item.unique_group,
        has_item_cooldown=item.has_item_cooldown,
        base_cooldown_s=item.base_cooldown_s,
        cooldown_leverage_weight=item.cooldown_leverage_weight,
        crit_chance=item.crit_chance,
        stat_scaling_tier=item.stat_scaling_tier,
        transformed_from=item.transformed_from,
        mechanic_tags=list(item.mechanic_tags),
        reference_prior_logit=0.0,
    )


def effective_champion_meta(
    champ: CatalogChampion, graph: RelationGraph
) -> ChampionGraphMeta:
    gmeta = graph.champions.get(champ.canonical_id)
    if gmeta is not None:
        return ChampionGraphMeta(
            canonical_id=champ.canonical_id,
            mechanic_tags=sorted(set(gmeta.mechanic_tags) | set(champ.mechanic_tags)),
            converter_specs=list(gmeta.converter_specs) or list(champ.converter_specs),
            stat_scaling_overrides={
                **gmeta.stat_scaling_overrides,
                **champ.stat_scaling_overrides,
            },
        )
    return ChampionGraphMeta(
        canonical_id=champ.canonical_id,
        mechanic_tags=list(champ.mechanic_tags),
        converter_specs=list(champ.converter_specs),
        stat_scaling_overrides=dict(champ.stat_scaling_overrides),
    )


# --- Exact Physical Mechanics Calculators ---


@dataclass(frozen=True)
class HastePhysics:
    base_cooldown_s: float | None
    effective_cooldown_s: float | None
    ideal_activation_rate_hz: float | None
    fractional_cdr: float
    haste_logit: float


def compute_item_haste_physics(
    item: ItemGraphMeta, total_item_haste: float
) -> HastePhysics:
    h = max(0.0, total_item_haste)
    cdr = h / (100.0 + h) if h > 0.0 else 0.0
    if not item.has_item_cooldown:
        return HastePhysics(
            base_cooldown_s=item.base_cooldown_s,
            effective_cooldown_s=item.base_cooldown_s,
            ideal_activation_rate_hz=(
                1.0 / item.base_cooldown_s if item.base_cooldown_s is not None else None
            ),
            fractional_cdr=0.0,
            haste_logit=0.0,
        )
    eff_cd = (
        item.base_cooldown_s / (1.0 + h / 100.0)
        if item.base_cooldown_s is not None
        else None
    )
    rate_hz = (1.0 / eff_cd) if eff_cd is not None else None
    haste_logit = item.cooldown_leverage_weight * cdr
    return HastePhysics(
        base_cooldown_s=item.base_cooldown_s,
        effective_cooldown_s=eff_cd,
        ideal_activation_rate_hz=rate_hz,
        fractional_cdr=cdr,
        haste_logit=haste_logit,
    )


@dataclass(frozen=True)
class CritPhysics:
    usable_crit_added: float
    wasted_crit_added: float
    converted_overflow_stats: dict[str, float]
    overcap_penalty_logit: float


def compute_crit_physics(
    *,
    crit_before: float,
    crit_added: float,
    converters_before: Sequence[ConverterSpec],
    converters_added: Sequence[ConverterSpec],
    k_crit: float,
) -> CritPhysics:
    crit_after = crit_before + crit_added
    all_after = list(converters_before) + list(converters_added)

    if not all_after:
        usable = min(1.0, crit_after) - min(1.0, crit_before)
        wasted = max(0.0, crit_after - 1.0) - max(0.0, crit_before - 1.0)
        return CritPhysics(
            usable_crit_added=usable,
            wasted_crit_added=wasted,
            converted_overflow_stats={},
            overcap_penalty_logit=k_crit * wasted,
        )

    usable = min(1.0, crit_after) - min(1.0, crit_before)
    before_rates: dict[str, float] = {}
    for spec in converters_before:
        before_rates[spec.stat_name] = before_rates.get(spec.stat_name, 0.0) + spec.units_per_crit_fraction
    after_rates: dict[str, float] = {}
    for spec in all_after:
        after_rates[spec.stat_name] = after_rates.get(spec.stat_name, 0.0) + spec.units_per_crit_fraction

    overflow_before = max(0.0, crit_before - 1.0)
    overflow_after = max(0.0, crit_after - 1.0)
    converted: dict[str, float] = {}
    for stat, r_after in after_rates.items():
        r_before = before_rates.get(stat, 0.0)
        delta_stat = overflow_after * r_after - overflow_before * r_before
        if abs(delta_stat) > 1e-12:
            converted[stat] = round(delta_stat, 9)

    return CritPhysics(
        usable_crit_added=usable,
        wasted_crit_added=0.0,
        converted_overflow_stats=converted,
        overcap_penalty_logit=0.0,
    )


@dataclass(frozen=True)
class WeaponDestinyResolution:
    status: Literal["inactive", "open", "locked", "ambiguous", "transform_ambiguous", "unknown"]
    bound_item_id: str | None = None
    warning: str | None = None


def _is_descendant_in_transform_dag(
    candidate_item_id: str,
    ancestor_id: str,
    items_by_id: dict[str, ItemGraphMeta],
) -> bool:
    curr = items_by_id.get(candidate_item_id)
    visited = {candidate_item_id}
    while curr is not None and curr.transformed_from is not None:
        parent_id = curr.transformed_from
        if parent_id == ancestor_id:
            return True
        if parent_id in visited:
            return False
        visited.add(parent_id)
        curr = items_by_id.get(parent_id)
    return False


def resolve_weapon_of_destiny_state(
    *,
    has_wod_augment: bool,
    current_items: Sequence[str],
    items_by_id: dict[str, ItemGraphMeta],
    bindings: AugmentBindings | None,
    slot_index: int | None,
) -> WeaponDestinyResolution:
    if not has_wod_augment:
        return WeaponDestinyResolution(status="inactive")

    state = bindings.weapon_of_destiny_state if bindings is not None else "auto"
    bound_id = bindings.weapon_of_destiny_item_id if bindings is not None else None

    if state == "sold_or_unknown":
        return WeaponDestinyResolution(
            status="unknown", warning="weapon_of_destiny_state_unknown"
        )

    if bound_id is not None:
        if bound_id in current_items:
            return WeaponDestinyResolution(status="locked", bound_item_id=bound_id)
        descendants = [
            iid
            for iid in dict.fromkeys(current_items)
            if _is_descendant_in_transform_dag(iid, bound_id, items_by_id)
        ]
        if len(descendants) == 1:
            return WeaponDestinyResolution(status="locked", bound_item_id=descendants[0])
        if len(descendants) >= 2:
            return WeaponDestinyResolution(
                status="transform_ambiguous", warning="weapon_of_destiny_transform_ambiguous"
            )
        return WeaponDestinyResolution(
            status="unknown", warning="weapon_of_destiny_state_unknown"
        )

    owned_legendaries = [
        iid
        for iid in dict.fromkeys(current_items)
        if iid in items_by_id and items_by_id[iid].is_legendary
    ]
    if len(owned_legendaries) == 1:
        return WeaponDestinyResolution(status="locked", bound_item_id=owned_legendaries[0])
    if len(owned_legendaries) >= 2:
        return WeaponDestinyResolution(
            status="ambiguous", warning="weapon_of_destiny_binding_ambiguous"
        )

    # 0 owned Legendaries and no explicit bound_id
    if state == "locked":
        return WeaponDestinyResolution(
            status="unknown", warning="weapon_of_destiny_state_unknown"
        )
    if state == "auto" and slot_index is not None and slot_index >= 2:
        return WeaponDestinyResolution(
            status="unknown", warning="weapon_of_destiny_state_unknown"
        )
    return WeaponDestinyResolution(status="open")


def _entity_matches(
    entity_id: str,
    entity_tags: set[str],
    required_entity: str | None,
    required_tag: str | None,
) -> bool:
    if required_entity is not None and entity_id != required_entity:
        return False
    if required_tag is not None and required_tag not in entity_tags:
        return False
    return True


def evaluate_augment_pair_edges(
    *,
    mode: GameMode,
    chosen_augments: Sequence[AugmentGraphMeta],
    candidate: AugmentGraphMeta,
    graph: RelationGraph,
) -> tuple[float, list[ReasonItem]]:
    """Evaluate augment_pair_synergy edges with canonical unordered deduplication for symmetric edges."""
    total = 0.0
    reasons: list[ReasonItem] = []
    seen_keys: set[tuple[str, str, str]] = set()
    cand_tags = set(candidate.mechanic_tags)

    for edge in graph.edges:
        if edge.edge_type != "augment_pair_synergy" or mode not in edge.modes:
            continue
        for chosen in chosen_augments:
            chosen_tags = set(chosen.mechanic_tags)
            forward = _entity_matches(
                chosen.canonical_id, chosen_tags, edge.source_entity_id, edge.source_tag
            ) and _entity_matches(
                candidate.canonical_id, cand_tags, edge.target_entity_id, edge.target_tag
            )
            reverse = edge.symmetric and _entity_matches(
                candidate.canonical_id, cand_tags, edge.source_entity_id, edge.source_tag
            ) and _entity_matches(
                chosen.canonical_id, chosen_tags, edge.target_entity_id, edge.target_tag
            )
            if not (forward or reverse):
                continue

            if edge.symmetric:
                a_id, b_id = sorted((chosen.canonical_id, candidate.canonical_id))
                key = (edge.edge_id, a_id, b_id)
            else:
                key = (edge.edge_id, chosen.canonical_id, candidate.canonical_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            total += edge.weight
            reasons.append(
                ReasonItem(
                    code=edge.edge_id,
                    message=edge.reason or f"Synergy with {chosen.canonical_id}",
                    delta_logit=round(edge.weight, 9),
                )
            )
    return total, reasons


def evaluate_context_to_augment_edges(
    *,
    mode: GameMode,
    current_items: Sequence[ItemGraphMeta],
    candidate: AugmentGraphMeta,
    graph: RelationGraph,
) -> tuple[float, list[ReasonItem]]:
    """Evaluate state-dependent item -> augment edges (champion-only fit belongs in S_ref prior)."""
    total = 0.0
    reasons: list[ReasonItem] = []
    seen_keys: set[tuple[str, str, str]] = set()
    cand_tags = set(candidate.mechanic_tags)

    for edge in graph.edges:
        if edge.edge_type != "context_to_augment" or mode not in edge.modes:
            continue
        for item in current_items:
            if _entity_matches(
                item.canonical_id, set(item.mechanic_tags), edge.source_entity_id, edge.source_tag
            ) and _entity_matches(
                candidate.canonical_id, cand_tags, edge.target_entity_id, edge.target_tag
            ):
                key = (edge.edge_id, item.canonical_id, candidate.canonical_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                total += edge.weight
                reasons.append(
                    ReasonItem(
                        code=edge.edge_id,
                        message=edge.reason or f"Item synergy with {item.canonical_id}",
                        delta_logit=round(edge.weight, 9),
                    )
                )
    return total, reasons


def evaluate_general_item_edges(
    *,
    mode: GameMode,
    chosen_augments: Sequence[AugmentGraphMeta],
    current_items: Sequence[ItemGraphMeta],
    candidate_item: ItemGraphMeta,
    graph: RelationGraph,
    indexed_edges: tuple[list[GraphEdge], list[GraphEdge]] | None = None,
) -> tuple[float, list[ReasonItem]]:
    """Evaluate augment_item_synergy and item_pair_synergy only (strictly excludes amplifies_locked_legendary)."""
    total = 0.0
    reasons: list[ReasonItem] = []
    seen_keys: set[tuple[str, str, str]] = set()
    cand_id = candidate_item.canonical_id
    cand_tags = set(candidate_item.mechanic_tags)

    if indexed_edges is not None:
        aug_edges, item_edges = indexed_edges
    else:
        aug_edges = [e for e in graph.edges if e.edge_type == "augment_item_synergy" and mode in e.modes]
        item_edges = [e for e in graph.edges if e.edge_type == "item_pair_synergy" and mode in e.modes]

    for edge in aug_edges:
        if edge.target_entity_id is not None and edge.target_entity_id != cand_id:
            continue
        if edge.target_tag is not None and edge.target_tag not in cand_tags:
            continue
        for aug in chosen_augments:
            if edge.source_entity_id is not None and edge.source_entity_id != aug.canonical_id:
                continue
            if edge.source_tag is not None and edge.source_tag not in aug.mechanic_tags:
                continue
            key = (edge.edge_id, aug.canonical_id, cand_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            total += edge.weight
            reasons.append(
                ReasonItem(
                    code=edge.edge_id,
                    message=edge.reason or f"Synergy with augment {aug.canonical_id}",
                    delta_logit=round(edge.weight, 9),
                )
            )

    for edge in item_edges:
        if edge.target_entity_id is not None and edge.target_entity_id != cand_id:
            continue
        if edge.target_tag is not None and edge.target_tag not in cand_tags:
            continue
        for item in current_items:
            if edge.source_entity_id is not None and edge.source_entity_id != item.canonical_id:
                continue
            if edge.source_tag is not None and edge.source_tag not in item.mechanic_tags:
                continue
            key = (edge.edge_id, item.canonical_id, cand_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            total += edge.weight
            reasons.append(
                ReasonItem(
                    code=edge.edge_id,
                    message=edge.reason or f"Item synergy with {item.canonical_id}",
                    delta_logit=round(edge.weight, 9),
                )
            )
    return total, reasons


def evaluate_locked_legendary_amplification(
    *,
    mode: GameMode,
    locked_item: ItemGraphMeta,
    candidate_item: ItemGraphMeta,
    graph: RelationGraph,
) -> tuple[float, list[ReasonItem]]:
    """Evaluate amplifies_locked_legendary edges ONLY (strictly partitioned from item_graph_logit)."""
    total = 0.0
    reasons: list[ReasonItem] = []
    seen_keys: set[tuple[str, str, str]] = set()
    cand_tags = set(candidate_item.mechanic_tags)
    locked_tags = set(locked_item.mechanic_tags)

    for edge in graph.edges:
        if edge.edge_type != "amplifies_locked_legendary" or mode not in edge.modes:
            continue
        if _entity_matches(
            candidate_item.canonical_id, cand_tags, edge.source_entity_id, edge.source_tag
        ) and _entity_matches(
            locked_item.canonical_id, locked_tags, edge.target_entity_id, edge.target_tag
        ):
            key = (edge.edge_id, candidate_item.canonical_id, locked_item.canonical_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            total += edge.weight
            reasons.append(
                ReasonItem(
                    code="weapon_of_destiny_locked_synergy",
                    message=edge.reason
                    or f"Amplifies Weapon of Destiny ({locked_item.canonical_id})",
                    delta_logit=round(edge.weight, 9),
                )
            )
    return total, reasons


def compute_reference_prior_mean(
    *,
    mode: GameMode,
    champion: ChampionGraphMeta,
    entity_type: Literal["augment", "item"],
    entity_meta: AugmentGraphMeta | ItemGraphMeta,
    graph: RelationGraph,
) -> float:
    """Compute h(S_ref, e): baseline reference fit + champion->entity static edges (no state)."""
    base = entity_meta.reference_prior_logit
    if entity_type == "augment":
        # Check champion-only context_to_augment edges
        champ_tags = set(champion.mechanic_tags)
        ent_tags = set(entity_meta.mechanic_tags)
        seen: set[tuple[str, str, str]] = set()
        for edge in graph.edges:
            if edge.edge_type != "context_to_augment" or mode not in edge.modes:
                continue
            if _entity_matches(
                champion.canonical_id, champ_tags, edge.source_entity_id, edge.source_tag
            ) and _entity_matches(
                entity_meta.canonical_id, ent_tags, edge.target_entity_id, edge.target_tag
            ):
                key = (edge.edge_id, champion.canonical_id, entity_meta.canonical_id)
                if key not in seen:
                    seen.add(key)
                    base += edge.weight
    return base
