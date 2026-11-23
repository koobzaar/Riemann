from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GameMode = Literal["aram_mayhem", "arena"]
PublicLocale = Literal["en_us", "pt_br"]
RequestLocale = Literal["en_us", "pt_br", "auto"]
EntityType = Literal["augment", "item", "champion"]
MetricType = Literal["win", "top4", "top1", "avg_placement"]
WeaponDestinyState = Literal["auto", "open", "locked", "sold_or_unknown"]


class NormalizedError(BaseModel):
    code: str
    detail: str
    field: str | None = None
    active_snapshot_preserved: bool | None = None


class ConverterSpec(BaseModel):
    stat_name: str
    units_per_crit_fraction: float = Field(gt=0.0)


class CatalogAugment(BaseModel):
    canonical_id: str
    modes: list[GameMode]
    rarity: str = "unknown"
    names: dict[PublicLocale, str]
    description: str = ""
    rework_patch: str | None = None
    mechanic_tags: list[str] = Field(default_factory=list)
    crit_chance: float = Field(default=0.0, ge=0.0)
    item_haste: float = Field(default=0.0, ge=0.0)
    converter_specs: list[ConverterSpec] = Field(default_factory=list)


class CatalogItem(BaseModel):
    canonical_id: str
    modes: list[GameMode]
    names: dict[PublicLocale, str]
    is_legendary: bool = False
    is_unique: bool = True
    unique_group: str | None = None
    has_item_cooldown: bool = False
    base_cooldown_s: float | None = Field(default=None, gt=0.0)
    cooldown_leverage_weight: float = Field(default=0.0, ge=0.0)
    crit_chance: float = Field(default=0.0, ge=0.0)
    stat_scaling_tier: float = Field(default=0.5, ge=0.0, le=1.0)
    transformed_from: str | None = None
    rework_patch: str | None = None
    mechanic_tags: list[str] = Field(default_factory=list)


class CatalogChampion(BaseModel):
    canonical_id: str
    modes: list[GameMode]
    names: dict[PublicLocale, str]
    rework_patch: str | None = None
    mechanic_tags: list[str] = Field(default_factory=list)
    converter_specs: list[ConverterSpec] = Field(default_factory=list)
    stat_scaling_overrides: dict[str, float] = Field(default_factory=dict)


class CatalogBundle(BaseModel):
    catalog_patch: str
    patch_calendar: list[str]
    mode_rework_patches: dict[GameMode, str | None] = Field(
        default_factory=lambda: {"aram_mayhem": None, "arena": None}
    )
    augments: list[CatalogAugment]
    items: list[CatalogItem]
    champions: list[CatalogChampion]

    @model_validator(mode="after")
    def validate_non_empty_catalog(self) -> CatalogBundle:
        if not self.catalog_patch or not self.patch_calendar:
            raise ValueError("CatalogBundle requires non-empty catalog_patch and patch_calendar")
        if not self.augments or not self.items or not self.champions:
            raise ValueError(
                "CatalogBundle requires non-empty augments, items, and champions"
            )
        return self


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    cohort_id: str = "default"
    overlap_group: str = "riot_global_pool"
    source_priority: int = 100
    is_disjoint_verified: bool = False
    mode: GameMode
    metric_type: MetricType
    patch: str
    champion_id: str
    entity_type: Literal["augment", "item"]
    entity_id: str
    slot_key: str = "all"

    # Binary fields
    games: int | None = None
    wins: int | None = None
    rate_str: str | None = None
    rate_decimals: int | None = None
    rounding_mode: Literal["ROUND_HALF_UP", "ROUND_HALF_EVEN"] = "ROUND_HALF_UP"
    baseline_wins: int | None = None
    baseline_games: int | None = None
    baseline_rate: float | None = None
    entity_equals_reference_cohort: bool = False
    champion_stratum_games: int | None = None
    pick_rate: float | None = None

    # Continuous Arena placement fields
    mean_placement: float | None = None
    baseline_mean_placement: float | None = None

    # Non-statistical metadata (never enters likelihood)
    curated_tier: str | None = None

    def primary_key(self) -> tuple[str, str, str, str, str, str, str, str, str]:
        return (
            self.source_id,
            self.cohort_id,
            self.mode,
            self.metric_type,
            self.patch,
            self.champion_id,
            self.entity_type,
            self.entity_id,
            self.slot_key,
        )


class AdapterPayload(BaseModel):
    schema_version: Literal["v1"] = "v1"
    adapter_id: str
    catalog_patch_hint: str | None = None
    observations: list[Observation] = Field(default_factory=list)
    adapter_metadata: dict[str, Any] = Field(default_factory=dict)


class AdapterRunReport(BaseModel):
    name: str
    status: Literal["ok", "error", "skipped"]
    last_sync_utc: str
    rows_loaded: int = 0
    rejected_rows: int = 0
    error: str | None = None


class CorroborationInfo(BaseModel):
    secondary_sources_checked: int = 0
    max_source_disagreement_logit: float | None = None
    status: Literal["no_secondary", "aligned", "divergent"] = "no_secondary"


class ArenaSecondary(BaseModel):
    top1_rate: float | None = None
    avg_placement: float | None = None


class PrecomputedPosterior(BaseModel):
    mode: GameMode
    champion_id: str
    entity_type: Literal["augment", "item"]
    entity_id: str
    slot_key: str
    prior_mean_logit: float
    association_mean_logit: float
    association_ci_90_logit: tuple[float, float]
    standardized_wr_delta: float
    effective_sample_size: float
    pick_rate: float | None = None
    min_baseline_games: int | None = None
    low_baseline_sample: bool = False
    corroboration: CorroborationInfo = Field(default_factory=CorroborationInfo)
    arena_secondary: ArenaSecondary | None = None


# --- HTTP Request / Response Models ---


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded_empty_catalog", "degraded_cache_reload_required"]
    snapshot_id: str | None = None
    catalog_patch: str | None = None
    modes: list[GameMode] = ["aram_mayhem", "arena"]


class ModeCounts(BaseModel):
    augments: int = 0
    items: int = 0
    champions: int = 0
    stats_rows: int = 0
    posteriors: int = 0


class CatalogStatusResponse(BaseModel):
    snapshot_id: str | None = None
    catalog_patch: str | None = None
    graph_version: str | None = None
    counts: dict[GameMode, ModeCounts]
    locales: list[PublicLocale] = ["en_us", "pt_br"]
    adapters: list[AdapterRunReport] = Field(default_factory=list)


class SyncRequest(BaseModel):
    adapters: list[str] | None = None
    catalog_source: Literal["cdragon", "fixture"] = "fixture"
    fixture_profile: str = "default"


class SyncResponse(BaseModel):
    snapshot_id: str
    catalog_patch: str
    active_snapshot_preserved: bool = False
    adapters: list[AdapterRunReport]


class ResolveAugmentRequest(BaseModel):
    text: str
    locale: RequestLocale = "auto"
    mode: GameMode


class ResolveCandidate(BaseModel):
    canonical_id: str
    display_name: str
    locale: PublicLocale
    confidence: float


class ResolveAugmentResponse(BaseModel):
    resolved: bool
    canonical_id: str | None = None
    display_name: str | None = None
    detected_locale: PublicLocale | None = None
    confidence: float
    reason: Literal["matched", "below_confidence_threshold", "ambiguous_match"]
    candidates: list[ResolveCandidate] = Field(default_factory=list)


class AugmentBindings(BaseModel):
    weapon_of_destiny_item_id: str | None = None
    weapon_of_destiny_state: WeaponDestinyState = "auto"

    @model_validator(mode="after")
    def validate_no_contradiction(self) -> AugmentBindings:
        if self.weapon_of_destiny_state == "open" and self.weapon_of_destiny_item_id is not None:
            raise ValueError("weapon_of_destiny_state='open' contradicts weapon_of_destiny_item_id")
        return self


class ReasonItem(BaseModel):
    code: str
    message: str
    delta_logit: float


class UnresolvedInput(BaseModel):
    raw_input: str
    reason: Literal["below_confidence_threshold", "ambiguous_match"]
    candidates: list[ResolveCandidate] = Field(default_factory=list)


class ItemPivotSummary(BaseModel):
    item_id: str
    display_name: str
    composite_score: float
    reason_codes: list[str] = Field(default_factory=list)


class AugmentPhysicalMetrics(BaseModel):
    usable_crit_added: float = 0.0
    wasted_crit_added: float = 0.0
    converted_overflow_stats: dict[str, float] = Field(default_factory=dict)


class AugmentBreakdown(BaseModel):
    pair_synergy_logit: float = 0.0
    context_synergy_logit: float = 0.0
    overcap_penalty_delta_logit: float = 0.0


class AugmentRecommendation(BaseModel):
    canonical_id: str
    display_name: str
    composite_score: float
    association_mean_logit: float
    association_ci_90_logit: tuple[float, float]
    standardized_wr_delta: float
    mechanistic_delta_logit: float
    bias_sensitivity_bound: float
    sensitivity_envelope_logit: tuple[float, float]
    mean_bias_margin: float | None = None
    is_bias_robust_on_means: bool | None = None
    envelope_gap: float | None = None
    is_envelope_separated: bool | None = None
    effective_sample_size: float
    corroboration: CorroborationInfo
    arena_secondary: ArenaSecondary | None = None
    physical_metrics: AugmentPhysicalMetrics
    prior_mean_logit: float
    breakdown: AugmentBreakdown
    reasons: list[ReasonItem] = Field(default_factory=list)
    top_item_pivots: list[ItemPivotSummary] = Field(default_factory=list)


class RecommendAugmentsRequest(BaseModel):
    mode: GameMode
    champion_id: str
    slot_index: int | None = Field(default=None, ge=1, le=4)
    chosen_augments: list[str] = Field(default_factory=list)
    current_items: list[str] = Field(default_factory=list, max_length=6)
    offered_augments: list[str]
    observed_crit_chance: float | None = None
    augment_bindings: AugmentBindings | None = None
    locale: RequestLocale = "en_us"
    top_item_pivots_k: int = Field(default=3, ge=0, le=10)

    @field_validator("observed_crit_chance")
    @classmethod
    def check_finite_nonneg_crit(cls, v: float | None) -> float | None:
        if v is not None:
            if not math.isfinite(v) or v < 0.0:
                raise ValueError("observed_crit_chance must be finite and >= 0")
        return v


class RecommendAugmentsResponse(BaseModel):
    snapshot_id: str
    mode: GameMode
    champion_id: str
    selection_bias_warning: bool
    warnings: list[str] = Field(default_factory=list)
    unresolved_inputs: list[UnresolvedInput] = Field(default_factory=list)
    recommendations: list[AugmentRecommendation]


class ItemPhysicalMetrics(BaseModel):
    base_cooldown_s: float | None = None
    effective_cooldown_s: float | None = None
    ideal_activation_rate_hz: float | None = None
    usable_crit_added: float = 0.0
    wasted_crit_added: float = 0.0
    converted_overflow_stats: dict[str, float] = Field(default_factory=dict)
    weapon_of_destiny_bound_item: str | None = None


class ItemBreakdown(BaseModel):
    item_haste_logit: float = 0.0
    weapon_destiny_logit: float = 0.0
    item_graph_logit: float = 0.0
    overcap_penalty_delta_logit: float = 0.0


class ItemRecommendation(BaseModel):
    item_id: str
    display_name: str
    composite_score: float
    association_mean_logit: float
    association_ci_90_logit: tuple[float, float]
    standardized_wr_delta: float
    mechanistic_delta_logit: float
    bias_sensitivity_bound: float
    sensitivity_envelope_logit: tuple[float, float]
    mean_bias_margin: float | None = None
    is_bias_robust_on_means: bool | None = None
    envelope_gap: float | None = None
    is_envelope_separated: bool | None = None
    effective_sample_size: float
    corroboration: CorroborationInfo
    arena_secondary: ArenaSecondary | None = None
    physical_metrics: ItemPhysicalMetrics
    prior_mean_logit: float
    breakdown: ItemBreakdown
    reasons: list[ReasonItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RecommendItemsRequest(BaseModel):
    mode: GameMode
    champion_id: str
    slot_index: int | None = Field(default=None, ge=1, le=6)
    chosen_augments: list[str] = Field(default_factory=list)
    current_items: list[str] = Field(default_factory=list, max_length=6)
    observed_crit_chance: float | None = None
    augment_bindings: AugmentBindings | None = None
    locale: RequestLocale = "en_us"
    top_k: int = Field(default=10, ge=1, le=50)

    @field_validator("observed_crit_chance")
    @classmethod
    def check_finite_nonneg_crit(cls, v: float | None) -> float | None:
        if v is not None:
            if not math.isfinite(v) or v < 0.0:
                raise ValueError("observed_crit_chance must be finite and >= 0")
        return v


class RecommendItemsResponse(BaseModel):
    snapshot_id: str
    mode: GameMode
    champion_id: str
    selection_bias_warning: bool
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[ItemRecommendation]
