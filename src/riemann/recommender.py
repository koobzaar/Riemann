from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from riemann.config import ModelHyperparams
from riemann.mechanics import (
    AugmentGraphMeta,
    ChampionGraphMeta,
    ItemGraphMeta,
    RelationGraph,
    compute_crit_physics,
    compute_item_haste_physics,
    compute_reference_prior_mean,
    effective_augment_meta,
    effective_champion_meta,
    effective_item_meta,
    evaluate_augment_pair_edges,
    evaluate_context_to_augment_edges,
    evaluate_general_item_edges,
    evaluate_locked_legendary_amplification,
    resolve_weapon_of_destiny_state,
)
from riemann.models import (
    AugmentBindings,
    AugmentBreakdown,
    AugmentPhysicalMetrics,
    AugmentRecommendation,
    CorroborationInfo,
    GameMode,
    ItemBreakdown,
    ItemPhysicalMetrics,
    ItemPivotSummary,
    ItemRecommendation,
    PrecomputedPosterior,
    PublicLocale,
    ReasonItem,
    RecommendAugmentsRequest,
    RecommendAugmentsResponse,
    RecommendItemsRequest,
    RecommendItemsResponse,
    UnresolvedInput,
)
from riemann.ocr import resolve_augment_text
from riemann.stats_engine import compute_posterior_quadrature
from riemann.storage import LoadedSnapshot


class DomainValidationError(ValueError):
    def __init__(self, code: str, detail: str, field: str | None = None) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.field = field


def _compute_sensitivity_bound(
    *,
    posterior: PrecomputedPosterior,
    used_slot_key: str,
    hyperparams: ModelHyperparams,
) -> float:
    # Precedence (maximum applicable):
    # 0.12 if N_eff == 0 OR pick_rate < 0.02 OR low_baseline_sample
    # else 0.08 if slot_key == "all"
    # else 0.05
    if (
        posterior.effective_sample_size <= 0.0
        or (
            posterior.pick_rate is not None
            and posterior.pick_rate < hyperparams.low_pick_rate_threshold
        )
        or posterior.low_baseline_sample
    ):
        b_base = hyperparams.sensitivity_base_sparse
    elif used_slot_key == "all":
        b_base = hyperparams.sensitivity_base_unstratified
    else:
        b_base = hyperparams.sensitivity_base_stratified

    extra = (
        posterior.corroboration.max_source_disagreement_logit
        if (
            posterior.corroboration.status == "divergent"
            and posterior.corroboration.max_source_disagreement_logit is not None
        )
        else 0.0
    )
    return b_base + extra


@lru_cache(maxsize=4096)
def _cached_zero_data_quadrature(
    m_e: float,
    pi_small: float,
    tau_small: float,
    tau_large: float,
    tol: float,
) -> tuple[float, tuple[float, float], float]:
    qres = compute_posterior_quadrature(
        [],
        m_e=m_e,
        mu_star=0.50,
        pi_small=pi_small,
        tau_small=tau_small,
        tau_large=tau_large,
        tol=tol,
    )
    return qres.mean_logit, qres.ci_90_logit, qres.standardized_wr_delta


def _lookup_posterior_or_prior(
    *,
    snapshot: LoadedSnapshot,
    graph: RelationGraph,
    mode: GameMode,
    champion: ChampionGraphMeta,
    entity_type: str,
    entity_meta: AugmentGraphMeta | ItemGraphMeta,
    slot_index: int | None,
    hyperparams: ModelHyperparams,
) -> tuple[PrecomputedPosterior, str, bool]:
    """Return (posterior, used_slot_key, selection_bias_warning)."""
    desired_slot = f"slot_{slot_index}" if slot_index is not None else "all"
    p = snapshot.posteriors.get(
        (mode, champion.canonical_id, entity_type, entity_meta.canonical_id, desired_slot)
    )
    if p is not None:
        return p, desired_slot, (desired_slot == "all")

    if desired_slot != "all":
        p_all = snapshot.posteriors.get(
            (mode, champion.canonical_id, entity_type, entity_meta.canonical_id, "all")
        )
        if p_all is not None:
            return p_all, "all", True

    # Zero-observation fallback under mechanistic reference prior m_e = h(S_ref, e)
    m_e = compute_reference_prior_mean(
        mode=mode,
        champion=champion,
        entity_type=entity_type,  # type: ignore[arg-type]
        entity_meta=entity_meta,
        graph=graph,
    )
    mean_logit, ci_90, std_delta = _cached_zero_data_quadrature(
        round(m_e, 12),
        hyperparams.pi_small,
        hyperparams.tau_small,
        hyperparams.tau_large,
        hyperparams.quadrature_tol,
    )
    return (
        PrecomputedPosterior(
            mode=mode,
            champion_id=champion.canonical_id,
            entity_type=entity_type,  # type: ignore[arg-type]
            entity_id=entity_meta.canonical_id,
            slot_key="all",
            prior_mean_logit=m_e,
            association_mean_logit=mean_logit,
            association_ci_90_logit=ci_90,
            standardized_wr_delta=std_delta,
            effective_sample_size=0.0,
            corroboration=CorroborationInfo(),
            arena_secondary=None,
        ),
        "all",
        True,
    )


def _display_name(
    snapshot: LoadedSnapshot,
    entity_type: str,
    canonical_id: str,
    locale: PublicLocale,
    fallback: str,
) -> str:
    row = snapshot.translations.get((entity_type, canonical_id, locale))
    if row is not None:
        return row[0]
    row_en = snapshot.translations.get((entity_type, canonical_id, "en_us"))
    if row_en is not None:
        return row_en[0]
    return fallback


@dataclass
class _ValidatedState:
    champion: ChampionGraphMeta
    chosen_augments: list[AugmentGraphMeta]
    current_items: list[ItemGraphMeta]
    all_mode_items: dict[str, ItemGraphMeta]
    crit_before: float
    total_item_haste: float
    converters_before: list


_mode_items_cache: dict[tuple[str, int, GameMode], dict[str, ItemGraphMeta]] = {}


def _get_mode_items(
    snapshot: LoadedSnapshot, graph: RelationGraph, mode: GameMode
) -> dict[str, ItemGraphMeta]:
    key = (snapshot.snapshot_id, id(graph), mode)
    cached = _mode_items_cache.get(key)
    if cached is not None:
        return cached
    out: dict[str, ItemGraphMeta] = {}
    for (m, iid), icat in snapshot.items.items():
        if m == mode:
            out[iid] = effective_item_meta(icat, graph)
    if len(_mode_items_cache) > 64:
        _mode_items_cache.clear()
    _mode_items_cache[key] = out
    return out


def _validate_shared_state(
    *,
    snapshot: LoadedSnapshot,
    graph: RelationGraph,
    mode: GameMode,
    champion_id: str,
    chosen_augments: Sequence[str],
    current_items: Sequence[str],
    observed_crit_chance: float | None,
    augment_bindings: AugmentBindings | None,
) -> _ValidatedState:
    champ_cat = snapshot.champions.get((mode, champion_id))
    if champ_cat is None:
        raise DomainValidationError(
            "unknown_champion_id", f"Unknown champion_id {champion_id!r} for mode {mode}", "champion_id"
        )
    champ_meta = effective_champion_meta(champ_cat, graph)

    # Validate chosen_augments (no duplicates, all valid for mode)
    seen_chosen: set[str] = set()
    chosen_metas: list[AugmentGraphMeta] = []
    for aid in chosen_augments:
        if aid in seen_chosen:
            raise DomainValidationError(
                "duplicate_chosen_augment", f"Duplicate chosen augment: {aid!r}", "chosen_augments"
            )
        seen_chosen.add(aid)
        aug_cat = snapshot.augments.get((mode, aid))
        if aug_cat is None:
            raise DomainValidationError(
                "augment_invalid_for_mode",
                f"Augment {aid!r} is unknown or invalid for mode {mode}",
                "chosen_augments",
            )
        chosen_metas.append(effective_augment_meta(aug_cat, graph))

    all_mode_items = _get_mode_items(snapshot, graph, mode)

    # Validate current_items (max 6, valid for mode, no duplicate unique items)
    if len(current_items) > 6:
        raise DomainValidationError(
            "inventory_limit_exceeded", "current_items cannot exceed 6 items", "current_items"
        )
    seen_unique_items: set[str] = set()
    current_item_metas: list[ItemGraphMeta] = []
    for iid in current_items:
        imeta = all_mode_items.get(iid)
        if imeta is None:
            raise DomainValidationError(
                "item_invalid_for_mode",
                f"Item {iid!r} is unknown or invalid for mode {mode}",
                "current_items",
            )
        if imeta.is_unique:
            if iid in seen_unique_items:
                raise DomainValidationError(
                    "duplicate_unique_item",
                    f"Duplicate unique item in current_items: {iid!r}",
                    "current_items",
                )
            seen_unique_items.add(iid)
        current_item_metas.append(imeta)

    # Validate augment_bindings
    if augment_bindings is not None and augment_bindings.weapon_of_destiny_item_id is not None:
        bid = augment_bindings.weapon_of_destiny_item_id
        bmeta = all_mode_items.get(bid)
        if bmeta is None or not bmeta.is_legendary:
            raise DomainValidationError(
                "invalid_weapon_of_destiny_binding",
                f"weapon_of_destiny_item_id {bid!r} must be a valid Legendary item for mode {mode}",
                "augment_bindings.weapon_of_destiny_item_id",
            )

    crit_before = (
        observed_crit_chance
        if observed_crit_chance is not None
        else sum(a.crit_chance for a in chosen_metas) + sum(i.crit_chance for i in current_item_metas)
    )
    total_haste = sum(a.item_haste for a in chosen_metas)
    converters_before = list(champ_meta.converter_specs)
    for a in chosen_metas:
        converters_before.extend(a.converter_specs)

    return _ValidatedState(
        champion=champ_meta,
        chosen_augments=chosen_metas,
        current_items=current_item_metas,
        all_mode_items=all_mode_items,
        crit_before=crit_before,
        total_item_haste=total_haste,
        converters_before=converters_before,
    )


def recommend_items(
    *,
    snapshot: LoadedSnapshot,
    graph: RelationGraph,
    request: RecommendItemsRequest,
    hyperparams: ModelHyperparams,
    resolved_locale_override: PublicLocale | None = None,
) -> RecommendItemsResponse:
    state = _validate_shared_state(
        snapshot=snapshot,
        graph=graph,
        mode=request.mode,
        champion_id=request.champion_id,
        chosen_augments=request.chosen_augments,
        current_items=request.current_items,
        observed_crit_chance=request.observed_crit_chance,
        augment_bindings=request.augment_bindings,
    )
    locale: PublicLocale = (
        resolved_locale_override
        if resolved_locale_override is not None
        else ("en_us" if request.locale == "auto" else request.locale)
    )

    owned_unique_ids = {i.canonical_id for i in state.current_items if i.is_unique}
    owned_unique_groups = {
        i.unique_group for i in state.current_items if i.unique_group is not None
    }

    has_wod = any("empowers_first_legendary" in a.mechanic_tags for a in state.chosen_augments)
    wod_res = resolve_weapon_of_destiny_state(
        has_wod_augment=has_wod,
        current_items=request.current_items,
        items_by_id=state.all_mode_items,
        bindings=request.augment_bindings,
        slot_index=request.slot_index,
    )
    top_warnings: list[str] = []
    if wod_res.warning is not None:
        top_warnings.append(wod_res.warning)

    locked_item_meta = (
        state.all_mode_items.get(wod_res.bound_item_id)
        if wod_res.status == "locked" and wod_res.bound_item_id is not None
        else None
    )

    any_selection_bias_warning = False
    recs: list[ItemRecommendation] = []

    # Pre-filter active edges once per request for fast item scoring (< 5 ms p95 on 100+ items)
    chosen_aug_ids = {a.canonical_id for a in state.chosen_augments}
    chosen_aug_tags = {t for a in state.chosen_augments for t in a.mechanic_tags}
    cur_item_ids = {i.canonical_id for i in state.current_items}
    cur_item_tags = {t for i in state.current_items for t in i.mechanic_tags}
    active_aug_item_edges = [
        e
        for e in graph.edges
        if e.edge_type == "augment_item_synergy"
        and request.mode in e.modes
        and (e.source_entity_id is None or e.source_entity_id in chosen_aug_ids)
        and (e.source_tag is None or e.source_tag in chosen_aug_tags)
    ]
    active_item_pair_edges = [
        e
        for e in graph.edges
        if e.edge_type == "item_pair_synergy"
        and request.mode in e.modes
        and (e.source_entity_id is None or e.source_entity_id in cur_item_ids)
        and (e.source_tag is None or e.source_tag in cur_item_tags)
    ]
    preindexed_edges = (active_aug_item_edges, active_item_pair_edges)

    for iid, imeta in state.all_mode_items.items():
        if imeta.is_unique and iid in owned_unique_ids:
            continue
        if imeta.unique_group is not None and imeta.unique_group in owned_unique_groups:
            continue

        post, used_slot, sb_warn = _lookup_posterior_or_prior(
            snapshot=snapshot,
            graph=graph,
            mode=request.mode,
            champion=state.champion,
            entity_type="item",
            entity_meta=imeta,
            slot_index=request.slot_index,
            hyperparams=hyperparams,
        )
        if sb_warn:
            any_selection_bias_warning = True

        reasons: list[ReasonItem] = []
        item_warnings: list[str] = []
        if post.low_baseline_sample:
            item_warnings.append("low_baseline_sample")

        # 1. Exact Item Haste physics
        haste_phys = compute_item_haste_physics(imeta, state.total_item_haste)
        if haste_phys.haste_logit > 0.0:
            reasons.append(
                ReasonItem(
                    code="apex_inventor_item_haste",
                    message=f"Item haste ({state.total_item_haste:.0f}) reduces cooldown by {haste_phys.fractional_cdr*100:.1f}%",
                    delta_logit=round(haste_phys.haste_logit, 9),
                )
            )

        # 2. Weapon of Destiny state machine
        wod_logit = 0.0
        if wod_res.status == "open" and imeta.is_legendary:
            tier = state.champion.stat_scaling_overrides.get(iid, imeta.stat_scaling_tier)
            wod_logit = hyperparams.weapon_destiny_rush_bonus * tier
            if wod_logit > 0.0:
                reasons.append(
                    ReasonItem(
                        code="weapon_of_destiny_first_legendary",
                        message=f"Locks in as Weapon of Destiny first Legendary (scaling tier {tier:.2f})",
                        delta_logit=round(wod_logit, 9),
                    )
                )
        elif wod_res.status == "locked" and locked_item_meta is not None:
            wod_logit, wod_reasons = evaluate_locked_legendary_amplification(
                mode=request.mode,
                locked_item=locked_item_meta,
                candidate_item=imeta,
                graph=graph,
            )
            reasons.extend(wod_reasons)

        # 3. General item graph edges (strictly excludes amplifies_locked_legendary)
        item_graph_logit, graph_reasons = evaluate_general_item_edges(
            mode=request.mode,
            chosen_augments=state.chosen_augments,
            current_items=state.current_items,
            candidate_item=imeta,
            graph=graph,
            indexed_edges=preindexed_edges,
        )
        reasons.extend(graph_reasons)

        # 4. Crit overcap physics (reference-delta!)
        crit_phys_live = compute_crit_physics(
            crit_before=state.crit_before,
            crit_added=imeta.crit_chance,
            converters_before=state.converters_before,
            converters_added=[],
            k_crit=hyperparams.crit_overcap_k,
        )
        crit_phys_ref = compute_crit_physics(
            crit_before=0.0,
            crit_added=imeta.crit_chance,
            converters_before=state.champion.converter_specs,
            converters_added=[],
            k_crit=hyperparams.crit_overcap_k,
        )
        overcap_delta = crit_phys_live.overcap_penalty_logit - crit_phys_ref.overcap_penalty_logit
        if overcap_delta > 0.0:
            reasons.append(
                ReasonItem(
                    code="crit_overcap_penalty",
                    message=f"Wastes {crit_phys_live.wasted_crit_added*100:.1f}% crit chance over 100% cap",
                    delta_logit=round(-overcap_delta, 9),
                )
            )

        mech_delta = haste_phys.haste_logit + wod_logit + item_graph_logit - overcap_delta
        comp_score = post.association_mean_logit + mech_delta
        b_e = _compute_sensitivity_bound(
            posterior=post, used_slot_key=used_slot, hyperparams=hyperparams
        )
        env = (
            post.association_ci_90_logit[0] + mech_delta - b_e,
            post.association_ci_90_logit[1] + mech_delta + b_e,
        )

        recs.append(
            ItemRecommendation(
                item_id=iid,
                display_name=_display_name(
                    snapshot, "item", iid, locale, snapshot.items[(request.mode, iid)].names["en_us"]
                ),
                composite_score=comp_score,
                association_mean_logit=post.association_mean_logit,
                association_ci_90_logit=post.association_ci_90_logit,
                standardized_wr_delta=post.standardized_wr_delta,
                mechanistic_delta_logit=mech_delta,
                bias_sensitivity_bound=b_e,
                sensitivity_envelope_logit=env,
                effective_sample_size=post.effective_sample_size,
                corroboration=post.corroboration,
                arena_secondary=post.arena_secondary,
                physical_metrics=ItemPhysicalMetrics(
                    base_cooldown_s=haste_phys.base_cooldown_s,
                    effective_cooldown_s=haste_phys.effective_cooldown_s,
                    ideal_activation_rate_hz=haste_phys.ideal_activation_rate_hz,
                    usable_crit_added=crit_phys_live.usable_crit_added,
                    wasted_crit_added=crit_phys_live.wasted_crit_added,
                    converted_overflow_stats=crit_phys_live.converted_overflow_stats,
                    weapon_of_destiny_bound_item=wod_res.bound_item_id,
                ),
                prior_mean_logit=post.prior_mean_logit,
                breakdown=ItemBreakdown(
                    item_haste_logit=haste_phys.haste_logit,
                    weapon_destiny_logit=wod_logit,
                    item_graph_logit=item_graph_logit,
                    overcap_penalty_delta_logit=overcap_delta,
                ),
                reasons=reasons,
                warnings=item_warnings,
            )
        )

    recs.sort(
        key=lambda r: (
            -round(r.composite_score, 9),
            -round(r.effective_sample_size, 6),
            r.item_id,
        )
    )
    recs = recs[: request.top_k]

    # Populate two-tier robustness against runner-up
    for i in range(len(recs) - 1):
        curr, nxt = recs[i], recs[i + 1]
        mb_margin = (curr.composite_score - nxt.composite_score) - (
            curr.bias_sensitivity_bound + nxt.bias_sensitivity_bound
        )
        env_gap = curr.sensitivity_envelope_logit[0] - nxt.sensitivity_envelope_logit[1]
        recs[i] = curr.model_copy(
            update={
                "mean_bias_margin": mb_margin,
                "is_bias_robust_on_means": mb_margin > 0.0,
                "envelope_gap": env_gap,
                "is_envelope_separated": env_gap > 0.0,
            }
        )

    return RecommendItemsResponse(
        snapshot_id=snapshot.snapshot_id,
        mode=request.mode,
        champion_id=request.champion_id,
        selection_bias_warning=any_selection_bias_warning,
        warnings=top_warnings,
        recommendations=recs,
    )


def recommend_augments(
    *,
    snapshot: LoadedSnapshot,
    graph: RelationGraph,
    request: RecommendAugmentsRequest,
    hyperparams: ModelHyperparams,
) -> RecommendAugmentsResponse:
    if not request.offered_augments:
        raise DomainValidationError(
            "empty_offered_augments", "offered_augments must not be empty", "offered_augments"
        )

    state = _validate_shared_state(
        snapshot=snapshot,
        graph=graph,
        mode=request.mode,
        champion_id=request.champion_id,
        chosen_augments=request.chosen_augments,
        current_items=request.current_items,
        observed_crit_chance=request.observed_crit_chance,
        augment_bindings=request.augment_bindings,
    )

    # Build OCR index for the requested mode
    mode_aug_ids = {cid for (m, cid) in snapshot.augments if m == request.mode}
    other_mode_aug_ids = {cid for (m, cid) in snapshot.augments if m != request.mode}
    ocr_entries: list[tuple[str, PublicLocale, str, str]] = []
    for (etype, cid, loc), (dname, nname) in snapshot.translations.items():
        if etype == "augment" and cid in mode_aug_ids:
            ocr_entries.append((cid, loc, dname, nname))

    resolved_ids: list[str] = []
    seen_offered: set[str] = set()
    unresolved: list[UnresolvedInput] = []
    top_warnings: list[str] = []
    detected_locale: PublicLocale | None = (
        None if request.locale == "auto" else request.locale
    )

    chosen_set = set(request.chosen_augments)
    for raw in request.offered_augments:
        if raw in mode_aug_ids:
            cid = raw
        elif raw in other_mode_aug_ids:
            raise DomainValidationError(
                "augment_invalid_for_mode",
                f"Offered augment {raw!r} is not valid for mode {request.mode}",
                "offered_augments",
            )
        else:
            res = resolve_augment_text(
                text=raw,
                locale=request.locale,
                entries=ocr_entries,
                confidence_threshold=hyperparams.ocr_confidence_threshold,
                margin_threshold=hyperparams.ocr_margin_threshold,
            )
            if not res.resolved or res.canonical_id is None:
                unresolved.append(
                    UnresolvedInput(
                        raw_input=raw,
                        reason=res.reason,  # type: ignore[arg-type]
                        candidates=res.candidates,
                    )
                )
                continue
            cid = res.canonical_id
            if detected_locale is None and res.detected_locale is not None:
                detected_locale = res.detected_locale

        if cid in chosen_set:
            raise DomainValidationError(
                "offered_augment_already_chosen",
                f"Offered augment {cid!r} is already in chosen_augments",
                "offered_augments",
            )
        if cid in seen_offered:
            if "duplicate_offered_augment" not in top_warnings:
                top_warnings.append("duplicate_offered_augment")
            continue
        seen_offered.add(cid)
        resolved_ids.append(cid)

    if not resolved_ids:
        raise DomainValidationError(
            "all_offered_augments_unresolved",
            "None of the offered_augments could be resolved",
            "offered_augments",
        )

    effective_locale: PublicLocale = detected_locale if detected_locale is not None else "en_us"
    if request.top_item_pivots_k > 0 and len(request.current_items) >= 6:
        top_warnings.append("inventory_full_no_item_pivots")

    any_selection_bias_warning = False
    recs: list[AugmentRecommendation] = []

    for cid in resolved_ids:
        aug_cat = snapshot.augments[(request.mode, cid)]
        ameta = effective_augment_meta(aug_cat, graph)

        post, used_slot, sb_warn = _lookup_posterior_or_prior(
            snapshot=snapshot,
            graph=graph,
            mode=request.mode,
            champion=state.champion,
            entity_type="augment",
            entity_meta=ameta,
            slot_index=request.slot_index,
            hyperparams=hyperparams,
        )
        if sb_warn:
            any_selection_bias_warning = True

        pair_logit, pair_reasons = evaluate_augment_pair_edges(
            mode=request.mode,
            chosen_augments=state.chosen_augments,
            candidate=ameta,
            graph=graph,
        )
        ctx_logit, ctx_reasons = evaluate_context_to_augment_edges(
            mode=request.mode,
            current_items=state.current_items,
            candidate=ameta,
            graph=graph,
        )

        # Reference-delta crit overcap penalty (so at S_ref even a +1.25 crit aug has delta 0.0!)
        crit_phys_live = compute_crit_physics(
            crit_before=state.crit_before,
            crit_added=ameta.crit_chance,
            converters_before=state.converters_before,
            converters_added=ameta.converter_specs,
            k_crit=hyperparams.crit_overcap_k,
        )
        crit_phys_ref = compute_crit_physics(
            crit_before=0.0,
            crit_added=ameta.crit_chance,
            converters_before=state.champion.converter_specs,
            converters_added=ameta.converter_specs,
            k_crit=hyperparams.crit_overcap_k,
        )
        overcap_delta = crit_phys_live.overcap_penalty_logit - crit_phys_ref.overcap_penalty_logit

        reasons = list(pair_reasons) + list(ctx_reasons)
        if overcap_delta > 0.0:
            reasons.append(
                ReasonItem(
                    code="crit_overcap_penalty",
                    message=f"Wastes {crit_phys_live.wasted_crit_added*100:.1f}% crit chance over 100% cap",
                    delta_logit=round(-overcap_delta, 9),
                )
            )

        mech_delta = pair_logit + ctx_logit - overcap_delta
        comp_score = post.association_mean_logit + mech_delta
        b_e = _compute_sensitivity_bound(
            posterior=post, used_slot_key=used_slot, hyperparams=hyperparams
        )
        env = (
            post.association_ci_90_logit[0] + mech_delta - b_e,
            post.association_ci_90_logit[1] + mech_delta + b_e,
        )

        # Post-pick top_item_pivots under S' = S U {a} (updating observed_crit_chance!)
        pivots: list[ItemPivotSummary] = []
        if request.top_item_pivots_k > 0 and len(request.current_items) < 6:
            post_pick_obs_crit = (
                request.observed_crit_chance + ameta.crit_chance
                if request.observed_crit_chance is not None
                else None
            )
            pivot_req = RecommendItemsRequest(
                mode=request.mode,
                champion_id=request.champion_id,
                slot_index=len(request.current_items) + 1,
                chosen_augments=list(request.chosen_augments) + [cid],
                current_items=list(request.current_items),
                observed_crit_chance=post_pick_obs_crit,
                augment_bindings=request.augment_bindings,
                locale=effective_locale,
                top_k=request.top_item_pivots_k,
            )
            pivot_resp = recommend_items(
                snapshot=snapshot,
                graph=graph,
                request=pivot_req,
                hyperparams=hyperparams,
                resolved_locale_override=effective_locale,
            )
            pivots = [
                ItemPivotSummary(
                    item_id=ir.item_id,
                    display_name=ir.display_name,
                    composite_score=ir.composite_score,
                    reason_codes=[r.code for r in ir.reasons],
                )
                for ir in pivot_resp.recommendations
            ]

        recs.append(
            AugmentRecommendation(
                canonical_id=cid,
                display_name=_display_name(
                    snapshot, "augment", cid, effective_locale, aug_cat.names["en_us"]
                ),
                composite_score=comp_score,
                association_mean_logit=post.association_mean_logit,
                association_ci_90_logit=post.association_ci_90_logit,
                standardized_wr_delta=post.standardized_wr_delta,
                mechanistic_delta_logit=mech_delta,
                bias_sensitivity_bound=b_e,
                sensitivity_envelope_logit=env,
                effective_sample_size=post.effective_sample_size,
                corroboration=post.corroboration,
                arena_secondary=post.arena_secondary,
                physical_metrics=AugmentPhysicalMetrics(
                    usable_crit_added=crit_phys_live.usable_crit_added,
                    wasted_crit_added=crit_phys_live.wasted_crit_added,
                    converted_overflow_stats=crit_phys_live.converted_overflow_stats,
                ),
                prior_mean_logit=post.prior_mean_logit,
                breakdown=AugmentBreakdown(
                    pair_synergy_logit=pair_logit,
                    context_synergy_logit=ctx_logit,
                    overcap_penalty_delta_logit=overcap_delta,
                ),
                reasons=reasons,
                top_item_pivots=pivots,
            )
        )

    recs.sort(
        key=lambda r: (
            -round(r.composite_score, 9),
            -round(r.effective_sample_size, 6),
            r.canonical_id,
        )
    )

    for i in range(len(recs) - 1):
        curr, nxt = recs[i], recs[i + 1]
        mb_margin = (curr.composite_score - nxt.composite_score) - (
            curr.bias_sensitivity_bound + nxt.bias_sensitivity_bound
        )
        env_gap = curr.sensitivity_envelope_logit[0] - nxt.sensitivity_envelope_logit[1]
        recs[i] = curr.model_copy(
            update={
                "mean_bias_margin": mb_margin,
                "is_bias_robust_on_means": mb_margin > 0.0,
                "envelope_gap": env_gap,
                "is_envelope_separated": env_gap > 0.0,
            }
        )

    return RecommendAugmentsResponse(
        snapshot_id=snapshot.snapshot_id,
        mode=request.mode,
        champion_id=request.champion_id,
        selection_bias_warning=any_selection_bias_warning,
        warnings=top_warnings,
        unresolved_inputs=unresolved,
        recommendations=recs,
    )
