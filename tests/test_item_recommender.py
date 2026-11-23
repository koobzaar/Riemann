from __future__ import annotations

from pathlib import Path

import pytest

from riemann.config import Settings
from riemann.mechanics import AugmentGraphMeta, GraphEdge, ItemGraphMeta, load_relation_graph
from riemann.models import AugmentBindings, RecommendItemsRequest, SyncRequest
from riemann.recommender import DomainValidationError
from riemann.service import RiemannService

FIX_DIR = Path(__file__).resolve().parent / "fixtures"


def _make_service(tmp_path: Path, graph=None) -> RiemannService:
    settings = Settings(
        db_path=tmp_path / "items.sqlite",
        graph_path=FIX_DIR / "relation_graph_sample.json",
        fixture_dir=FIX_DIR,
        block_outbound_on_inference=True,
        opgg_delay_s=0.0,
        opgg_timeout_s=5.0,
    )
    g = graph if graph is not None else load_relation_graph(FIX_DIR / "relation_graph_sample.json")
    svc = RiemannService(settings, graph=g)
    svc._sync_blocking(SyncRequest(catalog_source="fixture"))
    return svc


def test_exact_item_haste_physics_at_0_100_200(tmp_path: Path) -> None:
    g = load_relation_graph(FIX_DIR / "relation_graph_sample.json")
    # Add second +100 item haste augment to test H=200 cleanly
    g.augments["neutral_aug"] = AugmentGraphMeta(
        canonical_id="neutral_aug",
        mechanic_tags=["grants_item_haste"],
        item_haste=100.0,
    )
    svc = _make_service(tmp_path, graph=g)

    def get_hs(chosen: list[str]):
        resp = svc.recommend_items(
            RecommendItemsRequest(
                mode="aram_mayhem",
                champion_id="dr_mundo",
                slot_index=1,
                chosen_augments=chosen,
                top_k=20,
            )
        )
        return next(r for r in resp.recommendations if r.item_id == "heartsteel")

    h0 = get_hs([])
    assert h0.physical_metrics.effective_cooldown_s == 10.0
    assert abs((h0.physical_metrics.ideal_activation_rate_hz or 0.0) - 0.1) < 1e-9
    assert h0.breakdown.item_haste_logit == 0.0

    h100 = get_hs(["apex_inventor"])
    assert abs((h100.physical_metrics.effective_cooldown_s or 0.0) - 5.0) < 1e-6
    assert abs((h100.physical_metrics.ideal_activation_rate_hz or 0.0) - 0.2) < 1e-6
    assert abs(h100.breakdown.item_haste_logit - 0.12) < 1e-6

    h200 = get_hs(["apex_inventor", "neutral_aug"])
    assert abs((h200.physical_metrics.effective_cooldown_s or 0.0) - (10.0 / 3.0)) < 1e-5
    assert abs((h200.physical_metrics.ideal_activation_rate_hz or 0.0) - 0.3) < 1e-6
    assert abs(h200.breakdown.item_haste_logit - 0.16) < 1e-6


def test_weapon_of_destiny_full_state_machine(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)

    # 1. Negative control without weapon_of_destiny
    ctrl = svc.recommend_items(
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            slot_index=1,
            chosen_augments=[],
            top_k=20,
        )
    )
    assert all(r.breakdown.weapon_destiny_logit == 0.0 for r in ctrl.recommendations)

    # 2. Open slot (current_items=[], slot_index=1)
    r_open = svc.recommend_items(
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            slot_index=1,
            chosen_augments=["weapon_of_destiny"],
            current_items=[],
            top_k=20,
        )
    )
    by_open = {r.item_id: r for r in r_open.recommendations}
    assert abs(by_open["heartsteel"].breakdown.weapon_destiny_logit - 0.16) < 1e-6
    assert abs(by_open["thornmail"].breakdown.weapon_destiny_logit - 0.04) < 1e-6
    assert by_open["giants_belt"].breakdown.weapon_destiny_logit == 0.0
    assert by_open["heartsteel"].composite_score > by_open["thornmail"].composite_score

    # 3. 1-Legendary auto-lock (augment_bindings=None)
    r_auto = svc.recommend_items(
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            chosen_augments=["weapon_of_destiny"],
            current_items=["heartsteel"],
            augment_bindings=None,
            top_k=20,
        )
    )
    by_auto = {r.item_id: r for r in r_auto.recommendations}
    assert "heartsteel" not in by_auto
    assert abs(by_auto["overlords_bloodmail"].breakdown.weapon_destiny_logit - 0.12) < 1e-6
    assert by_auto["overlords_bloodmail"].breakdown.item_graph_logit == 0.0
    assert by_auto["rabadons_deathcap"].breakdown.weapon_destiny_logit == 0.0

    # 4. 2-Legendary explicit binding order invariance
    b_hs = AugmentBindings(weapon_of_destiny_item_id="heartsteel")
    r_ord1 = svc.recommend_items(
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            chosen_augments=["weapon_of_destiny"],
            current_items=["thornmail", "heartsteel"],
            augment_bindings=b_hs,
            top_k=20,
        )
    )
    r_ord2 = svc.recommend_items(
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            chosen_augments=["weapon_of_destiny"],
            current_items=["heartsteel", "thornmail"],
            augment_bindings=b_hs,
            top_k=20,
        )
    )
    s1 = {r.item_id: r.composite_score for r in r_ord1.recommendations}
    s2 = {r.item_id: r.composite_score for r in r_ord2.recommendations}
    assert s1 == s2
    assert abs(next(r for r in r_ord1.recommendations if r.item_id == "overlords_bloodmail").breakdown.weapon_destiny_logit - 0.12) < 1e-6

    # 5. Contradictory binding -> ValueError (422 at API layer)
    with pytest.raises(ValueError, match="contradicts"):
        AugmentBindings(weapon_of_destiny_item_id="heartsteel", weapon_of_destiny_state="open")

    # 6. Sold/absent/unknown states -> warning + 0.0 logit
    for req in [
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            chosen_augments=["weapon_of_destiny"],
            current_items=[],
            augment_bindings=AugmentBindings(weapon_of_destiny_item_id="heartsteel"),
            top_k=20,
        ),
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            chosen_augments=["weapon_of_destiny"],
            current_items=[],
            augment_bindings=AugmentBindings(weapon_of_destiny_state="sold_or_unknown"),
            top_k=20,
        ),
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            chosen_augments=["weapon_of_destiny"],
            current_items=[],
            augment_bindings=AugmentBindings(weapon_of_destiny_state="locked"),
            top_k=20,
        ),
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            slot_index=3,
            chosen_augments=["weapon_of_destiny"],
            current_items=[],
            top_k=20,
        ),
    ]:
        rr = svc.recommend_items(req)
        assert "weapon_of_destiny_state_unknown" in rr.warnings
        assert all(r.breakdown.weapon_destiny_logit == 0.0 for r in rr.recommendations)

    # 7. Transform DAG unique match (winters_approach -> fimbulwinter)
    r_tr = svc.recommend_items(
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            chosen_augments=["weapon_of_destiny"],
            current_items=["fimbulwinter"],
            augment_bindings=AugmentBindings(weapon_of_destiny_item_id="winters_approach"),
            top_k=20,
        )
    )
    ud = next(r for r in r_tr.recommendations if r.item_id == "unending_despair")
    assert ud.physical_metrics.weapon_of_destiny_bound_item == "fimbulwinter"
    assert abs(ud.breakdown.weapon_destiny_logit - 0.12) < 1e-6

    # 8. Transform DAG ambiguous (two descendants in current_items)
    g2 = load_relation_graph(FIX_DIR / "relation_graph_sample.json")
    g2.items["thornmail"] = ItemGraphMeta(
        canonical_id="thornmail",
        is_legendary=True,
        transformed_from="winters_approach",
    )
    svc2 = _make_service(tmp_path / "amb_tr", graph=g2)
    r_amb_tr = svc2.recommend_items(
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            chosen_augments=["weapon_of_destiny"],
            current_items=["fimbulwinter", "thornmail"],
            augment_bindings=AugmentBindings(weapon_of_destiny_item_id="winters_approach"),
            top_k=20,
        )
    )
    assert "weapon_of_destiny_transform_ambiguous" in r_amb_tr.warnings
    assert all(r.breakdown.weapon_destiny_logit == 0.0 for r in r_amb_tr.recommendations)

    # 9. Ambiguous 2-Legendary without binding
    r_amb = svc.recommend_items(
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            chosen_augments=["weapon_of_destiny"],
            current_items=["thornmail", "heartsteel"],
            augment_bindings=None,
            top_k=20,
        )
    )
    assert "weapon_of_destiny_binding_ambiguous" in r_amb.warnings
    assert all(r.breakdown.weapon_destiny_logit == 0.0 for r in r_amb.recommendations)


def test_uniqueness_general_item_edges_and_item_crit_overcap(tmp_path: Path) -> None:
    g = load_relation_graph(FIX_DIR / "relation_graph_sample.json")
    g.edges.extend(
        [
            GraphEdge(
                edge_id="aug_item_test",
                edge_type="augment_item_synergy",
                source_entity_id="ethereal_weapon",
                target_entity_id="overlords_bloodmail",
                weight=0.10,
            ),
            GraphEdge(
                edge_id="item_pair_test",
                edge_type="item_pair_synergy",
                source_entity_id="heartsteel",
                target_entity_id="overlords_bloodmail",
                weight=0.08,
            ),
        ]
    )
    svc = _make_service(tmp_path, graph=g)

    # Owned unique_group ("last_whisper") excludes mortal_reminder; stackable giants_belt allowed twice
    resp = svc.recommend_items(
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            chosen_augments=["ethereal_weapon"],
            current_items=["heartsteel", "lord_dominiks_regards", "giants_belt", "giants_belt"],
            observed_crit_chance=1.00,
            top_k=30,
        )
    )
    by_id = {r.item_id: r for r in resp.recommendations}
    assert "lord_dominiks_regards" not in by_id
    assert "mortal_reminder" not in by_id
    assert "giants_belt" in by_id
    assert abs(by_id["overlords_bloodmail"].breakdown.item_graph_logit - 0.18) < 1e-6
    assert by_id["overlords_bloodmail"].breakdown.weapon_destiny_logit == 0.0
    assert abs(by_id["infinity_edge"].breakdown.overcap_penalty_delta_logit - 0.08) < 1e-6

    # Duplicate unique item in current_items -> DomainValidationError (422)
    with pytest.raises(DomainValidationError) as ei:
        svc.recommend_items(
            RecommendItemsRequest(
                mode="aram_mayhem",
                champion_id="jinx",
                current_items=["heartsteel", "heartsteel"],
            )
        )
    assert ei.value.code == "duplicate_unique_item"
