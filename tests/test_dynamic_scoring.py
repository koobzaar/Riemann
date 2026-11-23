from __future__ import annotations

import json
from pathlib import Path

from riemann.adapters.cdragon import parse_cdragon_payloads
from riemann.config import ModelHyperparams, Settings
from riemann.mechanics import (
    AugmentGraphMeta,
    GraphEdge,
    RelationGraph,
    load_relation_graph,
)
from riemann.models import (
    CatalogAugment,
    ConverterSpec,
    Observation,
    RecommendAugmentsRequest,
    SyncRequest,
)
from riemann.adapters.base import FixtureStatsAdapter
from riemann.recommender import recommend_augments
from riemann.service import RiemannService

FIX_DIR = Path(__file__).resolve().parent / "fixtures"
HP = ModelHyperparams()


def _make_service(tmp_path: Path, graph: RelationGraph | None = None, obs: list[dict] | None = None) -> RiemannService:
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        db_path=tmp_path / "dyn.sqlite",
        graph_path=FIX_DIR / "relation_graph_sample.json",
        fixture_dir=FIX_DIR,
        block_outbound_on_inference=True,
        opgg_delay_s=0.0,
        opgg_timeout_s=5.0,
    )
    g = graph if graph is not None else load_relation_graph(FIX_DIR / "relation_graph_sample.json")

    if obs is not None:
        obs_path = tmp_path / "custom_obs.json"
        obs_path.write_text(json.dumps({"observations": obs}), encoding="utf-8")
        adapters = {"fixture": FixtureStatsAdapter(obs_path)}
    else:
        adapters = None

    svc = RiemannService(settings, graph=g, custom_adapters=adapters)
    svc._sync_blocking(SyncRequest(catalog_source="fixture"))
    return svc


def test_reference_subtraction_self_overcap_and_zero_data_reconstruction(tmp_path: Path) -> None:
    g = load_relation_graph(FIX_DIR / "relation_graph_sample.json")
    # Add a synthetic +1.25 crit augment with prior_mean_logit=0.15 and a pair edge +0.20
    g.augments["self_overcap_aug"] = AugmentGraphMeta(
        canonical_id="self_overcap_aug",
        mechanic_tags=["scales_with_crit"],
        crit_chance=1.25,
        reference_prior_logit=0.15,
    )

    def loader(_req: SyncRequest):
        cat = parse_cdragon_payloads(
            default_payload=json.loads((FIX_DIR / "cdragon_raw_default.json").read_text(encoding="utf-8")),
            pt_br_payload=json.loads((FIX_DIR / "cdragon_raw_pt_br.json").read_text(encoding="utf-8")),
        )
        cat.augments.append(
            CatalogAugment(
                canonical_id="self_overcap_aug",
                modes=["aram_mayhem", "arena"],
                names={"en_us": "Self Overcap", "pt_br": "Self Overcap"},
                crit_chance=1.25,
            )
        )
        return cat

    settings = Settings(
        db_path=tmp_path / "ref.sqlite",
        graph_path=FIX_DIR / "relation_graph_sample.json",
        fixture_dir=FIX_DIR,
        block_outbound_on_inference=True,
        opgg_delay_s=0.0,
        opgg_timeout_s=5.0,
    )
    svc = RiemannService(settings, graph=g, custom_catalog_loader=loader)
    svc._sync_blocking(SyncRequest(catalog_source="fixture"))

    # At S_ref: mechanistic_delta_logit == 0.0 (even for +1.25 crit!), association_mean_logit == 0.15
    resp_ref = svc.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            offered_augments=["self_overcap_aug"],
            top_item_pivots_k=0,
        )
    )
    rec = resp_ref.recommendations[0]
    assert abs(rec.prior_mean_logit - 0.15) < 1e-6
    assert abs(rec.association_mean_logit - 0.15) < 1e-6
    assert rec.mechanistic_delta_logit == 0.0
    assert rec.breakdown.overcap_penalty_delta_logit == 0.0
    assert abs(rec.composite_score - 0.15) < 1e-6

    # Live state with it_is_critical (+0.50 crit, triggers crit_synergy +0.20, plus incremental overcap)
    # Wait: test clean +0.20 without overcap using a non-crit scales_with_crit aug
    g.augments["pure_prior_aug"] = AugmentGraphMeta(
        canonical_id="pure_prior_aug",
        mechanic_tags=["scales_with_crit"],
        crit_chance=0.0,
        reference_prior_logit=0.15,
    )

    def loader2(_req: SyncRequest):
        cat = loader(_req)
        cat.augments.append(
            CatalogAugment(
                canonical_id="pure_prior_aug",
                modes=["aram_mayhem", "arena"],
                names={"en_us": "Pure Prior", "pt_br": "Pure Prior"},
            )
        )
        return cat

    svc2 = RiemannService(settings, graph=g, custom_catalog_loader=loader2)
    svc2._sync_blocking(SyncRequest(catalog_source="fixture"))
    resp_live = svc2.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            chosen_augments=["it_is_critical"],
            offered_augments=["pure_prior_aug"],
            top_item_pivots_k=0,
        )
    )
    rl = resp_live.recommendations[0]
    assert abs(rl.association_mean_logit - 0.15) < 1e-6
    assert abs(rl.mechanistic_delta_logit - 0.20) < 1e-6
    assert abs(rl.composite_score - 0.35) < 1e-6
    assert abs(rl.composite_score - (rl.association_mean_logit + rl.mechanistic_delta_logit)) < 1e-9


def test_counterfactual_synergy_and_symmetric_edge_deduplication(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)

    # Control: without it_is_critical, jeweled_gauntlet has pair_synergy_logit == 0.0
    ctrl = svc.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            slot_index=2,
            chosen_augments=[],
            offered_augments=["jeweled_gauntlet", "ethereal_weapon"],
            top_item_pivots_k=0,
        )
    )
    by_id_ctrl = {r.canonical_id: r for r in ctrl.recommendations}
    assert by_id_ctrl["jeweled_gauntlet"].breakdown.pair_synergy_logit == 0.0

    # With it_is_critical chosen: jeweled_gauntlet gets +0.20 and ranks #1 over equal-data ethereal_weapon
    live = svc.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            slot_index=2,
            chosen_augments=["it_is_critical"],
            offered_augments=["jeweled_gauntlet", "ethereal_weapon"],
            top_item_pivots_k=0,
        )
    )
    assert live.recommendations[0].canonical_id == "jeweled_gauntlet"
    assert abs(live.recommendations[0].breakdown.pair_synergy_logit - 0.20) < 1e-9

    # Symmetric deduplication: two augments BOTH carrying X and Y count symmetric X<->Y (+0.20) ONCE
    g = load_relation_graph(FIX_DIR / "relation_graph_sample.json")
    g.augments["it_is_critical"].mechanic_tags = ["X", "Y"]
    g.augments["jeweled_gauntlet"].mechanic_tags = ["X", "Y"]
    g.augments["ethereal_weapon"].mechanic_tags = ["X", "Y"]
    g.edges = [
        GraphEdge(
            edge_id="sym_xy",
            edge_type="augment_pair_synergy",
            source_tag="X",
            target_tag="Y",
            weight=0.20,
            symmetric=True,
        )
    ]
    svc_sym = _make_service(tmp_path / "sym", graph=g)
    r_one = svc_sym.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            chosen_augments=["it_is_critical"],
            offered_augments=["jeweled_gauntlet"],
            top_item_pivots_k=0,
        )
    )
    assert abs(r_one.recommendations[0].breakdown.pair_synergy_logit - 0.20) < 1e-9

    # Two distinct chosen augments triggering sym_xy against candidate contribute +0.40
    r_two = svc_sym.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            chosen_augments=["it_is_critical", "ethereal_weapon"],
            offered_augments=["jeweled_gauntlet"],
            top_item_pivots_k=0,
        )
    )
    assert abs(r_two.recommendations[0].breakdown.pair_synergy_logit - 0.40) < 1e-9


def test_incremental_crit_overcap_and_exact_converter_physics(tmp_path: Path) -> None:
    g = load_relation_graph(FIX_DIR / "relation_graph_sample.json")
    g.augments["neutral_aug"] = AugmentGraphMeta(
        canonical_id="neutral_aug",
        mechanic_tags=["crit_overflow_converter"],
        crit_chance=0.0,
        converter_specs=[ConverterSpec(stat_name="lifesteal_fraction", units_per_crit_fraction=0.35)],
    )
    svc = _make_service(tmp_path, graph=g)

    # Already overcapped observed_crit_chance=1.50, candidate +0.25 crit (no converter) -> wasted=0.25, pen=0.08
    r_over = svc.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            observed_crit_chance=1.50,
            offered_augments=["jeweled_gauntlet"],
            top_item_pivots_k=0,
        )
    )
    rec = r_over.recommendations[0]
    assert abs(rec.physical_metrics.usable_crit_added - 0.0) < 1e-9
    assert abs(rec.physical_metrics.wasted_crit_added - 0.25) < 1e-9
    assert abs(rec.breakdown.overcap_penalty_delta_logit - 0.08) < 1e-6

    # Converter active on champion senna (0.35 lifesteal/crit): +0.25 overflow crit -> exact 0.0875
    r_senna = svc.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="senna",
            observed_crit_chance=1.50,
            offered_augments=["jeweled_gauntlet"],
            top_item_pivots_k=0,
        )
    )
    rs = r_senna.recommendations[0]
    assert rs.physical_metrics.wasted_crit_added == 0.0
    assert rs.breakdown.overcap_penalty_delta_logit == 0.0
    assert abs(rs.physical_metrics.converted_overflow_stats["lifesteal_fraction"] - 0.0875) < 1e-6

    # Converter active on chosen augment (neutral_aug) -> same 0.0875
    r_chosen_conv = svc.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            chosen_augments=["neutral_aug"],
            observed_crit_chance=1.50,
            offered_augments=["jeweled_gauntlet"],
            top_item_pivots_k=0,
        )
    )
    assert abs(r_chosen_conv.recommendations[0].physical_metrics.converted_overflow_stats["lifesteal_fraction"] - 0.0875) < 1e-6

    # Candidate itself unlocks converter at crit_before=1.50 (+0.0 crit) -> exact 0.50 * 0.35 = 0.175
    r_cand_conv = svc.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            observed_crit_chance=1.50,
            offered_augments=["neutral_aug"],
            top_item_pivots_k=0,
        )
    )
    rc = r_cand_conv.recommendations[0]
    assert rc.breakdown.overcap_penalty_delta_logit == 0.0
    assert abs(rc.physical_metrics.converted_overflow_stats["lifesteal_fraction"] - 0.175) < 1e-6


def test_two_tier_robustness_and_post_pick_item_pivots(tmp_path: Path) -> None:
    # Two candidates with 50/100 (b=0, slot="all" -> B_e=0.08), mechanistic delta 0.20 vs 0.0
    obs = [
        {"mode": "aram_mayhem", "metric_type": "win", "patch": "26.22", "champion_id": "jinx", "entity_type": "augment", "entity_id": "jeweled_gauntlet", "slot_key": "all", "games": 100, "wins": 50, "baseline_rate": 0.50},
        {"mode": "aram_mayhem", "metric_type": "win", "patch": "26.22", "champion_id": "jinx", "entity_type": "augment", "entity_id": "ethereal_weapon", "slot_key": "all", "games": 100, "wins": 50, "baseline_rate": 0.50},
    ]
    svc = _make_service(tmp_path / "rob", obs=obs)
    resp = svc.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            chosen_augments=["it_is_critical"],
            offered_augments=["jeweled_gauntlet", "ethereal_weapon"],
            top_item_pivots_k=0,
        )
    )
    top1 = resp.recommendations[0]
    assert top1.canonical_id == "jeweled_gauntlet"
    assert abs((top1.mean_bias_margin or 0.0) - 0.04) < 1e-5
    assert top1.is_bias_robust_on_means is True
    assert (top1.envelope_gap or 0.0) < 0.0
    assert top1.is_envelope_separated is False

    # Increase edge to 0.60 -> BOTH is_bias_robust_on_means and is_envelope_separated are True
    g_big = load_relation_graph(FIX_DIR / "relation_graph_sample.json")
    g_big.edges[0].weight = 0.60
    svc_big = _make_service(tmp_path / "rob2", graph=g_big, obs=obs)
    resp2 = svc_big.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            chosen_augments=["it_is_critical"],
            offered_augments=["jeweled_gauntlet", "ethereal_weapon"],
            top_item_pivots_k=0,
        )
    )
    assert resp2.recommendations[0].is_bias_robust_on_means is True
    assert resp2.recommendations[0].is_envelope_separated is True

    # Post-pick item pivots: rank flip warmogs_armor -> heartsteel under apex_inventor
    svc_piv = _make_service(tmp_path / "piv")
    resp_piv = svc_piv.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            slot_index=1,
            offered_augments=["apex_inventor", "neutral_aug"],
            locale="auto",
            top_item_pivots_k=3,
        )
    )
    by_id = {r.canonical_id: r for r in resp_piv.recommendations}
    assert by_id["neutral_aug"].top_item_pivots[0].item_id in {"warmogs_armor", "thornmail"}
    assert by_id["apex_inventor"].top_item_pivots[0].item_id == "heartsteel"

    # Post-pick observed_crit_chance transition: 0.75 + 0.25 -> 1.00, so +0.25 crit item incurs 0.08 penalty
    resp_crit_piv = svc_piv.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            observed_crit_chance=0.75,
            offered_augments=["jeweled_gauntlet"],
            top_item_pivots_k=10,
        )
    )
    ie_piv = next(p for p in resp_crit_piv.recommendations[0].top_item_pivots if p.item_id == "infinity_edge")
    assert "crit_overcap_penalty" in ie_piv.reason_codes

    # 6-item full inventory -> empty pivots + warning (no 422!)
    resp_full = svc_piv.recommend_augments(
        RecommendAugmentsRequest(
            mode="aram_mayhem",
            champion_id="dr_mundo",
            current_items=[
                "heartsteel",
                "warmogs_armor",
                "thornmail",
                "overlords_bloodmail",
                "rabadons_deathcap",
                "unending_despair",
            ],
            offered_augments=["apex_inventor"],
            top_item_pivots_k=3,
        )
    )
    assert "inventory_full_no_item_pivots" in resp_full.warnings
    assert resp_full.recommendations[0].top_item_pivots == []
