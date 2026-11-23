from __future__ import annotations

import itertools
import json
from pathlib import Path

from hypothesis import given, settings as h_settings, strategies as st

from riemann.config import ModelHyperparams, Settings
from riemann.mechanics import load_relation_graph
from riemann.models import Observation, RecommendAugmentsRequest, RecommendItemsRequest, SyncRequest
from riemann.service import RiemannService
from riemann.stats_engine import (
    LikelihoodTerm,
    compute_entity_posterior,
    compute_posterior_quadrature,
    sigmoid,
    validate_observations,
)

FIX_DIR = Path(__file__).resolve().parent / "fixtures"
HP = ModelHyperparams()
CAL = ["26.22", "26.21", "26.20"]


@h_settings(max_examples=25, deadline=None)
@given(
    games=st.integers(min_value=20, max_value=5000),
    b=st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False),
    m_e=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    tau_s=st.floats(min_value=0.05, max_value=0.3),
    tau_l=st.floats(min_value=0.35, max_value=1.0),
    frac=st.floats(min_value=0.1, max_value=0.75),
)
def test_non_degenerate_and_likelihood_monotonic_posterior(
    games: int, b: float, m_e: float, tau_s: float, tau_l: float, frac: float
) -> None:
    w1 = int(frac * games)
    step = max(2, int(0.05 * games))
    w2 = min(games, w1 + step)
    if w2 <= w1:
        return

    q1 = compute_posterior_quadrature(
        [LikelihoodTerm(w1, games, b, 1.0)],
        m_e=m_e,
        tau_small=tau_s,
        tau_large=tau_l,
    )
    q2 = compute_posterior_quadrature(
        [LikelihoodTerm(w2, games, b, 1.0)],
        m_e=m_e,
        tau_small=tau_s,
        tau_large=tau_l,
    )
    assert q1.ci_90_logit[0] < q1.ci_90_logit[1]
    assert 0.0 < q1.standardized_wr_delta + 0.5 < 1.0
    assert q2.mean_logit > q1.mean_logit


@h_settings(max_examples=25, deadline=None)
@given(
    n=st.integers(min_value=10, max_value=500),
    b=st.floats(min_value=-1.0, max_value=1.0),
    lam=st.floats(min_value=0.1, max_value=0.8),
)
def test_patch_discounting_monotonicity_bounded_domain(n: int, b: float, lam: float) -> None:
    p_target = min(0.90, sigmoid(b) + 0.12)
    w = min(n, max(1, int(round(p_target * n))))
    if w / n <= sigmoid(b) + 0.05:
        return
    q_new = compute_posterior_quadrature([LikelihoodTerm(w, n, b, 1.0)], m_e=0.0)
    q_old = compute_posterior_quadrature([LikelihoodTerm(w, n, b, lam)], m_e=0.0)
    assert q_new.mean_logit - q_old.mean_logit > 1e-5


@h_settings(max_examples=15, deadline=None)
@given(k=st.integers(min_value=1, max_value=4), wins=st.integers(min_value=10, max_value=90))
def test_duplicate_overlapping_source_invariance(k: int, wins: int) -> None:
    rows = [
        Observation(
            source_id=f"src_{i}",
            overlap_group=f"pool_{i}",
            source_priority=10 + i,
            is_disjoint_verified=False,
            mode="aram_mayhem",
            metric_type="win",
            patch="26.22",
            champion_id="jinx",
            entity_type="augment",
            entity_id="a1",
            games=100,
            wins=wins,
            baseline_rate=0.50,
        )
        for i in range(k)
    ]
    b1, _ = validate_observations([rows[0]], patch_calendar=CAL)
    bk, _ = validate_observations(rows, patch_calendar=CAL)
    p1 = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="all",
        primary_metric_rows=b1,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    pk = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="all",
        primary_metric_rows=bk,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    assert abs(p1.association_mean_logit - pk.association_mean_logit) < 1e-12
    assert abs(p1.association_ci_90_logit[0] - pk.association_ci_90_logit[0]) < 1e-12
    assert abs(p1.effective_sample_size - pk.effective_sample_size) < 1e-12


def test_permutation_invariance_completeness_and_item_uniqueness(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "inv.sqlite",
        graph_path=FIX_DIR / "relation_graph_sample.json",
        fixture_dir=FIX_DIR,
        block_outbound_on_inference=True,
        opgg_delay_s=0.0,
        opgg_timeout_s=5.0,
    )
    svc = RiemannService(settings, graph=load_relation_graph(FIX_DIR / "relation_graph_sample.json"))
    svc._sync_blocking(SyncRequest(catalog_source="fixture"))

    offered = ["apex_inventor", "jeweled_gauntlet", "ethereal_weapon"]
    chosen = ["it_is_critical", "neutral_aug"]
    baseline_out = None
    for off_perm in itertools.permutations(offered):
        for ch_perm in itertools.permutations(chosen):
            resp = svc.recommend_augments(
                RecommendAugmentsRequest(
                    mode="aram_mayhem",
                    champion_id="jinx",
                    chosen_augments=list(ch_perm),
                    offered_augments=list(off_perm),
                    top_item_pivots_k=0,
                )
            )
            sig = [(r.canonical_id, round(r.composite_score, 9)) for r in resp.recommendations]
            assert len(sig) == len(offered)
            assert {x[0] for x in sig} == set(offered)
            if baseline_out is None:
                baseline_out = sig
            else:
                assert sig == baseline_out

    # Item completeness & uniqueness
    owned = ["heartsteel", "lord_dominiks_regards"]
    resp_items = svc.recommend_items(
        RecommendItemsRequest(
            mode="aram_mayhem",
            champion_id="jinx",
            current_items=owned,
            top_k=5,
        )
    )
    returned_ids = [r.item_id for r in resp_items.recommendations]
    assert len(returned_ids) == 5
    assert "heartsteel" not in returned_ids
    assert "lord_dominiks_regards" not in returned_ids
    assert "mortal_reminder" not in returned_ids
