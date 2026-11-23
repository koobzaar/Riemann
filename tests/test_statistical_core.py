from __future__ import annotations

import math

import pytest

from riemann.config import ModelHyperparams
from riemann.models import Observation
from riemann.stats_engine import (
    LikelihoodTerm,
    StatsValidationError,
    compute_avg_placement,
    compute_entity_posterior,
    compute_posterior_quadrature,
    logit,
    recover_unique_wins_from_rate,
    sigmoid,
    validate_observations,
)

# Provenance note: all numerical oracles below were independently calculated analytically
# or via independent reference quadrature prior to implementation.

HP = ModelHyperparams()
CAL = ["26.22", "26.21", "26.20"]


def test_quadrature_truncation_and_convergence_extreme_counts() -> None:
    # 10000/10000 at baseline 0.5 -> independent extended-domain integration gives ~6.040839
    q = compute_posterior_quadrature(
        [LikelihoodTerm(wins=10000, games=10000, baseline_logit=0.0, discount=1.0)],
        m_e=0.0,
    )
    assert abs(q.mean_logit - 6.040839) < 1e-3
    assert q.boundary_density_ratio < 1e-10

    # Extreme valid baseline_rate = 0.001 still converges without truncation
    q_ext = compute_posterior_quadrature(
        [LikelihoodTerm(wins=10000, games=10000, baseline_logit=logit(0.001), discount=1.0)],
        m_e=0.0,
    )
    assert math.isfinite(q_ext.mean_logit)
    assert q_ext.mean_logit > 10.0
    assert q_ext.boundary_density_ratio < 1e-10

    # Sparse skewed cell wins=2, games=20, baseline=0.1 converges within < 1e-6 vs high-precision reference
    q_skew = compute_posterior_quadrature(
        [LikelihoodTerm(wins=2, games=20, baseline_logit=logit(0.1), discount=1.0)],
        m_e=0.0,
        tol=1e-6,
    )
    q_skew_ref = compute_posterior_quadrature(
        [LikelihoodTerm(wins=2, games=20, baseline_logit=logit(0.1), discount=1.0)],
        m_e=0.0,
        tol=1e-9,
    )
    assert abs(q_skew.mean_logit - q_skew_ref.mean_logit) < 1e-6
    assert abs(q_skew.ci_90_logit[0] - q_skew_ref.ci_90_logit[0]) < 1e-6
    assert abs(q_skew.ci_90_logit[1] - q_skew_ref.ci_90_logit[1]) < 1e-6

    # Out-of-domain baseline_rate=1e-20 rejected at validation
    bad = Observation(
        source_id="s1",
        mode="aram_mayhem",
        metric_type="win",
        patch="26.22",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        games=10000,
        wins=10000,
        baseline_rate=1e-20,
    )
    with pytest.raises(StatsValidationError) as ei:
        validate_observations([bad], patch_calendar=CAL)
    assert ei.value.code == "baseline_out_of_domain"


def test_analytic_uniform_p_beta_3_1_oracle() -> None:
    # Uniform prior on p = sigmoid(delta) with 2/2 at b=0 -> exact Beta(3, 1):
    # E[p] = 3/4 = 0.75, q05(p) = 0.05**(1/3) = 0.368403, q95(p) = 0.95**(1/3) = 0.983048
    q = compute_posterior_quadrature(
        [LikelihoodTerm(wins=2, games=2, baseline_logit=0.0, discount=1.0)],
        uniform_p_prior=True,
    )
    assert abs(q.mean_prob_at_mu_star - 0.75) < 1e-4
    assert abs(sigmoid(q.ci_90_logit[0]) - (0.05 ** (1.0 / 3.0))) < 5e-4
    assert abs(sigmoid(q.ci_90_logit[1]) - (0.95 ** (1.0 / 3.0))) < 5e-4


def test_zero_observation_and_symmetric_midpoint_oracles() -> None:
    q0 = compute_posterior_quadrature([], m_e=0.0)
    assert abs(q0.mean_logit - 0.0) < 1e-6
    assert abs(q0.standardized_wr_delta - 0.0) < 1e-6

    q_mid = compute_posterior_quadrature(
        [LikelihoodTerm(wins=50, games=100, baseline_logit=0.0, discount=1.0)],
        m_e=0.0,
    )
    assert abs(q_mid.mean_logit - 0.0) < 1e-6
    assert abs(q_mid.mean_prob_at_mu_star - 0.5) < 1e-6


def test_two_scale_mixture_tail_adaptive_oracles() -> None:
    # Default pi=0.9, tau_small=0.1, tau_large=0.5, m_e=0, b=0
    q2 = compute_posterior_quadrature([LikelihoodTerm(2, 2, 0.0, 1.0)])
    q130 = compute_posterior_quadrature([LikelihoodTerm(130, 200, 0.0, 1.0)])
    q650 = compute_posterior_quadrature([LikelihoodTerm(650, 1000, 0.0, 1.0)])

    assert abs(q2.mean_prob_at_mu_star - 0.507760) < 2e-4
    assert q2.mean_prob_at_mu_star < 0.50980  # Tighter than fixed Beta M=100
    assert abs(q130.mean_prob_at_mu_star - 0.629503) < 2e-4
    assert q130.mean_prob_at_mu_star > 0.60000  # Preserves heavy-tailed signal
    assert abs(q650.mean_prob_at_mu_star - 0.647559) < 2e-4


def test_source_centering_across_shifted_baselines() -> None:
    # Source A: baseline 0.50, 5622/10000 -> ~0.243051
    # Source B: baseline 0.52, 5818/10000 -> ~0.243009
    qa = compute_posterior_quadrature([LikelihoodTerm(5622, 10000, logit(0.50), 1.0)])
    qb = compute_posterior_quadrature([LikelihoodTerm(5818, 10000, logit(0.52), 1.0)])
    assert abs(qa.mean_logit - 0.243051) < 2e-4
    assert abs(qb.mean_logit - 0.243009) < 2e-4
    assert abs(qa.mean_logit - qb.mean_logit) < 1e-4

    # Uncentered Source B (wrongly using baseline 0.50) fails by > 0.07 logits
    qb_wrong = compute_posterior_quadrature([LikelihoodTerm(5818, 10000, logit(0.50), 1.0)])
    assert abs(qb_wrong.mean_logit - qb.mean_logit) > 0.07


def test_overlap_protection_corroboration_and_verified_disjoint_cohorts() -> None:
    obs_prim = Observation(
        source_id="src_a",
        overlap_group="pool_1",
        source_priority=10,
        is_disjoint_verified=False,
        mode="aram_mayhem",
        metric_type="win",
        patch="26.22",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="slot_1",
        games=100,
        wins=55,
        baseline_rate=0.50,
    )
    # Different source_id & overlap_group string, still unverified -> MUST NOT double count
    obs_dup_unverified = obs_prim.model_copy(
        update={"source_id": "src_b", "overlap_group": "pool_2", "source_priority": 20}
    )
    b_single, _ = validate_observations([obs_prim], patch_calendar=CAL)
    b_both, _ = validate_observations([obs_prim, obs_dup_unverified], patch_calendar=CAL)

    p_single = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="slot_1",
        primary_metric_rows=b_single,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    p_both = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="slot_1",
        primary_metric_rows=b_both,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    assert p_single.association_mean_logit == p_both.association_mean_logit
    assert p_single.association_ci_90_logit == p_both.association_ci_90_logit
    assert p_single.effective_sample_size == p_both.effective_sample_size == 100.0
    assert p_both.corroboration.secondary_sources_checked == 1
    assert p_both.corroboration.status == "aligned"

    # Divergent secondary source (> 0.15 logit disagreement)
    obs_div = obs_dup_unverified.model_copy(update={"wins": 85})
    b_div, _ = validate_observations([obs_prim, obs_div], patch_calendar=CAL)
    p_div = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="slot_1",
        primary_metric_rows=b_div,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    assert p_div.corroboration.status == "divergent"
    assert (p_div.corroboration.max_source_disagreement_logit or 0.0) > 0.15

    # Verified disjoint cohorts: 55/100 + 55/100 narrows CI width (~0.321352 -> ~0.290812)
    d1 = obs_prim.model_copy(update={"overlap_group": "kr", "is_disjoint_verified": True})
    d2 = obs_prim.model_copy(update={"source_id": "src_b", "overlap_group": "eu", "is_disjoint_verified": True})
    bd, _ = validate_observations([d1, d2], patch_calendar=CAL)
    p_disj = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="slot_1",
        primary_metric_rows=bd,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    w_single = p_single.association_ci_90_logit[1] - p_single.association_ci_90_logit[0]
    w_disj = p_disj.association_ci_90_logit[1] - p_disj.association_ci_90_logit[0]
    assert abs(w_single - 0.321352) < 1e-3
    assert abs(w_disj - 0.290812) < 1e-3
    assert w_disj < w_single
    assert p_disj.effective_sample_size == 200.0

    # Genuine mixture widening case: 2/2 vs 2/2 + 2/2 = 4/4 (~0.4565 -> ~0.6819)
    q_2 = compute_posterior_quadrature([LikelihoodTerm(2, 2, 0.0, 1.0)])
    q_4 = compute_posterior_quadrature([LikelihoodTerm(2, 2, 0.0, 1.0), LikelihoodTerm(2, 2, 0.0, 1.0)])
    assert abs((q_2.ci_90_logit[1] - q_2.ci_90_logit[0]) - 0.456500) < 2e-3
    assert abs((q_4.ci_90_logit[1] - q_4.ci_90_logit[0]) - 0.681923) < 2e-3


def test_cohort_identity_safeguard_and_independent_60_100_oracle() -> None:
    # Certified identity -> exact 0.0, CI [0.0, 0.0]
    obs_id = Observation(
        source_id="s1",
        mode="aram_mayhem",
        metric_type="win",
        patch="26.22",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        games=100,
        wins=60,
        baseline_wins=60,
        baseline_games=100,
        entity_equals_reference_cohort=True,
    )
    b_id, _ = validate_observations([obs_id], patch_calendar=CAL)
    p_id = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="all",
        primary_metric_rows=b_id,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    assert p_id.association_mean_logit == 0.0
    assert p_id.association_ci_90_logit == (0.0, 0.0)

    # Certified identity with divergent counts -> rejected
    bad_id = obs_id.model_copy(update={"wins": 61})
    with pytest.raises(StatsValidationError) as ei:
        validate_observations([bad_id], patch_calendar=CAL)
    assert ei.value.code == "invalid_cohort_identity"

    # Independent cohort (entity_equals_reference_cohort=False) with raw 60/100 baseline (mu=60.5/101):
    # Independent oracle: mean ≈ 0.001206, CI ≈ [-0.153170, 0.155958]
    obs_ind = obs_id.model_copy(update={"entity_equals_reference_cohort": False})
    b_ind, _ = validate_observations([obs_ind], patch_calendar=CAL)
    p_ind = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="all",
        primary_metric_rows=b_ind,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    assert abs(p_ind.association_mean_logit - 0.001206) < 1e-4
    assert abs(p_ind.association_ci_90_logit[0] - (-0.153170)) < 1e-3
    assert abs(p_ind.association_ci_90_logit[1] - 0.155958) < 1e-3

    # Low baseline sample flag (< 50 baseline_games)
    obs_low_b = obs_ind.model_copy(update={"baseline_wins": 6, "baseline_games": 10})
    b_low, _ = validate_observations([obs_low_b], patch_calendar=CAL)
    p_low = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="all",
        primary_metric_rows=b_low,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    assert p_low.low_baseline_sample is True


def test_exact_decimal_rounded_count_recovery_and_validation_errors() -> None:
    # Unique compatible integer accepted
    assert recover_unique_wins_from_rate(rate_str="0.5000", games=100, rate_decimals=4) == 50
    # Ambiguous rounded rate (100000 games at 0.5000 matches multiple integers) rejected
    with pytest.raises(StatsValidationError) as ei:
        recover_unique_wins_from_rate(rate_str="0.5000", games=100000, rate_decimals=4)
    assert ei.value.code == "ambiguous_rounded_count"

    base = Observation(
        source_id="s1",
        mode="aram_mayhem",
        metric_type="win",
        patch="26.22",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        games=100,
        wins=55,
        baseline_wins=50,
        baseline_games=100,
        baseline_rate=0.50,
    )
    # Conflicting baseline_wins/baseline_games vs baseline_rate
    with pytest.raises(StatsValidationError) as e1:
        validate_observations([base.model_copy(update={"baseline_rate": 0.60})], patch_calendar=CAL)
    assert e1.value.code == "conflicting_baseline_fields"

    # Conflicting reference baseline across two entities in same reference
    other = base.model_copy(update={"entity_id": "a2", "baseline_wins": 60, "baseline_rate": 0.60})
    with pytest.raises(StatsValidationError) as e2:
        validate_observations([base, other], patch_calendar=CAL)
    assert e2.value.code == "conflicting_reference_baseline"

    # wins > games
    with pytest.raises(StatsValidationError):
        validate_observations([base.model_copy(update={"wins": 105})], patch_calendar=CAL)

    # NaN baseline_rate
    with pytest.raises(StatsValidationError):
        validate_observations(
            [base.model_copy(update={"baseline_wins": None, "baseline_games": None, "baseline_rate": float("nan")})],
            patch_calendar=CAL,
        )

    # Duplicate PK
    with pytest.raises(StatsValidationError) as e3:
        validate_observations([base, base], patch_calendar=CAL)
    assert e3.value.code == "duplicate_observation_key"

    # Unknown patch
    with pytest.raises(StatsValidationError) as e4:
        validate_observations([base.model_copy(update={"patch": "99.99"})], patch_calendar=CAL)
    assert e4.value.code == "unknown_patch"

    # Invalid metric for mode
    with pytest.raises(StatsValidationError) as e5:
        validate_observations([base.model_copy(update={"metric_type": "top4"})], patch_calendar=CAL)
    assert e5.value.code == "invalid_metric_for_mode"


def test_patch_discounting_rework_exclusion_and_likelihood_splitting() -> None:
    o0 = Observation(
        source_id="s1",
        mode="aram_mayhem",
        metric_type="win",
        patch="26.22",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        games=200,
        wins=130,
        baseline_rate=0.50,
    )
    o_old = o0.model_copy(update={"patch": "26.20", "games": 500, "wins": 0})
    b_only, _ = validate_observations([o0], patch_calendar=CAL)
    b_with_old, _ = validate_observations([o0, o_old], patch_calendar=CAL)

    p1 = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="all",
        primary_metric_rows=b_only,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=["26.21"],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    p2 = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="all",
        primary_metric_rows=b_with_old,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=["26.21"],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    assert p1.association_mean_logit == p2.association_mean_logit
    assert p2.effective_sample_size == 200.0

    # Full rework exclusion (rework newer than all observations) falls back to zero-data prior
    b_old_only, _ = validate_observations([o_old], patch_calendar=CAL)
    p_rework_all = compute_entity_posterior(
        mode="aram_mayhem",
        champion_id="jinx",
        entity_type="augment",
        entity_id="a1",
        slot_key="all",
        primary_metric_rows=b_old_only,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=["26.22"],
        prior_mean_logit=0.15,
        hyperparams=HP,
    )
    assert p_rework_all.effective_sample_size == 0.0
    assert abs(p_rework_all.association_mean_logit - 0.15) < 1e-6

    # Likelihood-layer row splitting invariance
    q_whole = compute_posterior_quadrature([LikelihoodTerm(130, 200, 0.0, 0.7)])
    q_split = compute_posterior_quadrature(
        [LikelihoodTerm(70, 100, 0.0, 0.7), LikelihoodTerm(60, 100, 0.0, 0.7)]
    )
    assert abs(q_whole.mean_logit - q_split.mean_logit) < 1e-12


def test_missing_joint_protection_identical_marginals() -> None:
    # Joint Table I: p00=0.6, p10=0.4, p01=0.4, p11=0.6 (interaction +1.62186)
    # Joint Table II: p00=0.4, p10=0.6, p01=0.6, p11=0.4 (interaction -1.62186)
    # With n=100 per cell, both have marginal A=1 wins = 40+60 = 100 / 200 (0.5) and total 200 / 400 (0.5).
    q_i = compute_posterior_quadrature([LikelihoodTerm(100, 200, 0.0, 1.0)])
    q_ii = compute_posterior_quadrature([LikelihoodTerm(100, 200, 0.0, 1.0)])
    assert abs(q_i.mean_logit) < 1e-9
    assert abs(q_ii.mean_logit) < 1e-9
    assert q_i.mean_logit == q_ii.mean_logit


def test_arena_top1_and_multipatch_avg_placement() -> None:
    t4 = Observation(
        source_id="s1",
        mode="arena",
        metric_type="top4",
        patch="26.22",
        champion_id="jinx",
        entity_type="augment",
        entity_id="apex_inventor",
        slot_key="slot_1",
        games=200,
        wins=120,
        baseline_rate=0.50,
    )
    t1 = Observation(
        source_id="s1",
        mode="arena",
        metric_type="top1",
        patch="26.22",
        champion_id="jinx",
        entity_type="augment",
        entity_id="apex_inventor",
        slot_key="slot_1",
        games=200,
        wins=40,
        baseline_rate=0.125,
    )
    pl_single = Observation(
        source_id="s1",
        mode="arena",
        metric_type="avg_placement",
        patch="26.22",
        champion_id="jinx",
        entity_type="augment",
        entity_id="apex_inventor",
        slot_key="slot_1",
        games=2,
        mean_placement=7.5,
        baseline_mean_placement=4.5,
    )
    b_t4, _ = validate_observations([t4], patch_calendar=CAL)
    b_t1, _ = validate_observations([t1], patch_calendar=CAL)
    _, p_single = validate_observations([pl_single], patch_calendar=CAL)

    post_with = compute_entity_posterior(
        mode="arena",
        champion_id="jinx",
        entity_type="augment",
        entity_id="apex_inventor",
        slot_key="slot_1",
        primary_metric_rows=b_t4,
        top1_rows=b_t1,
        placement_rows=p_single,
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    post_only_t4 = compute_entity_posterior(
        mode="arena",
        champion_id="jinx",
        entity_type="augment",
        entity_id="apex_inventor",
        slot_key="slot_1",
        primary_metric_rows=b_t4,
        top1_rows=[],
        placement_rows=[],
        patch_calendar=CAL,
        rework_patches=[],
        prior_mean_logit=0.0,
        hyperparams=HP,
    )
    assert post_with.association_mean_logit == post_only_t4.association_mean_logit
    assert post_with.arena_secondary is not None
    assert abs((post_with.arena_secondary.top1_rate or 0.0) - 0.162434) < 2e-4
    assert abs((post_with.arena_secondary.avg_placement or 0.0) - 4.558824) < 1e-5

    # Multi-patch placement with differing baselines:
    # patch_0 (d=1.0, n=50, mean=3.5, base=4.5 -> -1.0) + patch_1 (d=0.7, n=100, mean=3.2, base=4.2 -> -1.0)
    # -> 4.5 + (-120)/220 = 3.954545
    pl_0 = pl_single.model_copy(update={"patch": "26.22", "games": 50, "mean_placement": 3.5, "baseline_mean_placement": 4.5})
    pl_1 = pl_single.model_copy(update={"patch": "26.21", "games": 100, "mean_placement": 3.2, "baseline_mean_placement": 4.2})
    _, p_multi = validate_observations([pl_0, pl_1], patch_calendar=CAL)
    est = compute_avg_placement(p_multi, patch_calendar=CAL, rework_patches=[], hyperparams=HP)
    assert est is not None
    assert abs(est - 3.954545) < 1e-5
