from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from typing import Callable, Iterable, Sequence

from riemann.config import ModelHyperparams
from riemann.models import (
    ArenaSecondary,
    CorroborationInfo,
    GameMode,
    Observation,
    PrecomputedPosterior,
)

BASELINE_MIN = 0.001
BASELINE_MAX = 0.999
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class StatsValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def logit(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise StatsValidationError("invalid_probability", f"Probability out of (0,1): {p}")
    return math.log(p / (1.0 - p))


def log_sigmoid(z: float) -> float:
    if z >= 0.0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


def log_one_minus_sigmoid(z: float) -> float:
    if z >= 0.0:
        return -z - math.log1p(math.exp(-z))
    return -math.log1p(math.exp(z))


def recover_unique_wins_from_rate(
    *,
    rate_str: str,
    games: int,
    rate_decimals: int | None = None,
    rounding_mode: str = "ROUND_HALF_UP",
) -> int:
    """Recover exact integer wins from a published rounded rate string using Decimal arithmetic.

    Raises StatsValidationError('ambiguous_rounded_count') if 0 or >=2 integers match.
    """
    if games < 1:
        raise StatsValidationError("invalid_games", f"games must be >= 1, got {games}")
    try:
        target = Decimal(rate_str.strip())
    except Exception as exc:
        raise StatsValidationError("invalid_rate_str", f"Invalid rate_str: {rate_str}") from exc

    if rate_decimals is None:
        exp = target.as_tuple().exponent
        if not isinstance(exp, int):
            raise StatsValidationError("invalid_rate_str", f"Non-finite rate_str: {rate_str}")
        rate_decimals = max(0, -exp)

    quant = Decimal(10) ** (-rate_decimals)
    rmode = ROUND_HALF_UP if rounding_mode == "ROUND_HALF_UP" else ROUND_HALF_EVEN
    target_q = target.quantize(quant, rounding=rmode)

    step = Decimal(10) ** (-rate_decimals)
    lo = max(0, int(((target_q - step) * games).to_integral_value(rounding="ROUND_FLOOR")))
    hi = min(games, int(((target_q + step) * games).to_integral_value(rounding="ROUND_CEILING")))

    d_games = Decimal(games)
    matches: list[int] = []
    for w in range(lo, hi + 1):
        val = (Decimal(w) / d_games).quantize(quant, rounding=rmode)
        if val == target_q:
            matches.append(w)
            if len(matches) > 1:
                raise StatsValidationError(
                    "ambiguous_rounded_count",
                    f"Multiple integer wins match rate_str={rate_str} for games={games}",
                )
    if len(matches) != 1:
        raise StatsValidationError(
            "ambiguous_rounded_count",
            f"No integer wins match rate_str={rate_str} for games={games}",
        )
    return matches[0]


@dataclass(frozen=True)
class ValidatedBinaryObs:
    obs: Observation
    wins: int
    games: int
    baseline_rate: float
    baseline_logit: float
    baseline_games: int | None
    pick_rate: float | None


@dataclass(frozen=True)
class ValidatedPlacementObs:
    obs: Observation
    games: int
    mean_placement: float
    baseline_mean_placement: float


def validate_observations(
    observations: Sequence[Observation],
    *,
    patch_calendar: Sequence[str],
) -> tuple[list[ValidatedBinaryObs], list[ValidatedPlacementObs]]:
    """Validate all adapter observations strictly before staging/persistence."""
    valid_patches = set(patch_calendar)
    seen_pks: set[tuple[str, ...]] = set()
    ref_baselines: dict[tuple[str, str, str, str, str, str, str], float] = {}

    binary_out: list[ValidatedBinaryObs] = []
    placement_out: list[ValidatedPlacementObs] = []

    for obs in observations:
        pk = obs.primary_key()
        if pk in seen_pks:
            raise StatsValidationError("duplicate_observation_key", f"Duplicate observation key: {pk}")
        seen_pks.add(pk)

        if obs.patch not in valid_patches:
            raise StatsValidationError("unknown_patch", f"Patch {obs.patch!r} not in patch_calendar")

        if obs.mode == "aram_mayhem" and obs.metric_type != "win":
            raise StatsValidationError(
                "invalid_metric_for_mode",
                f"Metric {obs.metric_type!r} invalid for mode aram_mayhem",
            )
        if obs.mode == "arena" and obs.metric_type not in {"top4", "top1", "avg_placement"}:
            raise StatsValidationError(
                "invalid_metric_for_mode",
                f"Metric {obs.metric_type!r} invalid for mode arena",
            )

        if obs.slot_key != "all":
            if not obs.slot_key.startswith("slot_") or not obs.slot_key[5:].isdigit():
                raise StatsValidationError("invalid_slot_key", f"Invalid slot_key: {obs.slot_key}")
            slot_num = int(obs.slot_key[5:])
            if slot_num < 1 or slot_num > 6:
                raise StatsValidationError("invalid_slot_key", f"slot_key out of 1..6: {obs.slot_key}")

        # Tier-only metadata row (no countable denominator) -> skip statistical ingestion cleanly
        if obs.games is None and obs.curated_tier is not None:
            continue

        if obs.games is None or obs.games < 1:
            raise StatsValidationError("invalid_games", f"games must be integer >= 1, got {obs.games}")

        if obs.metric_type == "avg_placement":
            if obs.mean_placement is None or not math.isfinite(obs.mean_placement):
                raise StatsValidationError("invalid_placement", "mean_placement must be finite")
            if not (1.0 <= obs.mean_placement <= 8.0):
                raise StatsValidationError(
                    "invalid_placement", f"mean_placement out of [1,8]: {obs.mean_placement}"
                )
            b_mean = (
                obs.baseline_mean_placement
                if obs.baseline_mean_placement is not None
                else 4.5
            )
            if not math.isfinite(b_mean) or not (1.0 <= b_mean <= 8.0):
                raise StatsValidationError(
                    "invalid_placement_baseline", f"baseline_mean_placement out of [1,8]: {b_mean}"
                )
            placement_out.append(
                ValidatedPlacementObs(
                    obs=obs,
                    games=obs.games,
                    mean_placement=obs.mean_placement,
                    baseline_mean_placement=b_mean,
                )
            )
            continue

        # Binary observation
        if obs.wins is not None:
            wins = obs.wins
        elif obs.rate_str is not None:
            wins = recover_unique_wins_from_rate(
                rate_str=obs.rate_str,
                games=obs.games,
                rate_decimals=obs.rate_decimals,
                rounding_mode=obs.rounding_mode,
            )
        else:
            raise StatsValidationError(
                "missing_wins", "Binary observation requires integer wins or recoverable rate_str"
            )

        if wins < 0 or wins > obs.games:
            raise StatsValidationError(
                "invalid_wins", f"wins ({wins}) must satisfy 0 <= wins <= games ({obs.games})"
            )

        # Baseline validation
        has_raw_base = obs.baseline_wins is not None or obs.baseline_games is not None
        if has_raw_base:
            if obs.baseline_wins is None or obs.baseline_games is None:
                raise StatsValidationError(
                    "incomplete_baseline_counts",
                    "Both baseline_wins and baseline_games must be provided together",
                )
            if obs.baseline_games < 1 or obs.baseline_wins < 0 or obs.baseline_wins > obs.baseline_games:
                raise StatsValidationError(
                    "invalid_baseline_counts",
                    f"Invalid baseline counts: {obs.baseline_wins}/{obs.baseline_games}",
                )
            raw_ratio = obs.baseline_wins / obs.baseline_games
            if obs.baseline_rate is not None:
                if not math.isfinite(obs.baseline_rate):
                    raise StatsValidationError("invalid_baseline_rate", "baseline_rate must be finite")
                if abs(raw_ratio - obs.baseline_rate) > 1e-4:
                    raise StatsValidationError(
                        "conflicting_baseline_fields",
                        f"baseline_wins/baseline_games ({raw_ratio}) conflicts with baseline_rate ({obs.baseline_rate})",
                    )
            b_rate = (obs.baseline_wins + 0.5) / (obs.baseline_games + 1.0)
            b_games: int | None = obs.baseline_games
        elif obs.baseline_rate is not None:
            if not math.isfinite(obs.baseline_rate):
                raise StatsValidationError("invalid_baseline_rate", "baseline_rate must be finite")
            b_rate = obs.baseline_rate
            b_games = None
        else:
            raise StatsValidationError(
                "missing_baseline", "Binary observation requires baseline counts or baseline_rate"
            )

        if not (BASELINE_MIN <= b_rate <= BASELINE_MAX):
            raise StatsValidationError(
                "baseline_out_of_domain",
                f"Baseline rate {b_rate} outside [{BASELINE_MIN}, {BASELINE_MAX}]",
            )

        if obs.entity_equals_reference_cohort:
            if not has_raw_base or wins != obs.baseline_wins or obs.games != obs.baseline_games:
                raise StatsValidationError(
                    "invalid_cohort_identity",
                    "entity_equals_reference_cohort=True requires raw baseline counts matching wins and games",
                )

        ref_key = (
            obs.source_id,
            obs.cohort_id,
            obs.mode,
            obs.metric_type,
            obs.patch,
            obs.champion_id,
            obs.slot_key,
        )
        prev_base = ref_baselines.get(ref_key)
        if prev_base is not None and abs(prev_base - b_rate) > 1e-6:
            raise StatsValidationError(
                "conflicting_reference_baseline",
                f"Conflicting baselines for {ref_key}: {prev_base} vs {b_rate}",
            )
        ref_baselines[ref_key] = b_rate

        # Pick rate validation
        if obs.pick_rate is not None:
            if not math.isfinite(obs.pick_rate) or not (0.0 <= obs.pick_rate <= 1.0):
                raise StatsValidationError("invalid_pick_rate", f"pick_rate out of [0,1]: {obs.pick_rate}")
            pr: float | None = obs.pick_rate
        elif obs.champion_stratum_games is not None:
            if obs.champion_stratum_games < obs.games:
                raise StatsValidationError(
                    "invalid_pick_rate",
                    f"champion_stratum_games ({obs.champion_stratum_games}) < games ({obs.games})",
                )
            pr = obs.games / obs.champion_stratum_games
        else:
            pr = None

        binary_out.append(
            ValidatedBinaryObs(
                obs=obs,
                wins=wins,
                games=obs.games,
                baseline_rate=b_rate,
                baseline_logit=logit(b_rate),
                baseline_games=b_games,
                pick_rate=pr,
            )
        )

    return binary_out, placement_out


@dataclass(frozen=True)
class LikelihoodTerm:
    wins: int
    games: int
    baseline_logit: float
    discount: float


@dataclass(frozen=True)
class QuadratureResult:
    mean_logit: float
    ci_90_logit: tuple[float, float]
    mean_prob_at_mu_star: float
    standardized_wr_delta: float
    effective_sample_size: float
    boundary_density_ratio: float


def _log_two_scale_prior(
    delta: float, *, m_e: float, pi_small: float, tau_small: float, tau_large: float
) -> float:
    a = (
        math.log(pi_small)
        - 0.5 * ((delta - m_e) / tau_small) ** 2
        - math.log(tau_small)
        - LOG_SQRT_2PI
    )
    b = (
        math.log(1.0 - pi_small)
        - 0.5 * ((delta - m_e) / tau_large) ** 2
        - math.log(tau_large)
        - LOG_SQRT_2PI
    )
    mx = a if a >= b else b
    return mx + math.log(math.exp(a - mx) + math.exp(b - mx))


def _log_uniform_p_prior(delta: float) -> float:
    """Jacobian prior corresponding to Uniform(0,1) on p = sigmoid(delta)."""
    return log_sigmoid(delta) + log_one_minus_sigmoid(delta)


def _eval_simpson_panels(
    panels: Sequence[tuple[float, float, int]],
    log_fn: Callable[[float], float],
    b_star: float,
) -> tuple[float, float, float, float, float]:
    """Composite Simpson's rule (O(h^4) accuracy) over contiguous intervals with even sub-interval counts."""
    xs: list[float] = []
    panel_meta: list[tuple[int, int, float]] = []  # (start_idx, n_sub, h)
    for idx, (a, b, n_sub) in enumerate(panels):
        if b <= a or n_sub <= 0:
            continue
        if n_sub % 2 == 1:
            n_sub += 1
        h = (b - a) / n_sub
        start = len(xs) - 1 if xs else 0
        if not xs:
            xs.append(a)
        for k in range(1, n_sub + 1):
            xs.append(a + k * h)
        panel_meta.append((start, n_sub, h))

    lps = [log_fn(x) for x in xs]
    mx = max(lps)
    fs = [math.exp(lp - mx) for lp in lps]
    boundary_ratio = max(fs[0], fs[-1])

    # Accumulate Simpson weights and store per-segment quadratic interpolant for O(h^4) quantiles
    z = 0.0
    m_d_num = 0.0
    m_p_num = 0.0
    cdf_vals: list[float] = [0.0]
    seg_meta: list[tuple[float, float, float, float, float]] = []  # (x0, h, f0, f1, f2)

    for start, n_sub, h in panel_meta:
        h3 = h / 3.0
        for k in range(0, n_sub, 2):
            i0 = start + k
            i1 = i0 + 1
            i2 = i0 + 2
            x0, x1, x2 = xs[i0], xs[i1], xs[i2]
            f0, f1, f2 = fs[i0], fs[i1], fs[i2]
            seg_z = h3 * (f0 + 4.0 * f1 + f2)
            z += seg_z
            m_d_num += h3 * (x0 * f0 + 4.0 * x1 * f1 + x2 * f2)
            m_p_num += h3 * (
                sigmoid(b_star + x0) * f0
                + 4.0 * sigmoid(b_star + x1) * f1
                + sigmoid(b_star + x2) * f2
            )
            cdf_vals.append(z)
            seg_meta.append((x0, h, f0, f1, f2))

    mean_d = m_d_num / z
    mean_p = m_p_num / z

    def quantile(q: float) -> float:
        target = q * z
        lo_i, hi_i = 0, len(cdf_vals) - 1
        while lo_i + 1 < hi_i:
            mid = (lo_i + hi_i) // 2
            if cdf_vals[mid] <= target:
                lo_i = mid
            else:
                hi_i = mid
        rem = target - cdf_vals[lo_i]
        seg_total = cdf_vals[hi_i] - cdf_vals[lo_i]
        x0, h, f0, f1, f2 = seg_meta[lo_i]
        if seg_total <= 0.0:
            return x0
        # Exact integral of quadratic Lagrange interpolant through (0, f0), (h, f1), (2h, f2):
        # f(t) = f0 + c1*(t/h) + c2*(t/h)^2, u = t/h in [0, 2]
        # Integral_0^u f = h * (f0*u + 0.5*c1*u^2 + (1/3)*c2*u^3)
        c1 = -1.5 * f0 + 2.0 * f1 - 0.5 * f2
        c2 = 0.5 * f0 - f1 + 0.5 * f2
        u_lo, u_hi = 0.0, 2.0
        for _ in range(32):
            u_mid = 0.5 * (u_lo + u_hi)
            val = h * (f0 * u_mid + 0.5 * c1 * u_mid * u_mid + (c2 / 3.0) * u_mid * u_mid * u_mid)
            if val < rem:
                u_lo = u_mid
            else:
                u_hi = u_mid
        return x0 + 0.5 * (u_lo + u_hi) * h

    return mean_d, quantile(0.05), quantile(0.95), mean_p, boundary_ratio


def compute_posterior_quadrature(
    terms: Sequence[LikelihoodTerm],
    *,
    m_e: float = 0.0,
    mu_star: float = 0.5,
    pi_small: float = 0.90,
    tau_small: float = 0.10,
    tau_large: float = 0.50,
    uniform_p_prior: bool = False,
    tol: float = 1e-6,
) -> QuadratureResult:
    """Convergent adaptive Laplace-centered 3-panel Simpson quadrature (O(h^4))."""
    active_terms = [t for t in terms if t.discount > 0.0 and t.games > 0]
    n_eff = sum(t.discount * t.games for t in active_terms)

    # Precompute constants for tight inner loop
    log_pi_s = math.log(pi_small) - math.log(tau_small) - LOG_SQRT_2PI
    log_pi_l = math.log(1.0 - pi_small) - math.log(tau_large) - LOG_SQRT_2PI
    inv_2ts2 = 0.5 / (tau_small * tau_small)
    inv_2tl2 = 0.5 / (tau_large * tau_large)
    term_tuples = [(t.wins, t.games - t.wins, t.baseline_logit, t.discount) for t in active_terms]

    def log_post(delta: float) -> float:
        if uniform_p_prior:
            lp = log_sigmoid(delta) + log_one_minus_sigmoid(delta)
        else:
            d = delta - m_e
            d2 = d * d
            a = log_pi_s - d2 * inv_2ts2
            b = log_pi_l - d2 * inv_2tl2
            mx = a if a >= b else b
            lp = mx + math.log(math.exp(a - mx) + math.exp(b - mx))
        ll = 0.0
        for w, loss, bp, disc in term_tuples:
            z = bp + delta
            if z >= 0.0:
                emz = math.exp(-z)
                l1p = math.log1p(emz)
                ll += disc * (-w * l1p + loss * (-z - l1p))
            else:
                ez = math.exp(z)
                l1p = math.log1p(ez)
                ll += disc * (w * (z - l1p) - loss * l1p)
        return lp + ll

    def deriv(x: float, h: float = 1e-5) -> float:
        return (log_post(x + h) - log_post(x - h)) / (2.0 * h)

    # 1. Dynamic bracketing around m_e
    span = max(8.0, 10.0 * tau_large)
    left = m_e - span
    right = m_e + span
    while deriv(left) <= 0.0 and left > -200.0:
        left -= span
        span *= 1.5
    span = max(8.0, 10.0 * tau_large)
    while deriv(right) >= 0.0 and right < 200.0:
        right += span
        span *= 1.5

    scan_n = 80
    step = (right - left) / scan_n
    best_i = 0
    best_val = -math.inf
    for i in range(scan_n + 1):
        xi = left + i * step
        vi = log_post(xi)
        if vi > best_val:
            best_val = vi
            best_i = i
    a = max(left, left + (best_i - 1) * step)
    b = min(right, left + (best_i + 1) * step)

    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = log_post(c)
    fd = log_post(d)
    while (b - a) > 1e-8:
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = log_post(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = log_post(d)
    x_map = 0.5 * (a + b)
    f_map = log_post(x_map)

    eps = 1e-4
    curv = max(-(log_post(x_map + eps) - 2.0 * f_map + log_post(x_map - eps)) / (eps * eps), 1e-4)
    sd_loc = min(max(1.0 / math.sqrt(curv), 1e-4), 5.0)

    # 2. Expand integration bounds until boundary density < 1e-12 * peak
    lo = min(m_e - 8.0 * tau_large, x_map - 10.0 * sd_loc)
    hi = max(m_e + 8.0 * tau_large, x_map + 10.0 * sd_loc)
    if uniform_p_prior:
        lo = min(lo, -16.0)
        hi = max(hi, 16.0)

    pad = max(2.0 * tau_large, 4.0 * sd_loc)
    while (log_post(lo) - f_map) > -27.7 and lo > -250.0:
        lo -= pad
        pad *= 1.5
    pad = max(2.0 * tau_large, 4.0 * sd_loc)
    while (log_post(hi) - f_map) > -27.7 and hi < 250.0:
        hi += pad
        pad *= 1.5

    # 3. 3-panel Composite Simpson rule (tail-left, local peak, tail-right), doubled until max_diff < tol
    b_star = logit(mu_star)
    loc_lo = max(lo, x_map - 8.0 * sd_loc)
    loc_hi = min(hi, x_map + 8.0 * sd_loc)

    def make_panels(scale: int) -> list[tuple[float, float, int]]:
        panels: list[tuple[float, float, int]] = []
        if loc_lo > lo + 1e-12:
            panels.append((lo, loc_lo, 24 * scale))
        panels.append((loc_lo, loc_hi, 80 * scale))
        if hi > loc_hi + 1e-12:
            panels.append((loc_hi, hi, 24 * scale))
        return panels

    scale = 1
    prev = _eval_simpson_panels(make_panels(scale), log_post, b_star)
    for _ in range(8):
        scale *= 2
        curr = _eval_simpson_panels(make_panels(scale), log_post, b_star)
        max_diff = max(
            abs(curr[0] - prev[0]),
            abs(curr[1] - prev[1]),
            abs(curr[2] - prev[2]),
            abs(curr[3] - prev[3]),
        )
        prev = curr
        if max_diff < tol:
            break

    mean_d, q05, q95, mean_p, b_ratio = prev
    return QuadratureResult(
        mean_logit=mean_d,
        ci_90_logit=(q05, q95),
        mean_prob_at_mu_star=mean_p,
        standardized_wr_delta=mean_p - mu_star,
        effective_sample_size=n_eff,
        boundary_density_ratio=b_ratio,
    )


def compute_patch_discount(
    *,
    patch: str,
    patch_calendar: Sequence[str],
    rework_patches: Iterable[str | None],
    decay: float,
) -> float:
    """Compute d_p = 1{compatible patch} * lambda^age(p).

    patch_calendar is ordered newest-first (index 0 = age 0).
    """
    idx_map = {p: i for i, p in enumerate(patch_calendar)}
    if patch not in idx_map:
        raise StatsValidationError("unknown_patch", f"Patch {patch!r} not in patch_calendar")
    p_idx = idx_map[patch]
    for rp in rework_patches:
        if rp is not None and rp in idx_map:
            # Newer patches have smaller index; strictly older patch has index > rework index
            if p_idx > idx_map[rp]:
                return 0.0
    return decay**p_idx


def select_primary_and_secondary_binary(
    rows: Sequence[ValidatedBinaryObs],
) -> tuple[list[ValidatedBinaryObs], dict[tuple[str, str], list[ValidatedBinaryObs]]]:
    """Select primary cohorts and secondary corroboration cohorts for one stratum & metric.

    Rules:
    - Partition per patch into verified disjoint groups (is_disjoint_verified=True) and ONE
      shared unverified pool (is_disjoint_verified=False, regardless of overlap_group).
    - Within each verified overlap_group (per patch), select 1 winner by
      (source_priority ASC, source_id ASC, cohort_id ASC).
    - Within the shared unverified pool (per patch), select at most 1 overall winner by
      (source_priority ASC, source_id ASC, cohort_id ASC); all others route to corroboration.
    """
    by_patch: dict[str, list[ValidatedBinaryObs]] = {}
    for r in rows:
        by_patch.setdefault(r.obs.patch, []).append(r)

    primary: list[ValidatedBinaryObs] = []
    secondary_by_src: dict[tuple[str, str], list[ValidatedBinaryObs]] = {}

    def sort_key(r: ValidatedBinaryObs) -> tuple[int, str, str]:
        return (r.obs.source_priority, r.obs.source_id, r.obs.cohort_id)

    for patch_rows in by_patch.values():
        verified_groups: dict[str, list[ValidatedBinaryObs]] = {}
        unverified: list[ValidatedBinaryObs] = []
        for r in patch_rows:
            if r.obs.is_disjoint_verified:
                verified_groups.setdefault(r.obs.overlap_group, []).append(r)
            else:
                unverified.append(r)

        for grp_rows in verified_groups.values():
            grp_sorted = sorted(grp_rows, key=sort_key)
            primary.append(grp_sorted[0])
            for sec in grp_sorted[1:]:
                secondary_by_src.setdefault((sec.obs.source_id, sec.obs.cohort_id), []).append(sec)

        if unverified:
            unv_sorted = sorted(unverified, key=sort_key)
            primary.append(unv_sorted[0])
            for sec in unv_sorted[1:]:
                secondary_by_src.setdefault((sec.obs.source_id, sec.obs.cohort_id), []).append(sec)

    return primary, secondary_by_src


def compute_avg_placement(
    rows: Sequence[ValidatedPlacementObs],
    *,
    patch_calendar: Sequence[str],
    rework_patches: Iterable[str | None],
    hyperparams: ModelHyperparams,
) -> float | None:
    """Multi-patch centered shrinkage for Arena average placement."""
    if not rows:
        return None

    # Primary selection per patch (same deterministic priority)
    by_patch: dict[str, list[ValidatedPlacementObs]] = {}
    for r in rows:
        by_patch.setdefault(r.obs.patch, []).append(r)

    idx_map = {p: i for i, p in enumerate(patch_calendar)}
    selected: list[tuple[int, float, ValidatedPlacementObs]] = []
    for patch, p_rows in by_patch.items():
        d_p = compute_patch_discount(
            patch=patch,
            patch_calendar=patch_calendar,
            rework_patches=rework_patches,
            decay=hyperparams.patch_decay,
        )
        if d_p <= 0.0:
            continue
        p_sorted = sorted(
            p_rows, key=lambda r: (r.obs.source_priority, r.obs.source_id, r.obs.cohort_id)
        )
        winner = p_sorted[0]
        selected.append((idx_map[patch], d_p, winner))

    if not selected:
        return None

    selected.sort(key=lambda item: item[0])
    r_ref = selected[0][2].baseline_mean_placement
    num = 0.0
    den = 0.0
    for _, d_p, r in selected:
        num += d_p * r.games * (r.mean_placement - r.baseline_mean_placement)
        den += d_p * r.games
    est = r_ref + num / (den + hyperparams.placement_prior_weight)
    return min(8.0, max(1.0, est))


def compute_entity_posterior(
    *,
    mode: GameMode,
    champion_id: str,
    entity_type: str,
    entity_id: str,
    slot_key: str,
    primary_metric_rows: Sequence[ValidatedBinaryObs],
    top1_rows: Sequence[ValidatedBinaryObs],
    placement_rows: Sequence[ValidatedPlacementObs],
    patch_calendar: Sequence[str],
    rework_patches: Iterable[str | None],
    prior_mean_logit: float,
    hyperparams: ModelHyperparams,
) -> PrecomputedPosterior:
    """Compute the full PrecomputedPosterior for one (mode, champion, entity, slot)."""
    primary_rows, secondary_by_src = select_primary_and_secondary_binary(primary_metric_rows)

    # Filter through patch compatibility and cohort-identity rules
    primary_terms_by_patch: dict[str, list[LikelihoodTerm]] = {}
    surviving_identity_rows = 0
    surviving_non_identity_rows = 0
    pick_rates: list[float] = []
    baseline_games_list: list[int] = []

    for r in primary_rows:
        d_p = compute_patch_discount(
            patch=r.obs.patch,
            patch_calendar=patch_calendar,
            rework_patches=rework_patches,
            decay=hyperparams.patch_decay,
        )
        if d_p <= 0.0:
            continue
        if r.pick_rate is not None:
            pick_rates.append(r.pick_rate)
        if r.baseline_games is not None:
            baseline_games_list.append(r.baseline_games)
        if r.obs.entity_equals_reference_cohort:
            surviving_identity_rows += 1
            continue
        surviving_non_identity_rows += 1
        primary_terms_by_patch.setdefault(r.obs.patch, []).append(
            LikelihoodTerm(
                wins=r.wins,
                games=r.games,
                baseline_logit=r.baseline_logit,
                discount=d_p,
            )
        )

    all_primary_terms = [t for lst in primary_terms_by_patch.values() for t in lst]

    if surviving_identity_rows > 0 and surviving_non_identity_rows == 0:
        # Degenerate certified cohort identity
        assoc_mean = 0.0
        assoc_ci = (0.0, 0.0)
        std_delta = 0.0
        n_eff = sum(
            compute_patch_discount(
                patch=r.obs.patch,
                patch_calendar=patch_calendar,
                rework_patches=rework_patches,
                decay=hyperparams.patch_decay,
            )
            * r.games
            for r in primary_rows
            if r.obs.entity_equals_reference_cohort
        )
    else:
        qres = compute_posterior_quadrature(
            all_primary_terms,
            m_e=prior_mean_logit,
            mu_star=0.50,
            pi_small=hyperparams.pi_small,
            tau_small=hyperparams.tau_small,
            tau_large=hyperparams.tau_large,
            tol=hyperparams.quadrature_tol,
        )
        assoc_mean = qres.mean_logit
        assoc_ci = qres.ci_90_logit
        std_delta = qres.standardized_wr_delta
        n_eff = qres.effective_sample_size

    # Same-patch corroboration against secondary overlapping sources
    sec_checked = 0
    max_disagreement: float | None = None
    for sec_rows in secondary_by_src.values():
        sec_by_patch: dict[str, list[LikelihoodTerm]] = {}
        for r in sec_rows:
            if r.obs.patch not in primary_terms_by_patch:
                continue
            d_p = compute_patch_discount(
                patch=r.obs.patch,
                patch_calendar=patch_calendar,
                rework_patches=rework_patches,
                decay=hyperparams.patch_decay,
            )
            if d_p <= 0.0 or r.obs.entity_equals_reference_cohort:
                continue
            sec_by_patch.setdefault(r.obs.patch, []).append(
                LikelihoodTerm(
                    wins=r.wins,
                    games=r.games,
                    baseline_logit=r.baseline_logit,
                    discount=d_p,
                )
            )
        shared_patches = sorted(sec_by_patch.keys())
        if not shared_patches:
            continue
        sec_checked += 1
        prim_shared = [t for p in shared_patches for t in primary_terms_by_patch[p]]
        sec_shared = [t for p in shared_patches for t in sec_by_patch[p]]
        q_prim_shared = compute_posterior_quadrature(
            prim_shared,
            m_e=prior_mean_logit,
            mu_star=0.50,
            pi_small=hyperparams.pi_small,
            tau_small=hyperparams.tau_small,
            tau_large=hyperparams.tau_large,
            tol=hyperparams.quadrature_tol,
        )
        q_sec_shared = compute_posterior_quadrature(
            sec_shared,
            m_e=prior_mean_logit,
            mu_star=0.50,
            pi_small=hyperparams.pi_small,
            tau_small=hyperparams.tau_small,
            tau_large=hyperparams.tau_large,
            tol=hyperparams.quadrature_tol,
        )
        diff = abs(q_sec_shared.mean_logit - q_prim_shared.mean_logit)
        if max_disagreement is None or diff > max_disagreement:
            max_disagreement = diff

    if sec_checked == 0:
        corrob = CorroborationInfo(
            secondary_sources_checked=0,
            max_source_disagreement_logit=None,
            status="no_secondary",
        )
    else:
        status = (
            "divergent"
            if (max_disagreement is not None and max_disagreement > hyperparams.corroboration_divergent_threshold)
            else "aligned"
        )
        corrob = CorroborationInfo(
            secondary_sources_checked=sec_checked,
            max_source_disagreement_logit=max_disagreement,
            status=status,
        )

    # Arena secondary metrics (top1 & avg_placement)
    arena_sec: ArenaSecondary | None = None
    if mode == "arena":
        top1_rate: float | None = None
        if top1_rows:
            t1_prim, _ = select_primary_and_secondary_binary(top1_rows)
            t1_terms: list[LikelihoodTerm] = []
            for r in t1_prim:
                d_p = compute_patch_discount(
                    patch=r.obs.patch,
                    patch_calendar=patch_calendar,
                    rework_patches=rework_patches,
                    decay=hyperparams.patch_decay,
                )
                if d_p > 0.0 and not r.obs.entity_equals_reference_cohort:
                    t1_terms.append(
                        LikelihoodTerm(
                            wins=r.wins,
                            games=r.games,
                            baseline_logit=r.baseline_logit,
                            discount=d_p,
                        )
                    )
            if t1_terms:
                q_t1 = compute_posterior_quadrature(
                    t1_terms,
                    m_e=0.0,
                    mu_star=0.125,
                    pi_small=hyperparams.pi_small,
                    tau_small=hyperparams.tau_small,
                    tau_large=hyperparams.tau_large,
                    tol=hyperparams.quadrature_tol,
                )
                top1_rate = q_t1.mean_prob_at_mu_star

        avg_place = compute_avg_placement(
            placement_rows,
            patch_calendar=patch_calendar,
            rework_patches=rework_patches,
            hyperparams=hyperparams,
        )
        if top1_rate is not None or avg_place is not None:
            arena_sec = ArenaSecondary(top1_rate=top1_rate, avg_placement=avg_place)

    min_bg = min(baseline_games_list) if baseline_games_list else None
    low_bg = min_bg is not None and min_bg < hyperparams.min_baseline_games
    avg_pr = sum(pick_rates) / len(pick_rates) if pick_rates else None

    return PrecomputedPosterior(
        mode=mode,
        champion_id=champion_id,
        entity_type=entity_type,  # type: ignore[arg-type]
        entity_id=entity_id,
        slot_key=slot_key,
        prior_mean_logit=prior_mean_logit,
        association_mean_logit=assoc_mean,
        association_ci_90_logit=assoc_ci,
        standardized_wr_delta=std_delta,
        effective_sample_size=n_eff,
        pick_rate=avg_pr,
        min_baseline_games=min_bg,
        low_baseline_sample=low_bg,
        corroboration=corrob,
        arena_secondary=arena_sec,
    )
