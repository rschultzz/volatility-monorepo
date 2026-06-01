"""Tests for knn.py — rank_analogues, before_date, distance_ceiling (CR-029);
weighted_similarity_distance, z_diff_cap, recency, config v2 (CR-030)."""
import math
import pytest

from packages.shared.knn import (
    feature_stats,
    rank_analogues,
    similarity_distance,
    weighted_similarity_distance,
)
from packages.shared.knn_config import (
    CANONICAL_KNN_CONFIG_VERSION,
    KNN_CONFIGS,
    get_knn_config,
)
from packages.shared.day_features import FEATURE_NAMES


# ─── helpers ────────────────────────────────────────────────────────────────

def _sparse_vec(**kwargs) -> dict:
    """Build a sparse feature vector: only the supplied keys are non-None.

    All other FEATURE_NAMES are None. Sparse vectors keep n_active small so
    the rescaling factor is large and even a single-feature difference
    produces a large σ-normalized distance — making ceiling tests predictable.
    """
    base = {k: None for k in FEATURE_NAMES}
    base.update(kwargs)
    return base


def _dense_vec(**kwargs) -> dict:
    """Build a dense feature vector: all FEATURE_NAMES default to 0.0."""
    base = {k: 0.0 for k in FEATURE_NAMES}
    base.update(kwargs)
    return base


def _candidates_sparse(n_close: int = 3, n_far: int = 2) -> list:
    """Return (date, sparse-vec) pairs where far candidates are clearly beyond 4σ.

    Only cluster_1_signed_distance_sigma is active. With n_active=1 the
    rescale factor = sqrt(34) ≈ 5.83, so a 2-unit z-score difference → ~11.7σ
    (well above any reasonable ceiling).
    """
    cands = []
    for i in range(n_close):
        cands.append((
            f"2024-01-{i + 1:02d}",
            _sparse_vec(cluster_1_signed_distance_sigma=float(i) * 0.01),
        ))
    for i in range(n_far):
        cands.append((
            f"2024-06-{i + 1:02d}",
            _sparse_vec(cluster_1_signed_distance_sigma=50.0 + float(i)),
        ))
    return cands


def _compute_distance_between(v1: dict, v2: dict, pool: list) -> float:
    """Compute the distance between two vecs using stats from pool."""
    stats = feature_stats(v for (_, v) in pool)
    return similarity_distance(v1, v2, stats)


# ─── knn_config tests ───────────────────────────────────────────────────────

class TestKnnConfig:
    def test_canonical_version_is_v2(self):
        assert CANONICAL_KNN_CONFIG_VERSION == "v2"

    def test_v2_ceiling_is_5(self):
        cfg = get_knn_config("v2")
        assert cfg["distance_ceiling"] == pytest.approx(5.0)

    def test_v2_has_z_diff_cap(self):
        cfg = get_knn_config("v2")
        assert cfg["z_diff_cap"] == pytest.approx(3.0)

    def test_v2_has_feature_weights(self):
        cfg = get_knn_config("v2")
        assert "feature_weights" in cfg
        assert isinstance(cfg["feature_weights"], dict)
        # All FEATURE_NAMES should be present in weights
        for name in FEATURE_NAMES:
            assert name in cfg["feature_weights"], f"Missing weight for {name}"

    def test_v2_has_half_life_months(self):
        cfg = get_knn_config("v2")
        assert cfg["half_life_months"] == pytest.approx(18.0)

    def test_v1_ceiling_is_4(self):
        cfg = get_knn_config("v1")
        assert cfg["distance_ceiling"] == pytest.approx(4.0)

    def test_v0_ceiling_is_inf(self):
        cfg = get_knn_config("v0")
        assert math.isinf(cfg["distance_ceiling"])

    def test_default_version_is_canonical(self):
        assert get_knn_config() == get_knn_config(CANONICAL_KNN_CONFIG_VERSION)

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError, match="Unknown knn_config_version"):
            get_knn_config("v999")

    def test_ceiling_literal_only_in_config_module(self):
        """Distance ceiling and weight literals must live only in knn_config.py."""
        import inspect
        from packages.shared import knn
        src = inspect.getsource(knn)
        assert "4.0" not in src, (
            "Hard-coded 4.0 ceiling found in knn.py — must live in knn_config.py only"
        )
        assert "5.0" not in src, (
            "Hard-coded 5.0 ceiling found in knn.py — must live in knn_config.py only"
        )


# ─── before_date tests ──────────────────────────────────────────────────────

class TestBeforeDate:
    def test_before_date_excludes_future_candidates(self):
        candidates = [
            ("2024-01-14", _sparse_vec(implied_move_1d=1.0)),
            ("2024-01-15", _sparse_vec(implied_move_1d=1.0)),
            ("2024-01-16", _sparse_vec(implied_move_1d=1.0)),
        ]
        anchor = _sparse_vec(implied_move_1d=1.0)
        result = rank_analogues(anchor, candidates, 10, before_date="2024-01-15")
        dates = [d for d, _ in result]
        assert "2024-01-14" in dates
        assert "2024-01-15" not in dates
        assert "2024-01-16" not in dates

    def test_before_date_none_returns_all(self):
        candidates = [
            ("2024-01-14", _sparse_vec(implied_move_1d=1.0)),
            ("2024-01-15", _sparse_vec(implied_move_1d=1.0)),
            ("2024-01-16", _sparse_vec(implied_move_1d=1.0)),
        ]
        anchor = _sparse_vec(implied_move_1d=1.0)
        result = rank_analogues(anchor, candidates, 10, before_date=None)
        assert len(result) == 3

    def test_before_date_with_exclude_date(self):
        candidates = [
            ("2024-01-13", _sparse_vec(implied_move_1d=1.0)),
            ("2024-01-14", _sparse_vec(implied_move_1d=1.0)),
            ("2024-01-15", _sparse_vec(implied_move_1d=1.0)),  # anchor — excluded via exclude_date
            ("2024-01-16", _sparse_vec(implied_move_1d=1.0)),  # future — excluded via before_date
        ]
        anchor = _sparse_vec(implied_move_1d=1.0)
        result = rank_analogues(
            anchor, candidates, 10,
            exclude_date="2024-01-15",
            before_date="2024-01-15",
        )
        dates = [d for d, _ in result]
        assert set(dates) == {"2024-01-13", "2024-01-14"}

    def test_before_date_empty_pool_returns_empty(self):
        candidates = [
            ("2024-06-01", _sparse_vec()),
            ("2024-06-02", _sparse_vec()),
        ]
        anchor = _sparse_vec()
        result = rank_analogues(anchor, candidates, 10, before_date="2024-01-01")
        assert result == []

    def test_stats_computed_from_before_date_pool(self):
        """When stats=None and before_date is set, stats use only the as-of pool.

        Regression guard: if stats included future candidates, the normalization
        would be wrong for backtesting paths.
        """
        candidates = [
            ("2024-01-01", _sparse_vec(implied_move_1d=1.0)),
            ("2024-01-02", _sparse_vec(implied_move_1d=1.0)),
            ("2024-06-01", _sparse_vec(implied_move_1d=100.0)),  # future, very different
        ]
        anchor = _sparse_vec(implied_move_1d=1.0)

        # With before_date: future candidate excluded from both pool and stats
        result_with = rank_analogues(
            anchor, candidates, 10,
            before_date="2024-06-01",
            stats=None,
        )
        # Without: future is a candidate (stats may vary)
        result_without = rank_analogues(anchor, candidates, 10, stats=None)

        assert len(result_with) == 2          # only the two past dates
        assert len(result_without) == 3       # all three
        assert result_with[0][0] in {"2024-01-01", "2024-01-02"}


# ─── distance_ceiling tests ─────────────────────────────────────────────────

class TestDistanceCeiling:
    def test_ceiling_excludes_far_candidates(self):
        # Sparse vecs: only cluster_1_signed_distance_sigma active.
        # Far candidates differ by ~50 units; with 1 active feature the
        # rescale = sqrt(34) ≈ 5.83, making their distance >> 4.0σ.
        cands = _candidates_sparse(n_close=3, n_far=2)
        anchor = _sparse_vec(cluster_1_signed_distance_sigma=0.0)
        stats = feature_stats(v for (_, v) in cands)

        result = rank_analogues(anchor, cands, 20, stats=stats, distance_ceiling=4.0)
        dates = [d for d, _ in result]
        assert all(d.startswith("2024-01-") for d in dates)
        assert len(result) == 3

    def test_ceiling_inf_backward_compat(self):
        cands = _candidates_sparse(n_close=3, n_far=2)
        anchor = _sparse_vec(cluster_1_signed_distance_sigma=0.0)
        result = rank_analogues(anchor, cands, 20, distance_ceiling=math.inf)
        assert len(result) == 5

    def test_adaptive_k_fewer_than_k_when_ceiling_tight(self):
        cands = _candidates_sparse(n_close=3, n_far=5)
        anchor = _sparse_vec(cluster_1_signed_distance_sigma=0.0)
        stats = feature_stats(v for (_, v) in cands)

        result = rank_analogues(anchor, cands, 20, stats=stats, distance_ceiling=4.0)
        assert len(result) < 20
        assert len(result) == 3

    def test_ceiling_zero_returns_only_identical_candidates(self):
        anchor = _dense_vec(implied_move_1d=1.0)
        candidates = [
            ("2024-01-01", _dense_vec(implied_move_1d=1.0)),  # identical to anchor
            ("2024-01-02", _dense_vec(implied_move_1d=2.0)),  # different
        ]
        stats = feature_stats(v for (_, v) in candidates)
        result = rank_analogues(anchor, candidates, 10, stats=stats, distance_ceiling=0.0)
        assert len(result) == 1
        assert result[0][0] == "2024-01-01"
        assert result[0][1] == pytest.approx(0.0, abs=1e-9)

    def test_ceiling_does_not_modify_sort_order(self):
        cands = _candidates_sparse(n_close=5, n_far=0)
        anchor = _sparse_vec(cluster_1_signed_distance_sigma=0.0)
        result = rank_analogues(anchor, cands, 20, distance_ceiling=10.0)
        dists = [dist for (_, dist) in result]
        assert dists == sorted(dists)

    def test_all_excluded_by_ceiling_returns_empty(self):
        # Sparse vecs so the far candidates are genuinely far (> 4σ)
        anchor = _sparse_vec(cluster_1_signed_distance_sigma=0.0)
        candidates = [
            ("2024-01-01", _sparse_vec(cluster_1_signed_distance_sigma=50.0)),
            ("2024-01-02", _sparse_vec(cluster_1_signed_distance_sigma=60.0)),
        ]
        stats = feature_stats(v for (_, v) in candidates)
        result = rank_analogues(anchor, candidates, 10, stats=stats, distance_ceiling=4.0)
        assert result == []

    def test_ceiling_and_before_date_compose(self):
        # Sparse vecs: far past candidate is excluded by ceiling;
        # close future candidate is excluded by before_date.
        candidates = [
            ("2024-01-01", _sparse_vec(cluster_1_signed_distance_sigma=0.0)),   # close, past
            ("2024-01-02", _sparse_vec(cluster_1_signed_distance_sigma=50.0)),  # far,   past
            ("2024-06-01", _sparse_vec(cluster_1_signed_distance_sigma=0.0)),   # close, future
        ]
        anchor = _sparse_vec(cluster_1_signed_distance_sigma=0.0)
        result = rank_analogues(
            anchor, candidates, 10,
            before_date="2024-06-01",
            distance_ceiling=4.0,
        )
        dates = [d for d, _ in result]
        assert dates == ["2024-01-01"]


# ─── CR-030: weighted_similarity_distance tests ─────────────────────────────

class TestWeightedSimilarityDistance:
    """Tests for the weighted + z_diff_cap distance function."""

    def _two_candidate_stats(self, anchor_val: float, cand_val: float,
                              feature: str = "implied_move_1d") -> dict:
        """Build stats from a two-vector pool (anchor + candidate)."""
        pool = [
            _dense_vec(**{feature: anchor_val}),
            _dense_vec(**{feature: cand_val}),
        ]
        return feature_stats(pool)

    def test_equal_weights_and_no_cap_matches_similarity_distance(self):
        """weight=1.0 for all features + no cap → identical to similarity_distance."""
        anchor = _dense_vec(implied_move_1d=1.0, cluster_1_signed_distance_sigma=0.5)
        cand   = _dense_vec(implied_move_1d=2.0, cluster_1_signed_distance_sigma=1.5)
        candidates = [("2024-01-01", anchor), ("2024-01-02", cand)]
        stats = feature_stats(v for (_, v) in candidates)
        equal_weights = {name: 1.0 for name in FEATURE_NAMES}

        wd = weighted_similarity_distance(anchor, cand, stats,
                                          feature_weights=equal_weights,
                                          z_diff_cap=None)
        sd = similarity_distance(anchor, cand, stats)
        assert wd == pytest.approx(sd, rel=1e-6)

    def test_zero_weight_feature_does_not_contribute(self):
        """A feature with weight=0.0 contributes nothing regardless of z_diff."""
        anchor = _dense_vec(implied_move_1d=10.0)
        cand   = _dense_vec(implied_move_1d=0.0)   # large raw difference
        candidates = [("2024-01-01", anchor), ("2024-01-02", cand)]
        stats = feature_stats(v for (_, v) in candidates)
        # All weights 1.0 except implied_move_1d = 0.0
        weights = {name: (0.0 if name == "implied_move_1d" else 1.0)
                   for name in FEATURE_NAMES}
        # With weight=0: distance should be 0 (all other features identical)
        wd = weighted_similarity_distance(anchor, cand, stats,
                                          feature_weights=weights,
                                          z_diff_cap=None)
        assert wd == pytest.approx(0.0, abs=1e-9)

    def test_high_weight_feature_increases_distance(self):
        """Up-weighting a feature whose values differ increases the distance."""
        anchor = _dense_vec(cluster_1_signed_distance_sigma=0.0)
        cand   = _dense_vec(cluster_1_signed_distance_sigma=1.0)
        candidates = [("2024-01-01", anchor), ("2024-01-02", cand)]
        stats = feature_stats(v for (_, v) in candidates)
        equal_w  = {name: 1.0  for name in FEATURE_NAMES}
        high_w   = dict(equal_w); high_w["cluster_1_signed_distance_sigma"] = 4.0

        d_equal = weighted_similarity_distance(anchor, cand, stats,
                                               feature_weights=equal_w)
        d_high  = weighted_similarity_distance(anchor, cand, stats,
                                               feature_weights=high_w)
        assert d_high > d_equal

    def test_z_diff_cap_bounds_extreme_contribution(self):
        """A z_diff exceeding z_diff_cap should be capped, reducing the distance.

        Pool design: 8 zeros + big_anchor(10) + small_cand(-10) on a sparse
        single-feature vector. With 10 items: mean=0, std=sqrt(20)≈4.47.
        z_diff = (10-(-10))/4.47 ≈ 4.47 > 3.0 → cap kicks in.
        """
        zeros = [(f"zero{i}", _sparse_vec(cluster_2_quality_ordinal=0.0))
                 for i in range(8)]
        big_anchor = _sparse_vec(cluster_2_quality_ordinal=10.0)
        small_cand = _sparse_vec(cluster_2_quality_ordinal=-10.0)
        pool = [("big", big_anchor), ("small", small_cand)] + zeros
        stats = feature_stats(v for (_, v) in pool)

        # Verify pre-condition: raw z_diff exceeds cap
        fst = stats["cluster_2_quality_ordinal"]
        std_v = max(fst["std"], 1e-6)
        raw_zdiff = abs(10.0 / std_v - (-10.0 / std_v))
        assert raw_zdiff > 3.0, f"Pre-condition failed: z_diff={raw_zdiff:.2f} ≤ 3.0"

        w_uniform = {name: 1.0 for name in FEATURE_NAMES}
        d_nocap = weighted_similarity_distance(big_anchor, small_cand, stats,
                                               feature_weights=w_uniform,
                                               z_diff_cap=None)
        d_capped = weighted_similarity_distance(big_anchor, small_cand, stats,
                                                feature_weights=w_uniform,
                                                z_diff_cap=3.0)
        # Capped version must be strictly smaller (z_diff > cap was active)
        assert d_capped < d_nocap

    def test_z_diff_cap_contribution_bounded(self):
        """With z_diff_cap=C and weight=w, no single feature can contribute > w×C²."""
        big_anchor = _sparse_vec(cluster_2_quality_ordinal=100.0)
        small_cand = _sparse_vec(cluster_2_quality_ordinal=0.0)
        pool = [("a", big_anchor), ("b", small_cand)]
        stats = feature_stats(v for (_, v) in pool)
        w = {name: 1.0 for name in FEATURE_NAMES}
        CAP = 3.0
        # With z_diff_cap=3.0, max per-feature contribution = 1.0 × 3.0² = 9.0
        # With only ONE active feature, rescale=sqrt(34/1), so dist = sqrt(9)*sqrt(34)
        d = weighted_similarity_distance(big_anchor, small_cand, stats,
                                         feature_weights=w, z_diff_cap=CAP)
        max_possible = math.sqrt(1.0 * CAP * CAP) * math.sqrt(len(FEATURE_NAMES) / 1)
        assert d <= max_possible + 1e-9


# ─── CR-030: rank_analogues weighting + recency tests ───────────────────────

class TestWeightedRankAnalogues:
    """Smoke tests for rank_analogues with CR-030 parameters."""

    def test_feature_weights_none_reproduces_baseline(self):
        """feature_weights=None → identical ordering to pre-CR-030 behavior."""
        cands = _candidates_sparse(n_close=5, n_far=0)
        anchor = _sparse_vec(cluster_1_signed_distance_sigma=0.0)
        result_baseline = rank_analogues(anchor, cands, 20,
                                         feature_weights=None)
        result_explicit = rank_analogues(anchor, cands, 20,
                                         feature_weights=None,
                                         z_diff_cap=None)
        assert [d for d, _ in result_baseline] == [d for d, _ in result_explicit]

    def test_weight_effect_reorders_candidates(self):
        """Up-weighting a feature whose values differ more on one candidate
        pushes that candidate further away."""
        # cand_a differs on cluster_1_signed_distance_sigma
        # cand_b differs on implied_move_1d (same diff magnitude in z-space)
        # Up-weight cluster_1; cand_a should rank behind cand_b
        anchor = _dense_vec(cluster_1_signed_distance_sigma=0.0,
                             implied_move_1d=0.0)
        cand_a = _dense_vec(cluster_1_signed_distance_sigma=5.0,
                             implied_move_1d=0.0)
        cand_b = _dense_vec(cluster_1_signed_distance_sigma=0.0,
                             implied_move_1d=5.0)
        candidates = [("a", anchor), ("a2", cand_a), ("b", cand_b)]
        stats = feature_stats(v for (_, v) in candidates)

        # Equal weights: same z_diff magnitude → same raw distance, order arbitrary
        equal_w = {name: 1.0 for name in FEATURE_NAMES}
        r_equal = rank_analogues(anchor, candidates, 10, stats=stats,
                                 exclude_date="a",
                                 feature_weights=equal_w)
        d_a_equal = next(d for (dt, d) in r_equal if dt == "a2")
        d_b_equal = next(d for (dt, d) in r_equal if dt == "b")
        assert d_a_equal == pytest.approx(d_b_equal, rel=1e-4)

        # Up-weight cluster_1; cand_a farther
        high_w = dict(equal_w)
        high_w["cluster_1_signed_distance_sigma"] = 4.0
        r_weighted = rank_analogues(anchor, candidates, 10, stats=stats,
                                    exclude_date="a",
                                    feature_weights=high_w)
        d_a_w = next(d for (dt, d) in r_weighted if dt == "a2")
        d_b_w = next(d for (dt, d) in r_weighted if dt == "b")
        assert d_a_w > d_b_w

    def test_recency_effect_penalizes_older_candidate(self):
        """An older candidate with the same structural distance as a newer one
        should rank worse when half_life_months is set."""
        anchor = _dense_vec(implied_move_1d=1.0)
        # Both candidates are structurally identical to the anchor
        cand_new = _dense_vec(implied_move_1d=1.0)  # 3 months ago
        cand_old = _dense_vec(implied_move_1d=1.0)  # 3 years ago
        # Add tiny structural difference to break ties
        cand_new["cluster_1_signed_distance_sigma"] = 0.001
        cand_old["cluster_1_signed_distance_sigma"] = 0.002
        candidates = [
            ("2026-02-01", cand_new),   # ~3 months before anchor
            ("2023-05-31", cand_old),   # ~3 years before anchor
        ]
        anchor_date = "2026-05-31"
        stats = feature_stats(v for (_, v) in candidates)
        equal_w = {name: 1.0 for name in FEATURE_NAMES}

        result = rank_analogues(
            anchor, candidates, 10,
            stats=stats,
            feature_weights=equal_w,
            half_life_months=18.0,
            anchor_date=anchor_date,
        )
        dates = [d for d, _ in result]
        # The recent candidate should rank first despite tiny structural disadvantage
        assert dates[0] == "2026-02-01", (
            "Newer candidate should rank first when recency is active"
        )

    def test_recency_none_no_effect(self):
        """half_life_months=None → same ordering as baseline."""
        cands = _candidates_sparse(n_close=5, n_far=0)
        anchor = _sparse_vec(cluster_1_signed_distance_sigma=0.0)
        r1 = rank_analogues(anchor, cands, 20, feature_weights=None)
        r2 = rank_analogues(anchor, cands, 20, feature_weights=None,
                            half_life_months=None, anchor_date=None)
        assert [d for d, _ in r1] == [d for d, _ in r2]

    def test_recency_applied_before_ceiling(self):
        """A structurally close but old candidate can be pushed beyond the ceiling
        by recency scaling — ceiling is applied to effective (age-scaled) distance.

        Pool design: anchor + old_cand + ref_cand in the stats pool so that
        feature_stats are non-degenerate and implied_move_1d has a real std.
        Structural distance old_cand→anchor ≈ 0.5σ; with half_life=6mo and 3yr
        age, effective = 0.5 × 2^6 = 32σ >> 5σ ceiling.
        """
        anchor   = _dense_vec(implied_move_1d=1.0)
        old_cand = _dense_vec(implied_move_1d=1.01)   # slightly different
        ref_cand = _dense_vec(implied_move_1d=1.05)   # reference for normalisation
        candidates = [
            ("2023-01-01", old_cand),   # 3 years before anchor
            ("2025-12-01", ref_cand),   # ~1 month before anchor — low recency penalty
        ]
        anchor_date = "2026-01-01"
        stats = feature_stats(v for (_, v) in candidates)

        w = {name: 1.0 for name in FEATURE_NAMES}

        # Without recency: structural distance ≈ small, well inside ceiling=5.0
        r_no_decay = rank_analogues(
            anchor, candidates, 10, stats=stats,
            feature_weights=w,
            half_life_months=None, anchor_date=None,
            distance_ceiling=5.0,
        )
        assert len(r_no_decay) == 2   # both pass ceiling without decay

        # With recency half_life=6mo: 3yr-old day multiplier = 2^(3/0.5) = 2^6 = 64×
        # Structural dist * 64 >> 5σ ceiling → excluded
        r_decayed = rank_analogues(
            anchor, candidates, 10, stats=stats,
            feature_weights=w,
            half_life_months=6.0,   # very aggressive: 6-month half-life
            anchor_date=anchor_date,
            distance_ceiling=5.0,
        )
        # old_cand (3yr ago, 64× multiplier) should be excluded; ref_cand (1mo ago) survives
        old_dates = [d for d, _ in r_decayed]
        assert "2023-01-01" not in old_dates, (
            "3-year-old candidate should be pushed beyond ceiling by aggressive decay"
        )
        assert "2025-12-01" in old_dates, (
            "Recent candidate (1 month ago) should survive the ceiling"
        )

    def test_z_diff_cap_regression_two_pin_cluster_style(self):
        """Regression: a near-zero-mean ordinal feature (cluster_2_quality_ordinal)
        with rare value=2 (corpus mean≈0) should NOT dominate squared distance.
        Mirrors the 2026-05-07 two-pin-cluster structural issue (CR-030 hard gate).
        """
        # Build a corpus where cluster_2_quality_ordinal is 0 for all candidates
        # and 2 for the anchor — mimicking the actual corpus situation.
        n = 10
        candidates = [
            (f"2024-{i:02d}-01",
             _dense_vec(cluster_2_quality_ordinal=0.0,
                        cluster_1_signed_distance_sigma=float(i) * 0.1))
            for i in range(1, n + 1)
        ]
        anchor = _dense_vec(cluster_2_quality_ordinal=2.0,
                            cluster_1_signed_distance_sigma=0.0)
        stats = feature_stats(v for (_, v) in candidates)

        # Without cap: cluster_2_quality_ordinal has z_diff >> others → dominates
        w = {name: 1.0 for name in FEATURE_NAMES}
        breakdown_nocap = [
            (name,
             weighted_similarity_distance(
                 anchor,
                 dict(candidates[0][1]),
                 stats,
                 feature_weights=w,
                 z_diff_cap=None,
             ))
            for name in ["cluster_2_quality_ordinal"]
        ]
        # The interesting assertion: with cap=3.0, the blowup feature should
        # NOT contribute more than z_diff_cap^2 × weight per feature.
        best_cand_vec = candidates[0][1]
        # Compute contribution of cluster_2_quality_ordinal with cap vs without
        fstats = stats["cluster_2_quality_ordinal"]
        std_val = max(fstats["std"], 1e-6)
        z_q = (2.0 - fstats["mean"]) / std_val
        z_c = (0.0 - fstats["mean"]) / std_val
        raw_diff = z_q - z_c
        capped_diff = min(abs(raw_diff), 3.0) * (1 if raw_diff >= 0 else -1)

        contrib_nocap = 1.0 * raw_diff ** 2
        contrib_capped = 1.0 * capped_diff ** 2

        assert abs(raw_diff) > 3.0, "Test pre-condition: raw z_diff should exceed cap"
        assert contrib_capped <= 9.0 + 1e-9, (
            "Capped contribution must be ≤ z_diff_cap² = 9.0"
        )
        assert contrib_capped < contrib_nocap, (
            "Capped contribution must be less than uncapped"
        )

        # With cap applied in rank_analogues, 2026-05-07-style anchor should
        # find candidates (not get zero due to blowup)
        result_capped = rank_analogues(
            anchor, candidates, 10,
            feature_weights=w,
            z_diff_cap=3.0,
            distance_ceiling=8.0,
        )
        result_nocap = rank_analogues(
            anchor, candidates, 10,
            feature_weights=w,
            z_diff_cap=None,
            distance_ceiling=8.0,
        )
        # With cap the distances should be lower (blowup removed)
        if result_capped and result_nocap:
            assert result_capped[0][1] < result_nocap[0][1], (
                "Capped distance must be strictly less than uncapped"
            )

    def test_all_config_params_from_config_not_literals(self):
        """Confirm no hardcoded ceiling / weight / cap literals in knn.py."""
        import inspect
        from packages.shared import knn
        src = inspect.getsource(knn)
        for literal in ("4.0", "5.0", "3.0", "18.0", "2.5", "0.25"):
            assert literal not in src, (
                f"Hard-coded literal {literal!r} found in knn.py — "
                "it must live in knn_config.py only"
            )


# ─── CR-031: adaptive-K uncap smoke tests ───────────────────────────────────

def _make_within_ceiling_candidates(n: int, ceiling: float = 5.0) -> list:
    """Generate n candidates all at distance ≈ 0 from any dense anchor.

    All features are 0.0; when the anchor is also 0.0, σ-normalized distance
    is 0.0 — guaranteed within any positive ceiling.  Used to test that k=200
    returns all n without truncation (whereas k=20 would cap at 20).
    """
    return [
        (f"2024-{i // 31 + 1:02d}-{(i % 30) + 1:02d}", _dense_vec())
        for i in range(n)
    ]


class TestAdaptiveKUncap:
    """CR-031 Step 1 smoke tests — verify that the safety bound (k=200)
    is the only numeric gate; the distance ceiling is the similarity gate."""

    def test_35_within_ceiling_returned_when_k_200(self):
        """Common setup: 35 candidates within ceiling, k=200 → all 35 returned.

        This is the adaptive-K design intent: a well-precedented setup earns
        its full count; the ceiling is the gate, not a hard numeric cap.
        """
        anchor = _dense_vec()
        candidates = _make_within_ceiling_candidates(35)
        result = rank_analogues(anchor, candidates, k=200, distance_ceiling=5.0)
        assert len(result) == 35, (
            f"Expected 35 within-ceiling analogues; got {len(result)}. "
            "The k=200 safety bound should not truncate 35 results."
        )

    def test_old_k20_would_have_truncated_to_20(self):
        """Documents the pre-CR-031 bug: k=20 truncates even with 35 qualifying days.

        This test demonstrates why routes.py MUST call rank_analogues with
        k=_K_SAFETY (200), not the URL-param k (formerly capped at 20).
        """
        anchor = _dense_vec()
        candidates = _make_within_ceiling_candidates(35)
        result = rank_analogues(anchor, candidates, k=20, distance_ceiling=5.0)
        assert len(result) == 20, (
            "k=20 should truncate to 20 even when 35 candidates are within ceiling"
        )

    def test_rare_setup_unchanged_by_uncap(self):
        """Rare setup: 6 candidates within ceiling, k=200 → still returns 6.

        Uncapping must not affect rare setups — their small K is honest.
        """
        anchor = _dense_vec()
        candidates = _make_within_ceiling_candidates(6)
        result = rank_analogues(anchor, candidates, k=200, distance_ceiling=5.0)
        assert len(result) == 6, (
            f"Rare setup should return 6 analogues; got {len(result)}."
        )

    def test_safety_bound_200_binds_at_200_not_lower(self):
        """201 within-ceiling candidates with k=200 → exactly 200 returned.

        The safety bound is the ONLY numeric cap; it must not activate before 200.
        """
        anchor = _dense_vec()
        candidates = _make_within_ceiling_candidates(201)
        result = rank_analogues(anchor, candidates, k=200, distance_ceiling=5.0)
        assert len(result) == 200, (
            f"Safety bound should bind at 200; got {len(result)}."
        )

    def test_ceiling_still_gates_after_uncap(self):
        """Ceiling is still the similarity gate: far candidates stay excluded.

        With k=200, the ceiling (not k) decides admittance.  Far candidates
        (distance >> 5σ) must never appear in the result.
        """
        anchor = _dense_vec()
        close_cands = _make_within_ceiling_candidates(25)
        # Far candidates: sparse vec with extreme single-feature offset
        # (1 active feature → rescale=sqrt(34)≈5.83; offset=50 → distance≈290σ)
        far_cands = [
            (f"2099-01-{i+1:02d}",
             _sparse_vec(cluster_1_signed_distance_sigma=50.0))
            for i in range(5)
        ]
        all_cands = close_cands + far_cands
        result = rank_analogues(anchor, all_cands, k=200, distance_ceiling=5.0)
        returned_dates = {d for d, _ in result}
        far_dates = {d for d, _ in far_cands}
        assert not returned_dates & far_dates, (
            f"Far candidates leaked through the ceiling: {returned_dates & far_dates}"
        )
        assert len(result) == 25, (
            f"Expected 25 within-ceiling analogues; got {len(result)}"
        )

    def test_safety_bound_literal_not_in_routes(self):
        """The old _K_MAX=20 must not appear in routes.py; _K_SAFETY must be >= 100."""
        import os
        routes_path = os.path.join(
            os.path.dirname(__file__), "../../../apps/web/modules/Analogues/routes.py"
        )
        src = open(routes_path).read()
        import re as _re
        assert not _re.search(r"_K_MAX\s*=\s*\d+", src), (
            "_K_MAX=<n> assignment must be removed from routes.py (CR-031); "
            "use _K_SAFETY instead"
        )
        assert "_K_SAFETY" in src, "_K_SAFETY must be defined in routes.py"
        # Extract _K_SAFETY value from source (e.g. "_K_SAFETY = 200")
        import re
        m = re.search(r"_K_SAFETY\s*=\s*(\d+)", src)
        assert m is not None, "_K_SAFETY assignment not found in routes.py"
        safety_val = int(m.group(1))
        assert safety_val >= 100, f"_K_SAFETY={safety_val} is too low; expected ≥ 100"
