"""Tests for knn.py — rank_analogues, before_date, distance_ceiling (CR-029)."""
import math
import pytest

from packages.shared.knn import feature_stats, rank_analogues, similarity_distance
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
    def test_canonical_version_is_v1(self):
        assert CANONICAL_KNN_CONFIG_VERSION == "v1"

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
        """The literal 4.0 should live only in knn_config.py, not knn.py."""
        import inspect
        from packages.shared import knn
        src = inspect.getsource(knn)
        assert "4.0" not in src, (
            "Hard-coded ceiling found in knn.py — it must live in knn_config.py only"
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
