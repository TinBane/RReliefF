"""Tests for GPU-accelerated Relief implementations.

These tests verify that the vectorized GPU implementations (relieff_gpu.py)
produce results consistent with the original loop-based implementations
and correctly identify predictive features.

On non-Apple platforms, these tests exercise the vectorized NumPy fallback,
which uses the same algorithm and code paths (just runs on CPU).
"""

import numpy as np
import pytest

from relieff_gpu import (
    RReliefF_gpu,
    ReliefF_gpu,
    Relief_gpu,
    _get_backend,
    _HAS_MLX,
)
from relieff import RReliefF, ReliefF, Relief


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_2d_quadratic():
    """Two informative features (x, y) and one noise feature.
    Target: z = 5*x^2 + 5*y^2."""
    rng = np.random.RandomState(42)
    n = 200
    x = np.linspace(0, 5, n)
    y = np.linspace(0, 5, n)
    noise = rng.rand(n)
    X = np.column_stack([x, y, noise])
    target = (5 * x ** 2 + 5 * y ** 2).reshape(-1, 1)
    return X, target


@pytest.fixture
def regression_single_informative():
    """First feature matters, second is noise. Target: y = 3*x1."""
    rng = np.random.RandomState(7)
    n = 150
    x1 = np.linspace(-3, 3, n)
    x2 = rng.randn(n)
    X = np.column_stack([x1, x2])
    target = (3 * x1).reshape(-1, 1)
    return X, target


@pytest.fixture
def binary_classification():
    """First feature is informative, second is noise."""
    rng = np.random.RandomState(0)
    n = 100
    x1 = np.linspace(-5, 5, n)
    x2 = rng.randn(n)
    X = np.column_stack([x1, x2])
    y = (x1 > 0).astype(float)
    return X, y


@pytest.fixture
def multiclass_classification():
    """Three classes determined by first feature; second is noise."""
    rng = np.random.RandomState(3)
    n = 150
    x1 = np.linspace(-3, 3, n)
    x2 = rng.randn(n)
    X = np.column_stack([x1, x2])
    y = np.zeros(n)
    y[x1 > 1] = 2
    y[(x1 >= -1) & (x1 <= 1)] = 1
    return X, y


# ---------------------------------------------------------------------------
# Backend detection test
# ---------------------------------------------------------------------------

class TestBackend:
    def test_backend_returns_valid_string(self):
        backend = _get_backend()
        assert backend in ("mlx", "numpy")

    def test_mlx_detection_consistent(self):
        if _HAS_MLX:
            assert _get_backend() == "mlx"
        else:
            assert _get_backend() == "numpy"


# ---------------------------------------------------------------------------
# RReliefF GPU tests
# ---------------------------------------------------------------------------

class TestRReliefFGPU:
    def test_output_shape(self, regression_2d_quadratic):
        X, y = regression_2d_quadratic
        W = RReliefF_gpu(X, y)
        assert W.shape == (X.shape[1], 1)

    def test_informative_features_ranked_above_noise(self, regression_2d_quadratic):
        X, y = regression_2d_quadratic
        W = RReliefF_gpu(X, y, k=10, sigma=30)
        weights = W.ravel()
        assert weights[0] > weights[2], (
            f"Informative feature 0 ({weights[0]:.4f}) should outrank noise ({weights[2]:.4f})"
        )
        assert weights[1] > weights[2], (
            f"Informative feature 1 ({weights[1]:.4f}) should outrank noise ({weights[2]:.4f})"
        )

    def test_single_informative_feature(self, regression_single_informative):
        X, y = regression_single_informative
        W = RReliefF_gpu(X, y, k=10, sigma=30)
        weights = W.ravel()
        assert weights[0] > weights[1], (
            f"Informative ({weights[0]:.4f}) should outrank noise ({weights[1]:.4f})"
        )

    def test_agrees_with_original_ranking(self, regression_2d_quadratic):
        """GPU and original should agree on feature ranking order."""
        X, y = regression_2d_quadratic
        W_gpu = RReliefF_gpu(X, y, updates="all", k=10, sigma=30)
        W_orig = RReliefF(X, y, updates="all", k=10, sigma=30)
        # Both should rank features in the same order
        rank_gpu = np.argsort(W_gpu.ravel())
        rank_orig = np.argsort(W_orig.ravel())
        np.testing.assert_array_equal(rank_gpu, rank_orig)

    def test_updates_subset(self, regression_2d_quadratic):
        X, y = regression_2d_quadratic
        W = RReliefF_gpu(X, y, updates=50, k=10)
        assert W.shape == (X.shape[1], 1)

    def test_k_parameter(self, regression_2d_quadratic):
        X, y = regression_2d_quadratic
        for k_val in [5, 10, 20]:
            W = RReliefF_gpu(X, y, k=k_val)
            weights = W.ravel()
            assert weights[0] > weights[2], f"Failed for k={k_val}"

    def test_many_noise_features(self):
        """2 informative + 8 noise features."""
        rng = np.random.RandomState(42)
        n = 300
        x1 = np.linspace(0, 5, n)
        x2 = np.linspace(0, 5, n)
        noise = rng.randn(n, 8)
        X = np.column_stack([x1, x2, noise])
        y = (3 * x1 + 2 * x2).reshape(-1, 1)

        W = RReliefF_gpu(X, y, k=10, sigma=30)
        weights = W.ravel()
        top_2 = set(np.argsort(weights)[-2:])
        assert top_2 == {0, 1}, f"Top 2 should be {{0, 1}}, got {top_2}"

    def test_returns_finite_values(self, regression_2d_quadratic):
        X, y = regression_2d_quadratic
        W = RReliefF_gpu(X, y)
        assert np.all(np.isfinite(W))

    def test_reproducibility_all_updates(self, regression_2d_quadratic):
        X, y = regression_2d_quadratic
        W1 = RReliefF_gpu(X, y, updates="all")
        W2 = RReliefF_gpu(X, y, updates="all")
        np.testing.assert_array_almost_equal(W1, W2)


# ---------------------------------------------------------------------------
# ReliefF GPU tests
# ---------------------------------------------------------------------------

class TestReliefFGPU:
    def test_output_shape(self, multiclass_classification):
        X, y = multiclass_classification
        W = ReliefF_gpu(X, y)
        assert W.shape == (X.shape[1], 1)

    def test_informative_feature_ranked_higher(self, multiclass_classification):
        X, y = multiclass_classification
        W = ReliefF_gpu(X, y, k=10)
        weights = W.ravel()
        assert weights[0] > weights[1], (
            f"Informative ({weights[0]:.4f}) should outrank noise ({weights[1]:.4f})"
        )

    def test_binary_classification(self, binary_classification):
        X, y = binary_classification
        W = ReliefF_gpu(X, y, k=10)
        weights = W.ravel()
        assert weights[0] > weights[1]

    def test_agrees_with_original_ranking(self, multiclass_classification):
        """GPU and original should agree on which feature is most important."""
        X, y = multiclass_classification
        W_gpu = ReliefF_gpu(X, y, updates="all", k=10)
        W_orig = ReliefF(X, y, updates="all", k=10)
        assert np.argmax(W_gpu.ravel()) == np.argmax(W_orig.ravel())

    def test_many_noise_features(self):
        rng = np.random.RandomState(10)
        n = 200
        x1 = np.linspace(-3, 3, n)
        noise = rng.randn(n, 4)
        X = np.column_stack([x1, noise])
        y = (x1 > 0).astype(float)

        W = ReliefF_gpu(X, y, k=10)
        weights = W.ravel()
        assert np.argmax(weights) == 0

    def test_returns_finite_values(self, binary_classification):
        X, y = binary_classification
        W = ReliefF_gpu(X, y)
        assert np.all(np.isfinite(W))


# ---------------------------------------------------------------------------
# Relief GPU tests
# ---------------------------------------------------------------------------

class TestReliefGPU:
    def test_output_shape(self, binary_classification):
        X, y = binary_classification
        W = Relief_gpu(X, y)
        assert W.shape == (X.shape[1], 1)

    def test_informative_feature_ranked_higher(self, binary_classification):
        X, y = binary_classification
        W = Relief_gpu(X, y)
        weights = W.ravel()
        assert weights[0] > weights[1]

    def test_agrees_with_original_ranking(self, binary_classification):
        """GPU and original should agree on which feature is most important."""
        X, y = binary_classification
        W_gpu = Relief_gpu(X, y, updates="all")
        W_orig = Relief(X, y, updates="all")
        assert np.argmax(W_gpu.ravel()) == np.argmax(W_orig.ravel())

    def test_nonlinear_boundary(self):
        rng = np.random.RandomState(5)
        n = 100
        x1 = np.linspace(-3, 3, n)
        x2 = rng.randn(n)
        X = np.column_stack([x1, x2])
        y = (x1 ** 2 > 2).astype(float)

        W = Relief_gpu(X, y)
        weights = W.ravel()
        assert weights[0] > weights[1]

    def test_returns_finite_values(self, binary_classification):
        X, y = binary_classification
        W = Relief_gpu(X, y)
        assert np.all(np.isfinite(W))

    def test_updates_subset(self, binary_classification):
        X, y = binary_classification
        W = Relief_gpu(X, y, updates=20)
        assert W.shape == (X.shape[1], 1)


# ---------------------------------------------------------------------------
# Cross-algorithm consistency between GPU and original
# ---------------------------------------------------------------------------

class TestGPUOriginalConsistency:
    def test_relief_gpu_vs_original_same_top_feature(self, binary_classification):
        X, y = binary_classification
        W_gpu = Relief_gpu(X, y, updates="all")
        W_orig = Relief(X, y, updates="all")
        assert np.argmax(W_gpu.ravel()) == np.argmax(W_orig.ravel())

    def test_relieff_gpu_vs_original_same_top_feature(self, multiclass_classification):
        X, y = multiclass_classification
        W_gpu = ReliefF_gpu(X, y, updates="all", k=10)
        W_orig = ReliefF(X, y, updates="all", k=10)
        assert np.argmax(W_gpu.ravel()) == np.argmax(W_orig.ravel())

    def test_rrelieff_gpu_vs_original_same_ranking(self, regression_2d_quadratic):
        X, y = regression_2d_quadratic
        W_gpu = RReliefF_gpu(X, y, updates="all", k=10, sigma=30)
        W_orig = RReliefF(X, y, updates="all", k=10, sigma=30)
        rank_gpu = np.argsort(W_gpu.ravel())
        rank_orig = np.argsort(W_orig.ravel())
        np.testing.assert_array_equal(rank_gpu, rank_orig)
