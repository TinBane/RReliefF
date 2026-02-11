"""Tests for GPU-accelerated Relief implementations (multi-backend).

These tests verify that the vectorized GPU implementations (relieff_gpu.py)
produce results consistent with the original loop-based implementations
and correctly identify predictive features.

On this platform, the NumPy fallback backend is exercised. The same
vectorized algorithm runs identically on MLX, CuPy, JAX, and NumPy
through the _Ops abstraction layer.
"""

import numpy as np
import pytest

from relieff_gpu import (
    RReliefF_gpu,
    ReliefF_gpu,
    Relief_gpu,
    _get_backend,
    _HAS_MLX,
    _HAS_CUPY,
    _HAS_JAX,
    _VALID_BACKENDS,
    available_backends,
    set_backend,
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
# Backend detection and management tests
# ---------------------------------------------------------------------------

class TestBackend:
    def test_backend_returns_valid_string(self):
        backend = _get_backend()
        assert backend in _VALID_BACKENDS

    def test_available_backends_includes_numpy(self):
        backends = available_backends()
        assert "numpy" in backends

    def test_available_backends_consistent_with_flags(self):
        backends = available_backends()
        if _HAS_MLX:
            assert "mlx" in backends
        else:
            assert "mlx" not in backends
        if _HAS_CUPY:
            assert "cupy" in backends
        else:
            assert "cupy" not in backends
        if _HAS_JAX:
            assert "jax" in backends
        else:
            assert "jax" not in backends

    def test_set_backend_numpy(self):
        """Can always switch to numpy backend."""
        set_backend("numpy")
        assert _get_backend() == "numpy"
        set_backend(None)  # restore auto-detection

    def test_set_backend_none_restores_auto(self):
        set_backend("numpy")
        assert _get_backend() == "numpy"
        set_backend(None)
        # After reset, should auto-detect (whatever is available)
        assert _get_backend() in _VALID_BACKENDS

    def test_set_backend_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("opencl")

    def test_set_backend_unavailable_raises(self):
        """Requesting an uninstalled backend raises ImportError."""
        # Find a backend that is NOT installed
        for name in ("mlx", "cupy", "jax"):
            avail = {"mlx": _HAS_MLX, "cupy": _HAS_CUPY, "jax": _HAS_JAX}
            if not avail[name]:
                with pytest.raises(ImportError, match=name):
                    set_backend(name)
                break

    def test_set_backend_affects_computation(self, binary_classification):
        """Switching to numpy backend still produces correct results."""
        X, y = binary_classification
        set_backend("numpy")
        W = Relief_gpu(X, y)
        set_backend(None)
        assert W.shape == (X.shape[1], 1)
        assert W.ravel()[0] > W.ravel()[1]


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


# ---------------------------------------------------------------------------
# Explicit numpy backend tests (ensures forced-numpy path works correctly)
# ---------------------------------------------------------------------------

class TestNumpyBackendExplicit:
    """Run key tests with the numpy backend explicitly forced."""

    def test_rrelieff_numpy_backend(self, regression_2d_quadratic):
        X, y = regression_2d_quadratic
        set_backend("numpy")
        W = RReliefF_gpu(X, y, k=10, sigma=30)
        set_backend(None)
        weights = W.ravel()
        assert weights[0] > weights[2]

    def test_relieff_numpy_backend(self, multiclass_classification):
        X, y = multiclass_classification
        set_backend("numpy")
        W = ReliefF_gpu(X, y, k=10)
        set_backend(None)
        weights = W.ravel()
        assert weights[0] > weights[1]

    def test_relief_numpy_backend(self, binary_classification):
        X, y = binary_classification
        set_backend("numpy")
        W = Relief_gpu(X, y)
        set_backend(None)
        weights = W.ravel()
        assert weights[0] > weights[1]
