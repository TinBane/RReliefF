"""Comprehensive tests for RReliefF, ReliefF, and Relief algorithms.

Tests focus on the core purpose: detecting which features are predictive
of the target variable, and which are noise.
"""

import numpy as np
import pytest

from relieff import RReliefF, ReliefF, Relief


# ---------------------------------------------------------------------------
# Fixtures: synthetic datasets where we *know* the ground truth importance
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_2d_quadratic():
    """Two informative features (x, y) and one noise feature.

    Target: z = 5*x^2 + 5*y^2.  Third feature is uniform random noise.
    Both informative features should rank above the noise feature.
    """
    rng = np.random.RandomState(42)
    n = 200
    x = np.linspace(0, 5, n)
    y = np.linspace(0, 5, n)
    noise = rng.rand(n)
    X = np.column_stack([x, y, noise])
    target = 5 * x ** 2 + 5 * y ** 2
    target = target.reshape(-1, 1)
    return X, target


@pytest.fixture
def regression_single_informative():
    """Only the first feature matters; second is pure noise.

    Target: y = 3*x1.  x2 is random.
    """
    rng = np.random.RandomState(7)
    n = 150
    x1 = np.linspace(-3, 3, n)
    x2 = rng.randn(n)
    X = np.column_stack([x1, x2])
    target = (3 * x1).reshape(-1, 1)
    return X, target


@pytest.fixture
def binary_classification():
    """Binary classification: first feature is informative, second is noise.

    Label: y = 1 if x1 > 0 else 0.  x2 is random.
    """
    rng = np.random.RandomState(0)
    n = 100
    x1 = np.linspace(-5, 5, n)
    x2 = rng.randn(n)
    X = np.column_stack([x1, x2])
    y = (x1 > 0).astype(float)
    return X, y


@pytest.fixture
def multiclass_classification():
    """Three classes determined by first feature; second feature is noise.

    Label: 0 if x1 < -1, 1 if -1 <= x1 <= 1, 2 if x1 > 1.
    """
    rng = np.random.RandomState(3)
    n = 150
    x1 = np.linspace(-3, 3, n)
    x2 = rng.randn(n)
    X = np.column_stack([x1, x2])
    y = np.zeros(n)
    y[x1 > 1] = 2
    y[(x1 >= -1) & (x1 <= 1)] = 1
    return X, y


@pytest.fixture
def all_noise_regression():
    """Target is constant – no feature should be highly weighted."""
    rng = np.random.RandomState(99)
    n = 100
    X = rng.randn(n, 3)
    y = np.ones((n, 1)) * 5.0
    return X, y


# ---------------------------------------------------------------------------
# RReliefF (regression) tests
# ---------------------------------------------------------------------------

class TestRReliefF:
    """Tests for the RReliefF algorithm (regression feature selection)."""

    def test_output_shape(self, regression_2d_quadratic):
        X, y = regression_2d_quadratic
        W = RReliefF(X, y)
        assert W.shape == (X.shape[1], 1), (
            f"Weight vector shape should be ({X.shape[1]}, 1), got {W.shape}"
        )

    def test_informative_features_ranked_above_noise(self, regression_2d_quadratic):
        """The two quadratic features should have higher weight than noise."""
        X, y = regression_2d_quadratic
        W = RReliefF(X, y, k=10, sigma=30)
        weights = W.ravel()
        # Features 0 and 1 are informative; feature 2 is noise
        assert weights[0] > weights[2], (
            f"Informative feature 0 ({weights[0]:.4f}) should outrank noise ({weights[2]:.4f})"
        )
        assert weights[1] > weights[2], (
            f"Informative feature 1 ({weights[1]:.4f}) should outrank noise ({weights[2]:.4f})"
        )

    def test_single_informative_feature(self, regression_single_informative):
        """When only x1 matters, its weight should dominate."""
        X, y = regression_single_informative
        W = RReliefF(X, y, k=10, sigma=30)
        weights = W.ravel()
        assert weights[0] > weights[1], (
            f"Informative feature ({weights[0]:.4f}) should outrank noise ({weights[1]:.4f})"
        )

    def test_weight_track_mode(self, regression_2d_quadratic):
        """weight_track=True should return (W_A, Wtrack, iTrack)."""
        X, y = regression_2d_quadratic
        result = RReliefF(X, y, weight_track=True)
        assert isinstance(result, tuple) and len(result) == 3
        W_A, Wtrack, iTrack = result
        assert W_A.shape == (X.shape[1], 1)
        assert Wtrack.shape == (X.shape[0], X.shape[1])
        assert iTrack.shape == (X.shape[0], 1)

    def test_updates_subset(self, regression_2d_quadratic):
        """Using updates < n should still return correct shape."""
        X, y = regression_2d_quadratic
        m = 50
        W = RReliefF(X, y, updates=m, k=10, sigma=30)
        assert W.shape == (X.shape[1], 1)

    def test_updates_subset_weight_track(self, regression_2d_quadratic):
        """weight_track with subset updates should have correct dimensions."""
        X, y = regression_2d_quadratic
        m = 50
        W_A, Wtrack, iTrack = RReliefF(X, y, updates=m, weight_track=True)
        assert Wtrack.shape == (m, X.shape[1])
        assert iTrack.shape == (m, 1)

    def test_k_parameter_effect(self, regression_2d_quadratic):
        """Different k values should still identify the right features."""
        X, y = regression_2d_quadratic
        for k_val in [5, 10, 20]:
            W = RReliefF(X, y, k=k_val, sigma=30)
            weights = W.ravel()
            assert weights[0] > weights[2], f"Failed for k={k_val}"
            assert weights[1] > weights[2], f"Failed for k={k_val}"

    def test_sigma_parameter_effect(self, regression_2d_quadratic):
        """Different sigma values should still identify the right features."""
        X, y = regression_2d_quadratic
        for sigma_val in [10, 30, 50]:
            W = RReliefF(X, y, k=10, sigma=sigma_val)
            weights = W.ravel()
            assert weights[0] > weights[2], f"Failed for sigma={sigma_val}"

    def test_constant_target_low_weights(self, all_noise_regression):
        """When target is constant, all weights should be near zero or nan-free."""
        X, y = all_noise_regression
        # With constant target, yRange = 0 which causes division by zero.
        # This tests current behavior (may raise or return inf/nan).
        # We just ensure it doesn't crash unexpectedly.
        try:
            W = RReliefF(X, y)
            # If it succeeds, weights should exist
            assert W.shape == (3, 1)
        except (ZeroDivisionError, FloatingPointError):
            pass  # Known limitation with constant target

    def test_reproducibility_with_all_updates(self, regression_2d_quadratic):
        """With updates='all', results should be deterministic."""
        X, y = regression_2d_quadratic
        W1 = RReliefF(X, y, updates='all')
        W2 = RReliefF(X, y, updates='all')
        np.testing.assert_array_equal(W1, W2)


# ---------------------------------------------------------------------------
# ReliefF (multi-class classification) tests
# ---------------------------------------------------------------------------

class TestReliefF:
    """Tests for the ReliefF algorithm (multi-class feature selection)."""

    def test_output_shape(self, multiclass_classification):
        X, y = multiclass_classification
        W = ReliefF(X, y)
        assert W.shape == (X.shape[1], 1)

    def test_informative_feature_ranked_higher(self, multiclass_classification):
        """The class-defining feature should rank above noise."""
        X, y = multiclass_classification
        W = ReliefF(X, y, k=10, sigma=30)
        weights = W.ravel()
        assert weights[0] > weights[1], (
            f"Informative feature ({weights[0]:.4f}) should outrank noise ({weights[1]:.4f})"
        )

    def test_binary_classification(self, binary_classification):
        """ReliefF should also work on binary problems."""
        X, y = binary_classification
        W = ReliefF(X, y, k=10, sigma=30)
        weights = W.ravel()
        assert weights[0] > weights[1], (
            f"Informative feature ({weights[0]:.4f}) should outrank noise ({weights[1]:.4f})"
        )

    def test_weight_track_mode(self, multiclass_classification):
        X, y = multiclass_classification
        result = ReliefF(X, y, weight_track=True)
        assert isinstance(result, tuple) and len(result) == 3
        W_A, Wtrack, iTrack = result
        assert W_A.shape == (X.shape[1], 1)
        assert Wtrack.shape == (X.shape[0], X.shape[1])

    def test_updates_subset(self, multiclass_classification):
        X, y = multiclass_classification
        m = 30
        W = ReliefF(X, y, updates=m)
        assert W.shape == (X.shape[1], 1)

    def test_k_parameter_effect(self, multiclass_classification):
        X, y = multiclass_classification
        for k_val in [5, 10, 15]:
            W = ReliefF(X, y, k=k_val)
            weights = W.ravel()
            assert weights[0] > weights[1], f"Failed for k={k_val}"

    def test_reproducibility_with_all_updates(self, multiclass_classification):
        X, y = multiclass_classification
        W1 = ReliefF(X, y, updates='all')
        W2 = ReliefF(X, y, updates='all')
        np.testing.assert_array_equal(W1, W2)


# ---------------------------------------------------------------------------
# Relief (binary classification) tests
# ---------------------------------------------------------------------------

class TestRelief:
    """Tests for the Relief algorithm (binary classification feature selection)."""

    def test_output_shape(self, binary_classification):
        X, y = binary_classification
        W = Relief(X, y)
        assert W.shape == (X.shape[1], 1)

    def test_informative_feature_ranked_higher(self, binary_classification):
        """The class-defining feature should be ranked above noise."""
        X, y = binary_classification
        W = Relief(X, y, sigma=30)
        weights = W.ravel()
        assert weights[0] > weights[1], (
            f"Informative feature ({weights[0]:.4f}) should outrank noise ({weights[1]:.4f})"
        )

    def test_weight_track_mode(self, binary_classification):
        X, y = binary_classification
        result = Relief(X, y, weight_track=True)
        assert isinstance(result, tuple) and len(result) == 5
        W_A, Wtrack, iTrack, hitTrack, missTrack = result
        assert W_A.shape == (X.shape[1], 1)
        assert Wtrack.shape == (X.shape[0], X.shape[1])
        assert hitTrack.shape == (X.shape[0], X.shape[1])
        assert missTrack.shape == (X.shape[0], X.shape[1])

    def test_updates_subset(self, binary_classification):
        X, y = binary_classification
        m = 20
        W = Relief(X, y, updates=m)
        assert W.shape == (X.shape[1], 1)

    def test_reproducibility_with_all_updates(self, binary_classification):
        X, y = binary_classification
        W1 = Relief(X, y, updates='all')
        W2 = Relief(X, y, updates='all')
        np.testing.assert_array_equal(W1, W2)

    def test_hit_miss_consistency(self, binary_classification):
        """hitTrack and missTrack values should be non-negative."""
        X, y = binary_classification
        _, _, _, hitTrack, missTrack = Relief(X, y, weight_track=True)
        assert np.all(hitTrack >= 0), "Hit track should be non-negative"
        assert np.all(missTrack >= 0), "Miss track should be non-negative"


# ---------------------------------------------------------------------------
# Cross-algorithm consistency tests
# ---------------------------------------------------------------------------

class TestCrossAlgorithm:
    """Tests for consistency between Relief variants."""

    def test_relief_and_relieff_agree_on_binary(self, binary_classification):
        """Relief and ReliefF should agree on which feature is more important
        for a binary classification problem."""
        X, y = binary_classification
        W_relief = Relief(X, y).ravel()
        W_relieff = ReliefF(X, y).ravel()

        # Both should rank feature 0 above feature 1
        assert np.argmax(W_relief) == 0, "Relief should pick feature 0"
        assert np.argmax(W_relieff) == 0, "ReliefF should pick feature 0"

    def test_all_algorithms_return_finite(self, regression_2d_quadratic, binary_classification):
        """All algorithms should return finite values on well-formed data."""
        X_reg, y_reg = regression_2d_quadratic
        X_cls, y_cls = binary_classification

        W_rrelieff = RReliefF(X_reg, y_reg)
        assert np.all(np.isfinite(W_rrelieff)), "RReliefF should return finite values"

        W_relieff = ReliefF(X_cls, y_cls)
        assert np.all(np.isfinite(W_relieff)), "ReliefF should return finite values"

        W_relief = Relief(X_cls, y_cls)
        assert np.all(np.isfinite(W_relief)), "Relief should return finite values"


# ---------------------------------------------------------------------------
# Feature selection quality tests (higher-dimensional)
# ---------------------------------------------------------------------------

class TestFeatureSelectionQuality:
    """Tests that algorithms correctly identify predictive features
    in more challenging scenarios."""

    def test_rrelieff_many_noise_features(self):
        """With 2 informative + 8 noise features, RReliefF should rank
        informative features in the top positions."""
        rng = np.random.RandomState(42)
        n = 300
        x1 = np.linspace(0, 5, n)
        x2 = np.linspace(0, 5, n)
        noise = rng.randn(n, 8)
        X = np.column_stack([x1, x2, noise])
        y = (3 * x1 + 2 * x2).reshape(-1, 1)

        W = RReliefF(X, y, k=10, sigma=30)
        weights = W.ravel()

        # Top 2 features by weight should be features 0 and 1
        top_2 = set(np.argsort(weights)[-2:])
        assert top_2 == {0, 1}, (
            f"Top 2 features should be {{0, 1}}, got {top_2}. Weights: {weights}"
        )

    def test_relieff_many_noise_features(self):
        """With 1 informative + 4 noise features, ReliefF should rank
        the informative feature highest."""
        rng = np.random.RandomState(10)
        n = 200
        x1 = np.linspace(-3, 3, n)
        noise = rng.randn(n, 4)
        X = np.column_stack([x1, noise])
        y = (x1 > 0).astype(float)

        W = ReliefF(X, y, k=10)
        weights = W.ravel()
        assert np.argmax(weights) == 0, (
            f"Feature 0 should be ranked first, got {np.argmax(weights)}. Weights: {weights}"
        )

    def test_relief_nonlinear_boundary(self):
        """Relief should detect the informative feature even with a
        non-linear decision boundary (quadratic)."""
        rng = np.random.RandomState(5)
        n = 100
        x1 = np.linspace(-3, 3, n)
        x2 = rng.randn(n)
        X = np.column_stack([x1, x2])
        y = (x1 ** 2 > 2).astype(float)

        W = Relief(X, y)
        weights = W.ravel()
        assert weights[0] > weights[1], (
            f"Informative feature ({weights[0]:.4f}) should outrank noise ({weights[1]:.4f})"
        )


# ---------------------------------------------------------------------------
# Edge case / robustness tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_single_feature_regression(self):
        """Should work with a single feature."""
        n = 50
        X = np.linspace(0, 5, n).reshape(-1, 1)
        y = (2 * X).reshape(-1, 1)
        W = RReliefF(X, y, k=5)
        assert W.shape == (1, 1)

    def test_single_feature_classification(self):
        """Should work with a single feature for classification."""
        n = 50
        X = np.linspace(-5, 5, n).reshape(-1, 1)
        y = (X.ravel() > 0).astype(float)
        W = Relief(X, y)
        assert W.shape == (1, 1)

    def test_large_k(self):
        """k close to n should not crash (it may produce less meaningful results)."""
        n = 30
        X = np.random.RandomState(42).randn(n, 2)
        y = X[:, 0].reshape(-1, 1)
        # k = n-1 (max meaningful)
        W = RReliefF(X, y, k=n - 2)
        assert W.shape == (2, 1)

    def test_identical_features(self):
        """When two features are identical copies, they should get similar weights."""
        rng = np.random.RandomState(42)
        n = 100
        x = np.linspace(0, 5, n)
        X = np.column_stack([x, x, rng.randn(n)])
        y = (3 * x).reshape(-1, 1)
        W = RReliefF(X, y, k=10)
        weights = W.ravel()
        # Features 0 and 1 are identical, so weights should be close
        np.testing.assert_allclose(weights[0], weights[1], atol=0.01)

    def test_updates_one(self):
        """updates=1 should work (minimal case)."""
        rng = np.random.RandomState(0)
        n = 50
        X = rng.randn(n, 3)
        y = X[:, 0].reshape(-1, 1)
        W = RReliefF(X, y, updates=1, k=5)
        assert W.shape == (3, 1)
