"""GPU-accelerated Relief-based feature selection with multi-backend support.

This module provides fully vectorized implementations of RReliefF, ReliefF,
and Relief that run on GPU hardware from multiple vendors:

    - Apple Metal (M1/M2/M3/M4) via MLX  — pip install mlx
    - NVIDIA CUDA via CuPy                — pip install cupy-cuda12x
    - Any GPU (NVIDIA/AMD/Apple/TPU) via JAX — pip install jax jaxlib
    - CPU fallback via vectorized NumPy    — always available

Backend priority: MLX > CuPy > JAX > NumPy (auto-detected).
Override with: set_backend("jax"), set_backend("cupy"), etc.

Usage:
    from relieff_gpu import RReliefF_gpu, ReliefF_gpu, Relief_gpu
    W = RReliefF_gpu(X, y, k=10)  # returns numpy array, same shape as relieff.RReliefF
"""

from __future__ import annotations

from typing import Any, Literal, Union

import numpy as np
import numpy.typing as npt

NDFloat = npt.NDArray[np.float64]

# --- Backend detection ---

_HAS_MLX = False
_HAS_CUPY = False
_HAS_JAX = False

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    pass

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    pass

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    pass

# User-overridable backend selection
_BACKEND_OVERRIDE: str | None = None

_VALID_BACKENDS = ("mlx", "cupy", "jax", "numpy")


def set_backend(name: str | None) -> None:
    """Override the auto-detected backend.

    Parameters
    ----------
    name : str or None
        One of 'mlx', 'cupy', 'jax', 'numpy', or None (auto-detect).

    Raises
    ------
    ValueError
        If name is not a recognised backend.
    ImportError
        If the requested backend is not installed.
    """
    global _BACKEND_OVERRIDE
    if name is None:
        _BACKEND_OVERRIDE = None
        return
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend {name!r}. Choose from {_VALID_BACKENDS}")
    avail = {"mlx": _HAS_MLX, "cupy": _HAS_CUPY, "jax": _HAS_JAX, "numpy": True}
    if not avail[name]:
        raise ImportError(f"Backend {name!r} is not installed")
    _BACKEND_OVERRIDE = name


def _get_backend() -> str:
    """Return the name of the active backend.

    Priority: user override > MLX > CuPy > JAX > NumPy.
    """
    if _BACKEND_OVERRIDE is not None:
        return _BACKEND_OVERRIDE
    if _HAS_MLX:
        return "mlx"
    if _HAS_CUPY:
        return "cupy"
    if _HAS_JAX:
        return "jax"
    return "numpy"


def available_backends() -> list[str]:
    """Return a list of backends that are installed and available."""
    result = ["numpy"]  # always available
    if _HAS_JAX:
        result.append("jax")
    if _HAS_CUPY:
        result.append("cupy")
    if _HAS_MLX:
        result.append("mlx")
    return result


# ---------------------------------------------------------------------------
# Array operations abstraction layer
# ---------------------------------------------------------------------------
# Provides a thin wrapper so the same vectorized algorithm can run on
# MLX (Apple Metal), CuPy (NVIDIA CUDA), JAX (any GPU), or NumPy (CPU).


class _Ops:
    """Array operations backend interface.

    Uses Any types because the same interface serves multiple array libraries.
    The public API boundary always converts back to NDFloat.
    """

    def array(self, x: Any) -> Any: raise NotImplementedError
    def zeros(self, shape: Any) -> Any: raise NotImplementedError
    def arange(self, n: int) -> Any: raise NotImplementedError
    def sum(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: raise NotImplementedError
    def abs(self, x: Any) -> Any: raise NotImplementedError
    def exp(self, x: Any) -> Any: raise NotImplementedError
    def max(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: raise NotImplementedError
    def min(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: raise NotImplementedError
    def argsort(self, x: Any, axis: int = -1) -> Any: raise NotImplementedError
    def take_along_axis(self, x: Any, indices: Any, axis: int) -> Any: raise NotImplementedError
    def expand_dims(self, x: Any, axis: int) -> Any: raise NotImplementedError
    def unique(self, x: Any) -> Any: raise NotImplementedError
    def where(self, cond: Any, x: Any, y: Any) -> Any: raise NotImplementedError
    def to_numpy(self, x: Any) -> NDFloat: raise NotImplementedError
    def eval(self, *args: Any) -> None: raise NotImplementedError


class _NumpyOps(_Ops):
    """NumPy backend (CPU). Always available."""

    def array(self, x: Any) -> Any:
        return np.asarray(x, dtype=np.float64)

    def zeros(self, shape: Any) -> Any:
        return np.zeros(shape, dtype=np.float64)

    def arange(self, n: int) -> Any:
        return np.arange(n)

    def sum(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return np.sum(x, axis=axis, keepdims=keepdims)

    def abs(self, x: Any) -> Any:
        return np.abs(x)

    def exp(self, x: Any) -> Any:
        return np.exp(x)

    def max(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return np.max(x, axis=axis, keepdims=keepdims)

    def min(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return np.min(x, axis=axis, keepdims=keepdims)

    def argsort(self, x: Any, axis: int = -1) -> Any:
        return np.argsort(x, axis=axis)

    def take_along_axis(self, x: Any, indices: Any, axis: int) -> Any:
        return np.take_along_axis(x, indices, axis=axis)

    def expand_dims(self, x: Any, axis: int) -> Any:
        return np.expand_dims(x, axis=axis)

    def unique(self, x: Any) -> Any:
        return np.unique(x)

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        return np.where(cond, x, y)

    def to_numpy(self, x: Any) -> NDFloat:
        return np.asarray(x, dtype=np.float64)

    def eval(self, *args: Any) -> None:
        pass  # NumPy is eager, no-op


class _MLXOps(_Ops):
    """MLX backend for Apple Silicon (Metal GPU).

    Uses float32 (Metal does not support float64). Results are converted
    back to float64 numpy arrays at the API boundary.
    """

    def array(self, x: Any) -> Any:
        return mx.array(np.asarray(x, dtype=np.float32))

    def zeros(self, shape: Any) -> Any:
        if isinstance(shape, int):
            shape = (shape,)
        return mx.zeros(shape, dtype=mx.float32)

    def arange(self, n: int) -> Any:
        return mx.arange(n)

    def sum(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return mx.sum(x, axis=axis, keepdims=keepdims)

    def abs(self, x: Any) -> Any:
        return mx.abs(x)

    def exp(self, x: Any) -> Any:
        return mx.exp(x)

    def max(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return mx.max(x, axis=axis, keepdims=keepdims)

    def min(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return mx.min(x, axis=axis, keepdims=keepdims)

    def argsort(self, x: Any, axis: int = -1) -> Any:
        return mx.argsort(x, axis=axis)

    def take_along_axis(self, x: Any, indices: Any, axis: int) -> Any:
        return mx.take_along_axis(x, indices, axis=axis)

    def expand_dims(self, x: Any, axis: int) -> Any:
        return mx.expand_dims(x, axis=axis)

    def unique(self, x: Any) -> Any:
        # MLX does not have unique(); fall back to numpy for label discovery
        return mx.array(np.unique(np.array(x)))

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        return mx.where(cond, x, y)

    def to_numpy(self, x: Any) -> NDFloat:
        return np.array(x, copy=False).astype(np.float64)

    def eval(self, *args: Any) -> None:
        mx.eval(*args)


class _CuPyOps(_Ops):
    """CuPy backend for NVIDIA CUDA GPUs.

    CuPy mirrors the NumPy API almost exactly, running on CUDA.
    Uses float64 (NVIDIA GPUs support it natively).
    """

    def array(self, x: Any) -> Any:
        return cp.asarray(x, dtype=cp.float64)

    def zeros(self, shape: Any) -> Any:
        return cp.zeros(shape, dtype=cp.float64)

    def arange(self, n: int) -> Any:
        return cp.arange(n)

    def sum(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return cp.sum(x, axis=axis, keepdims=keepdims)

    def abs(self, x: Any) -> Any:
        return cp.abs(x)

    def exp(self, x: Any) -> Any:
        return cp.exp(x)

    def max(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return cp.max(x, axis=axis, keepdims=keepdims)

    def min(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return cp.min(x, axis=axis, keepdims=keepdims)

    def argsort(self, x: Any, axis: int = -1) -> Any:
        return cp.argsort(x, axis=axis)

    def take_along_axis(self, x: Any, indices: Any, axis: int) -> Any:
        return cp.take_along_axis(x, indices, axis=axis)

    def expand_dims(self, x: Any, axis: int) -> Any:
        return cp.expand_dims(x, axis=axis)

    def unique(self, x: Any) -> Any:
        return cp.unique(x)

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        return cp.where(cond, x, y)

    def to_numpy(self, x: Any) -> NDFloat:
        return cp.asnumpy(x).astype(np.float64)

    def eval(self, *args: Any) -> None:
        cp.cuda.Stream.null.synchronize()


class _JAXOps(_Ops):
    """JAX backend for cross-platform GPU acceleration.

    Supports NVIDIA (CUDA), AMD (ROCm), Apple (Metal via jax-metal),
    Google TPU, and CPU. Uses JIT compilation for performance.
    JAX arrays are immutable; operations build a computation trace
    that is compiled and executed on the available accelerator.
    """

    def array(self, x: Any) -> Any:
        return jnp.asarray(x, dtype=jnp.float32)

    def zeros(self, shape: Any) -> Any:
        return jnp.zeros(shape, dtype=jnp.float32)

    def arange(self, n: int) -> Any:
        return jnp.arange(n)

    def sum(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return jnp.sum(x, axis=axis, keepdims=keepdims)

    def abs(self, x: Any) -> Any:
        return jnp.abs(x)

    def exp(self, x: Any) -> Any:
        return jnp.exp(x)

    def max(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return jnp.max(x, axis=axis, keepdims=keepdims)

    def min(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return jnp.min(x, axis=axis, keepdims=keepdims)

    def argsort(self, x: Any, axis: int = -1) -> Any:
        return jnp.argsort(x, axis=axis)

    def take_along_axis(self, x: Any, indices: Any, axis: int) -> Any:
        return jnp.take_along_axis(x, indices, axis=axis)

    def expand_dims(self, x: Any, axis: int) -> Any:
        return jnp.expand_dims(x, axis=axis)

    def unique(self, x: Any) -> Any:
        # jnp.unique requires fixed-size output; use numpy for label discovery
        return jnp.asarray(np.unique(np.asarray(x)))

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        return jnp.where(cond, x, y)

    def to_numpy(self, x: Any) -> NDFloat:
        return np.asarray(x, dtype=np.float64)

    def eval(self, *args: Any) -> None:
        # JAX uses async dispatch; block until results are ready
        for a in args:
            a.block_until_ready()


_OPS_MAP: dict[str, type[_Ops]] = {
    "numpy": _NumpyOps,
    "mlx": _MLXOps,
    "cupy": _CuPyOps,
    "jax": _JAXOps,
}


def _get_ops() -> _Ops:
    """Return the appropriate ops backend instance."""
    return _OPS_MAP[_get_backend()]()


# ---------------------------------------------------------------------------
# Vectorized distance weight computation (Equation 8)
# ---------------------------------------------------------------------------

def _distance_weights(ops: _Ops, k: int, sigma: int) -> Any:
    """Compute exponential distance weights for k neighbours.

    w[j] = exp(-((j+1)/sigma)^2) / sum(w)
    """
    indices = ops.arange(k)
    # (indices + 1) / sigma, squared, negated, exponentiated
    raw = ops.exp(-((indices + 1.0) / sigma) ** 2)
    return raw / ops.sum(raw)


# ---------------------------------------------------------------------------
# Vectorized pairwise distance matrix
# ---------------------------------------------------------------------------

def _pairwise_sq_distances(ops: _Ops, X: Any) -> Any:
    """Compute (n, n) squared Euclidean distance matrix.

    D[i,j] = sum((X[i] - X[j])^2)

    Uses broadcasting: X[:, None, :] - X[None, :, :] -> (n, n, p)
    Then sum over features axis.
    """
    # X shape: (n, p)
    X_i = ops.expand_dims(X, 1)  # (n, 1, p)
    X_j = ops.expand_dims(X, 0)  # (1, n, p)
    diff = X_i - X_j             # (n, n, p) via broadcast
    return ops.sum(diff ** 2, axis=2)  # (n, n)


# ---------------------------------------------------------------------------
# Vectorized feature difference matrix (Equation 2)
# ---------------------------------------------------------------------------

def _feature_ranges(ops: _Ops, X: Any) -> Any:
    """Compute per-feature range for normalization: max - min along axis 0.

    Returns shape (p,).
    """
    feat_max = ops.max(X, axis=0)  # (p,)
    feat_min = ops.min(X, axis=0)  # (p,)
    ranges = feat_max - feat_min
    # Avoid division by zero for constant features
    ranges = ops.where(ranges == 0.0, ops.array(np.array([1.0])), ranges)
    return ranges


# ---------------------------------------------------------------------------
# RReliefF (Regression) - Vectorized GPU implementation
# ---------------------------------------------------------------------------

def RReliefF_gpu(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = "all",
    k: int = 10,
    sigma: int = 30,
) -> NDFloat:
    """GPU-accelerated RReliefF for regression feature selection.

    Fully vectorized implementation that auto-selects the best available
    backend: MLX (Apple Metal), CuPy (NVIDIA CUDA), JAX (any GPU), or NumPy.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,) or (n_samples, 1)
        Continuous target values.
    updates : 'all' or int
        Number of instances to sample. 'all' uses every instance.
    k : int
        Number of nearest neighbours (default 10).
    sigma : int
        Distance scaling factor (default 30).

    Returns
    -------
    W : ndarray of shape (n_features, 1)
        Feature importance weights. Higher = more predictive.
    """
    ops = _get_ops()

    y_np = np.asarray(y, dtype=np.float64).ravel()
    X_np = np.asarray(X, dtype=np.float64)
    n, p = X_np.shape

    # Determine which instances to process
    if updates == "all":
        sample_indices = np.arange(n)
    else:
        sample_indices = np.random.randint(0, n, size=int(updates))

    m = len(sample_indices)

    # Move data to GPU
    X_g = ops.array(X_np)
    y_g = ops.array(y_np)

    # Precompute
    y_range = ops.max(y_g) - ops.min(y_g)
    feat_ranges = _feature_ranges(ops, X_g)  # (p,)
    dist_w = _distance_weights(ops, k, sigma)  # (k,)
    dist_w_30 = _distance_weights(ops, k, 30)  # (k,) for N_dA (hardcoded sigma=30 in original)

    # Full pairwise distance matrix: (n, n)
    D = _pairwise_sq_distances(ops, X_g)

    # Set self-distance to infinity so self is never a neighbour
    big_val = ops.array(np.array([1e30]))
    identity = ops.array(np.eye(n, dtype=np.float32))
    D = D + identity * big_val

    # Find k nearest neighbours for all instances: argsort rows, take first k
    sorted_idx = ops.argsort(D, axis=1)  # (n, n) indices sorted by distance

    # Slice to k nearest: (n, k)
    # MLX/numpy slicing: sorted_idx[:, :k]
    knn_idx = sorted_idx[:, :k]  # type: ignore[index]

    # Gather neighbour features: X_neighbours[i, j, :] = X[knn_idx[i,j], :]
    # Shape: (n, k, p)
    # Convert indices to numpy for advanced indexing (works for all backends).
    knn_idx_np = np.asarray(ops.to_numpy(knn_idx)) if _get_backend() != "numpy" else knn_idx
    X_knn_np = X_np[knn_idx_np.astype(int)]  # (n, k, p) in numpy
    y_knn_np = y_np[knn_idx_np.astype(int).ravel()].reshape(n, k)  # (n, k)

    X_knn = ops.array(X_knn_np)
    y_knn = ops.array(y_knn_np)

    # Feature differences: |X[i, a] - X_knn[i, j, a]| / range(a)
    # X_g[:, None, :] has shape (n, 1, p), X_knn has shape (n, k, p)
    X_expanded = ops.expand_dims(X_g, 1)  # (n, 1, p)
    feat_diff = ops.abs(X_expanded - X_knn) / feat_ranges  # (n, k, p)

    # Output differences: |y[i] - y_knn[i, j]| / y_range
    y_expanded = ops.expand_dims(y_g, 1)  # (n, 1)
    y_diff = ops.abs(y_expanded - y_knn) / y_range  # (n, k)

    # Now restrict to sampled instances
    sample_idx_g = ops.array(np.array(sample_indices, dtype=np.float64))

    # For vectorized accumulation, we only need the sampled rows
    # feat_diff_sampled: (m, k, p), y_diff_sampled: (m, k)
    feat_diff_sampled_np = np.asarray(ops.to_numpy(feat_diff))[sample_indices]
    y_diff_sampled_np = np.asarray(ops.to_numpy(y_diff))[sample_indices]

    feat_diff_s = ops.array(feat_diff_sampled_np)
    y_diff_s = ops.array(y_diff_sampled_np)

    # dist_w shape: (k,) -> broadcast to (1, k) for (m, k) operations
    # N_dC = sum over all sampled instances and neighbours of y_diff * dist_w
    # N_dC = sum_{i in samples} sum_{j=0..k-1} y_diff[i,j] * dist_w[j]
    weighted_y_diff = y_diff_s * dist_w  # (m, k) broadcast
    N_dC = ops.sum(weighted_y_diff)  # scalar

    # N_dA[a] = sum_{i} sum_{j} feat_diff[i,j,a] * dist_w_30[j]
    # dist_w_30 shape (k,) -> (1, k, 1) for broadcast with (m, k, p)
    dw30_expanded = ops.expand_dims(ops.expand_dims(dist_w_30, 0), 2)  # (1, k, 1)
    N_dA = ops.sum(feat_diff_s * dw30_expanded, axis=(0, 1))  # (p,)

    # N_dCanddA[a] = sum_{i} sum_{j} y_diff[i,j] * dist_w[j] * feat_diff[i,j,a]
    # weighted_y_diff shape (m, k) -> (m, k, 1) for broadcast with (m, k, p)
    wy_expanded = ops.expand_dims(weighted_y_diff, 2)  # (m, k, 1)
    N_dCanddA = ops.sum(wy_expanded * feat_diff_s, axis=(0, 1))  # (p,)

    # Final weight: W[a] = N_dCanddA[a]/N_dC - (N_dA[a] - N_dCanddA[a])/(m - N_dC)
    W_A = N_dCanddA / N_dC - (N_dA - N_dCanddA) / (m - N_dC)

    # Force evaluation on GPU
    ops.eval(W_A)

    # Convert back to numpy (n_features, 1)
    result: NDFloat = ops.to_numpy(W_A).reshape(-1, 1)
    return result


# ---------------------------------------------------------------------------
# ReliefF (Classification) - Vectorized GPU implementation
# ---------------------------------------------------------------------------

def ReliefF_gpu(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = "all",
    k: int = 10,
    sigma: int = 30,
) -> NDFloat:
    """GPU-accelerated ReliefF for classification feature selection.

    Fully vectorized implementation that auto-selects the best available
    backend: MLX (Apple Metal), CuPy (NVIDIA CUDA), JAX (any GPU), or NumPy.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,) or (n_samples, 1)
        Class labels.
    updates : 'all' or int
        Number of instances to sample. 'all' uses every instance.
    k : int
        Number of nearest neighbours (default 10).
    sigma : int
        Distance scaling factor (default 30). Not used in weight computation
        for classification but kept for API consistency.

    Returns
    -------
    W : ndarray of shape (n_features, 1)
        Feature importance weights. Higher = more predictive.
    """
    ops = _get_ops()

    y_np = np.asarray(y, dtype=np.float64).ravel()
    X_np = np.asarray(X, dtype=np.float64)
    n, p = X_np.shape

    if updates == "all":
        sample_indices = np.arange(n)
    else:
        sample_indices = np.random.randint(0, n, size=int(updates))

    m = len(sample_indices)

    # Move to GPU
    X_g = ops.array(X_np)
    y_g = ops.array(y_np)

    feat_ranges = _feature_ranges(ops, X_g)  # (p,)
    labels_np = np.unique(y_np)

    # Precompute full pairwise distance matrix
    D = _pairwise_sq_distances(ops, X_g)
    big_val = ops.array(np.array([1e30]))
    identity = ops.array(np.eye(n, dtype=np.float32))
    D = D + identity * big_val

    # For each class label, create a masked distance matrix where non-members
    # have infinite distance. This lets us find k nearest hits/misses via argsort.
    # class_masks[label] = (n, n) where D[i,j] = inf if y[j] != label

    W_A_np = np.zeros(p, dtype=np.float64)

    for i_idx in range(m):
        i = sample_indices[i_idx]
        current_label = y_np[i]

        # --- Hits: same class ---
        hit_mask_np = (y_np == current_label).astype(np.float32)
        hit_mask = ops.array(hit_mask_np)
        # Mask: where y[j] != current_label, set distance to inf
        D_row_np = np.asarray(ops.to_numpy(D))[i:i + 1, :]  # (1, n)
        D_row = ops.array(D_row_np)
        D_hit = ops.where(hit_mask, D_row, big_val)  # (1, n)

        sorted_hit = ops.argsort(D_hit, axis=1)  # (1, n)
        hit_knn_idx = sorted_hit[0, :k]  # type: ignore[index]  # (k,)

        hit_knn_idx_np = np.asarray(ops.to_numpy(hit_knn_idx)).astype(int)
        X_hit = ops.array(X_np[hit_knn_idx_np])  # (k, p)

        X_i = ops.array(X_np[i:i + 1])  # (1, p)
        diff_hit = ops.abs(X_i - X_hit) / feat_ranges  # (k, p)
        sum_diff_hit_np = np.asarray(ops.to_numpy(ops.sum(diff_hit, axis=0)))  # (p,)

        # --- Misses: each other class ---
        diff_miss_weighted = np.zeros(p, dtype=np.float64)

        for label in labels_np:
            if label == current_label:
                continue

            prob_label = float(np.sum(y_np == label)) / n

            miss_mask_np = (y_np == label).astype(np.float32)
            miss_mask = ops.array(miss_mask_np)
            D_miss = ops.where(miss_mask, D_row, big_val)

            sorted_miss = ops.argsort(D_miss, axis=1)
            miss_knn_idx = sorted_miss[0, :k]  # type: ignore[index]

            miss_knn_idx_np = np.asarray(ops.to_numpy(miss_knn_idx)).astype(int)
            X_miss = ops.array(X_np[miss_knn_idx_np])  # (k, p)

            diff_miss = ops.abs(X_i - X_miss) / feat_ranges  # (k, p)
            sum_diff_miss_np = np.asarray(ops.to_numpy(ops.sum(diff_miss, axis=0)))

            diff_miss_weighted += prob_label * sum_diff_miss_np / (m * k)

        W_A_np -= sum_diff_hit_np / (m * k)
        W_A_np += diff_miss_weighted

    return W_A_np.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Relief (Binary Classification) - Vectorized GPU implementation
# ---------------------------------------------------------------------------

def Relief_gpu(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = "all",
    sigma: int = 30,
) -> NDFloat:
    """GPU-accelerated Relief for binary classification feature selection.

    Fully vectorized implementation that auto-selects the best available
    backend: MLX (Apple Metal), CuPy (NVIDIA CUDA), JAX (any GPU), or NumPy.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,) or (n_samples, 1)
        Binary class labels.
    updates : 'all' or int
        Number of instances to sample. 'all' uses every instance.
    sigma : int
        Distance scaling factor (default 30). Kept for API consistency.

    Returns
    -------
    W : ndarray of shape (n_features, 1)
        Feature importance weights. Higher = more predictive.
    """
    ops = _get_ops()

    y_np = np.asarray(y, dtype=np.float64).ravel()
    X_np = np.asarray(X, dtype=np.float64)
    n, p = X_np.shape

    if updates == "all":
        sample_indices = np.arange(n)
    else:
        sample_indices = np.random.randint(0, n, size=int(updates))

    m = len(sample_indices)

    X_g = ops.array(X_np)
    y_g = ops.array(y_np)

    feat_ranges = _feature_ranges(ops, X_g)

    # Full pairwise distances
    D = _pairwise_sq_distances(ops, X_g)
    big_val = ops.array(np.array([1e30]))
    identity = ops.array(np.eye(n, dtype=np.float32))
    D = D + identity * big_val

    # Build class masks: (n, n) boolean-like for hits and misses
    # hit_mask[i, j] = 1 if y[i] == y[j], else 0
    y_row = ops.expand_dims(y_g, 1)  # (n, 1)
    y_col = ops.expand_dims(y_g, 0)  # (1, n)

    # For all sampled instances simultaneously
    # D_hit[i,j] = D[i,j] if y[j]==y[i], else inf
    # D_miss[i,j] = D[i,j] if y[j]!=y[i], else inf
    same_class = ops.abs(y_row - y_col) < 0.5  # boolean-like (n, n)
    diff_class = ops.abs(y_row - y_col) >= 0.5

    D_hit_full = ops.where(same_class, D, big_val)   # (n, n)
    D_miss_full = ops.where(diff_class, D, big_val)  # (n, n)

    # Find nearest hit and nearest miss for each instance
    hit_sorted = ops.argsort(D_hit_full, axis=1)    # (n, n)
    miss_sorted = ops.argsort(D_miss_full, axis=1)  # (n, n)

    # Nearest hit/miss index for each instance
    nearest_hit_idx = hit_sorted[:, 0]   # type: ignore[index]  # (n,)
    nearest_miss_idx = miss_sorted[:, 0]  # type: ignore[index]  # (n,)

    ops.eval(nearest_hit_idx, nearest_miss_idx)

    nh_np = np.asarray(ops.to_numpy(nearest_hit_idx)).astype(int)
    nm_np = np.asarray(ops.to_numpy(nearest_miss_idx)).astype(int)

    # Gather nearest hit/miss feature vectors for sampled instances
    X_nh = X_np[nh_np[sample_indices]]  # (m, p)
    X_nm = X_np[nm_np[sample_indices]]  # (m, p)
    X_sampled = X_np[sample_indices]     # (m, p)

    feat_ranges_np = np.asarray(ops.to_numpy(feat_ranges))

    # Vectorized weight computation
    diff_hit = np.abs(X_sampled - X_nh) / feat_ranges_np   # (m, p)
    diff_miss = np.abs(X_sampled - X_nm) / feat_ranges_np  # (m, p)

    # W[a] = sum_i (-diff_hit[i,a] + diff_miss[i,a]) / m
    W_A: NDFloat = np.sum(-diff_hit + diff_miss, axis=0) / m  # (p,)

    return W_A.reshape(-1, 1)
