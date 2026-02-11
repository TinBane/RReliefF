from __future__ import annotations

from typing import Callable, Literal, Union, overload

import numpy as np
import numpy.typing as npt

"""
This code follows the algorithm for ReliefF as described in
"An adaptation of Relief for attribute estimation in regression"
by M. Robnik-Sikonja and I. Kononenko

Equation References in comments are based on the aforementioned article

To work with RReliefF, use RReliefF(X, y, opt)
opt can be replaced with the following optional arguments

- updates - This can be 'all' (default) or a positive integer depending
- k - The number of neighbours to look at. Default is 10.
- sigma - Distance scaling factor. Default is 50.
- weight_track - Returns a matrix which tracks the weight changes at each iteration. False by default
- categoricalx - This aspect has not been properly assimilated yet. Future work. Intended function:
You can specify if your inputs are categorial or not (False by default - assumes inputs are numeric).
Does not allow for the mixing of numeric and categorical predictors
"""

# Type aliases
NDFloat = npt.NDArray[np.float64]
DiffFunc = Callable[[int, NDFloat, NDFloat, NDFloat], float]


""" Multiple KNN Search functions for the different algorithms"""


def _knnsearchR(
    A: NDFloat, b: NDFloat, n: int
) -> tuple[NDFloat, npt.NDArray[np.intp]]:
    """Find the k nearest neighbours for regression."""
    difference: NDFloat = (A - b) ** 2
    sumDifference: NDFloat = np.sum(difference, axis=1)
    neighbourIndex: npt.NDArray[np.intp] = np.argsort(sumDifference)
    neighbours: NDFloat = A[neighbourIndex][1:]
    knn: NDFloat = neighbours[:n]
    return knn, neighbourIndex[1:]


def _knnsearchF(
    A: NDFloat, b: NDFloat, n: int, y: NDFloat, label: float
) -> tuple[NDFloat, npt.NDArray[np.intp]]:
    """Find the k nearest neighbours for classification (filtered by label)."""
    indToKeep: npt.NDArray[np.bool_] = y == label
    A = A[indToKeep.ravel(), :]

    difference: NDFloat = (A - b) ** 2
    sumDifference: NDFloat = np.sum(difference, axis=1)
    neighbourIndex: npt.NDArray[np.intp] = np.argsort(sumDifference)
    neighbours: NDFloat = A[neighbourIndex][1:]
    knn: NDFloat = neighbours[:n]
    return knn, neighbourIndex[1:]


def _knnsearch(
    A: NDFloat,
    b: NDFloat,
    n: int,
    opt: Literal["hit", "miss"],
    y: NDFloat,
    yRandomInstance: float,
) -> tuple[NDFloat, npt.NDArray[np.intp]]:
    """Find the k nearest neighbours with hit/miss label filtering."""
    if opt == "hit":
        indToKeep: npt.NDArray[np.bool_] = y == yRandomInstance
    else:
        indToKeep = y != yRandomInstance

    A = A[indToKeep, :]

    difference: NDFloat = (A - b) ** 2
    sumDifference: NDFloat = np.sum(difference, axis=1)
    neighbourIndex: npt.NDArray[np.intp] = np.argsort(sumDifference)
    neighbours: NDFloat = A[neighbourIndex][1:]
    knn: NDFloat = neighbours[:n]
    return knn, neighbourIndex[1:]


"""------------> Helper Functions <------------"""


def _distance(k: int, sigma: int) -> NDFloat:
    """Equation 8: exponential distance weighting for neighbours."""
    d1: list[np.float64] = [np.float64(np.exp(-((n + 1) / sigma) ** 2)) for n in range(k)]
    d: NDFloat = np.array(d1) / np.sum(d1)
    return d


def _diffNumeric(
    A: int,
    XRandomInstance: NDFloat,
    XKNNj: NDFloat,
    X: NDFloat,
) -> float:
    """Equation 2: normalized numeric attribute difference."""
    denominator: np.float64 = np.float64(np.max(X[:, A]) - np.min(X[:, A]))
    return float(np.abs(XRandomInstance[A] - XKNNj[A]) / denominator)


def _diffCategorical(
    A: int,
    XRandomInstance: NDFloat,
    XKNNj: NDFloat,
    X: NDFloat,
) -> int:
    """Categorical attribute difference (0 if equal, 1 if different)."""
    return int(not XRandomInstance[A] == XKNNj[A])


def _probability_class(y: NDFloat, currentLabel: float) -> float:
    """Calculate the probability of a class label in y."""
    numCurrentLabel: np.intp = np.sum(y == currentLabel)
    numTotal: int = len(y)
    return float(numCurrentLabel / numTotal)


"""------------> Main Relief related functions <------------"""


# --- RReliefF overloads ---
@overload
def RReliefF(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = ...,
    k: int = ...,
    sigma: int = ...,
    weight_track: Literal[False] = ...,
    categoricalx: bool = ...,
) -> NDFloat: ...


@overload
def RReliefF(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = ...,
    k: int = ...,
    sigma: int = ...,
    weight_track: Literal[True] = ...,
    categoricalx: bool = ...,
) -> tuple[NDFloat, NDFloat, NDFloat]: ...


def RReliefF(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = "all",
    k: int = 10,
    sigma: int = 30,
    weight_track: bool = False,
    categoricalx: bool = False,
) -> Union[NDFloat, tuple[NDFloat, NDFloat, NDFloat]]:

    # Ensure y is 1D to avoid shape issues with numpy 2.x
    y_flat: NDFloat = np.asarray(y, dtype=np.float64).ravel()

    # Check if user wants all values to be considered
    if updates == "all":
        m: int = X.shape[0]
    else:
        m = int(updates)

    # The constants need for RReliefF
    N_dC: float = 0.0
    N_dA: NDFloat = np.zeros(X.shape[1])
    N_dCanddA: NDFloat = np.zeros(X.shape[1])
    W_A_1d: NDFloat = np.zeros(X.shape[1])
    Wtrack: NDFloat = np.zeros([m, X.shape[1]])
    yRange: np.float64 = np.float64(np.max(y_flat) - np.min(y_flat))
    iTrack: NDFloat = np.zeros([m, 1])

    # Check if the input is categorical
    __diff: DiffFunc
    if categoricalx:
        __diff = _diffCategorical
    else:
        __diff = _diffNumeric

    # Repeat based on the total number of inputs or based on a user specified value
    for i in range(m):

        # Randomly access an instance
        if updates == "all":
            random_instance: int = i
        else:
            random_instance = int(np.random.randint(low=0, high=X.shape[0]))

        # Select a 'k' number in instances near the chosen random instance
        XKNN, neighbourIndex = _knnsearchR(X, X[random_instance, :], k)
        yKNN: NDFloat = y_flat[neighbourIndex]
        XRandomInstance: NDFloat = X[random_instance, :]
        yRandomInstance: np.float64 = np.float64(y_flat[random_instance])

        # Loop through all selected random instances
        for j in range(k):

            # Weight for different predictions
            N_dC += float(
                (np.abs(yRandomInstance - yKNN[j]) / yRange) * _distance(k, sigma)[j]
            )

            # Loop through all attributes
            for A in range(X.shape[1]):

                # Weight to account for different attributes
                N_dA[A] = N_dA[A] + __diff(A, XRandomInstance, XKNN[j], X) * _distance(k, 30)[j]

                # Concurrent examination of attributes and output
                N_dCanddA[A] = N_dCanddA[A] + float(
                    (np.abs(yRandomInstance - yKNN[j]) / yRange) * _distance(k, sigma)[j]
                ) * __diff(A, XRandomInstance, XKNN[j], X)

        # Track weights at each iteration
        for A in range(X.shape[1]):
            Wtrack[i, A] = N_dCanddA[A] / N_dC - ((N_dA[A] - N_dCanddA[A]) / (m - N_dC))

        # The index corresponding to the weight
        iTrack[i] = random_instance

    # Calculating the weights for all features
    for A in range(X.shape[1]):
        W_A_1d[A] = N_dCanddA[A] / N_dC - ((N_dA[A] - N_dCanddA[A]) / (m - N_dC))

    # Reshape to column vector for API compatibility
    W_A: NDFloat = W_A_1d.reshape(-1, 1)

    # Check if weight tracking is on
    if not weight_track:
        return W_A
    else:
        return W_A, Wtrack, iTrack


# --- ReliefF overloads ---
@overload
def ReliefF(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = ...,
    k: int = ...,
    sigma: int = ...,
    weight_track: Literal[False] = ...,
    categoricalx: bool = ...,
) -> NDFloat: ...


@overload
def ReliefF(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = ...,
    k: int = ...,
    sigma: int = ...,
    weight_track: Literal[True] = ...,
    categoricalx: bool = ...,
) -> tuple[NDFloat, NDFloat, NDFloat]: ...


def ReliefF(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = "all",
    k: int = 10,
    sigma: int = 30,
    weight_track: bool = False,
    categoricalx: bool = False,
) -> Union[NDFloat, tuple[NDFloat, NDFloat, NDFloat]]:
    # Ensure y is 1D to avoid shape issues with numpy 2.x
    y_flat: NDFloat = np.asarray(y, dtype=np.float64).ravel()

    # Check if user wants all values to be considered
    if updates == "all":
        m: int = X.shape[0]
    else:
        m = int(updates)

    # The constants need for ReliefF
    W_A_1d: NDFloat = np.zeros(X.shape[1])
    Wtrack: NDFloat = np.zeros([m, X.shape[1]])
    iTrack: NDFloat = np.zeros([m, 1])

    # Find unique labels
    labels: NDFloat = np.unique(y_flat)

    # Check if the input is categorical
    __diff: DiffFunc
    if categoricalx:
        __diff = _diffCategorical
    else:
        __diff = _diffNumeric

    # Repeat based on the total number of inputs or based on a user specified value
    for i in range(m):

        # Randomly access an instance
        if updates == "all":
            random_instance: int = i
        else:
            random_instance = int(np.random.randint(low=0, high=X.shape[0]))

        iTrack[i] = random_instance
        currentLabel: float = float(y_flat[random_instance])
        XKNNHit, neighbourIndexHit = _knnsearchF(X, X[random_instance, :], k, y_flat, currentLabel)
        missedLabels: NDFloat = labels[labels != currentLabel]
        XKNNMiss: list[NDFloat] = []
        neighbourIndexMiss: list[npt.NDArray[np.intp]] = []

        # Go through and find the misses
        for n in range(len(missedLabels)):
            XKNNCurrentMiss, neighbourIndexCurrentMiss = _knnsearchF(
                X, X[random_instance, :], k, y_flat, float(missedLabels[n])
            )
            XKNNMiss.append(XKNNCurrentMiss)
            neighbourIndexMiss.append(neighbourIndexCurrentMiss)
        XRandomInstance: NDFloat = X[random_instance, :]
        yRandomInstance: float = float(y_flat[random_instance])

        # Loop through all attributes
        for A in range(X.shape[1]):

            diffHit: float = 0.0
            # Loop through all neighbours
            for j in range(k):
                diffHit += __diff(A, XRandomInstance, XKNNHit[j], X)
            diffHit /= m * k

            diffMissVal: float = 0.0
            # Loop through the missed labels
            for n in range(len(missedLabels)):

                diffCurrentMiss: float = 0.0

                # Loop through the neighbours
                for j in range(k):
                    diffCurrentMiss += __diff(A, XRandomInstance, XKNNMiss[n][j], X)

                diffMissVal += _probability_class(y_flat, float(missedLabels[n])) * diffCurrentMiss / (m * k)

            # Calculate the weight
            W_A_1d[A] = W_A_1d[A] - diffHit + diffMissVal

            # Track the weights
            Wtrack[i, A] = W_A_1d[A]

    # Reshape to column vector for API compatibility
    W_A: NDFloat = W_A_1d.reshape(-1, 1)

    # Check if weight tracking is on
    if not weight_track:
        return W_A
    else:
        return W_A, Wtrack, iTrack


# --- Relief overloads ---
@overload
def Relief(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = ...,
    sigma: int = ...,
    weight_track: Literal[False] = ...,
    categoricalx: bool = ...,
) -> NDFloat: ...


@overload
def Relief(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = ...,
    sigma: int = ...,
    weight_track: Literal[True] = ...,
    categoricalx: bool = ...,
) -> tuple[NDFloat, NDFloat, NDFloat, NDFloat, NDFloat]: ...


def Relief(
    X: NDFloat,
    y: NDFloat,
    updates: Union[Literal["all"], int] = "all",
    sigma: int = 30,
    weight_track: bool = False,
    categoricalx: bool = False,
) -> Union[NDFloat, tuple[NDFloat, NDFloat, NDFloat, NDFloat, NDFloat]]:

    # Ensure y is 1D to avoid shape issues with numpy 2.x
    y_flat: NDFloat = np.asarray(y, dtype=np.float64).ravel()

    # Check if user wants all values to be considered
    if updates == "all":
        m: int = X.shape[0]
    else:
        m = int(updates)

    # The constants need for Relief
    W_A_1d: NDFloat = np.zeros(X.shape[1])
    Wtrack: NDFloat = np.zeros([m, X.shape[1]])
    hitTrack: NDFloat = np.zeros([m, X.shape[1]])
    missTrack: NDFloat = np.zeros([m, X.shape[1]])
    iTrack: NDFloat = np.zeros([m, 1])

    # Check if the input is categorical
    __diff: DiffFunc
    if categoricalx:
        __diff = _diffCategorical
    else:
        __diff = _diffNumeric

    # Repeat based on the total number of inputs or based on a user specified value
    for i in range(m):

        # Randomly access an instance
        if updates == "all":
            random_instance: int = i
        else:
            random_instance = int(np.random.randint(low=0, high=X.shape[0]))

        # Select a 'k' number in instances near the chosen random instance
        XKNNHit, neighbourIndexHit = _knnsearch(X, X[random_instance, :], 1, "hit", y_flat, float(y_flat[random_instance]))
        yKNNHit: NDFloat = y_flat[neighbourIndexHit]
        XKNNMiss, neighbourIndexMiss = _knnsearch(X, X[random_instance, :], 1, "miss", y_flat, float(y_flat[random_instance]))
        yKNNMiss: NDFloat = y_flat[neighbourIndexMiss]

        XRandomInstance: NDFloat = X[random_instance, :]
        yRandomInstance: float = float(y_flat[random_instance])

        iTrack[i] = random_instance

        # Loop through all attributes
        for A in range(X.shape[1]):
            # Calculate the weight
            W_A_1d[A] = W_A_1d[A] - __diff(A, XRandomInstance, XKNNHit[0], X) / m + __diff(
                A, XRandomInstance, XKNNMiss[0], X
            ) / m
            # Track the weights
            Wtrack[i, A] = W_A_1d[A]
            hitTrack[i, A] = __diff(A, XRandomInstance, XKNNHit[0], X) / m
            missTrack[i, A] = __diff(A, XRandomInstance, XKNNMiss[0], X) / m

    # Reshape to column vector for API compatibility
    W_A: NDFloat = W_A_1d.reshape(-1, 1)

    # Check if weight tracking is on
    if not weight_track:
        return W_A
    else:
        return W_A, Wtrack, iTrack, hitTrack, missTrack


# Backward compatibility aliases for the renamed private functions
__knnsearchR = _knnsearchR
__knnsearchF = _knnsearchF
__knnsearch = _knnsearch
__distance = _distance
__diffNumeric = _diffNumeric
__diffCaterogical = _diffCategorical
__probability_class = _probability_class
