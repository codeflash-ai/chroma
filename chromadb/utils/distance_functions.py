"""
These functions match what the spec of hnswlib is.
"""
from typing import Union, cast
import numpy as np
from numpy.typing import NDArray

Vector = NDArray[Union[np.int32, np.float32, np.int16, np.float16]]


def l2(x: Vector, y: Vector) -> float:
    return (np.linalg.norm(x - y) ** 2).item()


def cosine(x: Vector, y: Vector) -> float:
    # This epsilon is used to prevent division by zero, and the value is the same
    # https://github.com/nmslib/hnswlib/blob/359b2ba87358224963986f709e593d799064ace6/python_bindings/bindings.cpp#L238

    # We need to adapt the epsilon to the precision of the input
    NORM_EPS = 1e-30
    if x.dtype == np.float16 or y.dtype == np.float16:
        NORM_EPS = 1e-7

    # Avoid redundant norm calculations: compute each norm only once
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    denom = (x_norm * y_norm) + NORM_EPS

    # Use np.dot for both 1D and multidimensional (row vector) cases
    dot = np.dot(x, y)

    # Use fused multiply-add if available for numerical precision, but stay with simple version for performance
    # Directly perform subtraction and division without additional .item() dereference if already scalar
    # But in either case, guarantee returning a Python float
    result = 1.0 - dot / denom
    return cast(float, float(result))


def ip(x: Vector, y: Vector) -> float:
    return cast(float, (1.0 - np.dot(x, y)).item())
