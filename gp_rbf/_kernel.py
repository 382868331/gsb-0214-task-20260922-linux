"""RBF（平方指数）核：``k(x, x') = a^2 * exp(-||x-x'||^2 / (2 l^2))``。

参数化与 GP 标准不同：题目用 ``a``（标准差形式的输出尺度），故 ``k(0)=a^2``。
"""

from __future__ import annotations

import math
from typing import Sequence


def rbf_scalar(x: Sequence[float], xp: Sequence[float], a: float, l: float) -> float:
    inv2l2 = 0.5 / (l * l)
    s = 0.0
    for u, v in zip(x, xp):
        d = u - v
        s += d * d
    return a * a * math.exp(-s * inv2l2)


def rbf_matrix(
    X: Sequence[Sequence[float]],
    Xp: Sequence[Sequence[float]],
    a: float,
    l: float,
) -> list[list[float]]:
    """交叉核矩阵 ``K[i,j] = k(X[i], Xp[j])``（对称情形请用 :func:`rbf_symmetric`）。"""
    inv2l2 = 0.5 / (l * l)
    a2 = a * a
    out: list[list[float]] = []
    for xi in X:
        row: list[float] = []
        for xj in Xp:
            s = 0.0
            for u, v in zip(xi, xj):
                d = u - v
                s += d * d
            row.append(a2 * math.exp(-s * inv2l2))
        out.append(row)
    return out


def rbf_symmetric(
    X: Sequence[Sequence[float]], a: float, l: float
) -> list[list[float]]:
    """对称核矩阵 ``K(X, X)``，利用 ``K_ij = K_ji`` 且 ``K_ii = a^2``。

    逐元素独立计算，不做任何对角提升或截断，保证结果与
    :func:`rbf_matrix` 完全一致（逐位），便于对称性检查。
    """
    n = len(X)
    inv2l2 = 0.5 / (l * l)
    a2 = a * a
    K: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi = X[i]
        K[i][i] = a2
        for j in range(i):
            s = 0.0
            xj = X[j]
            for u, v in zip(xi, xj):
                d = u - v
                s += d * d
            v = a2 * math.exp(-s * inv2l2)
            K[i][j] = v
            K[j][i] = v
    return K
