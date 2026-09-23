"""独立参考实现：小矩阵直接求逆 + 显式标量公式。

本模块**不导入 gp_rbf**，用于在测试中独立给出期望值：

* :func:`inverse_logdet` —— 带部分选主元的 Gauss-Jordan 直接求逆与对数行列式；
* :func:`reference_lml`    —— 用直接逆计算 log 边际似然；
* :func:`reference_predict` —— 用直接逆计算后验均值与 latent/observed 方差；
* :func:`reference_kernel` —— 与题目公式逐字对应的 RBF 核。

只用于 n 很小的测试，不追求性能。
"""

from __future__ import annotations

import math


def reference_kernel(x, xp, a, l):
    d2 = sum((u - v) ** 2 for u, v in zip(x, xp))
    return a * a * math.exp(-d2 / (2.0 * l * l))


def kernel_matrix(X, a, l):
    n = len(X)
    return [[reference_kernel(X[i], X[j], a, l) for j in range(n)] for i in range(n)]


def cross_kernel(Xs, X, a, l):
    return [[reference_kernel(xs, xj, a, l) for xj in X] for xs in Xs]


def inverse_logdet(M):
    """Gauss-Jordan 消元求逆（部分选主元），同时返回 ``log|det(M)|``。"""
    n = len(M)
    A = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    logdet = 0.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) == 0.0:
            raise ZeroDivisionError("奇异矩阵")
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
        pivot = A[col][col]
        logdet += math.log(abs(pivot))
        invp = 1.0 / pivot
        prow = A[col]
        for j in range(2 * n):
            prow[j] *= invp
        for r in range(n):
            if r == col:
                continue
            factor = A[r][col]
            if factor == 0.0:
                continue
            rrow = A[r]
            for j in range(2 * n):
                rrow[j] -= factor * prow[j]
    Inv = [row[n:] for row in A]
    return Inv, logdet


def _mv(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def reference_lml(X, y, a, l, s, jitter=0.0):
    n = len(X)
    K = kernel_matrix(X, a, l)
    C = [row[:] for row in K]
    noise = s * s + jitter
    for i in range(n):
        C[i][i] += noise
    Cinv, logdet = inverse_logdet(C)
    alpha = _mv(Cinv, y)
    quad = sum(y[i] * alpha[i] for i in range(n))
    return -0.5 * (quad + logdet + n * math.log(2.0 * math.pi))


def reference_predict(X, y, Xs, a, l, s, jitter=0.0):
    n = len(X)
    K = kernel_matrix(X, a, l)
    C = [row[:] for row in K]
    noise = s * s + jitter
    for i in range(n):
        C[i][i] += noise
    Cinv, _ = inverse_logdet(C)
    Ks = cross_kernel(Xs, X, a, l)
    means, latent, observed = [], [], []
    s2 = s * s
    for q, ks in enumerate(Ks):
        Cinv_ks = _mv(Cinv, ks)
        mean = sum(ks[i] * _mv(Cinv, y)[i] for i in range(n))
        proj = sum(ks[i] * Cinv_ks[i] for i in range(n))
        lv = a * a - proj
        means.append(mean)
        latent.append(lv)
        observed.append(lv + s2)
    return means, latent, observed
