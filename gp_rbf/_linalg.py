"""最小稠密线性代数工具（纯标准库）。

仅服务于 :mod:`gp_rbf`，不对外提供通用线性代数库接口。
矩阵统一表示为 ``list[list[float]]``（行主序），向量为 ``list[float]``。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

Matrix = List[List[float]]
Vector = List[float]


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(A: Sequence[Sequence[float]]) -> Matrix:
    n = len(A)
    m = len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def matmul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> Matrix:
    n, k, m = len(A), len(A[0]), len(B[0])
    C: Matrix = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for t in range(k):
            a = Ai[t]
            if a == 0.0:
                continue
            Bt = B[t]
            for j in range(m):
                Ci[j] += a * Bt[j]
    return C


def mat_vec(A: Sequence[Sequence[float]], x: Sequence[float]) -> Vector:
    n, m = len(A), len(A[0])
    out: Vector = [0.0] * n
    for i in range(n):
        Ai = A[i]
        s = 0.0
        for j in range(m):
            s += Ai[j] * x[j]
        out[i] = s
    return out


def cholesky_pd(A: Sequence[Sequence[float]]) -> Optional[Matrix]:
    """对称正定矩阵的 Cholesky 分解 ``A = L L^T``。

    成功返回下三角 ``L``；只要任一主元非正（含 0、负数、NaN）即返回
    ``None`` 表示非正定。本函数绝不向对角线添加任何 jitter。
    """
    n = len(A)
    L: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j]
            if j:
                Li = L[i]
                Lj = L[j]
                for k in range(j):
                    s -= Li[k] * Lj[k]
            if i == j:
                if not (s > 0.0):
                    return None
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def solve_lower(L: Sequence[Sequence[float]], b: Sequence[float]) -> Vector:
    """前向代入解 ``L x = b``，``L`` 为对角元非零的下三角矩阵。"""
    n = len(L)
    x: Vector = [0.0] * n
    for i in range(n):
        s = b[i]
        Li = L[i]
        for j in range(i):
            s -= Li[j] * x[j]
        x[i] = s / Li[i]
    return x


def solve_lower_mat(L: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> Matrix:
    """多右端前向代入 ``L X = B``（B 按列存；这里直接按行索引等价处理）。"""
    return [solve_lower(L, col) for col in B]


def solve_upper(U: Sequence[Sequence[float]], b: Sequence[float]) -> Vector:
    """后向代入解 ``U x = b``，``U`` 为对角元非零的上三角矩阵。"""
    n = len(U)
    x: Vector = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        Ui = U[i]
        for j in range(i + 1, n):
            s -= Ui[j] * x[j]
        x[i] = s / Ui[i]
    return x


def solve_chol(L: Matrix, b: Sequence[float]) -> Vector:
    """解 ``L L^T x = b``。"""
    return solve_upper(transpose(L), solve_lower(L, b))


def chol_inverse(L: Sequence[Sequence[float]]) -> Matrix:
    """基于 Cholesky 因子求逆：``C^{-1} = L^{-T} L^{-1}``。

    解 ``L V_p = e_p`` 得到 ``L^{-1}`` 的第 ``p`` 列，记 ``V[p][i]``
    为该列第 ``i`` 个元素，即 ``V[p][i] = (L^{-1})[i][p]``。于是
    ``C^{-1}[i][j] = sum_p (L^{-1})[p][i] (L^{-1})[p][j]
                  = sum_p V[i][p] V[j][p]``。
    """
    cols = solve_lower_mat(L, identity(len(L)))
    n = len(L)
    Inv: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Vi = cols[i]
        for j in range(i + 1):
            Vj = cols[j]
            s = 0.0
            for p in range(n):
                s += Vi[p] * Vj[p]
            Inv[i][j] = s
            Inv[j][i] = s
    return Inv
