"""精确 RBF 高斯过程回归：边际似然、解析超参梯度、两类预测方差。

数学约定（零均值先验）：

* 核 ``k(x,x') = a^2 exp(-||x-x'||^2 / (2 l^2))``
* 训练协方差 ``C = K + s^2 I``
* 若显式给定 ``jitter > 0``，则实际使用 ``C = K + (s^2 + jitter) I``；
  jitter 只来自参数，内部绝不自动添加，且计入对数似然、梯度与全部预测公式。
* 超参梯度对 ``log a, log l, log s`` 给出。

Cholesky 失败（主元非正）时不抛异常，返回 ``status == "not_pd"`` 的
:class:`FitResult`；不回退、不加 jitter、不返回任何似然数值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import _linalg as la
from ._kernel import rbf_symmetric, rbf_matrix
from ._validate import (
    GPInputError,
    MAX_SAMPLES,
    check_matrix,
    check_vector,
    ensure_nonneg,
    ensure_positive,
)

#: 预测方差允许的负微量下限（绝对量）。落在此区间内的负方差钳为 0.0，
#: 并在 Prediction.clamped 中记录；小于该值视为数值/建模错误并抛错。
NEG_VAR_TOL = 1e-10

STATUS_OK = "ok"
STATUS_NOT_PD = "not_pd"


@dataclass(frozen=True)
class Hyperparams:
    """核超参（原值，非 log）。``jitter`` 是显式给定的对角附加方差。"""

    a: float
    l: float
    s: float
    jitter: float = 0.0

    @property
    def log_tuple(self) -> tuple[float, float, float]:
        return math.log(self.a), math.log(self.l), math.log(self.s)


@dataclass(frozen=True)
class Residuals:
    """数值自检残差（均为无量纲/原始尺度绝对量）。

    * ``symmetry``:        max|K - K^T|（构造保证应为 0）
    * ``chol_reconstruct``: ||C - L L^T||_F（Cholesky 成功时）
    * ``chol_solve``:       ||C alpha - y||_inf（Cholesky 成功时）
    """

    symmetry: float
    chol_reconstruct: Optional[float]
    chol_solve: Optional[float]


@dataclass(frozen=True)
class FitResult:
    """:func:`fit` 的结果。

    ``status == "not_pd"`` 时 ``log_marginal``、``grad_log`` 为 ``None``，
    Cholesky 相关残差也为 ``None``。
    """

    status: str
    params: Hyperparams
    n: int
    d: int
    log_marginal: Optional[float]
    grad_log: Optional[tuple[float, float, float]]
    residuals: Residuals
    noise_variance: float
    _L: Optional[List[List[float]]] = field(default=None, repr=False)
    _alpha: Optional[List[float]] = field(default=None, repr=False)
    _Cinv: Optional[List[List[float]]] = field(default=None, repr=False)
    _X: Optional[List[List[float]]] = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def grad(self) -> Optional[tuple[float, float, float]]:
        """``grad_log`` 的别名：``(dL/d log a, dL/d log l, dL/d log s)``。"""
        return self.grad_log


@dataclass(frozen=True)
class ClampedVariance:
    index: int
    raw: float
    kind: str  # "latent" 或 "observed"


@dataclass(frozen=True)
class Prediction:
    mean: List[float]
    latent_var: List[float]
    observed_var: List[float]
    clamped: tuple[ClampedVariance, ...]


def _symmetry_norm(K: Sequence[Sequence[float]]) -> float:
    n = len(K)
    m = 0.0
    for i in range(n):
        for j in range(i):
            d = abs(K[i][j] - K[j][i])
            if d > m:
                m = d
    return m


def fit(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    a: float,
    l: float,
    s: float,
    jitter: float = 0.0,
) -> FitResult:
    """用给定超参拟合 GP（计算 Cholesky、边际似然与解析梯度）。

    参数
    ----
    X:
        训练输入，``n`` 行 ``d`` 列；``1 <= n <= 80``，``1 <= d <= 3``。
    y:
        训练目标，长度 ``n``。拒绝 NaN/Infinity。
    a, l, s:
        核超参，必须严格为正（``bool`` 被拒绝）。
    jitter:
        显式附加到协方差对角的方差（非负，默认 0）。**仅此一处**可以
        增加对角噪声；算法失败时不会自动添加。
    """
    a_f = ensure_positive(a, "a")
    l_f = ensure_positive(l, "l")
    s_f = ensure_positive(s, "s")
    j_f = ensure_nonneg(jitter, "jitter")
    Xc = check_matrix(X, "X")
    yc = check_vector(y, "y", len(Xc))
    n = len(Xc)

    K = rbf_symmetric(Xc, a_f, l_f)
    sym = _symmetry_norm(K)

    noise_var = s_f * s_f + j_f
    C = [row[:] for row in K]
    for i in range(n):
        C[i][i] += noise_var

    L = la.cholesky_pd(C)
    params = Hyperparams(a_f, l_f, s_f, j_f)

    if L is None:
        return FitResult(
            status=STATUS_NOT_PD,
            params=params,
            n=n,
            d=len(Xc[0]),
            log_marginal=None,
            grad_log=None,
            residuals=Residuals(sym, None, None),
            noise_variance=noise_var,
        )

    # alpha = C^{-1} y
    alpha = la.solve_chol(L, yc)

    # 对数边际似然
    log_det = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    quad = sum(yc[i] * alpha[i] for i in range(n))
    lml = -0.5 * (quad + log_det + n * math.log(2.0 * math.pi))

    # C^{-1}（仅用于梯度；似然与预测走三角求解，不显式求逆）
    Cinv = la.chol_inverse(L)

    # ---- 解析梯度（对 log 超参）----
    # G = alpha alpha^T - C^{-1}；dL/dθ = 1/2 tr(G ∂C/∂θ)
    # dC/d log a = 2K                    -> dL/d log a = tr(GK)
    # dC/d log l = K ⊙ D / l^2（D 为平方距离）
    # dC/d log s = 2 s^2 I               -> dL/d log s = s^2 tr(G)
    # jitter 为显式常数，不产生超参导数。
    trace_G = 0.0
    tr_GK = 0.0
    tr_GKD2 = 0.0
    inv_l2 = 1.0 / (l_f * l_f)
    for i in range(n):
        for j in range(n):
            gij = alpha[i] * alpha[j] - Cinv[i][j]
            if i == j:
                # tr(G) = alpha'alpha - tr(Cinv)，只累加对角项
                trace_G += gij
            kij = K[i][j]
            tr_GK += gij * kij
            if i != j:
                xi, xj = Xc[i], Xc[j]
                d2 = 0.0
                for u, v in zip(xi, xj):
                    dd = u - v
                    d2 += dd * dd
                tr_GKD2 += gij * kij * d2
            # 对角元 d2=0，无需贡献
    grad_a = tr_GK  # 1/2 * tr(G 2K)
    grad_l = 0.5 * tr_GKD2 * inv_l2
    grad_s = (s_f * s_f) * trace_G

    # ---- 数值残差 ----
    # ||C - L L^T||_F
    LLt = la.matmul(L, la.transpose(L))
    chol_res = math.sqrt(sum((C[i][j] - LLt[i][j]) ** 2 for i in range(n) for j in range(n)))
    # ||C alpha - y||_inf
    Calpha = la.mat_vec(C, alpha)
    solve_res = max(abs(Calpha[i] - yc[i]) for i in range(n))

    return FitResult(
        status=STATUS_OK,
        params=params,
        n=n,
        d=len(Xc[0]),
        log_marginal=lml,
        grad_log=(grad_a, grad_l, grad_s),
        residuals=Residuals(sym, chol_res, solve_res),
        noise_variance=noise_var,
        _L=L,
        _alpha=alpha,
        _Cinv=Cinv,
        _X=Xc,
    )


def _validate_query(Xstar: Sequence[Sequence[float]], d: int) -> List[List[float]]:
    if not isinstance(Xstar, (list, tuple)) or isinstance(Xstar, (str, bytes)):
        raise GPInputError(f"Xstar: 必须是行的序列，收到 {type(Xstar).__name__}")
    if not (1 <= len(Xstar) <= MAX_SAMPLES):
        raise GPInputError(f"Xstar: 查询点数必须在 [1, {MAX_SAMPLES}]，收到 {len(Xstar)}")
    rows: List[List[float]] = []
    for i, row in enumerate(Xstar):
        if not isinstance(row, (list, tuple)) or isinstance(row, (str, bytes)):
            raise GPInputError(f"Xstar[{i}]: 必须是行序列，收到 {type(row).__name__}")
        if len(row) != d:
            raise GPInputError(
                f"Xstar[{i}]: 特征维度必须为 {d}，收到 {len(row)}"
            )
        out = []
        for j, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise GPInputError(
                    f"Xstar[{i}][{j}]: 必须是实数，收到 {type(v).__name__}"
                )
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                raise GPInputError(f"Xstar[{i}][{j}]: 拒绝 NaN/Infinity")
            out.append(fv)
        rows.append(out)
    return rows


def predict(result: FitResult, Xstar: Sequence[Sequence[float]]) -> Prediction:
    """后验预测。要求 ``result.status == "ok"``，否则抛 :class:`RuntimeError`。

    返回每个查询点的后验均值、

    * latent 方差 ``k(x*,x*) - k_*^T C^{-1} k_*``（隐函数值）；
    * observed 方差 ``latent + s^2``（含新观测噪声；不含常数 jitter）。
    """
    if not isinstance(result, FitResult):
        raise GPInputError("result: 必须是 fit() 返回的 FitResult")
    if result.status != STATUS_OK:
        raise RuntimeError(
            f"模型状态为 {result.status!r}：协方差矩阵非正定，无法预测；"
            "请检查噪声 s 或显式给定 jitter 后重新 fit"
        )
    L = result._L
    alpha = result._alpha
    X = result._X
    assert L is not None and alpha is not None and X is not None

    Xs = _validate_query(Xstar, result.d)
    a, l, s = result.params.a, result.params.l, result.params.s
    s2 = s * s

    Ks = rbf_matrix(Xs, X, a, l)
    means = la.mat_vec(Ks, alpha)

    latent_var: List[float] = []
    observed_var: List[float] = []
    clamped: List[ClampedVariance] = []
    a2 = a * a
    for q, ks in enumerate(Ks):
        v = la.solve_lower(L, ks)
        proj = sum(t * t for t in v)
        raw_latent = a2 - proj
        raw_obs = raw_latent + s2
        lv = raw_latent
        ov = raw_obs
        if lv < 0.0:
            if lv >= -NEG_VAR_TOL:
                clamped.append(ClampedVariance(q, lv, "latent"))
                lv = 0.0
                ov = lv + s2
            else:
                raise RuntimeError(
                    f"latent 预测方差为负且超过容差：Xstar[{q}] 处 {raw_latent:.3e} "
                    f"< -{NEG_VAR_TOL:.0e}；输入可能重复或数值不稳定"
                )
        if ov < 0.0:
            if ov >= -NEG_VAR_TOL:
                clamped.append(ClampedVariance(q, ov, "observed"))
                ov = 0.0
            else:
                raise RuntimeError(
                    f"observed 预测方差为负且超过容差：Xstar[{q}] 处 {raw_obs:.3e}"
                )
        latent_var.append(lv)
        observed_var.append(ov)

    return Prediction(means, latent_var, observed_var, tuple(clamped))


def predict_one(
    result: FitResult, x: Sequence[float]
) -> tuple[float, float, float]:
    """单点预测便捷接口，返回 ``(mean, latent_var, observed_var)``。"""
    p = predict(result, [x])
    return p.mean[0], p.latent_var[0], p.observed_var[0]
