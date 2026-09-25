"""
gprbf - 精确 RBF 高斯过程回归（标准库实现，无第三方依赖）。

核函数: k(x, x') = alpha^2 * exp(-||x-x'||^2 / (2 * lengthscale^2))
训练协方差: K_y = K(X, X) + (noise^2 + jitter) * I

接口契约见 README.md。要点:
  * 训练样本数 1 <= n <= 80, 输入维数 1 <= d <= 3。
  * alpha / lengthscale / noise 必须为有限正数; jitter 默认为 0,
    只能通过关键字参数显式给定 (有限, >= 0), 并作为方差项进入全部公式。
  * Cholesky 失败时返回 status="not_pd", 绝不自动加未记录的 jitter。
  * 数值输入拒绝 NaN/Infinity; bool 不作为数值接受。
  * 本模块无状态、不修改调用方传入的数据。
  * GPFit 为不可变拟合对象: delete(sample_id) 按稳定样本 ID 删除一条,
    用 Cholesky 行删除 (Givens 旋转) 做 O(n^2) 局部更新并返回新对象,
    旧对象保持可查询; 数值残差超限时回退全量分解并记入 diagnostics。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Hashable

__all__ = [
    "MAX_SAMPLES",
    "MAX_DIM",
    "MAX_TEST_POINTS",
    "NEG_VAR_REL_TOL",
    "DOWNDATE_RESIDUAL_RTOL",
    "Hyperparameters",
    "MarginalResult",
    "Prediction",
    "DeletionRecord",
    "GPFit",
    "log_marginal",
    "predict",
    "rbf_kernel",
    "classify_raw_variance",
]

#: 训练样本数硬上限（题面接口边界）。
MAX_SAMPLES = 80
#: 输入维数硬上限（题面接口边界）。
MAX_DIM = 3
#: 预测点数量的保护性上限（README 中记录的防护，非题面要求）。
MAX_TEST_POINTS = 10_000
#: 预测方差负微量判定容差: raw >= -NEG_VAR_REL_TOL * max(1, alpha^2) 视为数值抹零。
NEG_VAR_REL_TOL = 1e-10
#: GPFit 删除样本时 Cholesky 行删除更新的验收容差 (相对 max(1, max diag Ky))。
#: 超过则回退全量 Cholesky 并在 DeletionRecord 中记录 method="full_refit"。
DOWNDATE_RESIDUAL_RTOL = 1e-8

_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hyperparameters:
    """GP 超参数 (alpha, lengthscale, noise), 三者均须为有限正数。"""

    alpha: float
    lengthscale: float
    noise: float


@dataclass(frozen=True)
class MarginalResult:
    """
    log_marginal 的结果。

    status:
        "ok"      - Cholesky 成功, 其余字段已填充;
        "not_pd"  - 训练协方差非正定 (Cholesky 遇到 <=0 的根号项),
                    log_marginal / gradient / alpha / K_inv 为 None。
    log_marginal: 自然对数边际似然
        L = -1/2 y^T K_y^-1 y - 1/2 log|K_y| - n/2 log(2*pi)
    gradient: (dL/d log alpha, dL/d log lengthscale, dL/d log noise) 解析梯度。
    alpha_vec: K_y^-1 y (诊断/复用用)。
    K_inv: K_y 的显式逆 (诊断/独立核对用, 嵌套元组)。
    symmetry_residual: max|K_y - K_y^T| (绝对量纲)。
    chol_residual: max|K_y - L L^T| (绝对量纲), Cholesky 未执行为 None。
    solve_residual: max|K_y alpha_vec - y| (绝对量纲)。
    effective_noise_variance: 实际进入公式的 noise^2 + jitter。
    """

    status: str
    log_marginal: Optional[float]
    gradient: Optional[Tuple[float, float, float]]
    alpha_vec: Optional[Tuple[float, ...]]
    K_inv: Optional[Tuple[Tuple[float, ...], ...]]
    symmetry_residual: float
    chol_residual: Optional[float]
    solve_residual: Optional[float]
    effective_noise_variance: float
    jitter: float


@dataclass(frozen=True)
class Prediction:
    """
    predict 的结果。

    status:
        "ok"                - 全部预测点正常 (允许个别点发生负微量抹零, 见 clamped);
        "not_pd"            - 训练协方差非正定, 预测字段为 None;
        "negative_variance" - 至少一个点的原始隐方差比 -NEG_VAR_REL_TOL*max(1,a^2)
                              还小, 属于不可用数值误差解释的矛盾, 返回原始 (负) 值
                              供定位, 不做静默纠正。
    mean:               后验均值 E[f*|y] (观测后验, 噪声为零均值)。
    latent_variance:    隐函数后验方差 V[f*|y]。
    observed_variance: 观测后验方差 V[y*|y] = latent_variance + noise^2 + jitter。
    raw_latent_variance: 未经抹零处理的 a^2 - k_*^T K_y^-1 k_*。
    clamped: 被抹零的预测点下标 (raw 为幅度不超过容差的负微量, latent 置 0)。
    symmetry_residual / solve_residual: 数值诊断量。
    """

    status: str
    mean: Optional[Tuple[float, ...]]
    latent_variance: Optional[Tuple[float, ...]]
    observed_variance: Optional[Tuple[float, ...]]
    raw_latent_variance: Optional[Tuple[float, ...]]
    clamped: Tuple[int, ...]
    symmetry_residual: float
    solve_residual: Optional[float]
    effective_noise_variance: float
    jitter: float


# ---------------------------------------------------------------------------
# 输入校验（错误语义: TypeError 类型错误, ValueError 值非法; 信息带下标可定位）
# ---------------------------------------------------------------------------


def _finite_float(value, where: str) -> float:
    """把单个标量转为有限 float; bool 显式拒绝; NaN/Infinity 拒绝。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{where}: expected finite real int/float (bool rejected), "
            f"got {type(value).__name__}"
        )
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{where}: must be finite (NaN/Infinity rejected), got {value!r}")
    return f


def _validate_matrix(
    X,
    name: str,
    *,
    max_rows: int,
    expect_dim: Optional[int] = None,
) -> Tuple[Tuple[Tuple[float, ...], ...], int, int]:
    if not isinstance(X, (list, tuple)):
        raise TypeError(f"{name}: expected list/tuple of rows, got {type(X).__name__}")
    if len(X) == 0:
        raise ValueError(f"{name}: must contain at least one row")
    if len(X) > max_rows:
        raise ValueError(
            f"{name}: {len(X)} rows exceeds limit {max_rows}"
        )

    rows = []
    dim = None
    for i, row in enumerate(X):
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"{name}[{i}]: expected list/tuple row, got {type(row).__name__}")
        if dim is None:
            dim = len(row)
            if not 1 <= dim <= MAX_DIM:
                raise ValueError(
                    f"{name}[0]: dimension {dim} outside supported range 1..{MAX_DIM}"
                )
            if expect_dim is not None and dim != expect_dim:
                raise ValueError(
                    f"{name}[0]: dimension {dim} does not match training dimension {expect_dim}"
                )
        elif len(row) != dim:
            raise ValueError(
                f"{name}[{i}]: row length {len(row)} != {dim} (expected consistent dimension)"
            )
        rows.append(tuple(_finite_float(v, f"{name}[{i}][{j}]") for j, v in enumerate(row)))
    return tuple(rows), len(rows), dim


def _validate_vector(y, name: str, n: Optional[int] = None) -> Tuple[float, ...]:
    if not isinstance(y, (list, tuple)):
        raise TypeError(f"{name}: expected list/tuple, got {type(y).__name__}")
    if len(y) == 0:
        raise ValueError(f"{name}: must contain at least one element")
    if n is not None and len(y) != n:
        raise ValueError(f"{name}: length {len(y)} != number of samples {n}")
    return tuple(_finite_float(v, f"{name}[{i}]") for i, v in enumerate(y))


def _validate_hp(hp) -> Hyperparameters:
    if not isinstance(hp, Hyperparameters):
        raise TypeError(
            f"hp: expected Hyperparameters, got {type(hp).__name__}"
        )
    a = _finite_float(hp.alpha, "hp.alpha")
    ell = _finite_float(hp.lengthscale, "hp.lengthscale")
    s = _finite_float(hp.noise, "hp.noise")
    if a <= 0.0:
        raise ValueError(f"hp.alpha: must be > 0, got {a!r}")
    if ell <= 0.0:
        raise ValueError(f"hp.lengthscale: must be > 0, got {ell!r}")
    if s <= 0.0:
        raise ValueError(f"hp.noise: must be > 0, got {s!r}")
    return Hyperparameters(a, ell, s)


def _validate_jitter(jitter) -> float:
    j = _finite_float(jitter, "jitter")
    if j < 0.0:
        raise ValueError(f"jitter: must be >= 0, got {j!r}")
    return j


# ---------------------------------------------------------------------------
# 核函数与线性代数（全部为纯 Python, 仅依赖 math）
# ---------------------------------------------------------------------------


def rbf_kernel(
    X: Sequence[Sequence[float]], hp: Hyperparameters
) -> Tuple[Tuple[float, ...], ...]:
    """构造 RBF 核矩阵 (走完整输入校验, 便于独立使用)。"""
    Xv, _, _ = _validate_matrix(X, "X", max_rows=MAX_SAMPLES)
    hpv = _validate_hp(hp)
    K, _ = _kernel_and_d2(Xv, hpv)
    return tuple(tuple(row) for row in K)


def _kernel_and_d2(X, hp: Hyperparameters):
    """返回 K (list[list]) 与平方距离矩阵 D2 (list[list])。"""
    n = len(X)
    a2 = hp.alpha * hp.alpha
    inv2l2 = 1.0 / (2.0 * hp.lengthscale * hp.lengthscale)
    K = [[0.0] * n for _ in range(n)]
    D2 = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi = X[i]
        for j in range(i, n):
            xj = X[j]
            d2 = 0.0
            for u, v in zip(xi, xj):
                d = u - v
                d2 += d * d
            kij = a2 * math.exp(-d2 * inv2l2)
            D2[i][j] = d2
            D2[j][i] = d2
            K[i][j] = kij
            K[j][i] = kij  # 同一表达式镜像赋值, 保证逐位对称
    return K, D2


def _cross_kernel(X_train, X_test, hp: Hyperparameters):
    """K[i][t] = k(X_train[i], X_test[t])。"""
    n = len(X_train)
    m = len(X_test)
    a2 = hp.alpha * hp.alpha
    inv2l2 = 1.0 / (2.0 * hp.lengthscale * hp.lengthscale)
    Ks = [[0.0] * m for _ in range(n)]
    for i in range(n):
        xi = X_train[i]
        for t in range(m):
            xt = X_test[t]
            d2 = 0.0
            for u, v in zip(xi, xt):
                d = u - v
                d2 += d * d
            Ks[i][t] = a2 * math.exp(-d2 * inv2l2)
    return Ks


def _cholesky(A):
    """正定 Cholesky; 遇到根号项 <= 0 返回 None (不做任何隐式修补)。"""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            Li = L[i]
            Lj = L[j]
            for k in range(j):
                s += Li[k] * Lj[k]
            val = A[i][j] - s
            if i == j:
                if not (val > 0.0) or math.isnan(val):  # NaN 也不可接受
                    return None
                Li[j] = math.sqrt(val)
            else:
                Li[j] = val / Lj[j]
    return L


def _cho_solve(L, b):
    """解 L L^T x = b。"""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        Li = L[i]
        for j in range(i):
            s -= Li[j] * y[j]
        y[i] = s / Li[i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= L[j][i] * x[j]
        x[i] = s / L[i][i]
    return x


def _matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _max_abs(v):
    m = 0.0
    for z in v:
        z = abs(z)
        if z > m:
            m = z
    return m


def _symmetry_residual(A):
    n = len(A)
    m = 0.0
    for i in range(n):
        Ai = A[i]
        for j in range(i + 1, n):
            r = abs(Ai[j] - A[j][i])
            if r > m:
                m = r
    return m


def _chol_residual(A, L):
    n = len(A)
    m = 0.0
    for i in range(n):
        Li = L[i]
        for j in range(i + 1):
            s = 0.0
            for k in range(j + 1):
                s += Li[k] * L[j][k]
            r = abs(s - A[i][j])
            if r > m:
                m = r
    return m


def _inverse_via_cholesky(L):
    """K_y^-1: 对每个单位向量做 Cholesky 求解 (列与列之间相互独立)。"""
    n = len(L)
    cols = []
    e = [0.0] * n
    for j in range(n):
        if j:
            e[j - 1] = 0.0
        e[j] = 1.0
        cols.append(_cho_solve(L, e))
    Inv = [[cols[j][i] for j in range(n)] for i in range(n)]
    return Inv


def classify_raw_variance(raw: float, alpha2: float) -> Tuple[float, str]:
    """
    单个点原始隐方差 raw = alpha^2 - k_*^T K_y^-1 k_* 的分类规则
    (predict 内部使用, 单独暴露以便审计与直接测试):

      raw >= 0                              -> (raw, "ok")
      -tol <= raw < 0 (舍入负微量)          -> (0.0, "clamped"), 抹零
      raw < -tol (超过容差的矛盾负值)       -> (raw, "negative"), 原样保留
          tol = NEG_VAR_REL_TOL * max(1, alpha2)
    """
    tol = NEG_VAR_REL_TOL * max(1.0, alpha2)
    if raw < 0.0:
        if raw >= -tol:
            return 0.0, "clamped"
        return raw, "negative"
    return raw, "ok"


# ---------------------------------------------------------------------------
# 核心接口
# ---------------------------------------------------------------------------


def _prepare_training(X, y, hp, jitter):
    Xv, n, _d = _validate_matrix(X, "X", max_rows=MAX_SAMPLES)
    yv = _validate_vector(y, "y", n)
    hpv = _validate_hp(hp)
    jv = _validate_jitter(jitter)
    return Xv, yv, hpv, jv, hpv.noise * hpv.noise + jv


def _lml_not_pd(sym, var, jv) -> MarginalResult:
    return MarginalResult(
        status="not_pd",
        log_marginal=None,
        gradient=None,
        alpha_vec=None,
        K_inv=None,
        symmetry_residual=sym,
        chol_residual=None,
        solve_residual=None,
        effective_noise_variance=var,
        jitter=jv,
    )


def _lml_from_factor(Xv, yv, hpv, jv, var, Ky, L, sym) -> MarginalResult:
    """由已完成的 Cholesky 分解 L (Ky = L L^T) 计算 LML 与解析梯度。"""
    n = len(Xv)
    K, D2 = _kernel_and_d2(Xv, hpv)
    avec = _cho_solve(L, yv)

    chol_res = _chol_residual(Ky, L)
    solve_res = _max_abs([q - yv[i] for i, q in enumerate(_matvec(Ky, avec))])

    logdet = 0.0
    for i in range(n):
        logdet += 2.0 * math.log(L[i][i])
    quad = sum(yv[i] * avec[i] for i in range(n))
    lml = -0.5 * (quad + logdet + n * _LOG_2PI)

    Inv = _inverse_via_cholesky(L)
    tr_inv = sum(Inv[i][i] for i in range(n))

    K_alpha = _matvec(K, avec)
    quad_K = sum(avec[i] * K_alpha[i] for i in range(n))
    g_log_a = quad_K - (n - var * tr_inv)

    ell2 = hpv.lengthscale * hpv.lengthscale
    Kl = [[K[i][j] * D2[i][j] / ell2 for j in range(n)] for i in range(n)]
    Kl_alpha = _matvec(Kl, avec)
    quad_Kl = sum(avec[i] * Kl_alpha[i] for i in range(n))
    tr_inv_Kl = 0.0
    for i in range(n):
        for j in range(n):
            tr_inv_Kl += Inv[i][j] * Kl[j][i]
    g_log_l = 0.5 * (quad_Kl - tr_inv_Kl)

    s2 = hpv.noise * hpv.noise
    g_log_s = s2 * (sum(q * q for q in avec) - tr_inv)

    return MarginalResult(
        status="ok",
        log_marginal=lml,
        gradient=(g_log_a, g_log_l, g_log_s),
        alpha_vec=tuple(avec),
        K_inv=tuple(tuple(row) for row in Inv),
        symmetry_residual=sym,
        chol_residual=chol_res,
        solve_residual=solve_res,
        effective_noise_variance=var,
        jitter=jv,
    )


def log_marginal(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    hp: Hyperparameters,
    *,
    jitter: float = 0.0,
) -> MarginalResult:
    """
    计算 log p(y|X, hp) 及对 (log alpha, log lengthscale, log noise) 的解析梯度。

    梯度公式: dL/dθ = 1/2 alpha_vec^T (dK_y/dθ) alpha_vec
                    - 1/2 tr(K_y^-1 dK_y/dθ)
      dK_y/d log alpha   = 2 K
      dK_y/d log ell     = K ⊙ (D2 / ell^2)   (对角为 0)
      dK_y/d log noise   = 2 * noise^2 * I     (显式 jitter 为常数, 不产生梯度)
    """
    Xv, yv, hpv, jv, var = _prepare_training(X, y, hp, jitter)
    n = len(Xv)
    K, _D2 = _kernel_and_d2(Xv, hpv)
    Ky = [row[:] for row in K]
    for i in range(n):
        Ky[i][i] += var

    sym = _symmetry_residual(Ky)
    L = _cholesky(Ky)
    if L is None:
        return _lml_not_pd(sym, var, jv)
    return _lml_from_factor(Xv, yv, hpv, jv, var, Ky, L, sym)


def _predict_not_pd(sym, var, jv) -> Prediction:
    return Prediction(
        status="not_pd",
        mean=None,
        latent_variance=None,
        observed_variance=None,
        raw_latent_variance=None,
        clamped=(),
        symmetry_residual=sym,
        solve_residual=None,
        effective_noise_variance=var,
        jitter=jv,
    )


def _predict_from_factor(Xv, yv, hpv, jv, var, Ky, L, sym, Xt) -> Prediction:
    """由已完成的 Cholesky 分解 L (Ky = L L^T) 计算后验预测。"""
    n = len(Xv)
    m = len(Xt)
    avec = _cho_solve(L, yv)
    Ks = _cross_kernel(Xv, Xt, hpv)
    a2 = hpv.alpha * hpv.alpha

    means = [0.0] * m
    raw_var = [0.0] * m
    latent = [0.0] * m
    observed = [0.0] * m
    clamped = []
    hard_negative = False
    solve_res = 0.0

    for t in range(m):
        kvec = [Ks[i][t] for i in range(n)]
        w = _cho_solve(L, kvec)
        means[t] = sum(kvec[i] * avec[i] for i in range(n))
        raw = a2 - sum(kvec[i] * w[i] for i in range(n))
        raw_var[t] = raw

        Kv_minus_k = [q - kvec[i] for i, q in enumerate(_matvec(Ky, w))]
        r = _max_abs(Kv_minus_k)
        if r > solve_res:
            solve_res = r

        v, kind = classify_raw_variance(raw, a2)
        latent[t] = v
        observed[t] = v + var
        if kind == "clamped":
            clamped.append(t)
        elif kind == "negative":
            hard_negative = True

    status = "negative_variance" if hard_negative else "ok"
    return Prediction(
        status=status,
        mean=tuple(means),
        latent_variance=tuple(latent),
        observed_variance=tuple(observed),
        raw_latent_variance=tuple(raw_var),
        clamped=tuple(clamped),
        symmetry_residual=sym,
        solve_residual=solve_res,
        effective_noise_variance=var,
        jitter=jv,
    )


def predict(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    X_test: Sequence[Sequence[float]],
    hp: Hyperparameters,
    *,
    jitter: float = 0.0,
) -> Prediction:
    """
    RBF GP 后验预测。

    对每个测试点 x_* (k_* = k(X, x_*)):
        mean_t            = k_*^T K_y^-1 y
        raw latent var    = alpha^2 - k_*^T K_y^-1 k_*
        observed var      = latent_var + noise^2 + jitter

    负微量处理: raw 为负且幅度 <= NEG_VAR_REL_TOL*max(1, alpha^2) 时,
    latent 抹为 0 并在 clamped 中记录下标; 更负则 status="negative_variance"
    且原样返回负值, 不静默纠正。
    """
    Xv, yv, hpv, jv, var = _prepare_training(X, y, hp, jitter)
    Xt, _m, _d = _validate_matrix(
        X_test, "X_test", max_rows=MAX_TEST_POINTS, expect_dim=len(Xv[0])
    )
    n = len(Xv)

    K, _ = _kernel_and_d2(Xv, hpv)
    Ky = [row[:] for row in K]
    for i in range(n):
        Ky[i][i] += var
    sym = _symmetry_residual(Ky)

    L = _cholesky(Ky)
    if L is None:
        return _predict_not_pd(sym, var, jv)
    return _predict_from_factor(Xv, yv, hpv, jv, var, Ky, L, sym, Xt)


# ---------------------------------------------------------------------------
# GPFit: 不可变拟合对象与按稳定 ID 的样本删除 (Cholesky 行删除更新)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeletionRecord:
    """
    一次 GPFit.delete 的诊断记录。

    removed_id:         被删除样本的稳定 ID。
    method:
        "cholesky_downdate" - Cholesky 行删除 (Givens 旋转) O(n^2) 局部更新,
                              残差验收通过;
        "full_refit"        - 行删除结果未通过残差验收 (或对角非正),
                              回退到对压缩后的 Ky 做全量 Cholesky;
        "trivial_empty"     - 删除后无剩余样本, 无需分解。
    downdate_residual:  行删除路径的 max|Ky' - L'L'^T| (未走该路径为 None)。
    chol_residual:      最终采用分解的 max|Ky' - L L^T| (空拟合为 0.0)。
    """

    removed_id: Hashable
    method: str
    downdate_residual: Optional[float]
    chol_residual: float


def _cholesky_delete_row(L, i):
    """
    Cholesky 行删除更新: 已知 Ky = L L^T (n x n), 删除 Ky 的第 i 行/列后
    的新因子 L' ((n-1) x (n-1), Ky' = L' L'^T), 不重新分解。

    原理: 删去 L 的第 i 行得 M ((n-1) x n), 仍有 Ky' = M M^T; 对列对
    (c, c+1), c = i..n-2 依次施加 Givens 旋转 (右乘正交阵不改变 M M^T),
    消去尾随块上次对角线, 得到下三角因子并丢弃恒为零的最后一列。
    总复杂度 O(n^2)。
    """
    n = len(L)
    M = [list(L[r]) for r in range(n) if r != i]
    for c in range(i, n - 1):
        a = M[c][c]
        b = M[c][c + 1]
        if b == 0.0:
            continue
        r = math.hypot(a, b)
        if r == 0.0:
            return None
        cos = a / r
        sin = b / r
        for row in range(c, n - 1):
            u = M[row][c]
            v = M[row][c + 1]
            M[row][c] = cos * u + sin * v
            M[row][c + 1] = cos * v - sin * u
    Lnew = [row[: n - 1] for row in M]
    for d in range(n - 1):
        if not (Lnew[d][d] > 0.0) or math.isnan(Lnew[d][d]):
            return None
    return Lnew


def _delete_row_col(A, i):
    return [
        [A[r][c] for c in range(len(A)) if c != i]
        for r in range(len(A))
        if r != i
    ]


class GPFit:
    """
    不可变 GP 拟合对象: 持有训练数据 (带稳定样本 ID)、超参数、显式 jitter
    以及训练协方差 Ky 的 Cholesky 因子。

    构造: GPFit.fit(X, y, hp, *, jitter=0.0, sample_ids=None)
      sample_ids 缺省为 (0, 1, ..., n-1); 给定时必须可哈希且互不重复。
      ID 与样本终身绑定: delete 只删除不复用, 剩余 ID 始终指向原样本。

    删除: fit2 = fit.delete(sample_id)
      返回新 GPFit (训练点顺序被压缩, 但 ID 不变), 原 fit 完全不变、
      可继续查询。内部用 Cholesky 行删除 (Givens 旋转) 做 O(n^2) 局部更新,
      不把每次删除当作全量 fit; 残差验收失败时回退全量分解,
      两者都记入新对象的 diagnostics (旧对象诊断保持不变)。
      未知 ID 抛 KeyError。

    查询: log_marginal() / predict(X_test) 与模块级同名函数契约一致,
      全部使用当前剩余样本数 n。删除最后一点后 (n=0):
      log_marginal() 返回空数据似然 0.0、零梯度;
      predict() 返回先验预测 (均值 0, 隐方差 alpha^2,
      观测方差 alpha^2 + noise^2 + jitter), 不产生空矩阵索引错误。

    对象不可变: 所有内部状态在构造时定型, 无任何修改方法。
    """

    __slots__ = (
        "_X", "_y", "_ids", "_hp", "_jitter", "_var",
        "_Ky", "_L", "_status", "_sym", "_diagnostics", "_dim",
    )

    def __init__(self):
        raise TypeError("GPFit: use GPFit.fit(...) or fit.delete(...)")

    @classmethod
    def _from_state(cls, X, y, ids, hp, jitter, var, Ky, L, status, sym, diags, dim):
        obj = object.__new__(cls)
        obj._X = X
        obj._y = y
        obj._ids = ids
        obj._hp = hp
        obj._jitter = jitter
        obj._var = var
        obj._Ky = Ky
        obj._L = L
        obj._status = status
        obj._sym = sym
        obj._diagnostics = diags
        obj._dim = dim
        return obj

    @classmethod
    def fit(
        cls,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        hp: Hyperparameters,
        *,
        jitter: float = 0.0,
        sample_ids: Optional[Sequence[Hashable]] = None,
    ) -> "GPFit":
        """全量初始拟合 (唯一做完整 Cholesky 的入口)。"""
        Xv, yv, hpv, jv, var = _prepare_training(X, y, hp, jitter)
        n = len(Xv)
        ids = cls._validate_ids(sample_ids, n)

        K, _ = _kernel_and_d2(Xv, hpv)
        Ky = [row[:] for row in K]
        for i in range(n):
            Ky[i][i] += var
        sym = _symmetry_residual(Ky)
        L = _cholesky(Ky)
        status = "ok" if L is not None else "not_pd"
        return cls._from_state(
            Xv, yv, ids, hpv, jv, var, Ky, L, status, sym, (), len(Xv[0])
        )

    @staticmethod
    def _validate_ids(sample_ids, n) -> Tuple[Hashable, ...]:
        if sample_ids is None:
            return tuple(range(n))
        if not isinstance(sample_ids, (list, tuple)):
            raise TypeError(
                f"sample_ids: expected list/tuple, got {type(sample_ids).__name__}"
            )
        if len(sample_ids) != n:
            raise ValueError(
                f"sample_ids: length {len(sample_ids)} != number of samples {n}"
            )
        ids = tuple(sample_ids)
        seen = set()
        for k, sid in enumerate(ids):
            try:
                if sid in seen:
                    raise ValueError(f"sample_ids[{k}]: duplicate id {sid!r}")
                seen.add(sid)
            except TypeError:
                raise TypeError(f"sample_ids[{k}]: unhashable id {sid!r}") from None
        return ids

    # -- 只读属性 -----------------------------------------------------------

    @property
    def sample_ids(self) -> Tuple[Hashable, ...]:
        return self._ids

    @property
    def n_samples(self) -> int:
        return len(self._ids)

    @property
    def hyperparameters(self) -> Hyperparameters:
        return self._hp

    @property
    def jitter(self) -> float:
        return self._jitter

    @property
    def status(self) -> str:
        """初始/当前分解状态: "ok" 或 "not_pd"。"""
        return self._status

    @property
    def diagnostics(self) -> Tuple[DeletionRecord, ...]:
        return self._diagnostics

    @property
    def X(self) -> Tuple[Tuple[float, ...], ...]:
        return self._X

    @property
    def y(self) -> Tuple[float, ...]:
        return self._y

    # -- 删除 ---------------------------------------------------------------

    def delete(self, sample_id: Hashable) -> "GPFit":
        """按稳定 ID 删除一条样本, 返回新 GPFit; 本对象不变。"""
        try:
            i = self._ids.index(sample_id)
        except ValueError:
            raise KeyError(
                f"delete: unknown sample id {sample_id!r}; "
                f"known ids: {self._ids!r}"
            ) from None

        X2 = tuple(row for k, row in enumerate(self._X) if k != i)
        y2 = tuple(v for k, v in enumerate(self._y) if k != i)
        ids2 = tuple(sid for k, sid in enumerate(self._ids) if k != i)
        n2 = len(ids2)

        if n2 == 0:
            rec = DeletionRecord(sample_id, "trivial_empty", None, 0.0)
            return GPFit._from_state(
                X2, y2, ids2, self._hp, self._jitter, self._var,
                (), (), "ok", 0.0, self._diagnostics + (rec,), self._dim,
            )

        Ky2 = _delete_row_col(self._Ky, i)
        sym2 = _symmetry_residual(Ky2)
        tol = DOWNDATE_RESIDUAL_RTOL * max(
            1.0, max(abs(Ky2[d][d]) for d in range(n2))
        )

        L2 = None
        dd_res = None
        method = "cholesky_downdate"
        if self._L is not None:
            cand = _cholesky_delete_row(self._L, i)
            if cand is not None:
                dd_res = _chol_residual(Ky2, cand)
                if dd_res <= tol:
                    L2 = cand
        if L2 is None:
            # 数值验收失败: 明确回退全量分解, 并在诊断中记录
            method = "full_refit"
            L2 = _cholesky(Ky2)

        if L2 is None:
            status2 = "not_pd"
            chol_res = None
        else:
            status2 = "ok"
            chol_res = _chol_residual(Ky2, L2)
        rec = DeletionRecord(sample_id, method, dd_res, chol_res)
        return GPFit._from_state(
            X2, y2, ids2, self._hp, self._jitter, self._var,
            Ky2, L2, status2, sym2, self._diagnostics + (rec,), self._dim,
        )

    # -- 查询 (全部使用当前剩余样本数) --------------------------------------

    def log_marginal(self) -> MarginalResult:
        """当前剩余训练集上的 log 边际似然与解析梯度 (契约同模块级函数)。"""
        n = len(self._ids)
        if n == 0:
            # 空数据集: LML 为空和 0.0, 梯度为零向量
            return MarginalResult(
                status="ok",
                log_marginal=0.0,
                gradient=(0.0, 0.0, 0.0),
                alpha_vec=(),
                K_inv=(),
                symmetry_residual=0.0,
                chol_residual=0.0,
                solve_residual=0.0,
                effective_noise_variance=self._var,
                jitter=self._jitter,
            )
        if self._status != "ok":
            return _lml_not_pd(self._sym, self._var, self._jitter)
        return _lml_from_factor(
            self._X, self._y, self._hp, self._jitter, self._var,
            self._Ky, self._L, self._sym,
        )

    def predict(self, X_test: Sequence[Sequence[float]]) -> Prediction:
        """当前剩余训练集上的后验预测 (契约同模块级函数)。"""
        n = len(self._ids)
        Xt, _m, _d = _validate_matrix(
            X_test, "X_test", max_rows=MAX_TEST_POINTS, expect_dim=self._dim
        )
        if n == 0:
            # 先验预测: 均值 0, 隐方差 alpha^2, 观测方差 alpha^2 + s^2 + jitter
            m = len(Xt)
            a2 = self._hp.alpha * self._hp.alpha
            return Prediction(
                status="ok",
                mean=(0.0,) * m,
                latent_variance=(a2,) * m,
                observed_variance=(a2 + self._var,) * m,
                raw_latent_variance=(a2,) * m,
                clamped=(),
                symmetry_residual=0.0,
                solve_residual=0.0,
                effective_noise_variance=self._var,
                jitter=self._jitter,
            )
        if self._status != "ok":
            return _predict_not_pd(self._sym, self._var, self._jitter)
        return _predict_from_factor(
            self._X, self._y, self._hp, self._jitter, self._var,
            self._Ky, self._L, self._sym, Xt,
        )
