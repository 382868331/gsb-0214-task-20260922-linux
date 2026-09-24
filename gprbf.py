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

一维一阶导数观测 (本次迭代新增, 仅 d=1):
  predict_joint 把函数值观测 (X, y, 噪声 hp.noise) 与导数观测 (Xd, yd,
  噪声 noise_deriv) 拼成联合协方差做联合预测, 同时给出测试点处函数与
  导数的后验。rbf_kernel_df / rbf_kernel_dd 公开函数-导数与导数-导数
  协方差块, 便于独立审计。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

__all__ = [
    "MAX_SAMPLES",
    "MAX_DIM",
    "MAX_TEST_POINTS",
    "NEG_VAR_REL_TOL",
    "Hyperparameters",
    "MarginalResult",
    "Prediction",
    "JointPrediction",
    "log_marginal",
    "predict",
    "predict_joint",
    "rbf_kernel",
    "rbf_kernel_df",
    "rbf_kernel_dd",
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


@dataclass(frozen=True)
class JointPrediction:
    """
    predict_joint 的结果 (函数值观测 + 一维一阶导数观测的联合后验)。

    status:
        "ok"                - 全部预测点正常 (允许个别点负微量抹零);
        "not_pd"            - 联合协方差非正定, 预测字段为 None;
        "negative_variance" - 至少一个点的原始隐方差 (函数或导数) 比
                              -NEG_VAR_REL_TOL*max(1, 先验方差) 还小,
                              原样返回负值供定位, 不静默纠正。
    mean / latent_variance / observed_variance:
        测试点处函数值 f(x*) 的后验均值、隐方差、观测方差
        (observed = latent + noise^2 + jitter)。
    deriv_mean / deriv_latent_variance / deriv_observed_variance:
        测试点处导数 f'(x*) 的后验均值、隐方差、观测方差
        (observed = latent + noise_deriv^2 + jitter)。
    raw_latent_variance / raw_deriv_latent_variance: 未经抹零的原始值。
    clamped / deriv_clamped: 发生负微量抹零的预测点下标。
    symmetry_residual: max|C - C^T|, C 为 (n+m) 阶联合协方差。
    solve_residual: 训练求解与全部测试点求解残差的最大值。
    effective_noise_variance:        函数值观测实际噪声方差 noise^2 + jitter。
    effective_deriv_noise_variance:  导数观测实际噪声方差 noise_deriv^2 + jitter。
    """

    status: str
    mean: Optional[Tuple[float, ...]]
    latent_variance: Optional[Tuple[float, ...]]
    observed_variance: Optional[Tuple[float, ...]]
    deriv_mean: Optional[Tuple[float, ...]]
    deriv_latent_variance: Optional[Tuple[float, ...]]
    deriv_observed_variance: Optional[Tuple[float, ...]]
    raw_latent_variance: Optional[Tuple[float, ...]]
    raw_deriv_latent_variance: Optional[Tuple[float, ...]]
    clamped: Tuple[int, ...]
    deriv_clamped: Tuple[int, ...]
    symmetry_residual: float
    solve_residual: Optional[float]
    effective_noise_variance: float
    effective_deriv_noise_variance: float
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


# ---------------------------------------------------------------------------
# 一维导数协方差 (仅 d=1; 记 r = x - x', k = k(x, x'), ell = lengthscale)
#
#   Cov(f(x), f'(x')) = ∂k/∂x'          = k * r / ell^2
#   Cov(f'(x), f(x')) = ∂k/∂x           = -k * r / ell^2
#   Cov(f'(x), f'(x')) = ∂^2 k/∂x∂x'    = k * (1/ell^2 - r^2/ell^4)
#   Var(f'(x))        = alpha^2 / ell^2          (r = 0 时)
# ---------------------------------------------------------------------------


def _kernel_df_1d(Xf, Xd, hp: Hyperparameters):
    """Kd[i][j] = Cov(f(Xf[i]), f'(Xd[j])) (输入须已校验为一维)。"""
    a2 = hp.alpha * hp.alpha
    ell2 = hp.lengthscale * hp.lengthscale
    inv2l2 = 1.0 / (2.0 * ell2)
    out = []
    for xf in Xf:
        x = xf[0]
        row = []
        for xd in Xd:
            r = x - xd[0]
            k = a2 * math.exp(-r * r * inv2l2)
            row.append(k * r / ell2)
        out.append(row)
    return out


def _kernel_dd_1d(Xd1, Xd2, hp: Hyperparameters):
    """Kdd[i][j] = Cov(f'(Xd1[i]), f'(Xd2[j])) (输入须已校验为一维)。"""
    a2 = hp.alpha * hp.alpha
    ell2 = hp.lengthscale * hp.lengthscale
    inv2l2 = 1.0 / (2.0 * ell2)
    out = []
    for x1 in Xd1:
        x = x1[0]
        row = []
        for x2 in Xd2:
            r = x - x2[0]
            k = a2 * math.exp(-r * r * inv2l2)
            row.append(k * (1.0 / ell2 - (r * r) / (ell2 * ell2)))
        out.append(row)
    return out


def _validate_1d(Xv, dim: int, name: str) -> None:
    if dim != 1:
        raise ValueError(
            f"{name}: derivative observations are supported for 1D inputs only, "
            f"got dimension {dim}"
        )


def rbf_kernel_df(
    X_f: Sequence[Sequence[float]],
    X_d: Sequence[Sequence[float]],
    hp: Hyperparameters,
) -> Tuple[Tuple[float, ...], ...]:
    """
    函数-导数交叉协方差块 (仅一维):
        返回矩阵第 [i][j] 个元素为 Cov(f(X_f[i]), f'(X_d[j]))
        = k(X_f[i], X_d[j]) * (X_f[i] - X_d[j]) / lengthscale^2。
    走完整输入校验, 便于独立使用与审计。
    """
    Xf, _, df = _validate_matrix(X_f, "X_f", max_rows=MAX_SAMPLES)
    Xd, _, dd = _validate_matrix(X_d, "X_d", max_rows=MAX_SAMPLES)
    _validate_1d(Xf, df, "X_f")
    _validate_1d(Xd, dd, "X_d")
    hpv = _validate_hp(hp)
    return tuple(tuple(row) for row in _kernel_df_1d(Xf, Xd, hpv))


def rbf_kernel_dd(
    X_d1: Sequence[Sequence[float]],
    X_d2: Sequence[Sequence[float]],
    hp: Hyperparameters,
) -> Tuple[Tuple[float, ...], ...]:
    """
    导数-导数协方差块 (仅一维):
        返回矩阵第 [i][j] 个元素为 Cov(f'(X_d1[i]), f'(X_d2[j]))
        = k(X_d1[i], X_d2[j]) * (1/ell^2 - (X_d1[i]-X_d2[j])^2/ell^4)。
    走完整输入校验, 便于独立使用与审计。
    """
    X1, _, d1 = _validate_matrix(X_d1, "X_d1", max_rows=MAX_SAMPLES)
    X2, _, d2 = _validate_matrix(X_d2, "X_d2", max_rows=MAX_SAMPLES)
    _validate_1d(X1, d1, "X_d1")
    _validate_1d(X2, d2, "X_d2")
    hpv = _validate_hp(hp)
    return tuple(tuple(row) for row in _kernel_dd_1d(X1, X2, hpv))


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
    K, D2 = _kernel_and_d2(Xv, hpv)
    Ky = [row[:] for row in K]
    for i in range(n):
        Ky[i][i] += var

    sym = _symmetry_residual(Ky)
    L = _cholesky(Ky)
    if L is None:
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

    avec = _cho_solve(L, yv)

    # 数值残差诊断
    chol_res = _chol_residual(Ky, L)
    solve_res = _max_abs([q - yv[i] for i, q in enumerate(_matvec(Ky, avec))])

    logdet = 0.0
    for i in range(n):
        logdet += 2.0 * math.log(L[i][i])
    quad = sum(yv[i] * avec[i] for i in range(n))
    lml = -0.5 * (quad + logdet + n * _LOG_2PI)

    Inv = _inverse_via_cholesky(L)
    tr_inv = sum(Inv[i][i] for i in range(n))

    # g_log_alpha: K_y = K + var*I  => tr(K^-1 K) = n - var*tr(K^-1)
    K_alpha = _matvec(K, avec)
    quad_K = sum(avec[i] * K_alpha[i] for i in range(n))
    g_log_a = quad_K - (n - var * tr_inv)

    # g_log_ell: dK/d log ell = K ⊙ D2/ell^2 (无 2 倍因子, 故保留 1/2)
    ell2 = hpv.lengthscale * hpv.lengthscale
    Kl = [[K[i][j] * D2[i][j] / ell2 for j in range(n)] for i in range(n)]
    Kl_alpha = _matvec(Kl, avec)
    quad_Kl = sum(avec[i] * Kl_alpha[i] for i in range(n))
    tr_inv_Kl = 0.0
    for i in range(n):
        for j in range(n):
            tr_inv_Kl += Inv[i][j] * Kl[j][i]
    g_log_l = 0.5 * (quad_Kl - tr_inv_Kl)

    # g_log_noise: dK_y/d log s = 2 s^2 I
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
    m = len(Xt)

    K, _ = _kernel_and_d2(Xv, hpv)
    Ky = [row[:] for row in K]
    for i in range(n):
        Ky[i][i] += var
    sym = _symmetry_residual(Ky)

    L = _cholesky(Ky)
    if L is None:
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


# ---------------------------------------------------------------------------
# 一维导数观测的联合预测
# ---------------------------------------------------------------------------


def _validate_noise_deriv(noise_deriv) -> float:
    v = _finite_float(noise_deriv, "noise_deriv")
    if v <= 0.0:
        raise ValueError(f"noise_deriv: must be > 0, got {v!r}")
    return v


def _validate_derivative_observations(Xd, yd):
    """
    导数观测 (Xd, yd): 允许同时为空 (表示无导数观测), 非空时为一维点集,
    长度必须一致。返回 (Xdv, ydv, m)。
    """
    if not isinstance(Xd, (list, tuple)):
        raise TypeError(f"Xd: expected list/tuple of rows, got {type(Xd).__name__}")
    if not isinstance(yd, (list, tuple)):
        raise TypeError(f"yd: expected list/tuple, got {type(yd).__name__}")
    if len(Xd) == 0:
        if len(yd) != 0:
            raise ValueError(
                f"yd: length {len(yd)} != number of derivative observations 0 "
                "(Xd is empty)"
            )
        return (), (), 0
    Xdv, m, dd = _validate_matrix(Xd, "Xd", max_rows=MAX_SAMPLES)
    _validate_1d(Xdv, dd, "Xd")
    ydv = _validate_vector(yd, "yd", m)
    return Xdv, ydv, m


def _joint_covariance_1d(Xv, Xdv, hp: Hyperparameters, var_f: float, var_d: float):
    """
    联合协方差 C (阶 n+m), 观测顺序为 [y_1..y_n, yd_1..yd_m]:

        C = [[K_ff + var_f * I_n,  K_fd              ],
             [K_fd^T,               K_dd + var_d * I_m]]

    分块镜像赋值, 保证逐位对称。
    """
    n = len(Xv)
    m = len(Xdv)
    N = n + m
    a2 = hp.alpha * hp.alpha
    ell2 = hp.lengthscale * hp.lengthscale
    inv2l2 = 1.0 / (2.0 * ell2)
    C = [[0.0] * N for _ in range(N)]
    for i in range(n):
        xi = Xv[i][0]
        for j in range(i, n):
            r = xi - Xv[j][0]
            k = a2 * math.exp(-r * r * inv2l2)
            C[i][j] = k
            C[j][i] = k  # 同一表达式镜像赋值, 保证逐位对称
    for i in range(n):
        xi = Xv[i][0]
        for j in range(m):
            r = xi - Xdv[j][0]
            k = a2 * math.exp(-r * r * inv2l2)
            v = k * r / ell2
            C[i][n + j] = v
            C[n + j][i] = v
    for i in range(m):
        xi = Xdv[i][0]
        for j in range(i, m):
            r = xi - Xdv[j][0]
            k = a2 * math.exp(-r * r * inv2l2)
            v = k * (1.0 / ell2 - (r * r) / (ell2 * ell2))
            C[n + i][n + j] = v
            C[n + j][n + i] = v
    for i in range(n):
        C[i][i] += var_f
    for j in range(m):
        C[n + j][n + j] += var_d
    return C


def predict_joint(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    Xd: Sequence[Sequence[float]],
    yd: Sequence[float],
    X_test: Sequence[Sequence[float]],
    hp: Hyperparameters,
    *,
    noise_deriv: float,
    jitter: float = 0.0,
) -> JointPrediction:
    """
    函数值观测 + 一维一阶导数观测的联合 GP 后验预测 (仅 d=1)。

    观测向量 z = [y (n 个函数值, 噪声 hp.noise), yd (m 个导数值, 噪声
    noise_deriv)], 联合协方差见 _joint_covariance_1d。对每个测试点 x*:

        mean_f  = k_*^T C^-1 z,      latent_f  = alpha^2 - k_*^T C^-1 k_*
        mean_d  = kd_*^T C^-1 z,     latent_d  = alpha^2/ell^2 - kd_*^T C^-1 kd_*

    其中 k_* 为 Cov(z, f(x*)), kd_* 为 Cov(z, f'(x*))。观测方差分别在隐方差
    上加 noise^2 + jitter 与 noise_deriv^2 + jitter。

    Xd/yd 可同时为空列表 (退化为纯函数值观测, 结果与 predict 一致);
    noise_deriv 为必填关键字参数 (即使 m=0 也会校验, 不提供默认值)。
    联合观测总数 n + m <= MAX_SAMPLES。负微量抹零策略与 predict 相同,
    导数方差的容差按先验方差 alpha^2/ell^2 缩放。
    """
    Xv, n, d = _validate_matrix(X, "X", max_rows=MAX_SAMPLES)
    _validate_1d(Xv, d, "X")
    yv = _validate_vector(y, "y", n)
    Xdv, ydv, m = _validate_derivative_observations(Xd, yd)
    if n + m > MAX_SAMPLES:
        raise ValueError(
            f"X/Xd: total observations {n + m} exceeds limit {MAX_SAMPLES}"
        )
    hpv = _validate_hp(hp)
    sd = _validate_noise_deriv(noise_deriv)
    jv = _validate_jitter(jitter)
    Xt, _mt, dt = _validate_matrix(X_test, "X_test", max_rows=MAX_TEST_POINTS)
    _validate_1d(Xt, dt, "X_test")

    var_f = hpv.noise * hpv.noise + jv
    var_d = sd * sd + jv

    C = _joint_covariance_1d(Xv, Xdv, hpv, var_f, var_d)
    sym = _symmetry_residual(C)
    L = _cholesky(C)
    if L is None:
        return JointPrediction(
            status="not_pd",
            mean=None,
            latent_variance=None,
            observed_variance=None,
            deriv_mean=None,
            deriv_latent_variance=None,
            deriv_observed_variance=None,
            raw_latent_variance=None,
            raw_deriv_latent_variance=None,
            clamped=(),
            deriv_clamped=(),
            symmetry_residual=sym,
            solve_residual=None,
            effective_noise_variance=var_f,
            effective_deriv_noise_variance=var_d,
            jitter=jv,
        )

    z = list(yv) + list(ydv)
    avec = _cho_solve(L, z)
    solve_res = _max_abs([q - z[i] for i, q in enumerate(_matvec(C, avec))])

    a2 = hpv.alpha * hpv.alpha
    ell2 = hpv.lengthscale * hp.lengthscale
    inv2l2 = 1.0 / (2.0 * ell2)
    prior_var_d = a2 / ell2
    N = n + m

    mt = len(Xt)
    mean_f = [0.0] * mt
    raw_f = [0.0] * mt
    lat_f = [0.0] * mt
    obs_f = [0.0] * mt
    mean_d = [0.0] * mt
    raw_d = [0.0] * mt
    lat_d = [0.0] * mt
    obs_d = [0.0] * mt
    clamped_f = []
    clamped_d = []
    hard_negative = False

    for t in range(mt):
        xs = Xt[t][0]
        kv = [0.0] * N  # Cov(z_a, f(x*))
        kd = [0.0] * N  # Cov(z_a, f'(x*))
        for i in range(n):
            r = Xv[i][0] - xs
            k = a2 * math.exp(-r * r * inv2l2)
            kv[i] = k
            kd[i] = k * r / ell2
        for j in range(m):
            r = Xdv[j][0] - xs
            k = a2 * math.exp(-r * r * inv2l2)
            kv[n + j] = -k * r / ell2
            kd[n + j] = k * (1.0 / ell2 - (r * r) / (ell2 * ell2))

        w = _cho_solve(L, kv)
        u = _cho_solve(L, kd)
        mean_f[t] = sum(kv[i] * avec[i] for i in range(N))
        mean_d[t] = sum(kd[i] * avec[i] for i in range(N))
        rf = a2 - sum(kv[i] * w[i] for i in range(N))
        rd = prior_var_d - sum(kd[i] * u[i] for i in range(N))
        raw_f[t] = rf
        raw_d[t] = rd

        r1 = _max_abs([q - kv[i] for i, q in enumerate(_matvec(C, w))])
        r2 = _max_abs([q - kd[i] for i, q in enumerate(_matvec(C, u))])
        solve_res = max(solve_res, r1, r2)

        vf, kind_f = classify_raw_variance(rf, a2)
        vd, kind_d = classify_raw_variance(rd, prior_var_d)
        lat_f[t] = vf
        obs_f[t] = vf + var_f
        lat_d[t] = vd
        obs_d[t] = vd + var_d
        if kind_f == "clamped":
            clamped_f.append(t)
        elif kind_f == "negative":
            hard_negative = True
        if kind_d == "clamped":
            clamped_d.append(t)
        elif kind_d == "negative":
            hard_negative = True

    status = "negative_variance" if hard_negative else "ok"
    return JointPrediction(
        status=status,
        mean=tuple(mean_f),
        latent_variance=tuple(lat_f),
        observed_variance=tuple(obs_f),
        deriv_mean=tuple(mean_d),
        deriv_latent_variance=tuple(lat_d),
        deriv_observed_variance=tuple(obs_d),
        raw_latent_variance=tuple(raw_f),
        raw_deriv_latent_variance=tuple(raw_d),
        clamped=tuple(clamped_f),
        deriv_clamped=tuple(clamped_d),
        symmetry_residual=sym,
        solve_residual=solve_res,
        effective_noise_variance=var_f,
        effective_deriv_noise_variance=var_d,
        jitter=jv,
    )
