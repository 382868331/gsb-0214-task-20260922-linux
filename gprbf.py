"""
gprbf - 精确 RBF 高斯过程回归（标准库实现，无第三方依赖）。

核函数: k(x, x') = alpha^2 * exp(-||x-x'||^2 / (2 * lengthscale^2))
训练协方差: K_y = K(X, X) + (noise^2 + jitter) * I

另支持一维一阶导数观测 f'(x) 的联合 GP (predict_with_derivatives),
导数观测带独立噪声, 联合协方差分块:
    K_f f' = K(X,X) + (s_f^2+j)I,  K_f d = Cov(f(X), f'(Z)),
    K_d f = K_f d^T,               K_d d = Cov(f'(Z),f'(Z)) + (s_d^2+j)I
其中 (u = x - z, ell = lengthscale):
    Cov(f(x), f'(z)) = k(x,z) * u / ell^2
    Cov(f'(x), f(z)) = -k(x,z) * u / ell^2
    Cov(f'(x), f'(z)) = k(x,z) * (1/ell^2 - u^2/ell^4)

接口契约见 README.md。要点:
  * 训练样本数 1 <= n <= 80, 输入维数 1 <= d <= 3;
    导数联合功能仅支持一维 (d=1), 函数与导数观测总数 <= 80。
  * alpha / lengthscale / noise 必须为有限正数; jitter 默认为 0,
    只能通过关键字参数显式给定 (有限, >= 0), 并作为方差项进入全部公式。
  * Cholesky 失败时返回 status="not_pd", 绝不自动加未记录的 jitter。
  * 数值输入拒绝 NaN/Infinity; bool 不作为数值接受。
  * 本模块无状态、不修改调用方传入的数据。
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
    "DerivativePrediction",
    "log_marginal",
    "predict",
    "predict_with_derivatives",
    "rbf_kernel",
    "rbf_kernel_derivatives_1d",
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
class DerivativePrediction:
    """
    predict_with_derivatives 的联合后验预测结果 (仅一维)。

    训练观测由 n_f 个函数值 y (噪声 noise) 与 n_d 个导数值 dy
    (独立噪声 derivative_noise) 组成, 按 [函数观测 | 导数观测] 排列联合
    GP。预测点同时给出函数 f 与一阶导数 f' 的后验。

    status:
        "ok"                - 全部预测点的 f / f' 分量正常 (允许个别负微量抹零);
        "not_pd"            - 联合协方差非正定, 预测字段为 None;
        "negative_variance" - 至少一个分量的原始隐方差比容差还小 (不可用
                              数值误差解释), 原始 (负) 值原样返回。
    function_mean / function_latent_variance / function_observed_variance:
        各预测点 E[f*|y,dy]、V[f*]、V[y*] = V[f*] + noise^2 + jitter。
    derivative_mean / derivative_latent_variance / derivative_observed_variance:
        各预测点 E[f'*|y,dy]、V[f'*]、V[新导数观测] = V[f'*] + deriv_noise^2 + jitter。
    raw_function_variance / raw_derivative_variance: 抹零前的原始隐方差。
    clamped_function / clamped_derivative: 被抹零分量的预测点下标。
    symmetry_residual: 联合协方差矩阵 max|A - A^T|。
    solve_residual: 所有交叉协方差列求解 max|A w - k|。
    effective_function_noise_variance / effective_derivative_noise_variance:
        实际进入公式的 noise^2 + jitter 与 derivative_noise^2 + jitter。
    jitter: 实际使用的显式 jitter。
    """

    status: str
    function_mean: Optional[Tuple[float, ...]]
    function_latent_variance: Optional[Tuple[float, ...]]
    function_observed_variance: Optional[Tuple[float, ...]]
    derivative_mean: Optional[Tuple[float, ...]]
    derivative_latent_variance: Optional[Tuple[float, ...]]
    derivative_observed_variance: Optional[Tuple[float, ...]]
    raw_function_variance: Optional[Tuple[float, ...]]
    raw_derivative_variance: Optional[Tuple[float, ...]]
    clamped_function: Tuple[int, ...]
    clamped_derivative: Tuple[int, ...]
    symmetry_residual: float
    solve_residual: Optional[float]
    effective_function_noise_variance: float
    effective_derivative_noise_variance: float
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


def _scalar_rbf_1d(x: float, z: float, a2: float, inv2l2: float) -> float:
    d = x - z
    return a2 * math.exp(-(d * d) * inv2l2)


def rbf_kernel_derivatives_1d(
    X: Sequence[Sequence[float]],
    Z: Sequence[Sequence[float]],
    hp: Hyperparameters,
) -> Tuple[
    Tuple[Tuple[float, ...], ...],
    Tuple[Tuple[float, ...], ...],
    Tuple[Tuple[float, ...], ...],
]:
    """
    一维 RBF 核的函数/一阶导数交叉协方差块 (走完整输入校验)。

    对 k(x,z) = alpha^2 exp(-(x-z)^2/(2 ell^2)), 记 u = x - z:
      K[i][j]    = k(X_i, Z_j)                                   = Cov(f(X_i), f(Z_j))
      K_x_z[i][j]= k(X_i,Z_j) * u / ell^2     (对第二点 z 求导)  = Cov(f(X_i), f'(Z_j))
      K_d_d[i][j]= k(X_i,Z_j) * (1/ell^2 - u^2/ell^4)            = Cov(f'(X_i), f'(Z_j))

    符号关系 (可独立核对):
      Cov(f(x), f'(z)) = -Cov(f(z), f'(x))  (K_x_z 关于交换 X,Z 反号)
      Cov(f'(x), f'(z)) = Cov(f'(z), f'(x)) (K_d_d 对称)
    要求 X / Z 均为一维 (d=1)。
    """
    Xv, _, dx = _validate_matrix(X, "X", max_rows=MAX_SAMPLES)
    Zv, _, dz = _validate_matrix(Z, "Z", max_rows=MAX_SAMPLES)
    if dx != 1:
        raise ValueError(f"X: derivative kernels are 1D only, got dimension {dx}")
    if dz != 1:
        raise ValueError(f"Z: derivative kernels are 1D only, got dimension {dz}")
    hpv = _validate_hp(hp)
    K, K_x_z, K_d_d = _derivative_blocks_1d(Xv, Zv, hpv)
    return (
        tuple(tuple(row) for row in K),
        tuple(tuple(row) for row in K_x_z),
        tuple(tuple(row) for row in K_d_d),
    )


def _derivative_blocks_1d(X, Z, hp: Hyperparameters):
    """返回 list 形式的 (k(X,Z), Cov(f(X),f'(Z)), Cov(f'(X),f'(Z)))。"""
    n, m = len(X), len(Z)
    a2 = hp.alpha * hp.alpha
    ell = hp.lengthscale
    inv2l2 = 1.0 / (2.0 * ell * ell)
    inv_l2 = 1.0 / (ell * ell)
    K = [[0.0] * m for _ in range(n)]
    K_x_z = [[0.0] * m for _ in range(n)]
    K_d_d = [[0.0] * m for _ in range(n)]
    for i in range(n):
        xi = X[i][0]
        for j in range(m):
            u = xi - Z[j][0]
            kij = _scalar_rbf_1d(xi, Z[j][0], a2, inv2l2)
            K[i][j] = kij
            K_x_z[i][j] = kij * u * inv_l2
            K_d_d[i][j] = kij * (inv_l2 - u * u * inv_l2 * inv_l2)
    return K, K_x_z, K_d_d


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
# 一维函数值 + 一阶导数观测的联合 GP
# ---------------------------------------------------------------------------


def _assemble_joint_1d(Xf, Xd, hp: Hyperparameters, var_f: float, var_d: float):
    """
    组装按 [函数观测 | 导数观测] 排列的联合协方差矩阵 (list[list])。

        A = [ K_ff + var_f I     K_fd     ]
            [     K_df        K_dd + var_d I ]
    K_fd = Cov(f(Xf), f'(Xd)); K_df 为其转置 (用同一表达式转置填入, 保证逐位对称);
    K_dd = Cov(f'(Xd), f'(Xd))。
    """
    nf, nd = len(Xf), len(Xd)
    N = nf + nd
    _, Kfd, _ = _derivative_blocks_1d(Xf, Xd, hp)
    Kff_self, _, _ = _derivative_blocks_1d(Xf, Xf, hp)
    _, _, Kdd_d = _derivative_blocks_1d(Xd, Xd, hp)

    A = [[0.0] * N for _ in range(N)]
    for i in range(nf):
        for j in range(nf):
            A[i][j] = Kff_self[i][j]
        A[i][i] += var_f
    for i in range(nd):
        for j in range(nd):
            A[nf + i][nf + j] = Kdd_d[i][j]
        A[nf + i][nf + i] += var_d
    # 交叉块: A[i][nf+j] = K_fd[i][j], 镜像块由同一值转置填入 => 逐位对称
    for i in range(nf):
        for j in range(nd):
            v = Kfd[i][j]
            A[i][nf + j] = v
            A[nf + j][i] = v
    return A


def predict_with_derivatives(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    Xd: Sequence[Sequence[float]],
    dy: Sequence[float],
    X_test: Sequence[Sequence[float]],
    hp: Hyperparameters,
    *,
    derivative_noise: float,
    jitter: float = 0.0,
) -> DerivativePrediction:
    """
    一维 RBF GP 的函数值 + 一阶导数联合后验预测。

    观测模型:
        y_i  = f(X_i)  + eps_f,   eps_f ~ N(0, noise^2 + jitter)
        dy_j = f'(Xd_j) + eps_d,  eps_d ~ N(0, derivative_noise^2 + jitter), 独立
    仅支持一维输入。函数与导数观测各至少 1 个, 总数 nf + nd <= MAX_SAMPLES。
    预测点同时返回 f 与 f' 的后验均值、隐方差与观测方差。

    协方差块 (u = x - z, ell = lengthscale):
        Cov(f(x), f'(z))  = k(x,z) * u / ell^2
        Cov(f'(x), f(z))  = -k(x,z) * u / ell^2
        Cov(f'(x), f'(z)) = k(x,z) * (1/ell^2 - u^2/ell^4)
    f' 的先验方差为 alpha^2 / ell^2 (令 u=0)。负微量方差沿用 classify_raw_variance。

    所有输入先整体校验, 再开始计算; 本函数无状态、不修改入参。
    """
    Xfv, nf, df = _validate_matrix(X, "X", max_rows=MAX_SAMPLES)
    yv = _validate_vector(y, "y", nf)
    Xdv, nd, dd = _validate_matrix(Xd, "Xd", max_rows=MAX_SAMPLES)
    dyv = _validate_vector(dy, "dy", nd)
    hpv = _validate_hp(hp)
    sd = _finite_float(derivative_noise, "derivative_noise")
    if sd <= 0.0:
        raise ValueError(f"derivative_noise: must be > 0, got {sd!r}")
    jv = _validate_jitter(jitter)

    if df != 1:
        raise ValueError(
            f"X: derivative-joint GP is 1D only, got dimension {df}"
        )
    if dd != 1:
        raise ValueError(
            f"Xd: derivative-joint GP is 1D only, got dimension {dd}"
        )
    if nf + nd > MAX_SAMPLES:
        raise ValueError(
            f"total observations {nf}+{nd}={nf + nd} exceeds limit {MAX_SAMPLES}"
        )
    Xtv, m, dt = _validate_matrix(
        X_test, "X_test", max_rows=MAX_TEST_POINTS, expect_dim=1
    )

    var_f = hpv.noise * hpv.noise + jv
    var_d = sd * sd + jv
    a2 = hpv.alpha * hpv.alpha
    prior_f = a2
    prior_d = a2 / (hpv.lengthscale * hpv.lengthscale)

    A = _assemble_joint_1d(Xfv, Xdv, hpv, var_f, var_d)
    sym = _symmetry_residual(A)
    N = nf + nd
    y_joint = list(yv) + list(dyv)

    L = _cholesky(A)
    if L is None:
        return DerivativePrediction(
            status="not_pd",
            function_mean=None,
            function_latent_variance=None,
            function_observed_variance=None,
            derivative_mean=None,
            derivative_latent_variance=None,
            derivative_observed_variance=None,
            raw_function_variance=None,
            raw_derivative_variance=None,
            clamped_function=(),
            clamped_derivative=(),
            symmetry_residual=sym,
            solve_residual=None,
            effective_function_noise_variance=var_f,
            effective_derivative_noise_variance=var_d,
            jitter=jv,
        )

    avec = _cho_solve(L, y_joint)

    # 交叉协方差: 对训练点 Xf / Xd 与预测点 T 分别取导数块。
    _, Kf_dstar, _ = _derivative_blocks_1d(Xfv, Xtv, hpv)   # Cov(f(Xf), f'(T))
    Kff_star = _cross_kernel(Xfv, Xtv, hpv)                 # Cov(f(Xf), f(T))
    _, Kfd_T, Kdd_star = _derivative_blocks_1d(Xdv, Xtv, hpv)
    # blocks(Xd,T)[1] = Cov(f(Xd), f'(T)); 需要 Cov(f'(Xd), f(T)) = 其负值

    f_mean = [0.0] * m
    d_mean = [0.0] * m
    f_raw = [0.0] * m
    d_raw = [0.0] * m
    f_lat = [0.0] * m
    d_lat = [0.0] * m
    f_obs = [0.0] * m
    d_obs = [0.0] * m
    clamped_f = []
    clamped_d = []
    hard_negative = False
    solve_res = 0.0

    for t in range(m):
        # 预测 f(T_t): 联合交叉向量 [Cov(f(Xf),f*) ; Cov(f'(Xd),f*)]
        kf = [Kff_star[i][t] for i in range(nf)]
        kf += [-Kfd_T[i][t] for i in range(nd)]
        # 预测 f'(T_t): [Cov(f(Xf),f'*) ; Cov(f'(Xd),f'*)]
        kd = [Kf_dstar[i][t] for i in range(nf)]
        kd += [Kdd_star[i][t] for i in range(nd)]

        wf = _cho_solve(L, kf)
        wd = _cho_solve(L, kd)

        f_mean[t] = sum(kf[i] * avec[i] for i in range(N))
        d_mean[t] = sum(kd[i] * avec[i] for i in range(N))
        f_raw[t] = prior_f - sum(kf[i] * wf[i] for i in range(N))
        d_raw[t] = prior_d - sum(kd[i] * wd[i] for i in range(N))

        for w, kv in ((wf, kf), (wd, kd)):
            r = _max_abs([q - kv[i] for i, q in enumerate(_matvec(A, w))])
            if r > solve_res:
                solve_res = r

        vf, kindf = classify_raw_variance(f_raw[t], prior_f)
        vd, kindd = classify_raw_variance(d_raw[t], prior_d)
        f_lat[t], d_lat[t] = vf, vd
        f_obs[t] = vf + var_f
        d_obs[t] = vd + var_d
        if kindf == "clamped":
            clamped_f.append(t)
        elif kindf == "negative":
            hard_negative = True
        if kindd == "clamped":
            clamped_d.append(t)
        elif kindd == "negative":
            hard_negative = True

    status = "negative_variance" if hard_negative else "ok"
    return DerivativePrediction(
        status=status,
        function_mean=tuple(f_mean),
        function_latent_variance=tuple(f_lat),
        function_observed_variance=tuple(f_obs),
        derivative_mean=tuple(d_mean),
        derivative_latent_variance=tuple(d_lat),
        derivative_observed_variance=tuple(d_obs),
        raw_function_variance=tuple(f_raw),
        raw_derivative_variance=tuple(d_raw),
        clamped_function=tuple(clamped_f),
        clamped_derivative=tuple(clamped_d),
        symmetry_residual=sym,
        solve_residual=solve_res,
        effective_function_noise_variance=var_f,
        effective_derivative_noise_variance=var_d,
        jitter=jv,
    )
