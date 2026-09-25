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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

__all__ = [
    "MAX_SAMPLES",
    "MAX_DIM",
    "MAX_TEST_POINTS",
    "MAX_JOINT_POINTS",
    "NEG_VAR_REL_TOL",
    "PSD_EIGEN_REL_TOL",
    "RANK_REL_TOL",
    "Hyperparameters",
    "MarginalResult",
    "Prediction",
    "JointPosterior",
    "JointSamples",
    "log_marginal",
    "predict",
    "joint_posterior",
    "sample_posterior",
    "rbf_kernel",
    "classify_raw_variance",
    "jacobi_eigen",
    "psd_factor",
]

#: 训练样本数硬上限（题面接口边界）。
MAX_SAMPLES = 80
#: 输入维数硬上限（题面接口边界）。
MAX_DIM = 3
#: 预测点数量的保护性上限（README 中记录的防护，非题面要求）。
MAX_TEST_POINTS = 10_000
#: 联合后验（满协方差）查询点上限: 协方差为 m*m, 分解为 O(m^3)。
MAX_JOINT_POINTS = 80
#: 预测方差负微量判定容差: raw >= -NEG_VAR_REL_TOL * max(1, alpha^2) 视为数值抹零。
NEG_VAR_REL_TOL = 1e-10
#: 后验协方差特征值负微量修正容差: 小于该幅度的负特征值抹零并记录。
PSD_EIGEN_REL_TOL = 1e-10
#: 数值秩判定: 特征值 <= RANK_REL_TOL * max(最大特征值, 1) 视为零特征方向。
RANK_REL_TOL = 1e-12
#: 条件化时零空间相容性容差 (相对尺度): 投影超过该幅度即判定不相容。
COND_NULL_REL_TOL = 1e-9

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
class JointPosterior:
    """
    多个查询点的联合后验 N(mean, cov) (隐函数 f* 的满协方差)。

    status:
        "ok"                - 满协方差已分解 (允许零特征方向, 允许记录的负微量修正);
        "not_pd"            - 训练协方差无法 Cholesky 分解;
        "not_psd"           - 后验协方差存在超过容差的负特征值, 明确拒绝,
                              不添加观测噪声或 jitter 掩盖问题。
    mean: 后验均值向量 (嵌套元组长度 m)。
    covariance: 后验满协方差 (m*m 嵌套元组, 逐位对称)。
    factor: 协方差的相关结构因子 F (嵌套元组, m 行 rank 列),
            满足 F F^T = 修正后的协方差; 样本 = mean + Z F^T。
    rank: 非零特征方向数 (数值秩); 重复查询点会使 rank < m。
    eigenvalues: Jacobi 特征分解的原始特征值 (升序, 未修正), 供审计。
    corrected_eigenvalues: 被抹零修正的 (下标, 原始特征值) 列表。
    rejected_eigenvalue: status="not_psd" 时最负的那个特征值, 否则 None。
    symmetry_residual / solve_residual: 数值诊断量。
    effective_noise_variance: noise^2 + jitter。
    """

    status: str
    mean: Optional[Tuple[float, ...]]
    covariance: Optional[Tuple[Tuple[float, ...], ...]]
    factor: Optional[Tuple[Tuple[float, ...], ...]]
    rank: Optional[int]
    eigenvalues: Optional[Tuple[float, ...]]
    corrected_eigenvalues: Tuple[Tuple[int, float], ...]
    rejected_eigenvalue: Optional[float]
    symmetry_residual: float
    solve_residual: Optional[float]
    effective_noise_variance: float
    jitter: float


@dataclass(frozen=True)
class JointSamples:
    """
    sample_posterior 的结果: 给定标准正态数矩阵生成的相关函数样本。

    status:
        "ok"                - samples 已按满协方差相关结构生成 (或在固定坐标上完成条件化);
        "not_pd"            - 训练协方差无法分解;
        "not_psd"           - (条件) 后验协方差明显非半正定, 拒绝;
        "incompatible_fixed" - 固定坐标与其零方差/零空间约束不相容, 拒绝;
        "bad_shape"         - Z 形状与所需标准正态数不符。
    samples: (n_draws, m) 嵌套元组; 固定坐标逐位等于给定值, 自由坐标相关。
    mean: 条件化后的均值向量 (无 fixed 时即联合后验均值)。
    covariance: 条件化后的满协方差 (无 fixed 时即后验满协方差)。
    rank: 条件化后自由方向的数值秩 (即每次抽样所需的标准正态数列数)。
    fixed_indices: 参与条件化的查询点下标 (升序)。
    corrected_eigenvalues: 分解过程中记录的负微量抹零 ((下标, 原始值), ...)。
    rejected_eigenvalue / incompatible_detail: 失败原因诊断。
    """

    status: str
    samples: Optional[Tuple[Tuple[float, ...], ...]]
    mean: Optional[Tuple[float, ...]]
    covariance: Optional[Tuple[Tuple[float, ...], ...]]
    rank: Optional[int]
    fixed_indices: Tuple[int, ...]
    corrected_eigenvalues: Tuple[Tuple[int, float], ...]
    rejected_eigenvalue: Optional[float]
    incompatible_detail: Optional[str]


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
    allow_empty: bool = False,
) -> Tuple[Tuple[Tuple[float, ...], ...], int, int]:
    if not isinstance(X, (list, tuple)):
        raise TypeError(f"{name}: expected list/tuple of rows, got {type(X).__name__}")
    if len(X) == 0 and not allow_empty:
        raise ValueError(f"{name}: must contain at least one row")
    if len(X) > max_rows:
        raise ValueError(
            f"{name}: {len(X)} rows exceeds limit {max_rows}"
        )

    rows = []
    dim = expect_dim
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


def _validate_hp(hp, *, allow_zero_noise: bool = False) -> Hyperparameters:
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
    if s < 0.0 or (s == 0.0 and not allow_zero_noise):
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


def _prepare_training(X, y, hp, jitter, *, allow_zero_noise: bool = False):
    Xv, n, _d = _validate_matrix(X, "X", max_rows=MAX_SAMPLES)
    yv = _validate_vector(y, "y", n)
    hpv = _validate_hp(hp, allow_zero_noise=allow_zero_noise)
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
# 对称特征分解与半正定 (允许奇异) 因子
# ---------------------------------------------------------------------------


def jacobi_eigen(A, max_sweeps: int = 100):
    """
    实对称矩阵的循环 Jacobi 特征分解 (标准库纯 Python)。

    返回 (w, V): w 为升序特征值元组, V 的第 k 列为对应特征向量 (V[i][k]),
    满足 A V[:,k] = w[k] V[:,k]。不修改入参。空矩阵返回 ((), ())。
    """
    n = len(A)
    if n == 0:
        return (), ()
    S = [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += S[p][q] * S[p][q]
        if off <= 1e-30:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = S[p][q]
                if apq == 0.0:
                    continue
                tau = (S[q][q] - S[p][p]) / (2.0 * apq)
                # t = sign(tau) / (|tau| + sqrt(1+tau^2)); tau=0 取 t=1。
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                # S <- G^T S G; 先列旋转再行旋转, 行旋转必须用列旋转后的新值。
                for k in range(n):
                    skp = S[k][p]
                    skq = S[k][q]
                    S[k][p] = c * skp - s * skq
                    S[k][q] = s * skp + c * skq
                for k in range(n):
                    spk = S[p][k]
                    sqk = S[q][k]
                    S[p][k] = c * spk - s * sqk
                    S[q][k] = s * spk + c * sqk
                S[p][q] = 0.0
                S[q][p] = 0.0
                for k in range(n):
                    vkp = V[k][p]
                    vkq = V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq

    w = [S[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: w[i])
    w_sorted = tuple(w[i] for i in order)
    V_sorted = tuple(tuple(V[i][order[j]] for j in range(n)) for i in range(n))
    return w_sorted, V_sorted


def psd_factor(A):
    """
    对称半正定矩阵的带秩判断分解。

    先对称化 (C <- (A+A^T)/2) 消除逐位非对称, 再做 Jacobi 特征分解:
      * 特征值 lam 满足 lam < -PSD_EIGEN_REL_TOL*max(1, lam_max):
        明显非半正定 -> status="not_psd", 返回最负特征值, 不做修正;
      * 容差内的负特征值: 抹为 0 并在 corrected 中记录 (下标, 原始值);
      * 0 <= lam <= RANK_REL_TOL*max(1, lam_max, 1): 零特征方向, 保留秩亏损,
        重复查询点的完全相关性由此体现, 绝不通过加噪声解除;
      * 其余为正特征方向。

    返回字典: status, factor F (m 行 rank 列, F F^T = 修正后 C), rank,
    eigenvalues (原始, 升序), corrected, rejected, symmetrized C, symmetry_residual。
    """
    n = len(A)
    C = [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]
    sym_res = _symmetry_residual(A)
    if n == 0:
        return {
            "status": "ok", "factor": (), "rank": 0, "eigenvalues": (),
            "corrected": (), "rejected": None, "C": (), "symmetry_residual": 0.0,
        }
    w, V = jacobi_eigen(C)
    lam_max = max(w[-1], 1.0)
    neg_tol = PSD_EIGEN_REL_TOL * max(1.0, w[-1])
    rank_tol = RANK_REL_TOL * lam_max
    worst = w[0]
    if worst < -neg_tol:
        return {
            "status": "not_psd", "factor": None, "rank": None, "eigenvalues": w,
            "corrected": (), "rejected": worst, "C": None, "symmetry_residual": sym_res,
        }
    corrected = tuple((i, w[i]) for i in range(n) if w[i] < 0.0)
    pos_idx = [i for i in range(n) if w[i] > rank_tol]
    rank = len(pos_idx)
    # F[i][c] = V[i][i_c] * sqrt(lam_c);  samples(row) = mean + sum_c Z[c] F[:,c]
    F = [[0.0] * rank for _ in range(n)]
    for c, idx in enumerate(pos_idx):
        scale = math.sqrt(w[idx])
        for i in range(n):
            F[i][c] = V[i][idx] * scale
    # 重构修正后的对称协方差 (负微量/零方向均置 0), 保证 F F^T 与返回值一致。
    Crebuilt = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Fi = F[i]
        for j in range(i, n):
            Fj = F[j]
            v = sum(Fi[c] * Fj[c] for c in range(rank))
            Crebuilt[i][j] = v
            Crebuilt[j][i] = v
    return {
        "status": "ok",
        "factor": tuple(tuple(row) for row in F),
        "rank": rank,
        "eigenvalues": w,
        "corrected": corrected,
        "rejected": None,
        "C": tuple(tuple(row) for row in Crebuilt),
        "symmetry_residual": sym_res,
    }


# ---------------------------------------------------------------------------
# 联合后验与相关样本 (函数样本, 非逐点独立抽样)
# ---------------------------------------------------------------------------


def _validate_z(Z, n_draws, need_cols, name="Z"):
    """校验调用方提供的标准正态数矩阵: (n_draws, need_cols) 的有限实数列表/元组。"""
    if not isinstance(Z, (list, tuple)):
        raise TypeError(f"{name}: expected list/tuple rows, got {type(Z).__name__}")
    if len(Z) != n_draws:
        return f"{name}: {len(Z)} rows != number of draws {n_draws}"
    for i, row in enumerate(Z):
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"{name}[{i}]: expected list/tuple row, got {type(row).__name__}")
        if len(row) != need_cols:
            return (
                f"{name}[{i}]: {len(row)} standard normals != required {need_cols}"
                f" (posterior numerical rank)"
            )
        for j, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(
                    f"{name}[{i}][{j}]: expected finite real, got {type(v).__name__}"
                )
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                raise ValueError(f"{name}[{i}][{j}]: must be finite, got {v!r}")
    return None


def _joint_mean_cov(Xv, yv, Xt, hpv, var, *, allow_zero_noise=False):
    """独立供联合接口复用: 训练协方差 Cholesky + 查询集均值与满后验协方差。"""
    n = len(Xv)
    m = len(Xt)
    K, _ = _kernel_and_d2(Xv, hpv)
    Ky = [row[:] for row in K]
    for i in range(n):
        Ky[i][i] += var
    sym = _symmetry_residual(Ky)
    L = _cholesky(Ky)
    if L is None:
        return None
    avec = _cho_solve(L, list(yv))
    Ks = _cross_kernel(Xv, Xt, hpv)          # n x m
    Kss = None
    if m > 0:
        Kss, _ = _kernel_and_d2([tuple(r) for r in Xt], hpv)
    # 相同查询点必须共享完全相同的核列 (独立算两次会有 ~1e-16 漂移,
    # 破坏重复点的完全相关性/逐位相等): 首次出现坐标的列作为规范列。
    canonical = {}
    if m > 0:
        for t, xt in enumerate(Xt):
            key = tuple(xt)
            if key in canonical:
                c0 = canonical[key]
                for i in range(n):
                    Ks[i][t] = Ks[i][c0]
                for u in range(m):
                    Kss[t][u] = Kss[c0][u]
                    Kss[u][t] = Kss[u][c0]
            else:
                canonical[key] = t
    mean = [0.0] * m
    for t in range(m):
        mean[t] = sum(Ks[i][t] * avec[i] for i in range(n))
    for t, xt in enumerate(Xt):
        c0 = canonical.get(tuple(xt))
        if c0 is not None and c0 != t:
            mean[t] = mean[c0]
    # W = Ky^-1 Ks (每列一次 Cholesky 求解)
    W = [[0.0] * m for _ in range(n)]
    solve_res = 0.0
    for t in range(m):
        kvec = [Ks[i][t] for i in range(n)]
        w = _cho_solve(L, kvec)
        for i in range(n):
            W[i][t] = w[i]
        r = _max_abs([q - kvec[i] for i, q in enumerate(_matvec(Ky, w))])
        if r > solve_res:
            solve_res = r
    # 后验只条件于 y, Ks^T Ky^-1 Ks = Ks^T W (对称项用 Ks^T W 计算)
    Cov = [[0.0] * m for _ in range(m)]
    a2 = hpv.alpha * hpv.alpha
    for a in range(m):
        for b in range(a, m):
            v = Kss[a][b] - sum(Ks[i][a] * W[i][b] for i in range(n))
            Cov[a][b] = v
            Cov[b][a] = v
    if allow_zero_noise and var == 0.0:
        # 零噪声路径: 查询点与训练点逐位重合时后验必须精确确定 (方差 0)。
        # 仅对这种精确插值坐标, 把浮点残余 (如 1e-31) 抹零并整行整列置 0
        # (零方差变量与其完全相关副本对所有坐标协方差为 0); 不扩大到其他近零点。
        train_set = set(Xv)
        for a, xt in enumerate(Xt):
            if tuple(xt) not in train_set:
                continue
            v, kind = classify_raw_variance(Cov[a][a], a2)
            if kind != "negative":
                for b in range(m):
                    Cov[a][b] = 0.0
                    Cov[b][a] = 0.0
    # 重复坐标共享规范行, 必须在全部条目算完 (含零噪声抹零) 后统一同步,
    # 否则后续行会重新写入带漂移的镜像项。
    for t in range(m):
        c0 = canonical.get(tuple(Xt[t]))
        if c0 is not None and c0 != t:
            for b in range(m):
                Cov[t][b] = Cov[c0][b]
                Cov[b][t] = Cov[b][c0]
    return mean, Cov, sym, solve_res


def joint_posterior(
    X,
    y,
    X_test,
    hp,
    *,
    jitter: float = 0.0,
    allow_zero_noise: bool = False,
) -> JointPosterior:
    """
    计算多个查询点隐函数 f* 的联合后验 N(mean, C), 使用完整后验协方差。

    C = K(X*, X*) - K(X*, X) K_y^-1 K(X, X*)  (逐点边缘与 predict 一致)。
    重复查询点使 C 半正定而不满秩: 允许零特征方向 (样本在重复坐标上完全相关);
    容差内负特征值抹零并记录; 明显非半正定 (超过 PSD_EIGEN_REL_TOL) 返回
    status="not_psd", 绝不加观测噪声或 jitter 掩盖。

    allow_zero_noise=True 时允许 hp.noise == 0 (训练点处零噪声/精确插值路径);
    常规 predict/log_marginal 接口仍要求 noise > 0。X_test 允许为空 (返回 m=0 结果)。
    """
    Xv, yv, hpv, jv, var = _prepare_training(
        X, y, hp, jitter, allow_zero_noise=allow_zero_noise
    )
    Xt, _m, _d = _validate_matrix(
        X_test, "X_test", max_rows=MAX_JOINT_POINTS,
        expect_dim=len(Xv[0]), allow_empty=True,
    )
    prepared = _joint_mean_cov(
        Xv, yv, [tuple(r) for r in Xt], hpv, var,
        allow_zero_noise=allow_zero_noise,
    )
    if prepared is None:
        K0, _ = _kernel_and_d2(Xv, hpv)
        Ky0 = [row[:] for row in K0]
        for i in range(len(Xv)):
            Ky0[i][i] += var
        return JointPosterior(
            status="not_pd", mean=None, covariance=None, factor=None, rank=None,
            eigenvalues=None, corrected_eigenvalues=(), rejected_eigenvalue=None,
            symmetry_residual=_symmetry_residual(Ky0), solve_residual=None,
            effective_noise_variance=var, jitter=jv,
        )
    mean, Cov, sym, solve_res = prepared
    fac = psd_factor(Cov)
    if fac["status"] == "not_psd":
        return JointPosterior(
            status="not_psd", mean=None, covariance=None, factor=None, rank=None,
            eigenvalues=fac["eigenvalues"], corrected_eigenvalues=(),
            rejected_eigenvalue=fac["rejected"], symmetry_residual=sym,
            solve_residual=solve_res, effective_noise_variance=var, jitter=jv,
        )
    rebuilt = [list(row) for row in fac["C"]]
    Fmat = [list(row) for row in fac["factor"]]
    canon = {}
    for t, xt in enumerate(Xt):
        key = tuple(xt)
        if key in canon:
            c0 = canon[key]
            rebuilt[t] = rebuilt[c0][:]
            for b in range(len(Xt)):
                rebuilt[b][t] = rebuilt[b][c0]
            Fmat[t] = Fmat[c0][:]
        else:
            canon[key] = t
    return JointPosterior(
        status="ok",
        mean=tuple(mean),
        covariance=tuple(tuple(row) for row in rebuilt),
        factor=tuple(tuple(row) for row in Fmat),
        rank=fac["rank"],
        eigenvalues=fac["eigenvalues"],
        corrected_eigenvalues=fac["corrected"],
        rejected_eigenvalue=None,
        symmetry_residual=sym,
        solve_residual=solve_res,
        effective_noise_variance=var,
        jitter=jv,
    )


def _pinv_via_eigen(Csub):
    """对称半正定子块的 Moore-Penrose 伪逆 (Jacobi 特征分解 + 秩判定)。"""
    k = len(Csub)
    if k == 0:
        return (), 0, ()
    w, V = jacobi_eigen(Csub)
    lam_max = max(w[-1], 1.0)
    rank_tol = RANK_REL_TOL * lam_max
    pos = [i for i in range(k) if w[i] > rank_tol]
    P = [[0.0] * k for _ in range(k)]
    for a in range(k):
        for b in range(a, k):
            v = 0.0
            for i in pos:
                v += V[a][i] * V[b][i] / w[i]
            P[a][b] = v
            P[b][a] = v
    return tuple(tuple(row) for row in P), len(pos), w


def _validate_fixed(fixed, m):
    if fixed is None:
        return {}
    if isinstance(fixed, dict):
        items = list(fixed.items())
    elif isinstance(fixed, (list, tuple)):
        items = list(fixed)
    else:
        raise TypeError(
            f"fixed: expected dict {{index: value}} or list/tuple of pairs, "
            f"got {type(fixed).__name__}"
        )
    out = {}
    for pair in items:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TypeError("fixed: each entry must be a (index, value) pair")
        idx, val = pair
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError(f"fixed index: expected int, got {type(idx).__name__}")
        if not 0 <= idx < m:
            raise ValueError(f"fixed index {idx} outside query range 0..{m - 1}")
        fv = _finite_float(val, f"fixed[{idx}]")
        if idx in out:
            raise ValueError(f"fixed index {idx} given more than once")
        out[idx] = fv
    return out


def sample_posterior(
    X,
    y,
    X_test,
    hp,
    Z,
    *,
    fixed=None,
    jitter: float = 0.0,
    allow_zero_noise: bool = False,
) -> JointSamples:
    """
    联合后验函数抽样入口: 一次使用完整后验协方差生成相关样本。

    Z 为调用方提供的标准正态数矩阵 (n_draws, rank), rank 是 (条件化后) 后验协方差
    的数值秩, 由返回结果的 rank 字段给出审计; 本函数不读取任何随机种子配置。
    样本 = mean_cond + Z F^T, F 为满协方差经秩判定分解得到的相关结构因子,
    重复查询点共享同一随机方向 (完全相关), 不是逐点独立抽样。

    fixed: {查询点下标: 固定观测值} 或 ((idx, value), ...), 沿这些坐标做高斯条件化:
      * 条件系统只在非零协方差子空间上用伪逆求解 (奇异协方差不求普通逆);
      * 零方差坐标 (如重复点) 的观测必须与其零空间约束相容, 否则
        status="incompatible_fixed" 明确拒绝 (例如给零方差坐标指定不同的值);
      * 相容固定坐标在输出样本中逐位精确等于给定值 (不参与随机扰动)。
    X_test 允许为空: 此时不接受 fixed, Z 必须为 (n_draws, 0), 样本行为空元组。
    allow_zero_noise=True 允许训练点处零噪声 (hp.noise == 0) 的精确插值路径。
    """
    post = joint_posterior(
        X, y, X_test, hp, jitter=jitter, allow_zero_noise=allow_zero_noise
    )
    if post.status == "not_pd":
        return JointSamples(
            status="not_pd", samples=None, mean=None, covariance=None, rank=None,
            fixed_indices=(), corrected_eigenvalues=(), rejected_eigenvalue=None,
            incompatible_detail=None,
        )
    if post.status == "not_psd":
        return JointSamples(
            status="not_psd", samples=None, mean=None, covariance=None, rank=None,
            fixed_indices=(), corrected_eigenvalues=post.corrected_eigenvalues,
            rejected_eigenvalue=post.rejected_eigenvalue, incompatible_detail=None,
        )

    m = len(post.mean)
    fixed_map = _validate_fixed(fixed, m)
    idx_sorted = tuple(sorted(fixed_map))

    mean_c = list(post.mean)
    C = [list(row) for row in post.covariance]
    corrected_all = list(post.corrected_eigenvalues)

    if idx_sorted:
        k = len(idx_sorted)
        C_I = [[C[a][b] for b in idx_sorted] for a in idx_sorted]
        Pinv, _rank_I, _wI = _pinv_via_eigen(C_I)
        g = [fixed_map[a] - post.mean[a] for a in idx_sorted]
        # 相容性: g 在 C_I 列空间上的投影为 C_I (C_I^+ g), 残差 g - C_I C_I^+ g
        # 必须落在零空间容差内; 条件解本身为 C_I^+ g (只在非零子空间上求解)。
        sol = _matvec([list(r) for r in Pinv], g)
        proj = _matvec(C_I, sol)
        null_resid = [g[t] - proj[t] for t in range(k)]
        scale = max([1.0] + [abs(v) for v in g])
        worst_resid = _max_abs(null_resid)
        if worst_resid > COND_NULL_REL_TOL * scale:
            detail = (
                f"fixed observations incompatible with zero-variance subspace: "
                f"null-space residual {worst_resid:.3e} exceeds tolerance "
                f"{COND_NULL_REL_TOL:.0e} * {scale:.3e}"
            )
            return JointSamples(
                status="incompatible_fixed", samples=None, mean=None, covariance=None,
                rank=None, fixed_indices=idx_sorted, corrected_eigenvalues=(),
                rejected_eigenvalue=None, incompatible_detail=detail,
            )
        # K = C[:,I] C_I^+; 条件均值/协方差 (满高斯条件公式, 在非零子空间求解)
        Kgain = [[0.0] * k for _ in range(m)]
        for i in range(m):
            for t in range(k):
                Kgain[i][t] = sum(C[i][idx_sorted[u]] * Pinv[u][t] for u in range(k))
        for i in range(m):
            mean_c[i] += sum(Kgain[i][t] * g[t] for t in range(k))
        newC = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j2 in range(i, m):
                sub = sum(
                    Kgain[i][t] * C[idx_sorted[t]][j2] for t in range(k)
                )
                v = C[i][j2] - sub
                newC[i][j2] = v
                newC[j2][i] = v
        C = newC

    fac = psd_factor(C)
    if fac["status"] == "not_psd":
        return JointSamples(
            status="not_psd", samples=None, mean=None, covariance=None, rank=None,
            fixed_indices=idx_sorted,
            corrected_eigenvalues=tuple(corrected_all) + tuple(fac["corrected"]),
            rejected_eigenvalue=fac["rejected"], incompatible_detail=None,
        )
    corrected_all.extend(fac["corrected"])
    Fmat2 = [list(row) for row in fac["factor"]]
    Cmat2 = [list(row) for row in fac["C"]]
    canon2 = {}
    for t, xt in enumerate([tuple(r) for r in _validate_matrix(
        X_test, "X_test", max_rows=MAX_JOINT_POINTS, allow_empty=True,
    )[0]]):
        if xt in canon2:
            c0 = canon2[xt]
            Fmat2[t] = Fmat2[c0][:]
            Cmat2[t] = Cmat2[c0][:]
            for b in range(m):
                Cmat2[b][t] = Cmat2[b][c0]
        else:
            canon2[xt] = t
    F = tuple(tuple(row) for row in Fmat2)
    rank = fac["rank"]

    if isinstance(Z, (list, tuple)):
        n_draws = len(Z)
        z_err = _validate_z(Z, n_draws, rank)
    else:
        z_err = f"Z: expected list/tuple rows, got {type(Z).__name__}"
    if z_err is not None:
        return JointSamples(
            status="bad_shape", samples=None, mean=None, covariance=None, rank=None,
            fixed_indices=idx_sorted, corrected_eigenvalues=(),
            rejected_eigenvalue=None, incompatible_detail=z_err,
        )

    draws = []
    for row in Z:
        zrow = [float(v) for v in row]
        s = [mean_c[i] + sum(F[i][c] * zrow[c] for c in range(rank)) for i in range(m)]
        for idx in idx_sorted:
            s[idx] = fixed_map[idx]  # 固定坐标精确保持, 不做浮点近似
        draws.append(tuple(s))

    # 输出坐标的确定性赋值:
    #  * 调用方固定的坐标逐位保持给定值, 并传播给同一查询点的重复副本;
    #  * 未被固定但后验方差为 0 的坐标 (如零噪声训练点) 恒取其后验均值。
    forced = dict(fixed_map)
    Xt_coords, _nq, _dq = _validate_matrix(
        X_test, "X_test", max_rows=MAX_JOINT_POINTS, allow_empty=True,
    )
    for idx in idx_sorted:
        val = fixed_map[idx]
        for j in range(m):
            if j not in forced and Xt_coords[j] == Xt_coords[idx]:
                forced[j] = val
    for j in range(m):
        if j not in forced:
            row_sq = C[j][j]
            if row_sq == 0.0 and all(
                    C[j][t] == 0.0 and C[t][j] == 0.0 for t in range(m)):
                forced[j] = mean_c[j]
    return JointSamples(
        status="ok",
        samples=tuple(
            tuple(forced[i] if i in forced else s[i] for i in range(m)) for s in draws
        ),
        mean=tuple(mean_c),
        covariance=tuple(tuple(row) for row in Cmat2),
        rank=rank,
        fixed_indices=idx_sorted,
        corrected_eigenvalues=tuple(corrected_all),
        rejected_eigenvalue=None,
        incompatible_detail=None,
    )
