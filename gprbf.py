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
    "NEG_VAR_REL_TOL",
    "Hyperparameters",
    "MarginalResult",
    "Prediction",
    "GPFit",
    "fit_gp",
    "log_marginal",
    "predict",
    "rbf_kernel",
    "classify_raw_variance",
]

#: GPFit 删除更新后 Cholesky 因子的验收容差 (相对 max(1, alpha^2+var) 量级),
#: 超过则明确回退全量分解并在诊断中记录。
DOWNDATE_REL_TOL = 1e-8

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
# 不可变拟合对象 GPFit: 支持按稳定样本 ID 删除一条样本 (Cholesky 删除更新)
# ---------------------------------------------------------------------------


def _chol_delete_row(L, k):
    """
    Cholesky 删除更新: 已知 K = L L^T (n x n 下三角), 删除 K 的第 k 行/列后
    对应 (n-1) 阶矩阵的 Cholesky 因子。

    做法: 删去 L 的第 k 行得到 A ((n-1) x n, 满足 A A^T = K_new)。
    A 的行 i >= k 多出一个上对角元 A[i][i+1]; 对 i = k .. n-2 在相邻
    列 (i, i+1) 间做 Givens 旋转将其消去 (右乘正交矩阵, 保持 A A^T 不变),
    最后舍去全零末列。复杂度 O((n-k)*n), 不是全量重新分解。
    返回 list[list] 下三角因子 (对角恒正)。
    """
    n = len(L)
    A = [list(L[i]) for i in range(n) if i != k]
    for i in range(k, n - 1):
        a = A[i][i]
        b = A[i][i + 1]
        r = math.hypot(a, b)
        c = a / r
        s = b / r
        for row in range(n - 1):
            u = A[row][i]
            v = A[row][i + 1]
            A[row][i] = c * u + s * v
            A[row][i + 1] = c * v - s * u
    return [row[: n - 1] for row in A]


def _validate_ids(ids, n):
    if not isinstance(ids, (list, tuple)):
        raise TypeError(f"ids: expected list/tuple, got {type(ids).__name__}")
    if len(ids) != n:
        raise ValueError(f"ids: length {len(ids)} != number of samples {n}")
    out = []
    seen = set()
    for i, v in enumerate(ids):
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise TypeError(
                f"ids[{i}]: expected int/float/str sample id (bool rejected), "
                f"got {type(v).__name__}"
            )
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError(f"ids[{i}]: must be finite (NaN/Infinity rejected)")
        if v in seen:
            raise ValueError(f"ids[{i}]: duplicate sample id {v!r}")
        seen.add(v)
        out.append(v)
    return tuple(out)


def _summarize_from_factor(Xv, L, yv, hpv, var):
    """由已有 Cholesky 因子计算 lml/解析梯度/alpha_vec/K_inv 及残差诊断。"""
    n = len(yv)
    K, D2 = _kernel_and_d2(Xv, hpv)
    Ky = [row[:] for row in K]
    for i in range(n):
        Ky[i][i] += var
    avec = _cho_solve(L, yv)
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

    return lml, (g_log_a, g_log_l, g_log_s), avec, Inv, solve_res


class GPFit:
    """
    不可变 GP 拟合对象: 固定超参数与显式 jitter, 持有训练数据、稳定样本 ID
    与 Cholesky 因子。所有查询方法不修改对象; delete() 生成新对象,
    旧对象保持可查询。

    构造请用 fit_gp(X, y, hp, *, jitter=0.0, ids=None)。

    只读属性:
        n / ids / X / y / hyperparameters / jitter / effective_noise_variance
        status            - "ok" 或 "not_pd" (初始拟合即非正定)
        factor_source     - 本对象 Cholesky 因子来源:
                            "full"     初始全量分解;
                            "downdate" 由父对象经 Givens 旋转删除更新得到;
                            "refactor" 删除更新未通过数值验收 (或父对象 not_pd),
                                       已明确回退为对剩余样本的全量分解。
        downdate_residual - 删除更新因子 L_new 的 max|K_new - L_new L_new^T|
                            验收残差; 未做删除更新为 None。
    """

    __slots__ = (
        "_X", "_y", "_hp", "_jitter", "_var", "_ids", "_L", "_status",
        "_factor_source", "_downdate_residual",
    )

    def __init__(self, Xv, yv, hpv, jv, ids, L, status, factor_source,
                 downdate_residual):
        self._X = Xv
        self._y = yv
        self._hp = hpv
        self._jitter = jv
        self._var = hpv.noise * hpv.noise + jv
        self._ids = ids
        self._L = L
        self._status = status
        self._factor_source = factor_source
        self._downdate_residual = downdate_residual

    # -- 只读属性 ---------------------------------------------------------

    @property
    def n(self):
        return len(self._X)

    @property
    def ids(self):
        return self._ids

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def hyperparameters(self):
        return self._hp

    @property
    def jitter(self):
        return self._jitter

    @property
    def effective_noise_variance(self):
        return self._var

    @property
    def status(self):
        return self._status

    @property
    def factor_source(self):
        return self._factor_source

    @property
    def downdate_residual(self):
        return self._downdate_residual

    def __repr__(self):
        return (
            f"GPFit(n={self.n}, status={self._status!r}, "
            f"factor_source={self._factor_source!r}, ids={self._ids!r})"
        )

    # -- 查询 -------------------------------------------------------------

    def marginal(self) -> MarginalResult:
        """当前样本集上的 log 边际似然与解析梯度 (使用新样本数 n)。"""
        var = self._var
        if self._status == "not_pd":
            return MarginalResult(
                status="not_pd", log_marginal=None, gradient=None,
                alpha_vec=None, K_inv=None,
                symmetry_residual=_symmetry_residual(self._ky()),
                chol_residual=None, solve_residual=None,
                effective_noise_variance=var, jitter=self._jitter,
            )
        if self.n == 0:
            # 空样本集: log p(空 y) = 0, 梯度为零向量, 无矩阵可分解
            return MarginalResult(
                status="ok", log_marginal=0.0, gradient=(0.0, 0.0, 0.0),
                alpha_vec=(), K_inv=(),
                symmetry_residual=0.0, chol_residual=0.0, solve_residual=0.0,
                effective_noise_variance=var, jitter=self._jitter,
            )
        Ky = self._ky()
        lml, grad, avec, Inv, solve_res = _summarize_from_factor(
            self._X, self._L, self._y, self._hp, var
        )
        return MarginalResult(
            status="ok", log_marginal=lml, gradient=grad,
            alpha_vec=tuple(avec), K_inv=tuple(tuple(r) for r in Inv),
            symmetry_residual=_symmetry_residual(Ky),
            chol_residual=_chol_residual(Ky, self._L),
            solve_residual=solve_res,
            effective_noise_variance=var, jitter=self._jitter,
        )

    def predict(self, X_test) -> Prediction:
        """当前样本集上的 latent/observed 后验预测 (使用新样本数 n)。"""
        hpv = self._hp
        var = self._var
        d = len(self._X[0]) if self._X else None
        Xt, m, _ = _validate_matrix(
            X_test, "X_test", max_rows=MAX_TEST_POINTS, expect_dim=d
        )
        if self._status == "not_pd":
            return Prediction(
                status="not_pd", mean=None, latent_variance=None,
                observed_variance=None, raw_latent_variance=None, clamped=(),
                symmetry_residual=_symmetry_residual(self._ky()),
                solve_residual=None,
                effective_noise_variance=var, jitter=self._jitter,
            )
        a2 = hpv.alpha * hpv.alpha
        if self.n == 0:
            # 先验预测: 零均值, latent 方差 alpha^2, observed 再加噪声方差
            return Prediction(
                status="ok",
                mean=tuple(0.0 for _ in range(m)),
                latent_variance=tuple(a2 for _ in range(m)),
                observed_variance=tuple(a2 + var for _ in range(m)),
                raw_latent_variance=tuple(a2 for _ in range(m)),
                clamped=(),
                symmetry_residual=0.0, solve_residual=0.0,
                effective_noise_variance=var, jitter=self._jitter,
            )
        Ky = self._ky()
        avec = _cho_solve(self._L, self._y)
        Ks = _cross_kernel(self._X, Xt, hpv)
        n = self.n

        means = [0.0] * m
        raw_var = [0.0] * m
        latent = [0.0] * m
        observed = [0.0] * m
        clamped = []
        hard_negative = False
        solve_res = 0.0
        for t in range(m):
            kvec = [Ks[i][t] for i in range(n)]
            w = _cho_solve(self._L, kvec)
            means[t] = sum(kvec[i] * avec[i] for i in range(n))
            raw = a2 - sum(kvec[i] * w[i] for i in range(n))
            raw_var[t] = raw
            r = _max_abs([q - kvec[i] for i, q in enumerate(_matvec(Ky, w))])
            if r > solve_res:
                solve_res = r
            v, kind = classify_raw_variance(raw, a2)
            latent[t] = v
            observed[t] = v + var
            if kind == "clamped":
                clamped.append(t)
            elif kind == "negative":
                hard_negative = True
        return Prediction(
            status="negative_variance" if hard_negative else "ok",
            mean=tuple(means), latent_variance=tuple(latent),
            observed_variance=tuple(observed),
            raw_latent_variance=tuple(raw_var), clamped=tuple(clamped),
            symmetry_residual=_symmetry_residual(Ky), solve_residual=solve_res,
            effective_noise_variance=var, jitter=self._jitter,
        )

    # -- 删除 -------------------------------------------------------------

    def delete(self, sample_id) -> "GPFit":
        """
        按稳定样本 ID 删除一条样本, 返回新的 GPFit; 本对象不变、仍可查询。

        核参数与显式 jitter 原样保留; 剩余样本顺序压缩但 ID 不变、不复用。
        优先做 Cholesky 删除更新 (Givens 旋转, O(n^2)); 更新因子未通过
        数值验收 (或本对象 not_pd 无因子可用) 时明确回退全量分解,
        并在新对象的 factor_source / downdate_residual 诊断中记录。
        未知 ID 抛 KeyError。
        """
        if sample_id not in self._ids:
            raise KeyError(f"delete: unknown sample id {sample_id!r}")
        k = self._ids.index(sample_id)
        X_new = self._X[:k] + self._X[k + 1:]
        y_new = self._y[:k] + self._y[k + 1:]
        ids_new = self._ids[:k] + self._ids[k + 1:]

        if not X_new:
            # 删除最后一条: 空拟合对象, 预测退化为先验
            return GPFit((), (), self._hp, self._jitter, ids_new, None,
                         "ok", self._factor_source, None)

        if self._status == "ok" and self._L is not None:
            L_new = _chol_delete_row(self._L, k)
            Ky_new = _ky_of(X_new, self._hp, self._var)
            res = _chol_residual(Ky_new, L_new)
            ok = math.isfinite(res) and res <= DOWNDATE_REL_TOL * max(
                1.0, self._hp.alpha * self._hp.alpha + self._var
            )
            if ok:
                return GPFit(X_new, y_new, self._hp, self._jitter, ids_new,
                             L_new, "ok", "downdate", res)
            # 数值验收失败: 明确回退全量分解, 残差留在诊断中
            L_full = _cholesky(Ky_new)
            return GPFit(X_new, y_new, self._hp, self._jitter, ids_new,
                         L_full, "ok" if L_full is not None else "not_pd",
                         "refactor", res)

        # 本对象 not_pd (无可用因子): 对剩余样本全量分解, 可能恢复为正定
        L_full = _cholesky(_ky_of(X_new, self._hp, self._var))
        return GPFit(X_new, y_new, self._hp, self._jitter, ids_new,
                     L_full, "ok" if L_full is not None else "not_pd",
                     "refactor", None)

    # -- 内部 -------------------------------------------------------------

    def _ky(self):
        return _ky_of(self._X, self._hp, self._var)


def _ky_of(Xv, hpv, var):
    K, _ = _kernel_and_d2(Xv, hpv)
    for i in range(len(Xv)):
        K[i][i] += var
    return K


def fit_gp(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    hp: Hyperparameters,
    *,
    jitter: float = 0.0,
    ids: Optional[Sequence] = None,
) -> GPFit:
    """
    构造不可变 GPFit: 校验输入并做初始全量 Cholesky 分解。

    ids: 稳定样本 ID (int/float/str, 唯一, 非 bool, 数值须有限);
         缺省为 (0, 1, ..., n-1)。删除只按 ID 进行, ID 永不复用。
    """
    Xv, yv, hpv, jv, var = _prepare_training(X, y, hp, jitter)
    n = len(Xv)
    ids_v = _validate_ids(ids, n) if ids is not None else tuple(range(n))
    L = _cholesky(_ky_of(Xv, hpv, var))
    return GPFit(Xv, yv, hpv, jv, ids_v, L,
                 "ok" if L is not None else "not_pd", "full", None)
