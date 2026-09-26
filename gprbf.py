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
    "MAX_JOINT_SAMPLE_POINTS",
    "NEG_VAR_REL_TOL",
    "PSD_EIGEN_REL_TOL",
    "Hyperparameters",
    "MarginalResult",
    "Prediction",
    "SampleResult",
    "log_marginal",
    "predict",
    "sample_posterior",
    "rbf_kernel",
    "classify_raw_variance",
]

#: 训练样本数硬上限（题面接口边界）。
MAX_SAMPLES = 80
#: 输入维数硬上限（题面接口边界）。
MAX_DIM = 3
#: 预测点数量的保护性上限（README 中记录的防护，非题面要求）。
MAX_TEST_POINTS = 10_000
#: 联合函数抽样需要显式持有完整 m×m 后验协方差；标准库实现限制 m 为 500。
MAX_JOINT_SAMPLE_POINTS = 80
#: 半正定矩阵特征值的相对数值容差。
PSD_EIGEN_REL_TOL = 1e-11
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
class SampleResult:
    """联合高斯函数抽样结果；status 非 ok 时计算字段为 None。"""

    status: str
    samples: Optional[Tuple[Tuple[float, ...], ...]]
    mean: Optional[Tuple[float, ...]]
    covariance: Optional[Tuple[Tuple[float, ...], ...]]
    rank: Optional[int]
    corrected_eigenvalues: Tuple[float, ...]
    covariance_residual: Optional[float]
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
    allow_empty: bool = False,
) -> Tuple[Tuple[Tuple[float, ...], ...], int, int]:
    if not isinstance(X, (list, tuple)):
        raise TypeError(f"{name}: expected list/tuple of rows, got {type(X).__name__}")
    if len(X) == 0:
        if allow_empty:
            if expect_dim is None:
                raise ValueError(f"{name}: empty input requires an explicit expected dimension")
            return (), 0, expect_dim
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


def _symmetric_eigensystem(A, sweeps: int = 80):
    """实对称矩阵的循环 Jacobi 特征分解：A = V diag(w) V^T。"""

    n = len(A)
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    W = [list(row) for row in A]

    for _ in range(sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off = max(off, abs(W[p][q]))
        if off <= 1e-14:
            break

        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = W[p][q]
                if abs(apq) <= 1e-15:
                    continue
                tau = (W[q][q] - W[p][p]) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                for k in range(n):
                    wkp = W[k][p]
                    wkq = W[k][q]
                    W[k][p] = c * wkp - s * wkq
                    W[k][q] = s * wkp + c * wkq
                for k in range(n):
                    wpk = W[p][k]
                    wqk = W[q][k]
                    W[p][k] = c * wpk - s * wqk
                    W[q][k] = s * wpk + c * wqk
                for k in range(n):
                    vkp = V[k][p]
                    vkq = V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq

    values = [W[i][i] for i in range(n)]
    scale = max([abs(v) for v in values] + [abs(A[i][j]) for i in range(n) for j in range(n)] + [1.0])
    zero_tol = PSD_EIGEN_REL_TOL * scale
    rank = sum(1 for v in values if v > zero_tol)
    residual = _symmetry_residual(W)
    for i in range(n):
        residual = max(residual, abs(W[i][i] - values[i]))
        for j in range(i + 1, n):
            residual = max(residual, abs(W[i][j]), abs(W[j][i]))
    return tuple(values), V, rank, zero_tol, residual


def _psd_factor(A):
    """
    返回 (eigenvalues, vectors, rank, corrected, residual)。

    秩内特征向量构成平方根 C = V_r diag(sqrt(lambda_r)) (...)^T；
    零特征方向原样保留。明显负特征值返回 None。
    """

    n = len(A)
    if n == 0:
        return (), (), 0, (), 0.0

    values, V, rank, zero_tol, residual = _symmetric_eigensystem(A)
    corrected = []
    for i, value in enumerate(values):
        if value < 0.0:
            if value < -zero_tol:
                return None
            corrected.append(value)
            values = list(values)
            values[i] = 0.0
            values = tuple(values)

    reconstructed = _factor_covariance(V, values, rank)
    residual = max(residual, max(
        abs(reconstructed[i][j] - A[i][j]) for i in range(n) for j in range(n)
    ))
    return tuple(values), V, rank, tuple(corrected), residual


def _factor_covariance(V, values, rank):
    positive = [i for i, value in enumerate(values) if value > 0.0]
    positive = positive[:rank] if rank < len(positive) else positive
    n = len(V)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            total = 0.0
            for q in positive:
                total += V[i][q] * V[j][q] * values[q]
            C[i][j] = total
    return C


def _factor_transform(V, values, rank, Z):
    positive = [i for i, value in enumerate(values) if value > 0.0]
    positive = positive[:rank] if rank < len(positive) else positive
    scales = [math.sqrt(values[i]) for i in positive]
    out = []
    for zrow in Z:
        row = []
        for i in range(len(V)):
            total = 0.0
            for col, q in enumerate(positive):
                total += V[i][q] * scales[col] * zrow[col]
            row.append(total)
        out.append(tuple(row))
    return tuple(out)


def _psd_pseudoinverse(values, V, rank):
    positive = [i for i, value in enumerate(values) if value > 0.0][:rank]
    n = len(V)
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            inv[i][j] = sum(V[i][q] * V[j][q] / values[q] for q in positive)
    return inv


def _null_residual(values, V, rank, vector):
    positive = {i for i, value in enumerate(values) if value > 0.0}
    positive = set(sorted(positive)[:rank])
    result = 0.0
    n = len(V)
    for q, value in enumerate(values):
        if q in positive:
            continue
        coefficient = sum(V[i][q] * vector[i] for i in range(n))
        result = max(result, abs(coefficient))
    return result


def _validate_standard_matrix(Z, name, rows: Optional[int], cols: int):
    if not isinstance(Z, (list, tuple)):
        raise TypeError(f"{name}: expected list/tuple rows, got {type(Z).__name__}")
    out = []
    for i, row in enumerate(Z):
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"{name}[{i}]: expected list/tuple row")
        if len(row) != cols:
            raise ValueError(
                f"{name}[{i}]: length {len(row)} != number of query points {cols}"
            )
        out.append(tuple(_finite_float(v, f"{name}[{i}][{j}]") for j, v in enumerate(row)))
    if rows is not None and len(out) != rows:
        raise ValueError(f"{name}: expected {rows} rows, got {len(out)}")
    return tuple(out)


def _validate_fixed_values(fixed_values, m: int):
    if fixed_values is None:
        return {}
    if not isinstance(fixed_values, dict):
        raise TypeError("fixed_values: expected dict {index: value}")
    result = {}
    for key, value in fixed_values.items():
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError(f"fixed_values key {key!r}: expected int")
        if not 0 <= key < m:
            raise ValueError(f"fixed_values key {key}: outside 0..{m - 1}")
        result[key] = _finite_float(value, f"fixed_values[{key}]")
    return result


def _posterior_with_singular_training(Xv, yv, Xt, hpv, var):
    n = len(Xv)
    K, _ = _kernel_and_d2(Xv, hpv)
    Ky = [row[:] for row in K]
    for i in range(n):
        Ky[i][i] += var
    training = _psd_factor(Ky)
    if training is None:
        return None
    values, V, rank, corrected, residual = training
    mismatch = _null_residual(values, V, rank, yv)
    scale = max([1.0] + [abs(Ky[i][j]) for i in range(n) for j in range(n)])
    if mismatch > PSD_EIGEN_REL_TOL * scale:
        return "incompatible_observation"

    K_inv = _psd_pseudoinverse(values, V, rank)
    Ks = _cross_kernel(Xv, Xt, hpv) if Xt else []
    Ktt, _ = _kernel_and_d2(Xt, hpv) if Xt else ([], [])
    m = len(Xt)
    mean = []
    for t in range(m):
        mean.append(sum(Ks[i][t] * sum(K_inv[i][j] * yv[j] for j in range(n)) for i in range(n)))
    cov = []
    for t in range(m):
        row = []
        for u in range(m):
            reduction = sum(
                Ks[i][t] * sum(K_inv[i][j] * Ks[j][u] for j in range(n))
                for i in range(n)
            )
            value = Ktt[t][u] - reduction
            if u > t:
                mirror = Ktt[u][t] - sum(
                    Ks[i][u] * sum(K_inv[i][j] * Ks[j][t] for j in range(n))
                    for i in range(n)
                )
                value = 0.5 * (value + mirror)
            elif u < t:
                value = cov[u][t]
            row.append(value)
        cov.append(row)
    return mean, cov, corrected, residual


def _condition_factorized(mean, values, V, rank, fixed):
    n = len(V)
    fixed_idx = tuple(sorted(fixed))
    fixed_set = set(fixed_idx)
    free_idx = tuple(i for i in range(n) if i not in fixed_set)
    positive = [q for q, value in enumerate(values) if value > 0.0][:rank]
    U = [[V[i][q] * math.sqrt(values[q]) for q in positive] for i in range(n)]

    if not fixed_idx:
        free_cov = [
            [sum(U[i][q] * U[j][q] for q in range(len(positive))) for j in free_idx]
            for i in free_idx
        ]
        return tuple(mean), free_cov, free_idx, (), 0.0

    E = [[U[i][q] for q in range(len(positive))] for i in fixed_idx]
    B = [
        [sum(E[a][q] * E[c][q] for q in range(len(positive))) for c in range(len(fixed_idx))]
        for a in range(len(fixed_idx))
    ]
    bfact = _psd_factor(B)
    if bfact is None:
        return None
    bvalues, bV, brank, bcorrected, bresidual = bfact
    delta = [fixed[i] - mean[i] for i in fixed_idx]
    mismatch = _null_residual(bvalues, bV, brank, delta)
    scale = max([1.0] + [abs(B[i][j]) for i in range(len(B)) for j in range(len(B))])
    if mismatch > PSD_EIGEN_REL_TOL * scale:
        raise FloatingPointError("incompatible zero-variance observation")
    B_inv = _psd_pseudoinverse(bvalues, bV, brank)
    rvec = []
    for q in range(len(positive)):
        rvec.append(sum(
            E[a][q] * sum(B_inv[a][c] * delta[c] for c in range(len(fixed_idx)))
            for a in range(len(fixed_idx))
        ))
    shifted = [mean[i] + sum(U[i][q] * rvec[q] for q in range(len(positive))) for i in range(n)]

    projector = [[0.0] * len(positive) for _ in range(len(positive))]
    for q in range(len(positive)):
        projector[q][q] = 1.0
    for q in range(len(positive)):
        for p in range(len(positive)):
            adjustment = sum(
                E[a][q] * sum(B_inv[a][c] * E[c][p] for c in range(len(fixed_idx)))
                for a in range(len(fixed_idx))
            )
            projector[q][p] -= adjustment
    free_cov = []
    for ii, i in enumerate(free_idx):
        row = []
        for jj, j in enumerate(free_idx):
            total = 0.0
            for q in range(len(positive)):
                for p in range(len(positive)):
                    total += U[i][q] * projector[q][p] * U[j][p]
            row.append(total)
        free_cov.append(row)
    return tuple(shifted[i] for i in free_idx), free_cov, free_idx, bcorrected, bresidual


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


def sample_posterior(
    X,
    y,
    X_test,
    hp,
    standard_normal,
    *,
    jitter: float = 0.0,
    fixed_values: Optional[dict] = None,
) -> SampleResult:
    """
    一次抽出整条相关隐函数样本：f = mean + C Z^T，C C^T 是完整后验协方差。

    standard_normal 的一行对应一个样本，列数等于未固定查询点数；本函数不读取
    任何随机种子或全局 RNG。重复查询点产生半正定奇异协方差时，使用带秩判断
    的 Jacobi 特征分解；小负特征值只在相对容差内修正到 0 并记录，不添加噪声。
    """

    allow_zero_noise = True
    Xv, n, dim = _validate_matrix(X, "X", max_rows=MAX_SAMPLES)
    yv = _validate_vector(y, "y", n)
    hpv = _validate_hp(hp, allow_zero_noise=allow_zero_noise)
    jv = _validate_jitter(jitter)
    var = hpv.noise * hpv.noise + jv
    Xt, m, _ = _validate_matrix(
        X_test,
        "X_test",
        max_rows=MAX_JOINT_SAMPLE_POINTS,
        expect_dim=dim,
        allow_empty=True,
    )
    fixed = _validate_fixed_values(fixed_values, m)

    if m == 0:
        if fixed:
            raise ValueError("fixed_values: cannot index an empty query")
        zrows = _validate_standard_matrix(standard_normal, "standard_normal", None, 0)
        return SampleResult(
            status="ok",
            samples=tuple(() for _ in zrows),
            mean=(),
            covariance=(),
            rank=0,
            corrected_eigenvalues=(),
            covariance_residual=0.0,
            effective_noise_variance=var,
            jitter=jv,
        )

    posterior = _posterior_with_singular_training(Xv, yv, Xt, hpv, var)
    if posterior is None:
        return SampleResult(
            status="not_psd", samples=None, mean=None, covariance=None, rank=None,
            corrected_eigenvalues=(), covariance_residual=None,
            effective_noise_variance=var, jitter=jv,
        )
    if posterior == "incompatible_observation":
        return SampleResult(
            status="incompatible_observation", samples=None, mean=None,
            covariance=None, rank=None, corrected_eigenvalues=(),
            covariance_residual=None, effective_noise_variance=var, jitter=jv,
        )
    prior_mean, prior_cov, train_corrected, train_residual = posterior
    exact_values = {}
    train_by_point = {Xv[i]: yv[i] for i in range(n)}
    for t in range(m):
        if Xt[t] in train_by_point:
            exact_value = train_by_point[Xt[t]]
            prior_mean[t] = exact_value
            exact_values[t] = exact_value

    fact = _psd_factor(prior_cov)
    if fact is None:
        return SampleResult(
            status="not_psd", samples=None, mean=None, covariance=None, rank=None,
            corrected_eigenvalues=tuple(train_corrected), covariance_residual=None,
            effective_noise_variance=var, jitter=jv,
        )
    values, V, rank, query_corrected, query_residual = fact

    try:
        cond_mean, cond_cov, free_idx, fixed_corrected, fixed_residual = _condition_factorized(
            prior_mean, values, V, rank, fixed
        )
    except FloatingPointError:
        return SampleResult(
            status="incompatible_observation",
            samples=None, mean=None, covariance=None, rank=None,
            corrected_eigenvalues=tuple(train_corrected) + tuple(query_corrected),
            covariance_residual=None, effective_noise_variance=var, jitter=jv,
        )

    free_fact = _psd_factor(cond_cov)
    if free_fact is None:
        return SampleResult(
            status="not_psd", samples=None, mean=None, covariance=None, rank=None,
            corrected_eigenvalues=(
            tuple(train_corrected) + tuple(query_corrected) + tuple(fixed_corrected),
            ),
            covariance_residual=None, effective_noise_variance=var, jitter=jv,
        )
    free_values, free_V, free_rank, cond_corrected, cond_residual = free_fact
    zrows = _validate_standard_matrix(
        standard_normal, "standard_normal", None, len(free_idx)
    )
    free_noise = _factor_transform(free_V, free_values, free_rank, zrows)

    samples = []
    for row_noise in free_noise:
        sample = [0.0] * m
        for position, query_index in enumerate(free_idx):
            sample[query_index] = cond_mean[position] + row_noise[position]
        for query_index, value in fixed.items():
            sample[query_index] = value
        samples.append(tuple(sample))

    full_mean = [0.0] * m
    full_cov = [[0.0] * m for _ in range(m)]
    for position, query_index in enumerate(free_idx):
        full_mean[query_index] = cond_mean[position]
        for other_position, other_index in enumerate(free_idx):
            full_cov[query_index][other_index] = cond_cov[position][other_position]
    for query_index, value in fixed.items():
        full_mean[query_index] = value
    for query_index, value in exact_values.items():
        if query_index not in fixed:
            full_mean[query_index] = value
    residual = max(
        train_residual, query_residual, fixed_residual, cond_residual,
        _symmetry_residual(prior_cov), _symmetry_residual(full_cov),
    )
    all_corrected = tuple(train_corrected) + tuple(query_corrected) + tuple(fixed_corrected) + tuple(cond_corrected)
    return SampleResult(
        status="ok",
        samples=tuple(samples),
        mean=tuple(full_mean),
        covariance=tuple(tuple(row) for row in full_cov),
        rank=free_rank,
        corrected_eigenvalues=all_corrected,
        covariance_residual=residual,
        effective_noise_variance=var,
        jitter=jv,
    )
