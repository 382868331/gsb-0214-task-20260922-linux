"""
一维导数观测联合预测的独立参考测试。

本文件中的参考实现 (独立核公式 / Gauss-Jordan 直接逆 / 参考核的中心差分)
不导入也不调用 gprbf 的任何函数来产生期望值; gprbf 仅作为被测对象。
"""

import math
import unittest

import gprbf
from gprbf import (
    Hyperparameters,
    predict,
    predict_joint,
    rbf_kernel_df,
    rbf_kernel_dd,
)


# ---------------------------------------------------------------------------
# 独立参考实现 (与 gprbf 无关的另写代码, 供产生期望值)
# ---------------------------------------------------------------------------


def ref_k(x, z, a, ell):
    """一维 RBF 核 k(x, z)。"""
    return a * a * math.exp(-((x - z) ** 2) / (2.0 * ell * ell))


def ref_cov_f_d(x, z, a, ell):
    """Cov(f(x), f'(z)) = ∂k/∂z = k * (x - z) / ell^2 (独立公式)。"""
    return ref_k(x, z, a, ell) * (x - z) / (ell * ell)


def ref_cov_d_d(x, z, a, ell):
    """Cov(f'(x), f'(z)) = ∂²k/∂x∂z = k * (1/ell^2 - (x-z)^2/ell^4) (独立公式)。"""
    return ref_k(x, z, a, ell) * (1.0 / (ell * ell) - (x - z) ** 2 / (ell ** 4))


def ref_inverse(A):
    """带部分主元的 Gauss-Jordan 直接求逆 (独立实现)。"""
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = col
        best = abs(M[col][col])
        for r in range(col + 1, n):
            if abs(M[r][col]) > best:
                best = abs(M[r][col])
                piv = r
        if best <= 0.0:
            raise ZeroDivisionError("singular matrix in reference inverse")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if f != 0.0:
                M[r] = [M[r][k] - f * M[col][k] for k in range(2 * n)]
    return [row[n:] for row in M]


def ref_joint_covariance(X, Xd, a, ell, s, sd, jitter=0.0):
    """独立装配 (n+m) 阶联合协方差, 观测顺序 [y, yd]。"""
    n = len(X)
    m = len(Xd)
    N = n + m
    C = [[0.0] * N for _ in range(N)]
    for i in range(n):
        for j in range(n):
            C[i][j] = ref_k(X[i][0], X[j][0], a, ell)
        C[i][i] += s * s + jitter
    for i in range(n):
        for j in range(m):
            v = ref_cov_f_d(X[i][0], Xd[j][0], a, ell)
            C[i][n + j] = v
            C[n + j][i] = v
    for i in range(m):
        for j in range(m):
            C[n + i][n + j] = ref_cov_d_d(Xd[i][0], Xd[j][0], a, ell)
        C[n + i][n + i] += sd * sd + jitter
    return C


def ref_joint_predict_one(X, y, Xd, yd, xt, a, ell, s, sd, jitter=0.0):
    """
    单点联合预测参考: 显式逆矩阵公式 (独立期望值)。
    返回 (mean_f, latent_f, mean_d, latent_d)。
    """
    n = len(X)
    m = len(Xd)
    N = n + m
    C = ref_joint_covariance(X, Xd, a, ell, s, sd, jitter)
    Inv = ref_inverse(C)
    z = list(y) + list(yd)
    alpha = [sum(Inv[i][j] * z[j] for j in range(N)) for i in range(N)]
    kv = ([ref_k(X[i][0], xt, a, ell) for i in range(n)]
          + [ref_cov_f_d(xt, Xd[j][0], a, ell) for j in range(m)])
    kd = ([ref_cov_f_d(X[i][0], xt, a, ell) for i in range(n)]
          + [ref_cov_d_d(Xd[j][0], xt, a, ell) for j in range(m)])
    mean_f = sum(kv[i] * alpha[i] for i in range(N))
    mean_d = sum(kd[i] * alpha[i] for i in range(N))
    var_f = a * a - sum(kv[i] * sum(Inv[i][j] * kv[j] for j in range(N))
                        for i in range(N))
    var_d = a * a / (ell * ell) - sum(
        kd[i] * sum(Inv[i][j] * kd[j] for j in range(N)) for i in range(N)
    )
    return mean_f, var_f, mean_d, var_d


# ---------------------------------------------------------------------------
# 确定性数据 (解析函数合成: f = sin, f' = cos)
# ---------------------------------------------------------------------------


def make_joint_data():
    X = [(-2.0 + 0.5 * i,) for i in range(9)]
    y = [math.sin(x[0]) + 0.1 * x[0] for x in X]
    Xd = [(-1.5,), (-0.3,), (0.6,), (1.7,)]
    yd = [math.cos(x[0]) + 0.1 for x in Xd]
    return X, y, Xd, yd


HP = Hyperparameters(alpha=1.3, lengthscale=0.9, noise=0.15)
NOISE_D = 0.12


# ---------------------------------------------------------------------------
# 核导数块: 符号、反对称/对称、对参考核的中心差分
# ---------------------------------------------------------------------------


class TestKernelDerivativeBlocks(unittest.TestCase):
    def test_df_matches_central_difference_of_reference_kernel(self):
        # Cov(f(x), f'(z)) = ∂k/∂z, 对独立参考核做中心差分核对 (含符号)
        hp = HP
        Xf = [(-1.7,), (-0.2,), (0.9,)]
        Xd = [(-1.1,), (0.0,), (0.4,), (1.3,)]
        got = rbf_kernel_df(Xf, Xd, hp)
        h = 1e-6
        for i, (xf,) in enumerate(Xf):
            for j, (xd,) in enumerate(Xd):
                want = (ref_k(xf, xd + h, hp.alpha, hp.lengthscale)
                        - ref_k(xf, xd - h, hp.alpha, hp.lengthscale)) / (2 * h)
                self.assertAlmostEqual(got[i][j], want, delta=1e-8)

    def test_dd_matches_mixed_central_difference_of_reference_kernel(self):
        # Cov(f'(x), f'(z)) = ∂²k/∂x∂z, 对独立参考核做混合中心差分核对
        hp = HP
        X1 = [(-1.4,), (0.2,), (1.0,)]
        X2 = [(-0.8,), (0.5,)]
        got = rbf_kernel_dd(X1, X2, hp)
        h = 1e-4
        for i, (x,) in enumerate(X1):
            for j, (z,) in enumerate(X2):
                want = (ref_k(x + h, z + h, hp.alpha, hp.lengthscale)
                        - ref_k(x + h, z - h, hp.alpha, hp.lengthscale)
                        - ref_k(x - h, z + h, hp.alpha, hp.lengthscale)
                        + ref_k(x - h, z - h, hp.alpha, hp.lengthscale)) / (4 * h * h)
                self.assertAlmostEqual(got[i][j], want, delta=1e-5)

    def test_df_hand_values_sign_and_antisymmetry(self):
        # 手算: a=2, ell=0.5, x=1.0, z=0.0
        #   k = 4*exp(-2), Cov(f(x), f'(z)) = k*(x-z)/ell^2 = 16*exp(-2) > 0
        hp = Hyperparameters(2.0, 0.5, 0.3)
        got = rbf_kernel_df([(1.0,)], [(0.0,)], hp)[0][0]
        want = 16.0 * math.exp(-2.0)
        self.assertAlmostEqual(got, want, delta=1e-14)
        self.assertGreater(got, 0.0)  # x > z 时为正
        got_neg = rbf_kernel_df([(0.0,)], [(1.0,)], hp)[0][0]
        self.assertLess(got_neg, 0.0)  # x < z 时为负
        # 反对称: Cov(f(x), f'(z)) = -Cov(f(z), f'(x)), 同一表达式逐位相反
        self.assertEqual(got, -got_neg)
        # 同点处交叉协方差恰为 0 (r = 0)
        self.assertEqual(rbf_kernel_df([(0.7,)], [(0.7,)], hp)[0][0], 0.0)
        # 多块逐位反对称: df(A, B)[i][j] == -df(B, A)[j][i]
        A = [(-0.9,), (0.1,), (1.2,)]
        B = [(-0.4,), (0.8,)]
        ab = rbf_kernel_df(A, B, HP)
        ba = rbf_kernel_df(B, A, HP)
        for i in range(3):
            for j in range(2):
                self.assertEqual(ab[i][j], -ba[j][i])

    def test_dd_hand_values_symmetry_and_diagonal(self):
        # 手算: a=2, ell=0.5, x=1.0, z=0.0
        #   Cov(f'(x), f'(z)) = 4*exp(-2) * (1/0.25 - 1/0.0625) = -48*exp(-2)
        hp = Hyperparameters(2.0, 0.5, 0.3)
        got = rbf_kernel_dd([(1.0,)], [(0.0,)], hp)[0][0]
        want = -48.0 * math.exp(-2.0)
        self.assertAlmostEqual(got, want, delta=1e-13)
        self.assertLess(got, 0.0)  # |r| > ell 时为负
        # 对角恰为 a^2/ell^2 (先验导数方差)
        self.assertEqual(rbf_kernel_dd([(0.3,)], [(0.3,)], hp)[0][0],
                         2.0 ** 2 / 0.5 ** 2)
        # 同集自协方差逐位对称
        X = [(-1.1,), (0.0,), (0.6,), (1.4,)]
        K = rbf_kernel_dd(X, X, HP)
        for i in range(4):
            for j in range(4):
                self.assertEqual(K[i][j], K[j][i])
        # 交叉块转置一致: dd(A, B)[i][j] == dd(B, A)[j][i]
        A = [(-0.9,), (0.1,), (1.2,)]
        B = [(-0.4,), (0.8,)]
        ab = rbf_kernel_dd(A, B, HP)
        ba = rbf_kernel_dd(B, A, HP)
        for i in range(3):
            for j in range(2):
                self.assertEqual(ab[i][j], ba[j][i])


# ---------------------------------------------------------------------------
# 联合预测: 独立稠密参考 / 手算 2x2 / 解析关系
# ---------------------------------------------------------------------------


class TestJointPredictionAgainstReference(unittest.TestCase):
    def test_joint_matches_dense_reference(self):
        X, y, Xd, yd = make_joint_data()
        Xt = [(-1.7,), (-0.5,), (0.0,), (0.9,), (2.0,)]
        res = predict_joint(X, y, Xd, yd, Xt, HP, noise_deriv=NOISE_D)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.symmetry_residual, 0.0)
        self.assertLess(res.solve_residual, 1e-12)
        for t, (xt,) in enumerate(Xt):
            mf, vf, md, vd = ref_joint_predict_one(
                X, y, Xd, yd, xt, HP.alpha, HP.lengthscale, HP.noise, NOISE_D
            )
            self.assertAlmostEqual(res.mean[t], mf, delta=1e-10)
            self.assertAlmostEqual(res.latent_variance[t], vf, delta=1e-9)
            self.assertAlmostEqual(res.deriv_mean[t], md, delta=1e-10)
            self.assertAlmostEqual(res.deriv_latent_variance[t], vd, delta=1e-9)
            self.assertAlmostEqual(
                res.observed_variance[t], vf + HP.noise ** 2, delta=1e-9
            )
            self.assertAlmostEqual(
                res.deriv_observed_variance[t], vd + NOISE_D ** 2, delta=1e-9
            )

    def test_joint_with_jitter_matches_dense_reference(self):
        X, y, Xd, yd = make_joint_data()
        j = 1e-3
        res = predict_joint(X, y, Xd, yd, [(0.3,)], HP, noise_deriv=NOISE_D, jitter=j)
        self.assertEqual(res.status, "ok")
        mf, vf, md, vd = ref_joint_predict_one(
            X, y, Xd, yd, 0.3, HP.alpha, HP.lengthscale, HP.noise, NOISE_D, jitter=j
        )
        self.assertAlmostEqual(res.mean[0], mf, delta=1e-10)
        self.assertAlmostEqual(res.latent_variance[0], vf, delta=1e-9)
        self.assertAlmostEqual(res.deriv_mean[0], md, delta=1e-10)
        self.assertAlmostEqual(res.deriv_latent_variance[0], vd, delta=1e-9)
        # jitter 作为方差项同时进入两类观测方差
        self.assertAlmostEqual(
            res.observed_variance[0] - res.latent_variance[0],
            HP.noise ** 2 + j, delta=1e-15)
        self.assertAlmostEqual(
            res.deriv_observed_variance[0] - res.deriv_latent_variance[0],
            NOISE_D ** 2 + j, delta=1e-15)
        self.assertEqual(res.effective_noise_variance, HP.noise ** 2 + j)
        self.assertEqual(res.effective_deriv_noise_variance, NOISE_D ** 2 + j)

    def test_single_value_single_derivative_hand_computed(self):
        # 手算例子: 同点 x=0 处一个函数观测 + 一个导数观测。
        # r=0 => 交叉协方差为 0, 联合协方差为对角阵, 可完全手算。
        a, ell, s, sd = 1.5, 2.0, 0.5, 0.25
        hp = Hyperparameters(a, ell, s)
        y0, yd0 = 1.2, -0.7
        res = predict_joint([(0.0,)], [y0], [(0.0,)], [yd0], [(0.0,), (0.8,)],
                            hp, noise_deriv=sd)
        self.assertEqual(res.status, "ok")
        a2 = a * a
        ell2 = ell * ell
        c00 = a2 + s * s          # Var 观测 y0
        c11 = a2 / ell2 + sd * sd  # Var 观测 yd0
        # --- 测试点 x* = 0: k* = (a^2, 0), kd* = (0, a^2/ell^2) ---
        self.assertAlmostEqual(res.mean[0], a2 * y0 / c00, delta=1e-14)
        self.assertAlmostEqual(res.latent_variance[0],
                               a2 - a2 * a2 / c00, delta=1e-14)
        self.assertAlmostEqual(res.deriv_mean[0], (a2 / ell2) * yd0 / c11, delta=1e-14)
        self.assertAlmostEqual(res.deriv_latent_variance[0],
                               a2 / ell2 - (a2 / ell2) ** 2 / c11, delta=1e-14)
        # --- 测试点 x* = 0.8: 手算 k*, kd* 与对角逆 ---
        xs = 0.8
        k = a2 * math.exp(-(xs * xs) / (2 * ell2))
        kv0, kv1 = k, k * xs / ell2            # Cov(f(0),f*), Cov(f'(0),f*)
        kd0 = k * (0.0 - xs) / ell2            # Cov(f(0), f'*)
        kd1 = k * (1.0 / ell2 - xs * xs / (ell2 * ell2))  # Cov(f'(0), f'*)
        self.assertAlmostEqual(res.mean[1], kv0 * y0 / c00 + kv1 * yd0 / c11,
                               delta=1e-14)
        self.assertAlmostEqual(res.deriv_mean[1], kd0 * y0 / c00 + kd1 * yd0 / c11,
                               delta=1e-14)
        self.assertAlmostEqual(res.latent_variance[1],
                               a2 - (kv0 * kv0 / c00 + kv1 * kv1 / c11), delta=1e-14)
        self.assertAlmostEqual(res.deriv_latent_variance[1],
                               a2 / ell2 - (kd0 * kd0 / c00 + kd1 * kd1 / c11),
                               delta=1e-14)

    def test_derivative_mean_is_derivative_of_reference_mean(self):
        # 解析关系: E[f'(x*)|z] = d/dx* E[f(x*)|z]。
        # 对独立参考均值函数做中心差分 (不调用被测实现产生期望值)。
        X, y, Xd, yd = make_joint_data()
        Xt = [(-1.3,), (0.2,), (1.4,)]
        res = predict_joint(X, y, Xd, yd, Xt, HP, noise_deriv=NOISE_D)
        self.assertEqual(res.status, "ok")
        h = 1e-6
        for t, (xt,) in enumerate(Xt):
            mp = ref_joint_predict_one(X, y, Xd, yd, xt + h,
                                       HP.alpha, HP.lengthscale, HP.noise, NOISE_D)[0]
            mm = ref_joint_predict_one(X, y, Xd, yd, xt - h,
                                       HP.alpha, HP.lengthscale, HP.noise, NOISE_D)[0]
            self.assertAlmostEqual(res.deriv_mean[t], (mp - mm) / (2 * h), delta=1e-6)

    def test_empty_derivatives_reduces_to_predict(self):
        # Xd/yd 同时为空: 退化为纯函数值观测, 与旧入口 predict 完全一致
        X, y, Xd, yd = make_joint_data()
        Xt = [(-1.1,), (0.3,), (1.5,)]
        pj = predict_joint(X, y, [], [], Xt, HP, noise_deriv=0.5)
        p0 = predict(X, y, Xt, HP)
        self.assertEqual(pj.status, p0.status)
        self.assertEqual(pj.mean, p0.mean)
        self.assertEqual(pj.latent_variance, p0.latent_variance)
        self.assertEqual(pj.observed_variance, p0.observed_variance)
        self.assertEqual(pj.clamped, p0.clamped)
        # 导数后验仍存在 (由函数观测经交叉协方差约束), 且与参考一致
        mf, vf, md, vd = ref_joint_predict_one(
            X, y, [], [], 0.3, HP.alpha, HP.lengthscale, HP.noise, 0.5
        )
        self.assertAlmostEqual(pj.deriv_mean[1], md, delta=1e-10)
        self.assertAlmostEqual(pj.deriv_latent_variance[1], vd, delta=1e-9)

    def test_derivative_observations_reduce_latent_variance(self):
        # 解析关系: 增加观测不会增大隐后验方差 (条件方差单调不增)
        X, y, Xd, yd = make_joint_data()
        Xt = [(-1.75 + 0.25 * i,) for i in range(15)]
        pj = predict_joint(X, y, Xd, yd, Xt, HP, noise_deriv=NOISE_D)
        p0 = predict(X, y, Xt, HP)
        self.assertEqual(pj.status, "ok")
        strictly_less = 0
        for t in range(len(Xt)):
            self.assertLessEqual(pj.latent_variance[t],
                                 p0.latent_variance[t] + 1e-12)
            if pj.latent_variance[t] < p0.latent_variance[t] - 1e-9:
                strictly_less += 1
        self.assertGreater(strictly_less, 0)

    def test_analytic_function_recovery(self):
        # 解析函数合成样本: y=sin, yd=cos (无噪声合成, 小观测噪声),
        # 联合后验应同时逼近 sin 与 cos, 且优于纯函数值观测
        X = [(-2.0 + 0.5 * i,) for i in range(9)]
        y = [math.sin(x[0]) for x in X]
        Xd = [(-1.5,), (-0.5,), (0.5,), (1.5,)]
        yd = [math.cos(x[0]) for x in Xd]
        hp = Hyperparameters(1.2, 0.8, 0.05)
        Xt = [(-1.8 + 0.2 * i,) for i in range(19)]
        pj = predict_joint(X, y, Xd, yd, Xt, hp, noise_deriv=0.05)
        p0 = predict(X, y, Xt, hp)
        self.assertEqual(pj.status, "ok")
        sse_joint = sum((m - math.sin(x[0])) ** 2 for m, x in zip(pj.mean, Xt))
        sse_plain = sum((m - math.sin(x[0])) ** 2 for m, x in zip(p0.mean, Xt))
        self.assertLess(sse_joint, sse_plain)
        sse_deriv = sum((m - math.cos(x[0])) ** 2
                        for m, x in zip(pj.deriv_mean, Xt))
        self.assertLess(sse_deriv, 0.05)  # 19 点导数误差平方和小量级


# ---------------------------------------------------------------------------
# 边界与错误语义
# ---------------------------------------------------------------------------


class TestJointEdgeAndErrors(unittest.TestCase):
    def test_duplicate_derivative_points_tiny_noise_not_pd(self):
        # 重复导数点 + 极小噪声: d-d 块出现重复行 => 精确奇异 => not_pd
        X = [(0.0,), (1.0,)]
        y = [0.5, -0.5]
        Xd = [(0.5,), (0.5,)]
        yd = [0.3, 0.3]
        hp = Hyperparameters(1.0, 1.0, 1e-10)
        res = predict_joint(X, y, Xd, yd, [(0.5,)], hp, noise_deriv=1e-10)
        self.assertEqual(res.status, "not_pd")
        self.assertIsNone(res.mean)
        self.assertIsNone(res.deriv_mean)
        self.assertIsNone(res.latent_variance)
        self.assertIsNone(res.deriv_latent_variance)
        self.assertIsNone(res.solve_residual)
        # 有效噪声方差字段仍按实填充
        self.assertEqual(res.effective_noise_variance, 1e-10 ** 2)
        self.assertEqual(res.effective_deriv_noise_variance, 1e-10 ** 2)

    def test_explicit_jitter_recovers_not_pd(self):
        X = [(0.0,), (1.0,)]
        y = [0.5, -0.5]
        Xd = [(0.5,), (0.5,)]
        yd = [0.3, 0.3]
        hp = Hyperparameters(1.0, 1.0, 1e-10)
        res = predict_joint(X, y, Xd, yd, [(0.5,)], hp,
                            noise_deriv=1e-10, jitter=1e-6)
        self.assertEqual(res.status, "ok")
        self.assertTrue(all(math.isfinite(v) for v in res.mean))
        self.assertTrue(all(math.isfinite(v) for v in res.deriv_mean))

    def test_multidim_rejected(self):
        hp = HP
        with self.assertRaisesRegex(ValueError, "1D"):
            predict_joint([(0.0, 1.0), (1.0, 0.0)], [0.0, 1.0], [], [],
                          [(0.5, 0.5)], hp, noise_deriv=0.1)
        with self.assertRaisesRegex(ValueError, "1D"):
            predict_joint([(0.0,), (1.0,)], [0.0, 1.0], [(0.5, 1.0,)], [0.1],
                          [(0.5,)], hp, noise_deriv=0.1)
        with self.assertRaisesRegex(ValueError, "1D"):
            predict_joint([(0.0,), (1.0,)], [0.0, 1.0], [], [],
                          [(0.5, 0.5)], hp, noise_deriv=0.1)
        with self.assertRaisesRegex(ValueError, "1D"):
            rbf_kernel_df([(0.0, 1.0)], [(0.0,)], hp)
        with self.assertRaisesRegex(ValueError, "1D"):
            rbf_kernel_dd([(0.0,)], [(0.0, 1.0)], hp)

    def test_length_mismatch_and_partial_empty(self):
        hp = HP
        X, y = [(0.0,), (1.0,)], [0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "yd"):
            predict_joint(X, y, [(0.5,), (0.7,)], [0.1], [(0.5,)], hp,
                          noise_deriv=0.1)
        with self.assertRaisesRegex(ValueError, "yd"):
            predict_joint(X, y, [(0.5,)], [], [(0.5,)], hp, noise_deriv=0.1)
        with self.assertRaisesRegex(ValueError, "yd"):
            predict_joint(X, y, [], [0.1], [(0.5,)], hp, noise_deriv=0.1)
        with self.assertRaisesRegex(ValueError, "y"):
            predict_joint(X, [0.0], [], [], [(0.5,)], hp, noise_deriv=0.1)

    def test_noise_deriv_validation(self):
        X, y = [(0.0,), (1.0,)], [0.0, 1.0]
        for bad in (0.0, -0.1):
            with self.assertRaisesRegex(ValueError, "noise_deriv"):
                predict_joint(X, y, [], [], [(0.5,)], HP, noise_deriv=bad)
        with self.assertRaisesRegex(ValueError, "noise_deriv"):
            predict_joint(X, y, [], [], [(0.5,)], HP, noise_deriv=float("nan"))
        with self.assertRaises(TypeError):
            predict_joint(X, y, [], [], [(0.5,)], HP, noise_deriv=True)
        with self.assertRaises(TypeError):
            predict_joint(X, y, [], [], [(0.5,)], HP, noise_deriv="0.1")
        # noise_deriv 是必填关键字参数
        with self.assertRaises(TypeError):
            predict_joint(X, y, [], [], [(0.5,)], HP)  # noqa

    def test_total_observation_limit(self):
        X = [(float(i) * 0.1,) for i in range(60)]
        y = [0.0] * 60
        Xd = [(float(i) * 0.1,) for i in range(21)]
        yd = [0.0] * 21
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            predict_joint(X, y, Xd, yd, [(0.0,)], HP, noise_deriv=0.1)
        # 恰好 80 合法
        res = predict_joint(X, y, Xd[:20], yd[:20], [(0.0,)], HP, noise_deriv=0.1)
        self.assertIn(res.status, ("ok", "not_pd"))

    def test_nan_inf_and_bool_rejected_with_index(self):
        X, y = [(0.0,), (1.0,)], [0.0, 1.0]
        with self.assertRaisesRegex(ValueError, r"Xd\[1\]\[0\]"):
            predict_joint(X, y, [(0.5,), (float("nan"),)], [0.1, 0.2],
                          [(0.5,)], HP, noise_deriv=0.1)
        with self.assertRaisesRegex(ValueError, r"yd\[0\]"):
            predict_joint(X, y, [(0.5,)], [float("inf")], [(0.5,)], HP,
                          noise_deriv=0.1)
        with self.assertRaises(TypeError):
            predict_joint(X, y, [(True,)], [0.1], [(0.5,)], HP, noise_deriv=0.1)
        with self.assertRaises(TypeError):
            predict_joint(X, y, None, None, [(0.5,)], HP, noise_deriv=0.1)
        with self.assertRaises(TypeError):
            predict_joint(X, y, [(0.5,)], [0.1], [(0.5,)], (1.0, 1.0, 0.1),
                          noise_deriv=0.1)

    def test_negative_jitter_rejected(self):
        X, y = [(0.0,), (1.0,)], [0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "jitter"):
            predict_joint(X, y, [], [], [(0.5,)], HP, noise_deriv=0.1,
                          jitter=-1e-9)

    def test_inputs_not_mutated_and_deterministic(self):
        X, y, Xd, yd = make_joint_data()
        Xt = [(-0.4,), (0.8,)]
        snapshot = ([tuple(r) for r in X], list(y),
                    [tuple(r) for r in Xd], list(yd), [tuple(r) for r in Xt])
        r1 = predict_joint(X, y, Xd, yd, Xt, HP, noise_deriv=NOISE_D)
        r2 = predict_joint(X, y, Xd, yd, Xt, HP, noise_deriv=NOISE_D)
        self.assertEqual(r1, r2)
        self.assertEqual(([tuple(r) for r in X], list(y),
                          [tuple(r) for r in Xd], list(yd),
                          [tuple(r) for r in Xt]), snapshot)

    def test_kernel_block_validation(self):
        hp = HP
        with self.assertRaises(TypeError):
            rbf_kernel_df([(0.0,)], [(True,)], hp)
        with self.assertRaisesRegex(ValueError, r"X_d\[0\]\[0\]"):
            rbf_kernel_df([(0.0,)], [(float("nan"),)], hp)
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            rbf_kernel_dd([(float(i),) for i in range(81)], [(0.0,)], hp)
        with self.assertRaises(TypeError):
            rbf_kernel_df([(0.0,)], [(0.1,)], (1.0, 1.0, 0.1))  # noqa


if __name__ == "__main__":
    unittest.main()
