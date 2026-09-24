"""
独立参考测试: 本文件中的参考实现 (Gauss-Jordan 求逆/独立核函数/独立 LML)
不导入也不调用 gprbf 的任何函数来产生期望值。gprbf 仅作为被测对象。
"""

import math
import unittest

import gprbf
from gprbf import (
    Hyperparameters,
    log_marginal,
    predict,
    predict_with_derivatives,
    rbf_kernel,
    rbf_kernel_derivatives_1d,
)


# ---------------------------------------------------------------------------
# 独立参考实现 (与 gprbf 无关的另写代码, 供产生期望值)
# ---------------------------------------------------------------------------


def ref_rbf_matrix(X1, X2, a, ell):
    a2 = a * a
    out = []
    for x in X1:
        row = []
        for z in X2:
            d2 = sum((u - v) ** 2 for u, v in zip(x, z))
            row.append(a2 * math.exp(-d2 / (2.0 * ell * ell)))
        out.append(row)
    return out


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


def ref_matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def ref_log_marginal(X, y, a, ell, s, jitter=0.0):
    n = len(X)
    K = ref_rbf_matrix(X, X, a, ell)
    Ky = [list(K[i]) for i in range(n)]
    var = s * s + jitter
    for i in range(n):
        Ky[i][i] += var
    Inv = ref_inverse(Ky)
    ai = [sum(Inv[i][j] * y[j] for j in range(n)) for i in range(n)]
    quad = sum(y[i] * ai[i] for i in range(n))
    # logdet via reference inverse? 用行列式余子展开太昂贵; 用 Ky * Inv 校验后
    # 以 Cholesky 风格的独立平方根消去计算 logdet。
    logdet = _ref_logdet(Ky)
    return -0.5 * (quad + logdet + n * math.log(2.0 * math.pi)), Inv, ai


def _ref_logdet(A):
    """
    带部分主元的高斯消元计算 log|det A| = sum log|pivot| (独立实现)。
    参考只对正定矩阵使用 (det>0, log|det| 即 logdet)。
    """
    n = len(A)
    M = [list(row) for row in A]
    logdet = 0.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) == 0.0:
            raise ZeroDivisionError("singular in ref logdet")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        logdet += math.log(abs(pv))
        for r in range(col + 1, n):
            f = M[r][col] / pv
            M[r][col] = 0.0
            for c in range(col + 1, n):
                M[r][c] -= f * M[col][c]
    return logdet


def ref_predict_one(X, y, xt, a, ell, s, jitter=0.0):
    """单点解析预测: 显式逆矩阵公式 (独立期望值)。"""
    n = len(X)
    K = ref_rbf_matrix(X, X, a, ell)
    Ky = [list(K[i]) for i in range(n)]
    var = s * s + jitter
    for i in range(n):
        Ky[i][i] += var
    Inv = ref_inverse(Ky)
    ks = [ref_rbf_matrix(X, [xt], a, ell)[i][0] for i in range(n)]
    ai = [sum(Inv[i][j] * y[j] for j in range(n)) for i in range(n)]
    mean = sum(ks[i] * ai[i] for i in range(n))
    latent = a * a - sum(ks[i] * sum(Inv[i][j] * ks[j] for j in range(n)) for i in range(n))
    return mean, latent, latent + var


# ---------------------------------------------------------------------------
# 独立的一维函数/导数核参考 (在本文件内另行推导编写, 不调用 gprbf)
# u = x - z:
#   Cov(f(x), f'(z))  = k * u / ell^2
#   Cov(f'(x), f'(z)) = k * (1/ell^2 - u^2/ell^4)
# ---------------------------------------------------------------------------


def ref_deriv_blocks(X1, X2, a, ell):
    """返回 (K, Kf_fd=k 对第二点导数, Kdd), 全部用独立标量循环写出。"""
    l2 = ell * ell
    K0, G, H = [], [], []
    for row_x in X1:
        x = row_x[0]
        r0, rg, rh = [], [], []
        for row_z in X2:
            z = row_z[0]
            u = x - z
            kv = a * a * math.exp(-u * u / (2.0 * l2))
            r0.append(kv)
            rg.append(kv * u / l2)                    # d/dz k = +k u/l2
            rh.append(kv * (1.0 / l2 - u * u / l2 ** 2))
        K0.append(r0)
        G.append(rg)
        H.append(rh)
    return K0, G, H


def ref_joint_predict(Xf, yf, Xd, dy, T, a, ell, s, sd, jitter=0.0):
    """
    函数值+导数联合 GP 的独立参考: 直接拼联合协方差, Gauss-Jordan 显式逆,
    返回逐点 [(f_mean, f_latent, d_mean, d_latent), ...]。
    """
    nf, nd = len(Xf), len(Xd)
    N = nf + nd
    Kff = ref_rbf_matrix(Xf, Xf, a, ell)
    Kdd = ref_deriv_blocks(Xd, Xd, a, ell)[2]
    Kfd = ref_deriv_blocks(Xf, Xd, a, ell)[1]   # Cov(f(Xf), f'(Xd))
    A = [[0.0] * N for _ in range(N)]
    vf, vd = s * s + jitter, sd * sd + jitter
    for i in range(nf):
        for j in range(nf):
            A[i][j] = Kff[i][j]
        A[i][i] += vf
    for i in range(nd):
        for j in range(nd):
            A[nf + i][nf + j] = Kdd[i][j]
        A[nf + i][nf + i] += vd
    for i in range(nf):
        for j in range(nd):
            A[i][nf + j] = Kfd[i][j]
            A[nf + j][i] = Kfd[i][j]
    Inv = ref_inverse(A)
    obs = list(yf) + list(dy)
    a_vec = [sum(Inv[i][j] * obs[j] for j in range(N)) for i in range(N)]

    Kf_T = ref_rbf_matrix(Xf, T, a, ell)
    Gf_T = ref_deriv_blocks(Xf, T, a, ell)[1]   # Cov(f(Xf), f'(T))
    Gd_T = ref_deriv_blocks(Xd, T, a, ell)[1]   # Cov(f(Xd), f'(T))
    Hd_T = ref_deriv_blocks(Xd, T, a, ell)[2]   # Cov(f'(Xd), f'(T))
    Kd_T = ref_deriv_blocks(Xd, T, a, ell)[0]
    prior_d = a * a / (ell * ell)
    out = []
    for t, xt in enumerate(T):
        kf = [Kf_T[i][t] for i in range(nf)] + [-Gd_T[i][t] for i in range(nd)]
        kd = [Gf_T[i][t] for i in range(nf)] + [Hd_T[i][t] for i in range(nd)]
        fm = sum(kf[i] * a_vec[i] for i in range(N))
        dm = sum(kd[i] * a_vec[i] for i in range(N))
        fl = a * a - sum(kf[i] * sum(Inv[i][j] * kf[j] for j in range(N)) for i in range(N))
        dl = prior_d - sum(kd[i] * sum(Inv[i][j] * kd[j] for j in range(N)) for i in range(N))
        out.append((fm, fl, dm, dl))
    return out, Kd_T


# ---------------------------------------------------------------------------
# 确定性数据
# ---------------------------------------------------------------------------


def make_data():
    # 12 个 1D 点, y = sin(x) + 轻微确定性偏移, 全部手写常量生成
    X = [(-2.0 + 0.4 * i,) for i in range(12)]
    y = [math.sin(x[0]) + 0.15 * x[0] for x in X]
    return X, y


HP = Hyperparameters(alpha=1.3, lengthscale=0.9, noise=0.2)


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------


class TestAgainstDirectInverse(unittest.TestCase):
    def test_log_marginal_matches_direct_inverse(self):
        X, y = make_data()
        res = log_marginal(X, y, HP)
        self.assertEqual(res.status, "ok")
        ref_lml, ref_Inv, ref_ai = ref_log_marginal(
            X, y, HP.alpha, HP.lengthscale, HP.noise
        )
        self.assertTrue(math.isfinite(res.log_marginal))
        self.assertAlmostEqual(res.log_marginal, ref_lml, delta=abs(ref_lml) * 1e-11 + 1e-12)

        # alpha_vec 与显式逆对直接逆参考
        for i in range(len(X)):
            self.assertAlmostEqual(res.alpha_vec[i], ref_ai[i], delta=1e-11)
            for j in range(len(X)):
                self.assertAlmostEqual(res.K_inv[i][j], ref_Inv[i][j], delta=1e-10)

        # K_inv 对称性检查
        n = len(X)
        asym = max(abs(res.K_inv[i][j] - res.K_inv[j][i]) for i in range(n) for j in range(n))
        self.assertLessEqual(asym, 1e-10)

        # K_y 构造逐位对称; Cholesky 与求解残差
        self.assertEqual(res.symmetry_residual, 0.0)
        self.assertLess(res.chol_residual, 1e-13)
        self.assertLess(res.solve_residual, 1e-13)

    def test_predict_matches_single_point_analytic(self):
        X, y = make_data()
        for xt in [(-2.0,), (-0.73,), (0.11,), (1.57,), (2.4,)]:
            res = predict(X, y, [xt], HP)
            self.assertEqual(res.status, "ok")
            m, lv, ov = ref_predict_one(
                X, y, xt, HP.alpha, HP.lengthscale, HP.noise
            )
            self.assertAlmostEqual(res.mean[0], m, delta=1e-11)
            self.assertAlmostEqual(res.latent_variance[0], lv, delta=1e-10)
            self.assertAlmostEqual(res.observed_variance[0], ov, delta=1e-10)

    def test_training_point_variance_identity(self):
        # 训练点 x_i 处隐后验方差的精确恒等式 (K_y = K + s^2 I):
        #   v_i = s^2 * (1 - s^2 * (K_y^-1)_ii)
        # s^2->0 时 v_i->0 (内插); 大噪声时 v_i->s^2。期望值取自独立直接逆。
        X, y = make_data()
        res = predict(X, y, [X[3], X[7]], HP)
        self.assertEqual(res.status, "ok")
        _, ref_Inv, _ = ref_log_marginal(X, y, HP.alpha, HP.lengthscale, HP.noise)
        s2 = HP.noise * HP.noise
        for t, i in enumerate((3, 7)):
            want = s2 * (1.0 - s2 * ref_Inv[i][i])
            self.assertAlmostEqual(res.latent_variance[t], want, delta=1e-12)
            self.assertAlmostEqual(res.observed_variance[t], want + s2, delta=1e-12)

        # 噪声 -> 0 极限: 训练点隐方差 -> 0
        hp0 = Hyperparameters(1.0, 0.7, 1e-6)
        p0 = predict(X, y, [X[3]], hp0)
        self.assertEqual(p0.status, "ok")
        self.assertLess(p0.latent_variance[0], 1e-8)

    def test_3d_supported(self):
        X = [(0.0, 0.0, 0.0), (1.0, 0.5, -0.2), (-0.7, 2.0, 0.3), (1.0, 1.0, 1.0)]
        y = [0.5, -1.0, 2.0, 0.2]
        hp = Hyperparameters(1.1, 1.4, 0.1)
        res = log_marginal(X, y, hp)
        self.assertEqual(res.status, "ok")
        ref_lml, _, _ = ref_log_marginal(X, y, 1.1, 1.4, 0.1)
        self.assertAlmostEqual(res.log_marginal, ref_lml, delta=1e-11)
        p = predict(X, y, [(0.2, 0.2, 0.2)], hp)
        m, lv, ov = ref_predict_one(X, y, (0.2, 0.2, 0.2), 1.1, 1.4, 0.1)
        self.assertAlmostEqual(p.mean[0], m, delta=1e-11)
        self.assertAlmostEqual(p.latent_variance[0], lv, delta=1e-10)

    def test_kernel_matrix_reference_and_symmetry(self):
        X, _ = make_data()
        K = rbf_kernel(X, HP)
        ref = ref_rbf_matrix(X, X, HP.alpha, HP.lengthscale)
        for i in range(len(X)):
            for j in range(len(X)):
                self.assertAlmostEqual(K[i][j], ref[i][j], delta=1e-14)
                self.assertEqual(K[i][j], K[j][i])


class TestAnalyticGradient(unittest.TestCase):
    def _fd_gradient(self, X, y, hp, jitter, step=1e-6):
        def lml_at(a, ell, s):
            r = log_marginal(X, y, Hyperparameters(a, ell, s), jitter=jitter)
            assert r.status == "ok"
            return r.log_marginal

        la, le, ls = math.log(hp.alpha), math.log(hp.lengthscale), math.log(hp.noise)

        def fd(v):
            return (lml_at(math.exp(v + step), math.exp(le), math.exp(ls))
                    - lml_at(math.exp(v - step), math.exp(le), math.exp(ls))) / (2 * step)

        ga = fd(la)
        gl = (lml_at(hp.alpha, math.exp(le + step), hp.noise)
              - lml_at(hp.alpha, math.exp(le - step), hp.noise)) / (2 * step)
        gs = (lml_at(hp.alpha, hp.lengthscale, math.exp(ls + step))
              - lml_at(hp.alpha, hp.lengthscale, math.exp(ls - step))) / (2 * step)
        return ga, gl, gs

    def test_gradient_central_difference(self):
        X, y = make_data()
        res = log_marginal(X, y, HP)
        self.assertEqual(res.status, "ok")
        fd = self._fd_gradient(X, y, HP, 0.0)
        for got, want, name in zip(res.gradient, fd, ("d/dlog a", "d/dlog l", "d/dlog s")):
            self.assertAlmostEqual(got, want, delta=2e-5, msg=name)

    def test_gradient_with_jitter_central_difference(self):
        X, y = make_data()
        hp = Hyperparameters(0.8, 1.1, 0.05)
        res = log_marginal(X, y, hp, jitter=1e-4)
        self.assertEqual(res.status, "ok")
        fd = self._fd_gradient(X, y, hp, 1e-4)
        for got, want in zip(res.gradient, fd):
            self.assertAlmostEqual(got, want, delta=2e-5)

    def test_jitter_equivalent_to_extra_noise(self):
        # 显式 jitter 作为方差项: (s, j) 与 (sqrt(s^2+j), 0) 的 LML/预测相同
        X, y = make_data()
        j = 0.01
        r1 = log_marginal(X, y, HP, jitter=j)
        s2 = math.sqrt(HP.noise * HP.noise + j)
        r2 = log_marginal(X, y, Hyperparameters(HP.alpha, HP.lengthscale, s2))
        self.assertEqual(r1.status, r2.status)
        self.assertAlmostEqual(r1.log_marginal, r2.log_marginal, delta=1e-12)
        self.assertAlmostEqual(r1.effective_noise_variance,
                               r2.effective_noise_variance, delta=1e-15)
        p1 = predict(X, y, [(0.37,)], HP, jitter=j)
        p2 = predict(X, y, [(0.37,)], Hyperparameters(HP.alpha, HP.lengthscale, s2))
        self.assertAlmostEqual(p1.mean[0], p2.mean[0], delta=1e-13)
        self.assertAlmostEqual(p1.latent_variance[0], p2.latent_variance[0], delta=1e-13)
        # 梯度不同: jitter 不随 log noise 缩放
        self.assertNotAlmostEqual(r1.gradient[2], r2.gradient[2], delta=1e-8)

    def test_default_jitter_is_zero(self):
        X, y = make_data()
        self.assertEqual(log_marginal(X, y, HP).jitter, 0.0)
        self.assertEqual(log_marginal(X, y, HP).effective_noise_variance, HP.noise ** 2)


class TestEdgeCases(unittest.TestCase):
    def test_duplicate_inputs_tiny_noise_not_pd_without_jitter(self):
        # 重复输入 + 极小噪声: s^2 在浮点中被对角项吸收 => 精确零特征值 => not_pd
        X = [(0.0,), (0.0,), (1.0,), (1.0,)]
        y = [0.5, 0.5, -0.5, -0.5]
        hp = Hyperparameters(1.0, 1.0, 1e-10)
        res = log_marginal(X, y, hp)
        self.assertEqual(res.status, "not_pd")
        self.assertIsNone(res.log_marginal)
        self.assertIsNone(res.gradient)
        p = predict(X, y, [(0.0,)], hp)
        self.assertEqual(p.status, "not_pd")
        self.assertIsNone(p.mean)

    def test_explicit_jitter_recovers_not_pd(self):
        X = [(0.0,), (0.0,), (1.0,), (1.0,)]
        y = [0.5, 0.5, -0.5, -0.5]
        hp = Hyperparameters(1.0, 1.0, 1e-10)
        res = log_marginal(X, y, hp, jitter=1e-6)
        self.assertEqual(res.status, "ok")
        self.assertTrue(math.isfinite(res.log_marginal))
        p = predict(X, y, [(0.5,)], hp, jitter=1e-6)
        self.assertEqual(p.status, "ok")
        # jitter 是方差项, 直接计入观测方差: s^2 + jitter
        self.assertAlmostEqual(
            p.observed_variance[0] - p.latent_variance[0],
            1e-6 + 1e-20,
            delta=1e-15,
        )

    def test_tiny_noise_well_conditioned(self):
        # 无重复点, 极小噪声仍应求解成功且残差小
        X = [(float(i) * 0.37,) for i in range(8)]
        y = [math.sin(v[0]) for v in X]
        hp = Hyperparameters(1.0, 0.5, 1e-8)
        res = log_marginal(X, y, hp)
        self.assertEqual(res.status, "ok")
        self.assertLess(res.solve_residual, 1e-9)
        p = predict(X, y, [X[2]], hp)
        self.assertEqual(p.status, "ok")
        self.assertGreaterEqual(p.latent_variance[0], 0.0)
        self.assertLess(p.latent_variance[0], 1e-12)

    def test_negative_tiny_variance_policy_unit(self):
        # 负微量处理是独立公开的纯策略, 直接单测三个分支 (期望值手工给定):
        # tol = 1e-10 * max(1, alpha^2)
        f = gprbf.classify_raw_variance
        tol = gprbf.NEG_VAR_REL_TOL  # alpha=1 时 tol 即 1e-10
        # 正值原样
        self.assertEqual(f(0.5, 1.0), (0.5, "ok"))
        self.assertEqual(f(0.0, 1.0), (0.0, "ok"))
        # 负微量: 边界点 -tol 恰好抹零, 内部点抹零
        v, kind = f(-tol, 1.0)
        self.assertEqual((v, kind), (0.0, "clamped"))
        v, kind = f(-0.5 * tol, 1.0)
        self.assertEqual((v, kind), (0.0, "clamped"))
        # 超过容差的负值: 原样返回, 标记 negative
        v, kind = f(-tol * 1.0001, 1.0)
        self.assertEqual(kind, "negative")
        self.assertAlmostEqual(v, -tol * 1.0001)
        v, kind = f(-1.0, 1.0)
        self.assertEqual(kind, "negative")
        self.assertEqual(v, -1.0)
        # 容差随第二个参数 alpha^2 缩放: tol = 1e-10*max(1, alpha^2)
        self.assertEqual(f(-3e-10, 4.0)[1], "clamped")   # tol=4e-10
        self.assertEqual(f(-5e-10, 4.0)[1], "negative")  # 超过 4e-10
        self.assertEqual(f(-5e-10, 1.0)[1], "negative")  # tol=1e-10
        self.assertEqual(f(-3e-10, 100.0)[1], "clamped") # tol=1e-8

    def test_negative_policy_is_continuous_at_boundary(self):
        # 边界两侧分类单调: 从 -tol(含) 到 0 全部 clamped; 越过 -tol 立即 negative
        tol = gprbf.NEG_VAR_REL_TOL
        for frac in (0.25, 0.5, 0.999999, 1.0):
            self.assertEqual(gprbf.classify_raw_variance(-frac * tol, 1.0)[1],
                             "clamped")
        self.assertEqual(gprbf.classify_raw_variance(-1.000001 * tol, 1.0)[1],
                         "negative")

    def test_pipeline_negative_variance_status_mapping(self):
        # 端到端: 对接近奇异的重复构型做穷举 (噪声 1e-7..1e-13),
        # 断言状态机只可能返回三种已文档化状态, 且任何非负 raw 都 >= 0 输出;
        # raw 真的为负时: 微小 => clamped + latent==0; 大 => negative_variance + 原样。
        X = [(0.0,)] * 5 + [(1.0,)]
        y = [0.2] * 5 + [-0.2]
        observed_kinds = set()
        for s in (1e-7, 3e-8, 1e-8, 3e-9, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13):
            p = predict(X, y, [(0.0,), (1.0,), (0.5,)], Hyperparameters(1.0, 1.0, s))
            self.assertIn(p.status, ("ok", "not_pd", "negative_variance"))
            observed_kinds.add(p.status)
            if p.status == "not_pd":
                continue
            for t in range(3):
                raw = p.raw_latent_variance[t]
                if raw >= 0.0:
                    self.assertEqual(p.latent_variance[t], raw)
                elif t in p.clamped:
                    self.assertEqual(p.latent_variance[t], 0.0)
                    self.assertGreaterEqual(raw, -gprbf.NEG_VAR_REL_TOL)
                else:
                    self.assertEqual(p.status, "negative_variance")
                    self.assertEqual(p.latent_variance[t], raw)
        # 该平台/CPython 双精度下至少应观测到 ok 或 not_pd 之一
        self.assertTrue(observed_kinds & {"ok", "not_pd"})

    def test_duplicate_inputs_exact_fit_like_behavior(self):
        # 重复行且 y 一致: 小噪声下拟合接近观测值
        X = [(0.0,), (0.0,), (2.0,)]
        y = [1.0, 1.0, -1.0]
        hp = Hyperparameters(1.0, 0.7, 1e-4)
        p = predict(X, y, [(0.0,), (2.0,)], hp)
        self.assertEqual(p.status, "ok")
        self.assertAlmostEqual(p.mean[0], 1.0, delta=1e-6)
        self.assertAlmostEqual(p.mean[1], -1.0, delta=1e-6)
        self.assertTrue(all(v >= 0.0 for v in p.latent_variance))


class TestInputValidation(unittest.TestCase):
    def test_nan_inf_rejected(self):
        X, y = make_data()
        bad_X = [list(row) for row in X]
        bad_X[0] = (float("nan"),)
        with self.assertRaisesRegex(ValueError, "X\\[0\\]\\[0\\]"):
            log_marginal(bad_X, y, HP)
        bad_X2 = [list(row) for row in X]
        bad_X2[1] = (float("inf"),)
        with self.assertRaisesRegex(ValueError, "X\\[1\\]\\[0\\]"):
            log_marginal(bad_X2, y, HP)
        bad_y = list(y)
        bad_y[2] = float("-inf")
        with self.assertRaisesRegex(ValueError, "y\\[2\\]"):
            log_marginal(X, bad_y, HP)

    def test_bool_rejected(self):
        X, y = make_data()
        with self.assertRaises(TypeError):
            log_marginal(X, y, Hyperparameters(True, 1.0, 1.0))  # noqa
        with self.assertRaises(TypeError):
            log_marginal(X, y, Hyperparameters(1.0, 1.0, False))  # noqa
        bad_X = [list(row) for row in X]
        bad_X[0] = (1,)  # int 合法
        log_marginal(bad_X, y, HP)
        bad_X[0] = (True,)
        with self.assertRaises(TypeError):
            log_marginal(bad_X, y, HP)

    def test_nonpositive_hyperparameters(self):
        X, y = make_data()
        with self.assertRaisesRegex(ValueError, "hp.alpha"):
            log_marginal(X, y, Hyperparameters(0.0, 1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "hp.lengthscale"):
            log_marginal(X, y, Hyperparameters(1.0, -1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "hp.noise"):
            log_marginal(X, y, Hyperparameters(1.0, 1.0, 0.0))

    def test_shape_errors(self):
        X, y = make_data()
        with self.assertRaisesRegex(ValueError, "y"):
            log_marginal(X, y[:-1], HP)
        with self.assertRaisesRegex(ValueError, "dimension"):
            predict(X, y, [(0.0, 0.0)], HP)
        ragged = list(X)
        ragged = [list(r) for r in ragged]
        ragged[1] = (0.0, 0.0)
        with self.assertRaisesRegex(ValueError, r"X\[1\]"):
            log_marginal(ragged, y, HP)
        with self.assertRaises(TypeError):
            log_marginal([1, 2, 3], y, HP)

    def test_limits(self):
        X, y = make_data()
        # 81 样本拒绝
        X81 = [(float(i) * 0.1,) for i in range(81)]
        y81 = [0.0] * 81
        with self.assertRaisesRegex(ValueError, "exceeds limit 80"):
            log_marginal(X81, y81, HP)
        # 4 维拒绝
        with self.assertRaisesRegex(ValueError, "1\\.\\.3"):
            log_marginal([(0.0, 0.0, 0.0, 0.0)], [0.0], HP)
        # 空输入拒绝
        with self.assertRaises(ValueError):
            log_marginal([], [], HP)

    def test_negative_jitter_rejected(self):
        X, y = make_data()
        with self.assertRaisesRegex(ValueError, "jitter"):
            log_marginal(X, y, HP, jitter=-1e-12)
        with self.assertRaisesRegex(ValueError, "jitter"):
            predict(X, y, [(0.0,)], HP, jitter=float("nan"))

    def test_wrong_hp_type(self):
        X, y = make_data()
        with self.assertRaises(TypeError):
            log_marginal(X, y, (1.0, 1.0, 1.0))  # noqa


class TestDeterminismAndImmutability(unittest.TestCase):
    def test_repeated_calls_identical_and_inputs_unchanged(self):
        X, y = make_data()
        Xcopy = [tuple(v) for v in X]
        ycopy = tuple(y)
        r1 = log_marginal(X, y, HP)
        r2 = log_marginal(X, y, HP)
        self.assertEqual(r1.log_marginal, r2.log_marginal)
        self.assertEqual(r1.gradient, r2.gradient)
        self.assertEqual([tuple(v) for v in X], Xcopy)
        self.assertEqual(tuple(y), ycopy)
        p1 = predict(X, y, [(0.13,), (-1.2,)], HP)
        p2 = predict(X, y, [(0.13,), (-1.2,)], HP)
        self.assertEqual(p1.mean, p2.mean)
        self.assertEqual(p1.latent_variance, p2.latent_variance)

    def test_integer_inputs_accepted(self):
        X = [(0, 0), (1, 0), (0, 1), (1, 1)]
        y = [1, -1, -1, 1]
        res = log_marginal(X, y, Hyperparameters(1, 1, 0.1))
        self.assertEqual(res.status, "ok")
        ref_lml, _, _ = ref_log_marginal(
            [(float(a), float(b)) for a, b in X],
            [float(v) for v in y], 1.0, 1.0, 0.1,
        )
        self.assertAlmostEqual(res.log_marginal, ref_lml, delta=1e-12)


# ---------------------------------------------------------------------------
# 一维函数值 + 一阶导数联合 GP 的测试
# ---------------------------------------------------------------------------


def make_deriv_data():
    # f(x) = sin(x) + 0.15 x,  f'(x) = cos(x) + 0.15, 确定性 1D 数据
    Xf = [(-2.0 + 0.5 * i,) for i in range(9)]          # -2 .. 2, 9 点
    yf = [math.sin(x[0]) + 0.15 * x[0] for x in Xf]
    Xd = [(-1.5,), (-0.5,), (0.5,), (1.5,)]
    dy = [math.cos(x[0]) + 0.15 for x in Xd]
    T = [(-1.8,), (-0.95,), (0.0,), (0.95,), (1.8,)]
    return Xf, yf, Xd, dy, T


DHP = Hyperparameters(alpha=1.3, lengthscale=0.9, noise=0.2)
DSD = 0.15


class TestDerivativeKernels(unittest.TestCase):
    def test_blocks_match_independent_reference(self):
        X = [(0.3,), (1.1,), (-0.7,)]
        Z = [(-0.2,), (0.75,), (1.4,), (0.3,)]
        K, G, H = rbf_kernel_derivatives_1d(X, Z, DHP)
        rK, rG, rH = ref_deriv_blocks(
            X, Z, DHP.alpha, DHP.lengthscale
        )
        for i in range(len(X)):
            for j in range(len(Z)):
                self.assertAlmostEqual(K[i][j], rK[i][j], delta=1e-14)
                self.assertAlmostEqual(G[i][j], rG[i][j], delta=1e-14)
                self.assertAlmostEqual(H[i][j], rH[i][j], delta=1e-14)

    def test_blocks_vs_central_differences_signs(self):
        # 期望值由核标量函数中心差分独立给出:
        #   G = d/dz k;  d/dx k = -G;  H = d^2/(dx dz) k
        a, ell = DHP.alpha, DHP.lengthscale
        X = [(0.3,), (1.1,)]
        Z = [(-0.2,), (0.75,), (1.4,)]
        K, G, H = rbf_kernel_derivatives_1d(X, Z, DHP)
        h = 1e-4  # 二阶混合差分在 h 更小时相消误差增大, 1e-4 处截断/舍入平衡
        xs = [r[0] for r in X]
        zs = [r[0] for r in Z]
        for i, x in enumerate(xs):
            for j, z in enumerate(zs):
                d_dz = (ref_rbf_matrix([(x,)], [(z + h,)], a, ell)[0][0]
                        - ref_rbf_matrix([(x,)], [(z - h,)], a, ell)[0][0]) / (2 * h)
                d_dx = (ref_rbf_matrix([(x + h,)], [(z,)], a, ell)[0][0]
                        - ref_rbf_matrix([(x - h,)], [(z,)], a, ell)[0][0]) / (2 * h)
                d2 = (
                    ref_rbf_matrix([(x + h,)], [(z + h,)], a, ell)[0][0]
                    - ref_rbf_matrix([(x + h,)], [(z - h,)], a, ell)[0][0]
                    - ref_rbf_matrix([(x - h,)], [(z + h,)], a, ell)[0][0]
                    + ref_rbf_matrix([(x - h,)], [(z - h,)], a, ell)[0][0]
                ) / (4 * h * h)
                self.assertAlmostEqual(G[i][j], d_dz, delta=1e-7, msg="d/dz sign")
                self.assertAlmostEqual(-G[i][j], d_dx, delta=1e-7, msg="d/dx sign")
                self.assertAlmostEqual(H[i][j], d2, delta=1e-6, msg="d2/dxdz")

    def test_exact_values_and_diagonal_identities(self):
        # 手算: x=z 时 u=0 => G=0, H = alpha^2/ell^2 = K/ell^2
        a, ell = 1.3, 0.9
        X = [(0.0,)]
        _, G, H = rbf_kernel_derivatives_1d(X, X, DHP)
        self.assertEqual(G[0][0], 0.0)
        self.assertAlmostEqual(H[0][0], a * a / (ell * ell), delta=1e-14)

        # 手算单点: x=1, z=0, ell=1, alpha=1 => k=e^-1/2
        #   G =  k * 1      = e^-1/2
        #   H =  k * (1-1)  = 0
        hp1 = Hyperparameters(1.0, 1.0, 0.1)
        K, G, H = rbf_kernel_derivatives_1d([(1.0,)], [(0.0,)], hp1)
        e = math.exp(-0.5)
        self.assertAlmostEqual(K[0][0], e, delta=1e-15)
        self.assertAlmostEqual(G[0][0], e, delta=1e-15)
        self.assertEqual(H[0][0], 0.0)

    def test_symmetry_and_antisymmetry(self):
        X = [(0.3,), (1.1,), (-0.7,)]
        Z = [(-0.2,), (0.75,)]
        _, G, H = rbf_kernel_derivatives_1d(X, Z, DHP)
        _, Gt, Ht = rbf_kernel_derivatives_1d(Z, X, DHP)
        # Cov(f(x), f'(z)) 交换两点反号; Cov(f'(x), f'(z)) 交换对称
        for i in range(len(X)):
            for j in range(len(Z)):
                self.assertEqual(G[i][j], -Gt[j][i])
                self.assertEqual(H[i][j], Ht[j][i])
        # 方阵: G 反对称 (对角 0), H 对称, H 对角为正先验导数方差
        _, Gs, Hs = rbf_kernel_derivatives_1d(X, X, DHP)
        n = len(X)
        for i in range(n):
            self.assertEqual(Gs[i][i], 0.0)
            self.assertGreater(Hs[i][i], 0.0)
            for j in range(n):
                self.assertEqual(Gs[i][j], -Gs[j][i])
                self.assertEqual(Hs[i][j], Hs[j][i])


class TestJointPrediction(unittest.TestCase):
    def test_joint_predict_matches_explicit_inverse(self):
        Xf, yf, Xd, dy, T = make_deriv_data()
        res = predict_with_derivatives(
            Xf, yf, Xd, dy, T, DHP, derivative_noise=DSD
        )
        self.assertEqual(res.status, "ok")
        ref, _ = ref_joint_predict(
            Xf, yf, Xd, dy, T,
            DHP.alpha, DHP.lengthscale, DHP.noise, DSD,
        )
        prior_d = DHP.alpha ** 2 / DHP.lengthscale ** 2
        for t in range(len(T)):
            fm, fl, dm, dl = ref[t]
            self.assertAlmostEqual(res.function_mean[t], fm, delta=1e-11)
            self.assertAlmostEqual(res.derivative_mean[t], dm, delta=1e-11)
            self.assertAlmostEqual(res.function_latent_variance[t], fl, delta=1e-10)
            self.assertAlmostEqual(res.derivative_latent_variance[t], dl, delta=1e-10)
            self.assertGreaterEqual(res.function_latent_variance[t], 0.0)
            self.assertGreaterEqual(res.derivative_latent_variance[t], 0.0)
            # 观测方差 = 隐方差 + 各自噪声方差 (两种噪声相互独立)
            self.assertAlmostEqual(
                res.function_observed_variance[t]
                - res.function_latent_variance[t],
                DHP.noise ** 2, delta=1e-14,
            )
            self.assertAlmostEqual(
                res.derivative_observed_variance[t]
                - res.derivative_latent_variance[t],
                DSD ** 2, delta=1e-14,
            )
            self.assertLessEqual(res.function_latent_variance[t], DHP.alpha ** 2 + 1e-12)
            self.assertLessEqual(res.derivative_latent_variance[t], prior_d + 1e-12)
        self.assertEqual(res.symmetry_residual, 0.0)
        self.assertLess(res.solve_residual, 1e-12)
        self.assertEqual(res.clamped_function, ())
        self.assertEqual(res.clamped_derivative, ())

    def test_derivative_of_posterior_mean_equals_posterior_derivative(self):
        # 解析恒等式 (符号端到端核对): d/dx E[f*|data] = E[f'*|data]
        # 用后验函数均值的中心差分独立核对导数后验均值。
        Xf, yf, Xd, dy, _ = make_deriv_data()
        probes = [-1.3, -0.4, 0.27, 0.8, 1.42]
        h = 1e-5
        for x0 in probes:
            rp = predict_with_derivatives(
                Xf, yf, Xd, dy, [(x0 - h,), (x0,), (x0 + h,)],
                DHP, derivative_noise=DSD,
            )
            self.assertEqual(rp.status, "ok")
            slope = (rp.function_mean[2] - rp.function_mean[0]) / (2 * h)
            self.assertAlmostEqual(
                slope, rp.derivative_mean[1], delta=2e-6,
                msg=f"posterior mean slope mismatch at {x0}",
            )

    def test_derivative_observation_interpolation(self):
        # 导数噪声很小时, 导数观测点处的后验导数应贴近观测值
        Xf = [(-2.0 + 0.5 * i,) for i in range(9)]
        yf = [math.sin(x[0]) + 0.15 * x[0] for x in Xf]
        Xd = [(-1.5,), (0.0,), (1.5,)]
        dy = [math.cos(x[0]) + 0.15 for x in Xd]
        res = predict_with_derivatives(
            Xf, yf, Xd, dy, Xd, DHP, derivative_noise=1e-5
        )
        self.assertEqual(res.status, "ok")
        for j in range(3):
            self.assertAlmostEqual(res.derivative_mean[j], dy[j], delta=1e-3)
            self.assertLess(res.derivative_latent_variance[j], 1e-4)

    def test_recovers_true_function_derivative_better_than_noise_free_prior(self):
        # 正常结果核对: 后验导数均值在网格内逼近解析导数, 误差远小于先验 sd
        Xf, yf, Xd, dy, T = make_deriv_data()
        res = predict_with_derivatives(
            Xf, yf, Xd, dy, T, DHP, derivative_noise=DSD
        )
        prior_d_sd = DHP.alpha / DHP.lengthscale
        for t, xt in enumerate(T):
            truth = math.cos(xt[0]) + 0.15
            self.assertLess(abs(res.derivative_mean[t] - truth), 0.05)
        self.assertLess(
            max(res.derivative_latent_variance) ** 0.5, 0.5 * prior_d_sd
        )

    def test_infinite_derivative_noise_matches_function_only_predict(self):
        # sd_d -> inf 时导数观测被完全忽略, 联合后验的 f 分量必须与旧入口
        # predict 的结果一致 (独立入口交叉核对)。
        Xf, yf, Xd, dy, T = make_deriv_data()
        res = predict_with_derivatives(
            Xf, yf, Xd, dy, T, DHP, derivative_noise=1e6
        )
        self.assertEqual(res.status, "ok")
        old = predict(Xf, yf, T, DHP)
        self.assertEqual(old.status, "ok")
        for t in range(len(T)):
            self.assertAlmostEqual(
                res.function_mean[t], old.mean[t], delta=1e-9
            )
            self.assertAlmostEqual(
                res.function_latent_variance[t],
                old.latent_variance[t], delta=1e-9,
            )

    def test_jitter_enters_both_diagonal_blocks(self):
        Xf, yf, Xd, dy, T = make_deriv_data()
        j = 1e-3
        res = predict_with_derivatives(
            Xf, yf, Xd, dy, T, DHP, derivative_noise=DSD, jitter=j
        )
        self.assertEqual(res.status, "ok")
        self.assertAlmostEqual(
            res.effective_function_noise_variance, DHP.noise ** 2 + j, delta=1e-15
        )
        self.assertAlmostEqual(
            res.effective_derivative_noise_variance, DSD ** 2 + j, delta=1e-15
        )
        for t in range(len(T)):
            self.assertAlmostEqual(
                res.function_observed_variance[t]
                - res.function_latent_variance[t],
                DHP.noise ** 2 + j, delta=1e-14,
            )
            self.assertAlmostEqual(
                res.derivative_observed_variance[t]
                - res.derivative_latent_variance[t],
                DSD ** 2 + j, delta=1e-14,
            )

    def test_prior_variance_far_from_data(self):
        # 边界: 预测点远离全部观测 => 后验退化回先验
        # V[f*] = alpha^2, V[f'*] = alpha^2/ell^2, 均值 ~ 0
        Xf, yf, Xd, dy, _ = make_deriv_data()
        T = [(100.0,), (-80.0,)]
        res = predict_with_derivatives(
            Xf, yf, Xd, dy, T, DHP, derivative_noise=DSD
        )
        self.assertEqual(res.status, "ok")
        a2 = DHP.alpha ** 2
        for t in range(2):
            self.assertAlmostEqual(res.function_mean[t], 0.0, delta=1e-12)
            self.assertAlmostEqual(res.derivative_mean[t], 0.0, delta=1e-12)
            self.assertAlmostEqual(res.function_latent_variance[t], a2, delta=1e-10)
            self.assertAlmostEqual(
                res.derivative_latent_variance[t], a2 / DHP.lengthscale ** 2,
                delta=1e-10,
            )

    def test_minimal_one_plus_one_observations(self):
        # 边界: 最小构型 1 个函数观测 + 1 个导数观测, 与显式逆参考一致
        Xf, yf = [(0.3,)], [0.8]
        Xd, dy = [(-0.4,)], [-0.2]
        T = [(0.0,), (0.7,)]
        res = predict_with_derivatives(
            Xf, yf, Xd, dy, T, DHP, derivative_noise=0.1
        )
        self.assertEqual(res.status, "ok")
        ref, _ = ref_joint_predict(
            Xf, yf, Xd, dy, T, DHP.alpha, DHP.lengthscale, DHP.noise, 0.1
        )
        for t in range(2):
            fm, fl, dm, dl = ref[t]
            self.assertAlmostEqual(res.function_mean[t], fm, delta=1e-12)
            self.assertAlmostEqual(res.derivative_mean[t], dm, delta=1e-12)
            self.assertAlmostEqual(res.function_latent_variance[t], fl, delta=1e-11)
            self.assertAlmostEqual(res.derivative_latent_variance[t], dl, delta=1e-11)


class TestJointEdgeAndFailure(unittest.TestCase):
    def test_duplicate_derivative_points_tiny_noise_not_pd(self):
        # 重复导数观测 + 极小导数噪声 => 导数块秩亏, 联合协方差非正定
        Xf = [(-1.0,), (1.0,)]
        yf = [0.0, 0.0]
        Xd = [(0.0,), (0.0,), (0.5,), (0.5,)]
        dy = [0.3, 0.3, -0.3, -0.3]
        hp = Hyperparameters(1.0, 1.0, 1e-4)
        res = predict_with_derivatives(
            Xf, yf, Xd, dy, [(0.2,)], hp, derivative_noise=1e-10
        )
        self.assertEqual(res.status, "not_pd")
        self.assertIsNone(res.function_mean)
        self.assertIsNone(res.derivative_mean)
        self.assertIsNone(res.function_latent_variance)
        # 显式 jitter 恢复
        fixed = predict_with_derivatives(
            Xf, yf, Xd, dy, [(0.2,)], hp,
            derivative_noise=1e-10, jitter=1e-6,
        )
        self.assertEqual(fixed.status, "ok")
        self.assertTrue(math.isfinite(fixed.function_mean[0]))
        self.assertTrue(math.isfinite(fixed.derivative_mean[0]))

    def test_status_mapping_under_tiny_derivative_noises(self):
        # 小规模枚举: 状态机只可能返回三种已文档化状态; raw 与输出一致
        Xf = [(0.0,)]
        yf = [0.2]
        Xd = [(1.0,), (1.0,), (1.0,)]
        dy = [-0.2, -0.2, -0.2]
        seen = set()
        for sd_ in (1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12):
            p = predict_with_derivatives(
                Xf, yf, Xd, dy, [(0.0,), (1.0,)],
                Hyperparameters(1.0, 1.0, 0.05), derivative_noise=sd_,
            )
            self.assertIn(p.status, ("ok", "not_pd", "negative_variance"))
            seen.add(p.status)
            if p.status == "not_pd":
                continue
            for t in range(2):
                for raw, lat, clamped in (
                    (p.raw_function_variance[t], p.function_latent_variance[t],
                     t in p.clamped_function),
                    (p.raw_derivative_variance[t], p.derivative_latent_variance[t],
                     t in p.clamped_derivative),
                ):
                    if raw >= 0.0:
                        self.assertEqual(lat, raw)
                    elif clamped:
                        self.assertEqual(lat, 0.0)
                    else:
                        self.assertEqual(p.status, "negative_variance")
                        self.assertEqual(lat, raw)
        self.assertTrue(seen & {"ok", "not_pd"})

    def test_derivative_observations_reuse_allowed(self):
        # 同一点位同时作为函数观测与导数观测是合法的 (不同随机量)
        X = [(-1.0,), (0.0,), (1.0,)]
        y = [0.5, 0.0, -0.5]
        dy = [-0.1, 0.2, -0.1]
        res = predict_with_derivatives(
            X, y, X, dy, [(0.5,)], DHP, derivative_noise=0.1
        )
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.symmetry_residual, 0.0)

    def test_only_1d_supported(self):
        Xf = [(0.0, 0.0), (1.0, 0.0)]
        yf = [0.0, 1.0]
        Xd = [(0.5,)]
        dy = [0.1]
        with self.assertRaisesRegex(ValueError, "1D only"):
            predict_with_derivatives(
                Xf, yf, Xd, dy, [(0.0,)], DHP, derivative_noise=0.1
            )
        Xf2 = [(0.0,), (1.0,)]
        Xd2 = [(0.5, 0.5)]
        with self.assertRaisesRegex(ValueError, "1D only"):
            predict_with_derivatives(
                Xf2, yf, Xd2, [0.1], [(0.0,)], DHP, derivative_noise=0.1
            )
        with self.assertRaisesRegex(ValueError, "dimension"):
            predict_with_derivatives(
                Xf2, yf, Xd, dy, [(0.0, 0.0)], DHP, derivative_noise=0.1
            )

    def test_kernel_helper_requires_1d(self):
        with self.assertRaisesRegex(ValueError, "1D only"):
            rbf_kernel_derivatives_1d(
                [(0.0, 0.0)], [(0.0, 0.0)], DHP
            )

    def test_empty_and_length_mismatch(self):
        Xf, yf, Xd, dy, _ = make_deriv_data()
        with self.assertRaises(ValueError):
            predict_with_derivatives(
                [], [], Xd, dy, [(0.0,)], DHP, derivative_noise=0.1
            )
        with self.assertRaises(ValueError):
            predict_with_derivatives(
                Xf, yf, [], [], [(0.0,)], DHP, derivative_noise=0.1
            )
        with self.assertRaisesRegex(ValueError, "dy"):
            predict_with_derivatives(
                Xf, yf, Xd, dy[:-1], [(0.0,)], DHP, derivative_noise=0.1
            )
        with self.assertRaisesRegex(ValueError, "y"):
            predict_with_derivatives(
                Xf, yf[:-1], Xd, dy, [(0.0,)], DHP, derivative_noise=0.1
            )

    def test_total_observation_limit(self):
        # 函数+导数合计上限 80
        Xf = [(float(i) * 0.1,) for i in range(40)]
        yf = [0.0] * 40
        Xd = [(float(i) * 0.1 + 0.05,) for i in range(41)]
        dy = [0.0] * 41
        with self.assertRaisesRegex(ValueError, "exceeds limit 80"):
            predict_with_derivatives(
                Xf, yf, Xd, dy, [(0.0,)], DHP, derivative_noise=0.1
            )
        # 合计恰为 80 合法
        res = predict_with_derivatives(
            Xf, yf, Xd[:40], dy[:40], [(0.0,)], DHP, derivative_noise=0.1
        )
        self.assertEqual(res.status, "ok")

    def test_derivative_noise_validation(self):
        Xf, yf, Xd, dy, _ = make_deriv_data()
        for bad in (0.0, -0.1, float("nan"), float("inf")):
            with self.assertRaises((ValueError, TypeError)):
                predict_with_derivatives(
                    Xf, yf, Xd, dy, [(0.0,)], DHP, derivative_noise=bad
                )
        with self.assertRaises(TypeError):
            predict_with_derivatives(
                Xf, yf, Xd, dy, [(0.0,)], DHP, derivative_noise=True  # noqa
            )
        with self.assertRaisesRegex(ValueError, "jitter"):
            predict_with_derivatives(
                Xf, yf, Xd, dy, [(0.0,)], DHP,
                derivative_noise=0.1, jitter=-1e-9,
            )

    def test_nan_in_derivative_observations_rejected(self):
        Xf, yf, Xd, dy, _ = make_deriv_data()
        bad_dy = list(dy)
        bad_dy[1] = float("nan")
        with self.assertRaisesRegex(ValueError, "dy\\[1\\]"):
            predict_with_derivatives(
                Xf, yf, Xd, bad_dy, [(0.0,)], DHP, derivative_noise=0.1
            )
        bad_Xd = [list(v) for v in Xd]
        bad_Xd[2] = (float("inf"),)
        with self.assertRaisesRegex(ValueError, "Xd\\[2\\]\\[0\\]"):
            predict_with_derivatives(
                Xf, yf, bad_Xd, dy, [(0.0,)], DHP, derivative_noise=0.1
            )

    def test_integer_derivative_inputs_accepted(self):
        Xf = [(0,), (1,), (2,)]
        yf = [1, -1, 0]
        Xd = [(0,), (2,)]
        dy = [0, 1]
        res = predict_with_derivatives(
            Xf, yf, Xd, dy, [(1,)], Hyperparameters(1, 1, 0.1),
            derivative_noise=0.1,
        )
        self.assertEqual(res.status, "ok")
        ref, _ = ref_joint_predict(
            [(float(r[0]),) for r in Xf], [float(v) for v in yf],
            [(float(r[0]),) for r in Xd], [float(v) for v in dy],
            [(1.0,)], 1.0, 1.0, 0.1, 0.1,
        )
        self.assertAlmostEqual(res.function_mean[0], ref[0][0], delta=1e-12)
        self.assertAlmostEqual(res.derivative_mean[0], ref[0][2], delta=1e-12)


class TestJointDeterminismImmutability(unittest.TestCase):
    def test_repeated_calls_identical_and_inputs_unchanged(self):
        Xf, yf, Xd, dy, T = make_deriv_data()
        Xf_c = [tuple(v) for v in Xf]
        Xd_c = [tuple(v) for v in Xd]
        yf_c, dy_c, T_c = tuple(yf), tuple(dy), [tuple(v) for v in T]
        r1 = predict_with_derivatives(
            Xf, yf, Xd, dy, T, DHP, derivative_noise=DSD
        )
        r2 = predict_with_derivatives(
            Xf, yf, Xd, dy, T, DHP, derivative_noise=DSD
        )
        self.assertEqual(r1.function_mean, r2.function_mean)
        self.assertEqual(r1.derivative_mean, r2.derivative_mean)
        self.assertEqual(r1.function_latent_variance, r2.function_latent_variance)
        self.assertEqual(r1.derivative_latent_variance, r2.derivative_latent_variance)
        self.assertEqual([tuple(v) for v in Xf], Xf_c)
        self.assertEqual([tuple(v) for v in Xd], Xd_c)
        self.assertEqual([tuple(v) for v in T], T_c)
        self.assertEqual(tuple(yf), yf_c)
        self.assertEqual(tuple(dy), dy_c)

    def test_legacy_entrypoints_unchanged(self):
        # 旧纯函数值入口签名与行为回归 (导入与调用方式不变)
        X = [(-2.0 + 0.4 * i,) for i in range(12)]
        y = [math.sin(x[0]) + 0.15 * x[0] for x in X]
        r = log_marginal(X, y, DHP)
        p = predict(X, y, [(0.37,)], DHP)
        self.assertEqual(r.status, "ok")
        self.assertEqual(p.status, "ok")
        self.assertFalse(hasattr(p, "derivative_mean"))


if __name__ == "__main__":
    unittest.main()
