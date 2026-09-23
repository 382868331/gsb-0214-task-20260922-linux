"""
独立参考测试: 本文件中的参考实现 (Gauss-Jordan 求逆/独立核函数/独立 LML)
不导入也不调用 gprbf 的任何函数来产生期望值。gprbf 仅作为被测对象。
"""

import math
import unittest

import gprbf
from gprbf import Hyperparameters, log_marginal, predict, rbf_kernel


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


if __name__ == "__main__":
    unittest.main()
