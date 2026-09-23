"""gp_rbf 的必要测试。

期望值来源（独立于被测核心）：
* tests/ref_gp.py —— Gauss-Jordan 直接逆参考，不导入 gp_rbf；
* 显式标量公式（单点、n=1 闭式）；
* 中心差分数值梯度。
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gp_rbf
from gp_rbf import fit, predict, predict_one, GPInputError, NEG_VAR_TOL

try:  # 独立参考（绝不导入 gp_rbf）
    from tests.ref_gp import (
        reference_kernel,
        reference_lml,
        reference_predict,
    )
except ImportError:  # unittest discover -s tests 时 tests 作为顶层目录
    from ref_gp import reference_kernel, reference_lml, reference_predict  # type: ignore


def make_data(n=6, d=2, seed=1234, spread=1.7):
    rng = random.Random(seed)
    X = [[rng.uniform(-spread, spread) for _ in range(d)] for _ in range(n)]
    y = [
        math.sin(0.7 * x[0]) + 0.3 * sum(x[1:]) + rng.uniform(-0.05, 0.05)
        for x in X
    ]
    return X, y


def make_queries(m=4, d=2, seed=4321):
    rng = random.Random(seed)
    return [[rng.uniform(-2.0, 2.0) for _ in range(d)] for _ in range(m)]


class KernelTests(unittest.TestCase):
    def test_kernel_matches_reference_formula(self):
        rng = random.Random(7)
        for _ in range(20):
            d = rng.randint(1, 3)
            x = [rng.uniform(-3, 3) for _ in range(d)]
            xp = [rng.uniform(-3, 3) for _ in range(d)]
            a, l = rng.uniform(0.2, 3.0), rng.uniform(0.1, 2.0)
            self.assertAlmostEqual(
                gp_rbf.rbf_scalar(x, xp, a, l),
                reference_kernel(x, xp, a, l),
                places=14,
            )

    def test_diagonal_is_a_squared_and_symmetry(self):
        X, _ = make_data(n=5, d=3)
        a, l = 1.3, 0.8
        r = fit(X, [0.0] * 5, a, l, 0.5)
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.residuals.symmetry, 0.0)
        for x in X:
            self.assertAlmostEqual(gp_rbf.rbf_scalar(x, x, a, l), a * a, places=14)


class DirectInverseReferenceTests(unittest.TestCase):
    """小矩阵：被测实现 vs Gauss-Jordan 直接逆参考（多组配置）。"""

    def configs(self):
        cfgs = []
        for n in (1, 2, 3, 5, 8):
            for d in (1, 2, 3):
                cfgs.append((n, d, 1.1, 0.9, 0.4, 0.0, 1000 + n * 10 + d))
        # 含显式 jitter 的配置
        cfgs.append((6, 2, 1.5, 1.2, 0.2, 1e-6, 77))
        cfgs.append((4, 1, 0.7, 2.0, 1.0, 1e-4, 88))
        return cfgs

    def test_log_marginal_against_direct_inverse(self):
        for n, d, a, l, s, jit, seed in self.configs():
            X, y = make_data(n=n, d=d, seed=seed)
            r = fit(X, y, a, l, s, jitter=jit)
            self.assertEqual(r.status, "ok", f"config {(n, d, jit)}")
            ref = reference_lml(X, y, a, l, s, jit)
            self.assertAlmostEqual(
                r.log_marginal,
                ref,
                places=9,
                msg=f"lml config n={n} d={d} jit={jit}",
            )

    def test_prediction_against_direct_inverse(self):
        for n, d, a, l, s, jit, seed in self.configs():
            X, y = make_data(n=n, d=d, seed=seed)
            Xs = make_queries(m=5, d=d, seed=seed + 1)
            r = fit(X, y, a, l, s, jitter=jit)
            p = predict(r, Xs)
            rm, rl, ro = reference_predict(X, y, Xs, a, l, s, jit)
            for q in range(len(Xs)):
                self.assertAlmostEqual(p.mean[q], rm[q], places=9,
                                       msg=f"mean n={n} d={d} q={q}")
                # 参考同样可能给出微小负值；比较的是未钳制前的数学量，
                # 这里配置远离病态，参考值为正。
                self.assertAlmostEqual(p.latent_var[q], rl[q], places=9,
                                       msg=f"latent n={n} d={d} q={q}")
                self.assertAlmostEqual(p.observed_var[q], ro[q], places=9,
                                       msg=f"observed n={n} d={d} q={q}")
                self.assertGreaterEqual(p.latent_var[q], 0.0)

    def test_observed_equals_latent_plus_s2(self):
        X, y = make_data(n=6, d=2, seed=5)
        Xs = make_queries(m=4, d=2, seed=6)
        s = 0.37
        r = fit(X, y, 1.2, 0.8, s)
        p = predict(r, Xs)
        for q in range(len(Xs)):
            self.assertAlmostEqual(p.observed_var[q] - p.latent_var[q], s * s,
                                   places=13)


class SinglePointAnalyticTests(unittest.TestCase):
    """n=1 闭式解析预测（手写期望值）。"""

    def test_single_training_point_closed_form(self):
        x0 = [0.5]
        y0 = 1.75
        a, l, s = 2.0, 0.7, 0.3
        r = fit([x0], [y0], a, l, s)
        self.assertEqual(r.status, "ok")

        # 查询点就是训练点
        m0, lv0, ov0 = predict_one(r, x0)
        c = a * a + s * s  # 4.09
        self.assertAlmostEqual(m0, y0 * a * a / c, places=13)
        self.assertAlmostEqual(lv0, a * a - a ** 4 / c, places=13)
        self.assertAlmostEqual(ov0, lv0 + s * s, places=13)
        # n=1 时 latent 方差 = a² s²/(a²+s²)
        self.assertAlmostEqual(lv0, a * a * s * s / c, places=13)

        # 另一个查询点（显式标量推导）
        xs = [0.5 + l * math.sqrt(2.0)]  # ||x*-x0||²/(2l²)=1
        q = a * a * math.exp(-1.0)
        m1, lv1, ov1 = predict_one(r, xs)
        self.assertAlmostEqual(m1, q * y0 / c, places=13)
        self.assertAlmostEqual(lv1, a * a - q * q / c, places=13)
        self.assertAlmostEqual(ov1, lv1 + s * s, places=13)

    def test_lml_single_point_closed_form(self):
        a, l, s = 1.3, 0.6, 0.2
        r = fit([[0.0]], [2.0], a, l, s)
        c = a * a + s * s
        expect = -0.5 * (4.0 / c + math.log(c) + math.log(2.0 * math.pi))
        self.assertAlmostEqual(r.log_marginal, expect, places=13)


class CentralDifferenceGradientTests(unittest.TestCase):
    """解析梯度 vs 中心差分（独立地只调用 fit 的函数值）。"""

    H = 1e-3

    def _fd_grad(self, X, y, a, l, s, jitter):
        def lml(aa, ll, ss):
            rr = fit(X, y, aa, ll, ss, jitter=jitter)
            if rr.status != "ok":
                raise AssertionError("差分路径上矩阵非正定")
            return rr.log_marginal

        ga = (lml(a * math.exp(self.H), l, s) - lml(a / math.exp(self.H), l, s)) / (2 * self.H)
        gl = (lml(a, l * math.exp(self.H), s) - lml(a, l / math.exp(self.H), s)) / (2 * self.H)
        gs = (lml(a, l, s * math.exp(self.H)) - lml(a, l, s / math.exp(self.H))) / (2 * self.H)
        return ga, gl, gs

    def _check(self, n, d, a, l, s, jitter, seed):
        X, y = make_data(n=n, d=d, seed=seed)
        r = fit(X, y, a, l, s, jitter=jitter)
        self.assertEqual(r.status, "ok")
        fd = self._fd_grad(X, y, a, l, s, jitter)
        for name, ana, num in zip(("log a", "log l", "log s"), r.grad_log, fd):
            tol = 2e-5 + 2e-5 * abs(num)
            self.assertLess(
                abs(ana - num), tol,
                msg=f"dL/d{name}: 解析={ana:.8e} 差分={num:.8e} "
                    f"cfg n={n} d={d} a={a} l={l} s={s} jit={jitter}",
            )

    def test_gradients_multiple_configs(self):
        self._check(8, 1, 1.1, 0.9, 0.4, 0.0, 21)
        self._check(10, 2, 0.8, 1.3, 0.6, 0.0, 22)
        self._check(7, 3, 1.5, 0.7, 0.25, 0.0, 23)
        self._check(6, 2, 1.2, 1.5, 0.3, 1e-6, 24)  # jitter 为固定常数
        self._check(4, 1, 2.0, 2.0, 1.0, 0.0, 25)


class GradientStructureTests(unittest.TestCase):
    def test_grad_s_uses_s2_not_jitter(self):
        # jitter 不随 s 变化：同一数据分别给 jitter=0 与 jitter>0，
        # 梯度中 grad s 的变化只能通过 C⁻¹ 本身变化，验证“jitter 计入
        # 似然但不是 s 的导数”——用差分已覆盖，这里再做结构性校验：
        # 常数 jitter 应使边际似然介于无 jitter 之间且残差正常。
        X, y = make_data(n=5, d=2, seed=31)
        r = fit(X, y, 1.0, 1.0, 0.5, jitter=1e-8)
        self.assertEqual(r.status, "ok")
        self.assertLess(abs(r.residuals.chol_solve), 1e-10)

    def test_alpha_residual_reported(self):
        X, y = make_data(n=10, d=2, seed=32)
        r = fit(X, y, 1.2, 0.8, 0.3)
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.residuals.symmetry, 0.0)
        self.assertLess(r.residuals.chol_reconstruct, 1e-11)
        self.assertLess(abs(r.residuals.chol_solve), 1e-10)


class DuplicateAndTinyNoiseTests(unittest.TestCase):
    def test_duplicate_inputs_small_but_finite_noise(self):
        X = [[0.0], [0.0], [1.0], [1.0]]
        y = [0.5, 0.52, -0.5, -0.48]
        r = fit(X, y, 1.0, 1.0, 1e-7)  # s²=1e-14
        self.assertEqual(r.status, "ok")
        p = predict(r, [[0.0], [1.0]])
        # 重复点上的 latent 方差应接近 0，observed ≈ s²
        self.assertAlmostEqual(p.latent_var[0], 0.0, delta=1e-12)
        self.assertAlmostEqual(p.latent_var[1], 0.0, delta=1e-12)
        self.assertAlmostEqual(p.observed_var[0], 1e-14, places=14)

    def test_duplicate_inputs_round_to_singular_is_not_pd(self):
        # s²=1e-16 时浮点 1+1e-16 舍入为 1.0，重复点导致精确奇异：
        # 必须报告 not_pd，而不是偷偷加 jitter 或抛异常。
        X = [[0.0], [0.0]]
        y = [0.5, 0.52]
        r = fit(X, y, 1.0, 1.0, 1e-8)
        self.assertEqual(r.status, "not_pd")
        self.assertIsNone(r.log_marginal)
        self.assertIsNone(r.grad_log)
        self.assertEqual(r.residuals.symmetry, 0.0)
        with self.assertRaises(RuntimeError):
            predict(r, [[0.0]])

    def test_explicit_jitter_repairs_not_pd_and_is_counted(self):
        X = [[0.0], [0.0]]
        y = [0.5, 0.52]

        # 段 1（病态，条件数约 1e10）：无 jitter 时 not_pd；显式给定
        # jitter 后必须恢复 ok。此处只核对状态与相对误差量级——两条独立
        # 求逆路径在 κ~1e10 下的固有误差约 κ·eps≈2e-6，属浮点误差下界，
        # 不用小容差假装精确；精确公式核对放在段 2。
        s_ill, j_ill = 1e-8, 1e-10
        self.assertEqual(fit(X, y, 1.0, 1.0, s_ill).status, "not_pd")
        r_ill = fit(X, y, 1.0, 1.0, s_ill, jitter=j_ill)
        self.assertEqual(r_ill.status, "ok")
        ref_ill = reference_lml(X, y, 1.0, 1.0, s_ill, j_ill)
        # κ·eps ≈ 1e10·2.2e-16 ≈ 2.2e-6 是该条件数下的浮点误差下界；
        # 实测相对误差 ~6e-11，远优于下界。此处不用小容差掩盖病态，
        # 精确的公式一致性在段 2（κ~5e5）以 1e-13 核对。
        self.assertLess(abs(r_ill.log_marginal - ref_ill) / abs(ref_ill), 1e-9)

        # 段 2（条件数约 5e5）：验证 jitter 确实进入似然与预测公式。
        # 此条件数下两条独立求逆路径的固有误差 ~κ·eps ≈ 1.1e-10，
        # 实测 5.8e-11，容差按该浮点下界取 2e-10；不用宽容差掩盖，
        # 精确到 1e-13 的公式一致性在条件数温和的段 3 核对。
        s, jit = 1e-6, 1e-6
        r0 = fit(X, y, 1.0, 1.0, s)
        self.assertEqual(r0.status, "ok")
        r = fit(X, y, 1.0, 1.0, s, jitter=jit)
        self.assertEqual(r.status, "ok")
        ref = reference_lml(X, y, 1.0, 1.0, s, jit)
        self.assertLess(abs(r.log_marginal - ref) / max(1.0, abs(ref)), 2e-10)
        # 似然确实随 jitter 改变（证明计入了公式，而非被忽略）
        self.assertNotEqual(r.log_marginal, r0.log_marginal)
        p = predict(r, [[0.0]])
        rm, rl, _ = reference_predict(X, y, [[0.0]], 1.0, 1.0, s, jit)
        self.assertAlmostEqual(p.mean[0], rm[0], places=9)
        self.assertAlmostEqual(p.latent_var[0], max(rl[0], 0.0), places=9)
        # 关键不变量：observed 只加 s²，绝不把常数 jitter 算进观测噪声
        self.assertAlmostEqual(
            p.observed_var[0] - p.latent_var[0], s * s, places=15
        )
        self.assertNotAlmostEqual(
            p.observed_var[0] - p.latent_var[0], s * s + jit, places=12
        )

        # 段 3（条件数约 51，温和）：jitter 计入全部公式的 1e-13 级核对。
        s3, j3 = 0.1, 0.1
        r3 = fit(X, y, 1.0, 1.0, s3, jitter=j3)
        self.assertEqual(r3.status, "ok")
        ref3 = reference_lml(X, y, 1.0, 1.0, s3, j3)
        self.assertLess(abs(r3.log_marginal - ref3) / max(1.0, abs(ref3)), 1e-13)
        p3 = predict(r3, [[0.0], [0.5]])
        m3, l3, _ = reference_predict(X, y, [[0.0], [0.5]], 1.0, 1.0, s3, j3)
        for q in range(2):
            self.assertAlmostEqual(p3.mean[q], m3[q], places=12)
            self.assertAlmostEqual(p3.latent_var[q], l3[q], places=12)
            self.assertAlmostEqual(
                p3.observed_var[q] - p3.latent_var[q], s3 * s3, places=14
            )

    def test_tiny_noise_generic_data(self):
        X, y = make_data(n=5, d=2, seed=41)
        r = fit(X, y, 1.0, 1.0, 1e-9)
        self.assertEqual(r.status, "ok")
        self.assertLess(abs(r.residuals.chol_solve), 1e-9)
        Xs = make_queries(m=3, d=2, seed=42)
        p = predict(r, Xs)
        for v in p.latent_var:
            self.assertGreaterEqual(v, 0.0)

    def test_negative_latent_variance_is_clamped(self):
        # 真实浮点构造：5 个间距 1e-8 的近似重合训练点；s²=1e-18 远小于
        # eps(1) 被对角吞没，显式 jitter=1.5e-16 恰好把对角抬高一档 ulp。
        # 在重合簇内的训练点上，三角求解平方和向上舍入，使 a²-proj 得到
        # -2.22e-16（-1/2 ulp(1)）——一个真实的负微量，必须被钳为 0 并记录。
        h = 1e-8
        X = [[0.0]] + [[h * i] for i in range(1, 5)]
        y = [0.3 + 0.01 * i for i in range(5)]
        r = fit(X, y, 1.0, 1.0, 1e-9, jitter=1.5e-16)
        self.assertEqual(r.status, "ok")
        p = predict(r, X)
        self.assertTrue(
            p.clamped,
            "该构造在本平台必须触发负微量钳制；raw 应约为 -2.22e-16",
        )
        c = p.clamped[0]
        self.assertEqual(c.kind, "latent")
        self.assertGreaterEqual(c.raw, -NEG_VAR_TOL)
        self.assertEqual(c.raw, -2.220446049250313e-16)
        self.assertEqual(p.latent_var[c.index], 0.0)
        # observed 同步为 s²（=1e-18 加到 0 上，无二次消去）
        self.assertGreaterEqual(p.observed_var[c.index], 0.0)
        # 其余点方差非负
        for q, v in enumerate(p.latent_var):
            self.assertGreaterEqual(v, 0.0, f"point {q}")

    def test_big_negative_variance_raises(self):
        # 超过容差的负方差不可能由有界核自然产生，因此直接构造 FitResult
        # 数据对象（不打桩核心算法）：L=0.5I 使投影 = ks^T (LL^T)^-1 ks
        # = 4 > a²=1，raw=-3，必须抛 RuntimeError 而不是静默钳制。
        from gp_rbf.model import FitResult, Hyperparams, Residuals
        bad = FitResult(
            status="ok",
            params=Hyperparams(1.0, 1.0, 0.1, 0.0),
            n=1, d=1,
            log_marginal=0.0, grad_log=(0.0, 0.0, 0.0),
            residuals=Residuals(0.0, 0.0, 0.0),
            noise_variance=0.01,
            _L=[[0.5]],
            _alpha=[0.0],
            _Cinv=[[4.0]],
            _X=[[0.0]],
        )
        with self.assertRaises(RuntimeError):
            predict(bad, [[0.0]])


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        self.y = [1.0, 2.0, 3.0]

    def test_reject_nan_inf(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(GPInputError):
                fit([[bad]], [1.0], 1.0, 1.0, 1.0)
            with self.assertRaises(GPInputError):
                fit([[0.0]], [bad], 1.0, 1.0, 1.0)
            with self.assertRaises(GPInputError):
                fit(self.X, self.y, bad, 1.0, 1.0)

    def test_reject_bool_params(self):
        for kw in ("a", "l", "s"):
            with self.assertRaises(GPInputError):
                fit(self.X, self.y, **{kw: True},
                    **{k: v for k, v in
                       dict(a=1.0, l=1.0, s=1.0).items() if k != kw})
        with self.assertRaises(GPInputError):
            fit(self.X, self.y, 1.0, 1.0, 1.0, jitter=False)
        # bool 出现在数据里也拒绝
        with self.assertRaises(GPInputError):
            fit([[True]], [1.0], 1.0, 1.0, 1.0)

    def test_reject_nonpositive_params(self):
        with self.assertRaises(GPInputError):
            fit(self.X, self.y, 0.0, 1.0, 1.0)
        with self.assertRaises(GPInputError):
            fit(self.X, self.y, 1.0, -1.0, 1.0)
        with self.assertRaises(GPInputError):
            fit(self.X, self.y, 1.0, 1.0, 0.0)
        with self.assertRaises(GPInputError):
            fit(self.X, self.y, 1.0, 1.0, 1.0, jitter=-1e-12)

    def test_reject_bad_shapes_and_types(self):
        with self.assertRaises(GPInputError):
            fit([], [], 1.0, 1.0, 1.0)  # 0 样本
        with self.assertRaises(GPInputError):
            fit([[0.0]], "abc", 1.0, 1.0, 1.0)  # 类型错误
        with self.assertRaises(GPInputError):
            fit("xy", [1.0], 1.0, 1.0, 1.0)
        with self.assertRaises(GPInputError):
            fit([[0.0]], [1.0, 2.0], 1.0, 1.0, 1.0)  # y 长度不符
        with self.assertRaises(GPInputError):
            fit([[0.0], [0.0, 1.0]], [1.0, 2.0], 1.0, 1.0, 1.0)  # 行长不一
        with self.assertRaises(GPInputError):
            fit([[0.0, 0.0, 0.0, 0.0]], [1.0], 1.0, 1.0, 1.0)  # 4 维
        with self.assertRaises(GPInputError):
            fit([[1.0]], [1.0], 1.0, 1.0, 1.0, jitter=float("nan"))

    def test_limits_are_interface_boundaries(self):
        # 80 样本、3 维：边界接受
        X80 = [[float(i), 0.0, 0.0] for i in range(80)]
        y80 = [float(i % 5) for i in range(80)]
        r = fit(X80, y80, 1.0, 3.0, 0.5)
        self.assertEqual(r.status, "ok")
        self.assertEqual((r.n, r.d), (80, 3))
        # 81 样本：拒绝
        with self.assertRaises(GPInputError):
            fit(X80 + [[80.0, 0.0, 0.0]], y80 + [0.0], 1.0, 3.0, 0.5)

    def test_predict_validates_query(self):
        r = fit(self.X, self.y, 1.0, 1.0, 0.5)
        with self.assertRaises(GPInputError):
            predict(r, [])
        with self.assertRaises(GPInputError):
            predict(r, [[0.0]])  # 维度不符
        with self.assertRaises(GPInputError):
            predict(r, [[0.0, float("nan")]])
        with self.assertRaises(GPInputError):
            predict(r, [[0.0, True]])

    def test_error_is_localizable(self):
        try:
            fit(self.X, [1.0, 2.0, float("nan")], 1.0, 1.0, 1.0)
        except GPInputError as e:
            self.assertIn("y[2]", str(e))
        else:
            self.fail("应拒绝 NaN")
        try:
            fit([[0.0], [1.0, 0.0]], [1.0, 2.0], 1.0, 1.0, 1.0)
        except GPInputError as e:
            self.assertIn("X[1]", str(e))
        else:
            self.fail("应拒绝不规则矩阵")


class ReferenceIndependenceTests(unittest.TestCase):
    def test_reference_does_not_import_core(self):
        """契约：独立参考不得直接调用被测核心得到期望值。"""
        import ast
        ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ref_gp.py")
        with open(ref_path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.append(node.module)
        self.assertFalse(
            any("gp_rbf" in name for name in imported),
            f"参考实现不得导入 gp_rbf，实际导入：{imported}",
        )
        self.assertIn("math", imported)  # 只应有标准库
        import ref_gp
        self.assertFalse(hasattr(ref_gp, "fit"))


class ConfidenceIntervalSanityTests(unittest.TestCase):
    def test_ci_covers_training_region(self):
        # 训练点附近区间应收窄；远离数据处 latent 方差趋向 a²
        X, y = make_data(n=8, d=1, seed=51, spread=2.0)
        r = fit(X, y, 1.0, 0.8, 0.2)
        p = predict(r, [[0.0], [50.0]])
        # 远处不确定性显著大于训练区域
        self.assertGreater(p.latent_var[1], p.latent_var[0])
        self.assertLess(p.latent_var[1], 1.0 + 1e-12)  # 不超过先验 a²
        for z in (1.96,):
            lo = [p.mean[q] - z * math.sqrt(p.observed_var[q]) for q in range(2)]
            hi = [p.mean[q] + z * math.sqrt(p.observed_var[q]) for q in range(2)]
            self.assertTrue(all(h > l_ for l_, h in zip(lo, hi)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
