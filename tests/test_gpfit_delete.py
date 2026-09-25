"""
GPFit 不可变删除测试: 重复坐标有噪声、删除首/中/尾、连续删除、未知 ID、
删除至空的先验预测。期望值来自本文件独立参考实现 (Gauss-Jordan 直接逆,
与 tests/test_gprbf.py 同一独立参考风格) 以及 gprbf 模块级全量重建路径。
"""

import math
import unittest

import gprbf
from gprbf import GPFit, Hyperparameters, log_marginal, predict

HP = Hyperparameters(alpha=1.2, lengthscale=0.8, noise=0.15)
JIT = 1e-6


# ---------------------------------------------------------------------------
# 独立参考实现 (不调用 gprbf 产生期望值)
# ---------------------------------------------------------------------------


def ref_rbf(X1, X2, a, ell):
    a2 = a * a
    return [
        [a2 * math.exp(-sum((u - v) ** 2 for u, v in zip(x, z)) / (2 * ell * ell))
         for z in X2]
        for x in X1
    ]


def ref_inverse(A):
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) == 0.0:
            raise ZeroDivisionError("singular")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [M[r][k] - f * M[col][k] for k in range(2 * n)]
    return [row[n:] for row in M]


def ref_logdet(A):
    n = len(A)
    M = [list(row) for row in A]
    out = 0.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        out += math.log(abs(pv))
        for r in range(col + 1, n):
            f = M[r][col] / pv
            M[r][col] = 0.0
            for c in range(col + 1, n):
                M[r][c] -= f * M[col][c]
    return out


def ref_lml_and_predict(X, y, X_test, hp, jitter):
    n = len(X)
    var = hp.noise ** 2 + jitter
    Ky = ref_rbf(X, X, hp.alpha, hp.lengthscale)
    for i in range(n):
        Ky[i][i] += var
    Inv = ref_inverse(Ky)
    ai = [sum(Inv[i][j] * y[j] for j in range(n)) for i in range(n)]
    quad = sum(y[i] * ai[i] for i in range(n))
    lml = -0.5 * (quad + ref_logdet(Ky) + n * math.log(2 * math.pi))
    Ks = ref_rbf(X, X_test, hp.alpha, hp.lengthscale)
    means, lats = [], []
    for t in range(len(X_test)):
        ks = [Ks[i][t] for i in range(n)]
        means.append(sum(ks[i] * ai[i] for i in range(n)))
        lats.append(hp.alpha ** 2
                    - sum(ks[i] * sum(Inv[i][j] * ks[j] for j in range(n))
                          for i in range(n)))
    return lml, means, lats, var


def dup_data():
    # 重复坐标 + 有噪声 (y 在重复点上不一致, 模拟观测噪声)
    X = [(0.0,), (0.0,), (1.0,), (2.0,), (2.0,), (3.0,), (4.0,)]
    y = [0.5, 0.62, 1.1, -0.4, -0.55, 0.3, 0.9]
    ids = ("p0", "p1", "p2", "p3", "p4", "p5", "p6")
    return X, y, ids


def assert_matches_reference(test, fit, X_test):
    r = fit.log_marginal()
    test.assertEqual(r.status, "ok")
    lml, means, lats, var = ref_lml_and_predict(
        list(fit.X), list(fit.y), X_test, fit.hyperparameters, fit.jitter
    )
    test.assertAlmostEqual(r.log_marginal, lml, delta=abs(lml) * 1e-10 + 1e-11)
    p = fit.predict(X_test)
    test.assertEqual(p.status, "ok")
    for t in range(len(X_test)):
        test.assertAlmostEqual(p.mean[t], means[t], delta=1e-10)
        test.assertAlmostEqual(p.latent_variance[t], lats[t], delta=1e-9)
        test.assertAlmostEqual(p.observed_variance[t], lats[t] + var, delta=1e-9)
    # 数值残差诊断必须小
    test.assertLess(r.chol_residual, 1e-11)
    test.assertLess(r.solve_residual, 1e-11)
    return r, p


class TestDeletePositions(unittest.TestCase):
    def test_delete_first_middle_last(self):
        X, y, ids = dup_data()
        X_test = [(0.5,), (2.0,), (3.7,)]
        for pos, name in ((0, "first"), (3, "middle"), (6, "last")):
            with self.subTest(position=name):
                fit = GPFit.fit(X, y, HP, jitter=JIT, sample_ids=ids)
                before = fit.log_marginal().log_marginal
                new = fit.delete(ids[pos])
                # 稳定 ID: 删除后剩余 ID 不变且顺序压缩
                self.assertEqual(new.sample_ids, ids[:pos] + ids[pos + 1:])
                self.assertEqual(new.n_samples, len(X) - 1)
                # 与独立参考 (直接逆) 对照
                assert_matches_reference(self, new, X_test)
                # 与模块级全量重建对照 (梯度也一致)
                Xr = [x for k, x in enumerate(X) if k != pos]
                yr = [v for k, v in enumerate(y) if k != pos]
                ref = log_marginal(Xr, yr, HP, jitter=JIT)
                got = new.log_marginal()
                self.assertAlmostEqual(got.log_marginal, ref.log_marginal,
                                       delta=1e-10)
                for g, w in zip(got.gradient, ref.gradient):
                    self.assertAlmostEqual(g, w, delta=1e-8)
                # 旧对象保持原样可查询
                self.assertEqual(fit.n_samples, len(X))
                self.assertEqual(fit.sample_ids, ids)
                self.assertEqual(fit.log_marginal().log_marginal, before)
                self.assertEqual(fit.diagnostics, ())
                # 删除路径诊断: 走了 Cholesky 行删除而非全量
                rec = new.diagnostics[-1]
                self.assertEqual(rec.removed_id, ids[pos])
                self.assertEqual(rec.method, "cholesky_downdate")
                self.assertLess(rec.downdate_residual, 1e-8)

    def test_hyperparams_and_jitter_unchanged(self):
        X, y, ids = dup_data()
        fit = GPFit.fit(X, y, HP, jitter=JIT, sample_ids=ids)
        new = fit.delete("p2")
        self.assertEqual(new.hyperparameters, HP)
        self.assertEqual(new.jitter, JIT)
        self.assertEqual(new.log_marginal().effective_noise_variance,
                         HP.noise ** 2 + JIT)


class TestSequentialDeletes(unittest.TestCase):
    def test_delete_down_to_empty(self):
        X, y, ids = dup_data()
        X_test = [(1.5,)]
        fit = GPFit.fit(X, y, HP, jitter=JIT, sample_ids=ids)
        order = ["p3", "p0", "p6", "p1", "p5", "p2", "p4"]  # 中/首/尾交错
        for sid in order:
            new = fit.delete(sid)
            self.assertNotIn(sid, new.sample_ids)
            self.assertEqual(new.n_samples, fit.n_samples - 1)
            if new.n_samples > 0:
                assert_matches_reference(self, new, X_test)
            fit = new
        # 全部删完: 先验预测, 无空矩阵索引错误
        self.assertEqual(fit.n_samples, 0)
        r = fit.log_marginal()
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.log_marginal, 0.0)
        self.assertEqual(r.gradient, (0.0, 0.0, 0.0))
        p = fit.predict([(0.0,), (2.5,)])
        self.assertEqual(p.status, "ok")
        a2 = HP.alpha ** 2
        self.assertEqual(p.mean, (0.0, 0.0))
        self.assertEqual(p.latent_variance, (a2, a2))
        for ov in p.observed_variance:
            self.assertAlmostEqual(ov, a2 + HP.noise ** 2 + JIT, delta=1e-15)
        # 诊断链完整: 6 次 downdate + 1 次 trivial_empty
        self.assertEqual(len(fit.diagnostics), len(order))
        self.assertEqual(fit.diagnostics[-1].method, "trivial_empty")
        self.assertTrue(all(rec.method == "cholesky_downdate"
                            for rec in fit.diagnostics[:-1]))

    def test_ids_never_reused(self):
        X, y, ids = dup_data()
        fit = GPFit.fit(X, y, HP, jitter=JIT, sample_ids=ids)
        f2 = fit.delete("p1")
        with self.assertRaises(KeyError):
            f2.delete("p1")  # 已删除的 ID 不可再删 (不复用)
        # 剩余 ID 仍指向原样本
        idx = f2.sample_ids.index("p2")
        self.assertEqual(f2.X[idx], X[2])
        self.assertEqual(f2.y[idx], y[2])


class TestUnknownIdAndValidation(unittest.TestCase):
    def test_unknown_id_keyerror(self):
        X, y, ids = dup_data()
        fit = GPFit.fit(X, y, HP, jitter=JIT, sample_ids=ids)
        with self.assertRaises(KeyError):
            fit.delete("nope")
        with self.assertRaises(KeyError):
            fit.delete(0)  # ID 是字符串, 整数 0 未知
        self.assertEqual(fit.n_samples, len(X))  # 失败不留部分变更

    def test_default_ids_and_duplicate_ids_rejected(self):
        X, y, _ = dup_data()
        fit = GPFit.fit(X, y, HP)
        self.assertEqual(fit.sample_ids, tuple(range(len(X))))
        f2 = fit.delete(2)
        self.assertEqual(f2.sample_ids, (0, 1, 3, 4, 5, 6))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            GPFit.fit(X, y, HP, sample_ids=["a", "a", "b", "c", "d", "e", "f"])
        with self.assertRaisesRegex(ValueError, "length"):
            GPFit.fit(X, y, HP, sample_ids=["a", "b"])
        with self.assertRaises(TypeError):
            GPFit.fit(X, y, HP, sample_ids=[["x"]] * 7)

    def test_fit_not_pd_status_preserved(self):
        X = [(0.0,), (0.0,), (1.0,)]
        y = [0.5, 0.5, -0.5]
        hp = Hyperparameters(1.0, 1.0, 1e-10)
        fit = GPFit.fit(X, y, hp)
        self.assertEqual(fit.status, "not_pd")
        self.assertEqual(fit.log_marginal().status, "not_pd")
        self.assertEqual(fit.predict([(0.0,)]).status, "not_pd")


class TestFallbackRecorded(unittest.TestCase):
    def test_fallback_full_refit_recorded(self):
        # 人为构造: 破坏内部因子使行删除验收失败, 验证回退路径被记录且结果正确
        X, y, ids = dup_data()
        fit = GPFit.fit(X, y, HP, jitter=JIT, sample_ids=ids)
        bad_L = [list(row) for row in fit._L]
        bad_L[3][3] = -bad_L[3][3]  # 非法对角, 行删除必触发验收失败
        broken = GPFit._from_state(
            fit.X, fit.y, fit.sample_ids, HP, JIT, HP.noise ** 2 + JIT,
            [list(r) for r in fit._Ky], bad_L, "ok", 0.0, (), 1,
        )
        new = broken.delete("p5")
        rec = new.diagnostics[-1]
        self.assertEqual(rec.method, "full_refit")
        # 回退后结果仍与全量重建一致
        Xr = [x for k, x in enumerate(X) if k != 5]
        yr = [v for k, v in enumerate(y) if k != 5]
        ref = log_marginal(Xr, yr, HP, jitter=JIT)
        self.assertAlmostEqual(new.log_marginal().log_marginal,
                               ref.log_marginal, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
