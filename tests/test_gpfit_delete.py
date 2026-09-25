"""
GPFit 不可变删除测试: 重复坐标有噪声、删除首/中/尾、连续删除、未知 ID、
删除最后一条后的先验预测; 全部与独立全量重建 (gprbf.log_marginal /
gprbf.predict 在剩余样本上重新分解) 对照数值与残差, 并检查旧对象不变。
"""

import math
import unittest

import gprbf
from gprbf import GPFit, Hyperparameters, fit_gp, log_marginal, predict


HP = Hyperparameters(alpha=1.2, lengthscale=0.8, noise=0.15)
JITTER = 1e-6


def make_dup_data():
    # 重复坐标 + 有噪声: (0,0) 与 (1,1) 各重复一次, y 为确定性生成
    X = [(0.0,), (0.0,), (0.5,), (1.0,), (1.0,), (1.7,), (2.6,)]
    y = [0.31, 0.27, 0.66, 1.02, 0.98, 1.35, 1.71]
    ids = ("p0", "p1", "p2", "p3", "p4", "p5", "p6")
    return X, y, ids


def rebuild(X, y, keep_idx):
    return [X[i] for i in keep_idx], [y[i] for i in keep_idx]


class TestDeleteAgainstRebuild(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.ids = make_dup_data()
        self.fit = fit_gp(self.X, self.y, HP, jitter=JITTER, ids=self.ids)
        self.Xt = [(0.25,), (0.9,), (2.0,)]

    def _check_against_rebuild(self, fit_new, keep_idx):
        Xr, yr = rebuild(self.X, self.y, keep_idx)
        m_new = fit_new.marginal()
        m_ref = log_marginal(Xr, yr, HP, jitter=JITTER)
        self.assertEqual(m_new.status, "ok")
        self.assertEqual(m_ref.status, "ok")
        self.assertAlmostEqual(m_new.log_marginal, m_ref.log_marginal, delta=1e-9)
        for got, want in zip(m_new.gradient, m_ref.gradient):
            self.assertAlmostEqual(got, want, delta=1e-8)
        for got, want in zip(m_new.alpha_vec, m_ref.alpha_vec):
            self.assertAlmostEqual(got, want, delta=1e-9)
        # 残差诊断量级不劣于全量重建
        self.assertLessEqual(m_new.chol_residual, 1e-10)
        self.assertLessEqual(m_new.solve_residual, 1e-10)

        p_new = fit_new.predict(self.Xt)
        p_ref = predict(Xr, yr, self.Xt, HP, jitter=JITTER)
        self.assertEqual(p_new.status, "ok")
        for got, want in zip(p_new.mean, p_ref.mean):
            self.assertAlmostEqual(got, want, delta=1e-9)
        for got, want in zip(p_new.latent_variance, p_ref.latent_variance):
            self.assertAlmostEqual(got, want, delta=1e-9)
        for got, want in zip(p_new.observed_variance, p_ref.observed_variance):
            self.assertAlmostEqual(got, want, delta=1e-9)
        self.assertLessEqual(p_new.solve_residual, 1e-10)

    def test_delete_first_middle_last(self):
        n = len(self.X)
        for pos, del_id in ((0, "p0"), (n // 2, "p3"), (n - 1, "p6")):
            with self.subTest(position=pos):
                fit_new = self.fit.delete(del_id)
                self.assertIsInstance(fit_new, GPFit)
                self.assertEqual(fit_new.n, n - 1)
                self.assertEqual(fit_new.factor_source, "downdate")
                self.assertIsNotNone(fit_new.downdate_residual)
                # 核参数与显式 jitter 不变
                self.assertEqual(fit_new.hyperparameters, HP)
                self.assertEqual(fit_new.jitter, JITTER)
                self.assertEqual(fit_new.effective_noise_variance,
                                 HP.noise ** 2 + JITTER)
                # ID 顺序压缩且不复用
                want_ids = self.ids[:pos] + self.ids[pos + 1:]
                self.assertEqual(fit_new.ids, want_ids)
                self.assertNotIn(del_id, fit_new.ids)
                self._check_against_rebuild(
                    fit_new, [i for i in range(n) if i != pos]
                )

    def test_consecutive_deletes(self):
        # 连续删除三条 (含重复坐标中的一个), 每步都与全量重建对照
        fit = self.fit
        alive = list(range(len(self.X)))
        for del_id in ("p1", "p5", "p2"):
            fit = fit.delete(del_id)
            alive.remove(self.ids.index(del_id))
            self.assertEqual(fit.factor_source, "downdate")
            self.assertEqual(fit.ids, tuple(self.ids[i] for i in alive))
            self._check_against_rebuild(fit, alive)
        self.assertEqual(fit.n, len(self.X) - 3)

    def test_old_fit_unchanged_after_delete(self):
        before_ids = self.fit.ids
        before_lml = self.fit.marginal().log_marginal
        before_pred = self.fit.predict(self.Xt)
        self.fit.delete("p2")
        self.fit.delete("p2")  # 重复删除同一 ID 同样合法 (作用于旧对象)
        self.assertEqual(self.fit.ids, before_ids)
        self.assertEqual(self.fit.n, len(self.X))
        self.assertEqual(self.fit.marginal().log_marginal, before_lml)
        after = self.fit.predict(self.Xt)
        self.assertEqual(after.mean, before_pred.mean)
        self.assertEqual(after.latent_variance, before_pred.latent_variance)

    def test_unknown_id_raises_keyerror(self):
        for bad in ("nope", 999, None):
            with self.subTest(bad=bad):
                with self.assertRaises(KeyError):
                    self.fit.delete(bad)
        # 已删除的 ID 在新对象上即未知 (ID 不复用)
        fit_new = self.fit.delete("p0")
        with self.assertRaises(KeyError):
            fit_new.delete("p0")

    def test_delete_last_point_gives_prior(self):
        fit1 = fit_gp([(1.0,)], [2.0], HP, jitter=JITTER)
        fit0 = fit1.delete(0)
        self.assertEqual(fit0.n, 0)
        self.assertEqual(fit0.ids, ())
        self.assertEqual(fit0.status, "ok")
        m = fit0.marginal()
        self.assertEqual(m.status, "ok")
        self.assertEqual(m.log_marginal, 0.0)
        self.assertEqual(m.gradient, (0.0, 0.0, 0.0))
        p = fit0.predict([(0.0,), (3.3,)])
        a2 = HP.alpha * HP.alpha
        self.assertEqual(p.status, "ok")
        self.assertEqual(p.mean, (0.0, 0.0))
        self.assertEqual(p.latent_variance, (a2, a2))
        var = HP.noise ** 2 + JITTER
        for ov in p.observed_variance:
            self.assertAlmostEqual(ov, a2 + var, delta=1e-15)
        # 空对象上再删任何 ID 都是未知 ID
        with self.assertRaises(KeyError):
            fit0.delete(0)

    def test_delete_down_to_empty_via_chain(self):
        X = [(0.0,), (1.0,)]
        fit = fit_gp(X, [0.5, -0.5], HP)
        fit = fit.delete(0)
        self.assertEqual(fit.ids, (1,))
        fit = fit.delete(1)
        self.assertEqual(fit.n, 0)
        p = fit.predict([(0.5,)])
        self.assertEqual(p.status, "ok")
        self.assertEqual(p.mean, (0.0,))

    def test_not_pd_fit_delete_refactors(self):
        # 重复坐标 + 极小噪声无 jitter: 初始 not_pd; 删除一个重复点后恢复
        Xd = [(0.0,), (0.0,), (1.0,)]
        yd = [0.5, 0.5, -0.5]
        hp_bad = Hyperparameters(1.0, 1.0, 1e-10)
        fit_bad = fit_gp(Xd, yd, hp_bad)
        self.assertEqual(fit_bad.status, "not_pd")
        self.assertEqual(fit_bad.marginal().status, "not_pd")
        self.assertEqual(fit_bad.predict([(0.5,)]).status, "not_pd")
        fit_ok = fit_bad.delete(0)
        self.assertEqual(fit_ok.factor_source, "refactor")
        self.assertEqual(fit_ok.status, "ok")
        m = fit_ok.marginal()
        m_ref = log_marginal(Xd[1:], yd[1:], hp_bad)
        self.assertAlmostEqual(m.log_marginal, m_ref.log_marginal, delta=1e-9)


class TestFitConstruction(unittest.TestCase):
    def test_default_ids_and_immutability_of_inputs(self):
        X, y, _ = make_dup_data()
        X_snapshot = [tuple(r) for r in X]
        y_snapshot = list(y)
        fit = fit_gp(X, y, HP, jitter=JITTER)
        self.assertEqual(fit.ids, tuple(range(len(X))))
        fit.delete(3)
        self.assertEqual([tuple(r) for r in X], X_snapshot)
        self.assertEqual(list(y), y_snapshot)

    def test_fit_matches_module_functions(self):
        X, y, _ = make_dup_data()
        fit = fit_gp(X, y, HP, jitter=JITTER)
        m_fit = fit.marginal()
        m_ref = log_marginal(X, y, HP, jitter=JITTER)
        self.assertEqual(m_fit.log_marginal, m_ref.log_marginal)
        self.assertEqual(m_fit.gradient, m_ref.gradient)
        p_fit = fit.predict([(0.1,)])
        p_ref = predict(X, y, [(0.1,)], HP, jitter=JITTER)
        self.assertEqual(p_fit.mean, p_ref.mean)
        self.assertEqual(p_fit.observed_variance, p_ref.observed_variance)

    def test_invalid_ids_rejected(self):
        X, y, _ = make_dup_data()
        n = len(X)
        with self.assertRaises(ValueError):  # 长度不符
            fit_gp(X, y, HP, ids=("a", "b"))
        with self.assertRaises(ValueError):  # 重复 ID
            fit_gp(X, y, HP, ids=("a",) * n)
        with self.assertRaises(TypeError):  # bool ID
            fit_gp(X, y, HP, ids=(True,) + tuple(range(1, n)))
        with self.assertRaises(ValueError):  # NaN ID
            fit_gp(X, y, HP, ids=(float("nan"),) + tuple(range(1, n)))
        with self.assertRaises(TypeError):  # 非 list/tuple
            fit_gp(X, y, HP, ids="abcdefg")

    def test_invalid_training_input_still_rejected(self):
        with self.assertRaises(ValueError):
            fit_gp([(float("nan"),)], [0.0], HP)
        with self.assertRaises(ValueError):
            fit_gp([(0.0,)], [0.0], HP, jitter=-1.0)
        with self.assertRaises(TypeError):
            fit_gp([(0.0,)], [0.0], (1.0, 1.0, 0.1))


if __name__ == "__main__":
    unittest.main()
