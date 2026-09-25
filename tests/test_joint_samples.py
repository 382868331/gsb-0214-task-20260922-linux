"""
联合后验相关抽样的独立参考测试。

期望值全部由本文件内的独立实现产生 (Gauss-Jordan 求逆/独立核矩阵/显式满高斯
条件公式), 不调用 gprbf 的内部线性代数; gprbf 仅作为被测对象。
"""

import math
import unittest

import gprbf
from gprbf import Hyperparameters, joint_posterior, sample_posterior


def ref_kernel(X1, X2, a, ell):
    a2 = a * a
    out = []
    for x in X1:
        row = []
        for z in X2:
            d2 = sum((u - v) ** 2 for u, v in zip(x, z))
            row.append(a2 * math.exp(-d2 / (2.0 * ell * ell)))
        out.append(row)
    return out


def ref_gj_inverse(A):
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) <= 0.0:
            raise ZeroDivisionError("singular")
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


def ref_posterior(X, y, Xt, a, ell, s, jitter=0.0):
    """独立计算联合后验均值与满协方差 (Gauss-Jordan 显式逆)。"""
    n = len(X)
    Ky = ref_kernel(X, X, a, ell)
    var = s * s + jitter
    for i in range(n):
        Ky[i][i] += var
    Inv = ref_gj_inverse(Ky)
    Ks = ref_kernel(X, Xt, a, ell)
    Kss = ref_kernel(Xt, Xt, a, ell) if Xt else []
    m = len(Xt)
    mean = []
    for t in range(m):
        ai_y = [sum(Inv[i][j] * y[j] for j in range(n)) for i in range(n)]
        mean.append(sum(Ks[i][t] * ai_y[i] for i in range(n)))
    cov = [[0.0] * m for _ in range(m)]
    for u in range(m):
        for v in range(m):
            term = sum(
                Ks[i][u] * sum(Inv[i][j] * Ks[j][v] for j in range(n))
                for i in range(n)
            )
            cov[u][v] = Kss[u][v] - term
    return mean, cov


def ref_jacobi(A):
    n = len(A)
    S = [list(r) for r in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(100):
        off = sum(S[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
        if off <= 1e-30:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = S[p][q]
                if apq == 0.0:
                    continue
                tau = (S[q][q] - S[p][p]) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for kk in range(n):
                    sp, sq = S[kk][p], S[kk][q]
                    S[kk][p] = c * sp - s * sq
                    S[kk][q] = s * sp + c * sq
                for kk in range(n):
                    sp, sq = S[p][kk], S[q][kk]
                    S[p][kk] = c * sp - s * sq
                    S[q][kk] = s * sp + c * sq
                for kk in range(n):
                    vp, vq = V[kk][p], V[kk][q]
                    V[kk][p] = c * vp - s * vq
                    V[kk][q] = s * vp + c * vq
                S[p][q] = S[q][p] = 0.0
    w = [S[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: w[i])
    V = [[V[i][order[j]] for j in range(n)] for i in range(n)]
    return [w[i] for i in order], V


def ref_conditional(mean, cov, fixed_map):
    """独立满高斯条件公式 (截断谱伪逆, 仅在非零协方差子空间求解)。"""
    m = len(mean)
    idx = sorted(fixed_map)
    k = len(idx)
    CII = [[cov[i][j] for j in idx] for i in idx]
    w, V = ref_jacobi(CII) if k else ([], [])
    tol = 1e-12 * max(1.0, (w[-1] if k else 0.0))
    pos = [i for i in range(k) if w[i] > tol]
    Pinv = [[0.0] * k for _ in range(k)]
    for r in range(k):
        for c in range(k):
            Pinv[r][c] = sum(V[r][i] * V[c][i] / w[i] for i in pos)
    g = [fixed_map[idx[t]] - mean[idx[t]] for t in range(k)]
    sol = [sum(Pinv[u][t] * g[t] for t in range(k)) for u in range(k)]
    proj = [sum(CII[u][v] * sol[v] for v in range(k)) for u in range(k)]
    null_res = [g[u] - proj[u] for u in range(k)]
    scale = max([1.0] + [abs(v) for v in g])
    worst = max((abs(v) for v in null_res), default=0.0)
    if worst > 1e-9 * scale:
        return None, None, None, True, worst
    K = [[sum(cov[r][idx[u]] * Pinv[u][t] for u in range(k)) for t in range(k)]
         for r in range(m)]
    mean_c = [mean[r] + sum(K[r][t] * g[t] for t in range(k)) for r in range(m)]
    cov_c = [[cov[r][c] - sum(K[r][t] * cov[idx[t]][c] for t in range(k))
              for c in range(m)] for r in range(m)]
    w2, _ = ref_jacobi(cov_c) if m else ([], None)
    tol2 = 1e-12 * max(1.0, (w2[-1] if m else 0.0))
    rank_c = sum(1 for lam in w2 if lam > tol2)
    return mean_c, cov_c, rank_c, False, 0.0


X = [(0.0,), (1.0,), (2.0,), (3.0,), (4.0,)]
y = [1.0, 0.0, -1.0, 0.5, -0.5]
HP0 = Hyperparameters(alpha=1.2, lengthscale=1.1, noise=0.0)
HPP = Hyperparameters(alpha=1.2, lengthscale=1.1, noise=0.3)
Q_DUP = [(0.5,), (1.7,), (1.7,), (2.5,)]


class TestJointPosteriorAgainstReference(unittest.TestCase):
    def test_mean_and_full_covariance_match_reference(self):
        p = joint_posterior(X, y, Q_DUP, HPP)
        m_ref, c_ref = ref_posterior(
            X, y, Q_DUP, HPP.alpha, HPP.lengthscale, HPP.noise
        )
        self.assertEqual(p.status, "ok")
        for t in range(4):
            self.assertAlmostEqual(p.mean[t], m_ref[t], delta=1e-11)
            for u in range(4):
                self.assertAlmostEqual(p.covariance[t][u], c_ref[t][u], delta=1e-10)

    def test_marginal_matches_pointwise_predict(self):
        pred = gprbf.predict(X, y, Q_DUP, HPP)
        p = joint_posterior(X, y, Q_DUP, HPP)
        for t in range(4):
            self.assertAlmostEqual(p.mean[t], pred.mean[t], delta=1e-12)
            self.assertAlmostEqual(
                p.covariance[t][t], pred.latent_variance[t], delta=1e-10
            )

    def test_duplicate_query_points_rank_deficient_but_psd(self):
        p = joint_posterior(X, y, Q_DUP, HPP)
        # 两个完全相同的查询点 (下标 1,2) => 秩比 m 小 1, 但协方差半正定
        self.assertEqual(p.rank, 3)
        self.assertGreaterEqual(p.eigenvalues[0], -1e-12)
        self.assertAlmostEqual(
            p.covariance[1][2], p.covariance[1][1], delta=1e-11
        )
        # 因子的两行逐位相同 => 样本在重复坐标上完全相关
        self.assertEqual(p.factor[1], p.factor[2])

    def test_zero_noise_training_points_are_deterministic(self):
        q = [(2.0,), (0.0,), (1.3,)]  # 前两个是训练点
        p = joint_posterior(X, y, q, HP0, allow_zero_noise=True)
        self.assertEqual(p.status, "ok")
        self.assertEqual(p.covariance[0][0], 0.0)
        self.assertEqual(p.covariance[1][1], 0.0)
        self.assertAlmostEqual(p.mean[0], y[2], delta=1e-11)
        self.assertAlmostEqual(p.mean[1], y[0], delta=1e-11)
        self.assertGreater(p.covariance[2][2], 0.0)

    def test_zero_noise_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "hp.noise"):
            joint_posterior(X, y, Q_DUP, HP0)

    def test_empty_query(self):
        p = joint_posterior(X, y, [], HPP)
        self.assertEqual(p.status, "ok")
        self.assertEqual(p.mean, ())
        self.assertEqual(p.covariance, ())
        self.assertEqual(p.factor, ())
        self.assertEqual(p.rank, 0)
        p0 = joint_posterior(X, y, [], HP0, allow_zero_noise=True)
        self.assertEqual(p0.status, "ok")


class TestSamplesUseFullCovariance(unittest.TestCase):
    Z = tuple(
        (0.37 * (i + 1) - 0.21 * j, 1.13 * i - 0.07 * (j + 2),
         -0.55 * i + 0.83 * j)
        for i in range(6) for j in range(1)
    )

    def test_samples_equal_mean_plus_Z_FT_exactly(self):
        p = joint_posterior(X, y, Q_DUP, HPP)
        s = sample_posterior(X, y, Q_DUP, HPP, self.Z)
        self.assertEqual(s.status, "ok")
        self.assertEqual(s.rank, p.rank)
        F = p.factor
        for r, row in enumerate(self.Z):
            for t in range(4):
                expect = p.mean[t] + sum(F[t][c] * row[c] for c in range(p.rank))
                self.assertAlmostEqual(s.samples[r][t], expect, delta=1e-12)

    def test_duplicate_coordinates_are_bitwise_identical(self):
        s = sample_posterior(X, y, Q_DUP, HPP, self.Z)
        for row in s.samples:
            self.assertEqual(row[1], row[2])

    def test_samples_not_independent_empirical_cross_covariance(self):
        # 20 万次确定性线性同余生成的标准正态数, 经验协方差必须重现满协方差
        # (包括非零的交叉项; 独立逐点抽样无法通过此核对)。
        seed = 1234567
        rows = []
        for _ in range(200_000):
            vals = []
            for _ in range(3):
                # Box-Muller, 独立 LCG 流
                seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
                u1 = (seed >> 8) / 8388608.0 + 1e-9
                seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
                u2 = (seed >> 8) / 8388608.0
                vals.append(math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2))
            rows.append(tuple(vals))
        s = sample_posterior(X, y, Q_DUP, HPP, rows)
        p = joint_posterior(X, y, Q_DUP, HPP)
        n = len(rows)
        max_err = 0.0
        cross_seen = 0.0
        for a in range(4):
            for b in range(a, 4):
                cov = sum(
                    (s.samples[i][a] - p.mean[a]) * (s.samples[i][b] - p.mean[b])
                    for i in range(n)
                ) / n
                max_err = max(max_err, abs(cov - p.covariance[a][b]))
                if a != b:
                    cross_seen = max(cross_seen, abs(cov))
        self.assertLess(max_err, 5e-4)
        self.assertGreater(cross_seen, 0.01)


class TestConditioning(unittest.TestCase):
    def test_conditional_matches_full_gaussian_formula(self):
        m_ref, c_ref = ref_posterior(
            X, y, Q_DUP, HPP.alpha, HPP.lengthscale, HPP.noise
        )
        fixed = {0: 0.88, 3: -0.41}
        mc, cc, rc, bad, _ = ref_conditional(m_ref, c_ref, fixed)
        self.assertFalse(bad)
        z = tuple(tuple(0.11 * i * (c + 1) - 0.37 * c for c in range(rc))
                  for i in range(5))
        s = sample_posterior(X, y, Q_DUP, HPP, z, fixed=fixed)
        self.assertEqual(s.status, "ok")
        self.assertEqual(s.rank, rc)
        for t in range(4):
            self.assertAlmostEqual(s.mean[t], mc[t], delta=1e-10)
            for u in range(4):
                self.assertAlmostEqual(s.covariance[t][u], cc[t][u], delta=1e-10)

    def test_fixed_coordinates_held_exactly_including_duplicates(self):
        fixed = {1: 0.625}
        z = ((0.4, -0.9),)
        s = sample_posterior(X, y, Q_DUP, HPP, z, fixed=fixed)
        self.assertEqual(s.status, "ok")
        for row in s.samples:
            self.assertEqual(row[1], 0.625)
            self.assertEqual(row[2], 0.625)  # 重复点精确同步

    def test_incompatible_duplicate_values_rejected(self):
        s = sample_posterior(
            X, y, Q_DUP, HPP, ((),), fixed={1: 0.5, 2: 0.6}
        )
        self.assertEqual(s.status, "incompatible_fixed")
        self.assertIsNone(s.samples)

    def test_incompatible_zero_variance_observation_rejected(self):
        q = [(2.0,), (0.0,), (1.3,)]  # 前两个是零噪声训练点, 零方差
        s = sample_posterior(
            X, y, q, HP0, ((),), fixed={0: y[2] + 1.0},
            allow_zero_noise=True,
        )
        self.assertEqual(s.status, "incompatible_fixed")
        self.assertIn("zero-variance", s.incompatible_detail)

    def test_compatible_zero_variance_observation_held_exactly(self):
        q = [(2.0,), (1.3,)]
        s = sample_posterior(
            X, y, q, HP0, ((0.7,), (-1.2,)), fixed={0: y[2]},
            allow_zero_noise=True,
        )
        self.assertEqual(s.status, "ok")
        for row in s.samples:
            self.assertEqual(row[0], float(y[2]))

    def test_conditional_samples_reproduce_conditional_law(self):
        m_ref, c_ref = ref_posterior(
            X, y, Q_DUP, HPP.alpha, HPP.lengthscale, HPP.noise
        )
        fixed = {0: 0.88}
        mc, cc, rc, bad, _ = ref_conditional(m_ref, c_ref, fixed)
        self.assertFalse(bad)
        seed = 987654
        rows = []
        for _ in range(200_000):
            vals = []
            for _ in range(rc):
                seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
                u1 = (seed >> 8) / 8388608.0 + 1e-9
                seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
                u2 = (seed >> 8) / 8388608.0
                vals.append(math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2))
            rows.append(tuple(vals))
        s = sample_posterior(X, y, Q_DUP, HPP, rows, fixed=fixed)
        n = len(rows)
        max_err = 0.0
        for a in range(4):
            mhat = sum(r[a] for r in s.samples) / n
            max_err = max(max_err, abs(mhat - mc[a]))
            for b in range(a, 4):
                cov = sum(
                    (s.samples[i][a] - mc[a]) * (s.samples[i][b] - mc[b])
                    for i in range(n)
                ) / n
                max_err = max(max_err, abs(cov - cc[a][b]))
        self.assertLess(max_err, 5e-4)

    def test_empty_query_samples_and_no_fixed_allowed(self):
        s = sample_posterior(X, y, [], HPP, ())
        self.assertEqual(s.status, "ok")
        self.assertEqual(s.samples, ())
        self.assertEqual(s.rank, 0)
        s2 = sample_posterior(X, y, [], HPP, ((), ()))
        self.assertEqual(s2.status, "ok")
        self.assertEqual(len(s2.samples), 2)
        with self.assertRaises(ValueError):
            sample_posterior(X, y, [], HPP, (), fixed={0: 1.0})


class TestFactorAndInputValidation(unittest.TestCase):
    def test_not_pd_propagates_and_no_jitter_silently_added(self):
        Xd = [(0.0,), (0.0,)]
        yd = [1.0, 1.0]
        hp_tiny = Hyperparameters(1.0, 1.0, 1e-10)
        p = joint_posterior(Xd, yd, [(0.5,)], hp_tiny)
        self.assertEqual(p.status, "not_pd")
        s = sample_posterior(Xd, yd, [(0.5,)], hp_tiny, ((),))
        self.assertEqual(s.status, "not_pd")
        # 显式 jitter 后恢复
        p2 = joint_posterior(Xd, yd, [(0.5,)], hp_tiny, jitter=1e-6)
        self.assertEqual(p2.status, "ok")

    def test_jacobi_reconstructs_matrix(self):
        A = [[2.0, -1.0, 0.3], [-1.0, 2.0, -0.5], [0.3, -0.5, 1.0]]
        w, V = gprbf.jacobi_eigen(A)
        for k in range(3):
            Av = [sum(A[i][j] * V[j][k] for j in range(3)) for i in range(3)]
            for i in range(3):
                self.assertAlmostEqual(Av[i], w[k] * V[i][k], delta=1e-11)
        # 特征向量正交
        for k in range(3):
            for h in range(k, 3):
                dot = sum(V[i][k] * V[i][h] for i in range(3))
                self.assertAlmostEqual(dot, 1.0 if k == h else 0.0, delta=1e-11)

    def test_small_negative_eigenvalue_corrected_and_recorded(self):
        f = gprbf.psd_factor([[1.0, 0.0], [0.0, -1e-12]])
        self.assertEqual(f["status"], "ok")
        self.assertEqual(f["rank"], 1)
        self.assertEqual(len(f["corrected"]), 1)
        self.assertAlmostEqual(f["corrected"][0][1], -1e-12, delta=1e-18)
        FFt = [
            [sum(f["factor"][i][c] * f["factor"][j][c] for c in range(f["rank"]))
             for j in range(2)] for i in range(2)
        ]
        self.assertAlmostEqual(FFt[0][0], 1.0, delta=1e-12)
        self.assertEqual(FFt[1][1], 0.0)

    def test_clearly_non_psd_rejected_without_noise_masking(self):
        p = gprbf.psd_factor([[1.0, 0.0], [0.0, -0.5]])
        self.assertEqual(p["status"], "not_psd")
        self.assertAlmostEqual(p["rejected"], -0.5, delta=1e-12)
        self.assertIsNone(p["factor"])

    def test_bad_Z_shape_is_status_not_exception(self):
        p = joint_posterior(X, y, Q_DUP, HPP)
        s = sample_posterior(X, y, Q_DUP, HPP, ((0.0, 0.0),))
        self.assertEqual(s.status, "bad_shape")
        self.assertIn("standard normals", s.incompatible_detail)
        s2 = sample_posterior(X, y, Q_DUP, HPP, ((0.0,) * p.rank, (0.0,)))
        self.assertEqual(s2.status, "bad_shape")

    def test_nonfinite_Z_rejected(self):
        with self.assertRaisesRegex(ValueError, r"Z\[0\]\[1\]"):
            sample_posterior(X, y, Q_DUP, HPP, ((0.0, float("nan"), 0.0),))

    def test_caller_supplied_Z_used_deterministically(self):
        z1 = ((0.5, -0.5, 0.25),)
        z2 = ((0.5, -0.5, 0.25),)
        z3 = ((-0.5, 0.5, -0.25),)
        s1 = sample_posterior(X, y, Q_DUP, HPP, z1)
        s2 = sample_posterior(X, y, Q_DUP, HPP, z2)
        s3 = sample_posterior(X, y, Q_DUP, HPP, z3)
        self.assertEqual(s1.samples, s2.samples)
        for t in range(4):
            self.assertAlmostEqual(
                s1.samples[0][t] - joint_posterior(X, y, Q_DUP, HPP).mean[t],
                -(s3.samples[0][t] - joint_posterior(X, y, Q_DUP, HPP).mean[t]),
                delta=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
