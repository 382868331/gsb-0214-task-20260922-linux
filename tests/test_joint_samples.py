import math
import unittest

import gprbf
from gprbf import Hyperparameters, sample_posterior


def kernel(a, b, alpha=1.2, ell=0.8):
    d2 = sum((u - v) ** 2 for u, v in zip(a, b))
    return alpha * alpha * math.exp(-d2 / (2.0 * ell * ell))


def matrix(x1, x2, alpha=1.2, ell=0.8):
    return [[kernel(a, b, alpha, ell) for b in x2] for a in x1]


def inverse(A):
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        value = M[piv][col]
        if abs(value) < 1e-12:
            raise AssertionError("reference expected a positive-definite matrix")
        M[col], M[piv] = M[piv], M[col]
        M[col] = [v / M[col][col] for v in M[col]]
        for row in range(n):
            if row == col:
                continue
            factor = M[row][col]
            M[row] = [M[row][j] - factor * M[col][j] for j in range(2 * n)]
    return [row[n:] for row in M]


def posterior_reference(X, y, Xq):
    K = matrix(X, X)
    Ks = matrix(X, Xq)
    Kqq = matrix(Xq, Xq)
    Ki = inverse(K)
    n = len(X)
    m = len(Xq)
    mean = [
        sum(Ks[i][t] * sum(Ki[i][j] * y[j] for j in range(n)) for i in range(n))
        for t in range(m)
    ]
    cov = [
        [
            Kqq[t][u]
            - sum(Ks[i][t] * sum(Ki[i][j] * Ks[j][u] for j in range(n)) for i in range(n))
            for u in range(m)
        ]
        for t in range(m)
    ]
    return mean, cov


class TestJointSamples(unittest.TestCase):
    def test_duplicate_queries_use_full_correlated_covariance(self):
        X = [(0.0,), (1.0,), (2.0,)]
        y = [0.2, -0.3, 0.5]
        hp = Hyperparameters(1.2, 0.8, 0.0)
        Xq = [(0.5,), (0.5,), (1.0,)]
        ref_mean, ref_cov = posterior_reference(X, y, Xq)

        result = sample_posterior(X, y, Xq, hp, [(2.0, 0.0, 0.0), (0.0, 3.0, 7.0)])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.corrected_eigenvalues, ())
        self.assertLess(result.covariance_residual, 1e-10)

        for i in range(3):
            self.assertAlmostEqual(result.mean[i], ref_mean[i], delta=1e-11)
            for j in range(3):
                self.assertAlmostEqual(result.covariance[i][j], ref_cov[i][j], delta=1e-10)

        deviation = 2.0 * math.sqrt(max(ref_cov[0][0], 0.0))
        self.assertAlmostEqual(result.samples[0][0], ref_mean[0] + deviation, delta=1e-10)
        self.assertEqual(result.samples[0][0], result.samples[0][1])
        self.assertAlmostEqual(result.samples[0][2], -0.3, delta=1e-11)
        self.assertEqual(result.samples[1][0], result.samples[1][1])
        self.assertAlmostEqual(result.samples[1][2], -0.3, delta=1e-11)

    def test_fixed_values_condition_only_nonzero_subspace(self):
        X = [(0.0,), (1.0,), (2.0,)]
        y = [0.2, -0.3, 0.5]
        hp = Hyperparameters(1.2, 0.8, 0.0)
        Xq = [(0.25,), (0.25,), (0.75,)]
        mean, cov = posterior_reference(X, y, Xq)
        fixed_value = 0.4
        conditional_mean = mean[2] + cov[2][0] / cov[0][0] * (fixed_value - mean[0])
        conditional_var = cov[2][2] - cov[2][0] * cov[0][2] / cov[0][0]

        result = sample_posterior(
            X, y, Xq, hp, [(0.7,)], fixed_values={0: fixed_value, 1: fixed_value}
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.samples[0][0], fixed_value)
        self.assertEqual(result.samples[0][1], fixed_value)
        self.assertAlmostEqual(
            result.samples[0][2],
            conditional_mean + 0.7 * math.sqrt(conditional_var),
            delta=1e-10,
        )
        self.assertAlmostEqual(result.mean[2], conditional_mean, delta=1e-10)
        self.assertAlmostEqual(result.covariance[2][2], conditional_var, delta=1e-10)
        self.assertEqual(max(max(abs(v) for v in row) for row in result.covariance[:2]), 0.0)

        rejected = sample_posterior(
            X, y, Xq, hp, [(0.7,)], fixed_values={0: 0.4, 1: 0.5}
        )
        self.assertEqual(rejected.status, "incompatible_observation")
        self.assertIsNone(rejected.samples)

    def test_duplicate_zero_noise_training_points(self):
        X = [(0.0,), (0.0,), (1.0,)]
        y = [0.2, 0.2, -0.1]
        hp = Hyperparameters(1.0, 0.7, 0.0)
        result = sample_posterior(
            X, y, [(0.5,), (0.5,), (0.0,)], hp, [(1.5, 0.0, 0.0)]
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.samples[0][0], result.samples[0][1])
        self.assertAlmostEqual(result.samples[0][2], 0.2, delta=1e-12)

        incompatible = sample_posterior(
            [(0.0,), (0.0,), (1.0,)], [0.2, 0.25, -0.1],
            [(0.5,)], hp, [(1.0,)],
        )
        self.assertEqual(incompatible.status, "incompatible_observation")

    def test_empty_query_returns_empty_samples_without_rng(self):
        result = sample_posterior(
            [(0.0,), (1.0,)], [0.2, -0.3], [],
            Hyperparameters(1.0, 1.0, 0.0), [(), (), ()],
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.samples), 3)
        self.assertEqual(result.samples, ((), (), ()))
        self.assertEqual(result.covariance, ())
        self.assertEqual(result.rank, 0)
        with self.assertRaises(ValueError):
            sample_posterior([(0.0,)], [0.0], [], Hyperparameters(1.0, 1.0, 0.0), [(1.0,)])

    def test_psd_rank_factor_records_tiny_negativity_and_rejects_hard_negativity(self):
        tiny = gprbf._psd_factor([[-1e-13, 0.0], [0.0, 1.0]])
        values, vectors, rank, corrected, residual = tiny
        self.assertEqual(rank, 1)
        self.assertEqual(len(corrected), 1)
        self.assertGreaterEqual(values[0], 0.0)
        self.assertLess(residual, 1e-11)
        self.assertIsNone(gprbf._psd_factor([[-1.0, 0.0], [0.0, 1.0]]))

    def test_caller_normal_matrix_shape_validation(self):
        X = [(0.0,), (1.0,)]
        y = [0.0, 0.1]
        hp = Hyperparameters(1.0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            sample_posterior(X, y, [(0.5,), (0.5,)], hp, [(1.0,)])
        with self.assertRaises(TypeError):
            sample_posterior(X, y, [(0.5,)], hp, "not a matrix")


if __name__ == "__main__":
    unittest.main()
