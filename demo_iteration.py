#!/usr/bin/env python3
"""
demo_iteration.py - 联合后验相关函数抽样的真实离线演示 (标准库, 无 sleep)。

运行: python demo_iteration.py   (实测约 1~3 秒, 上限 90 秒)

内容全部当场计算, 不使用账号/密钥/网络/预录结果:
  1. 零噪声训练点处的精确插值路径 (allow_zero_noise=True);
  2. 含两个重复查询点的满协方差联合后验: 秩亏损、相关结构因子;
  3. 调用方给定标准正态数矩阵 -> 相关函数样本, 重复坐标逐位相同;
  4. 固定观测值的路径条件化: 相容固定坐标精确保持, 不相容零方差观测被拒绝;
  5. 空查询边界; 6. 独立 Gauss-Jordan 参考核对样本均值/协方差。
"""

import math
import time

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


def ref_inverse(A):
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        pv = M[piv][col]
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [M[r][k] - f * M[col][k] for k in range(2 * n)]
    return [row[n:] for row in M]


def ref_posterior(X, y, Xt, a, ell):
    """零噪声满后验 (Gauss-Jordan 显式逆), 仅用于演示中的独立核对。"""
    Ky = ref_kernel(X, X, a, ell)
    Inv = ref_inverse(Ky)
    Ks = ref_kernel(X, Xt, a, ell)
    Kss = ref_kernel(Xt, Xt, a, ell)
    n, m = len(X), len(Xt)
    mean = [
        sum(Ks[i][t] * sum(Inv[i][j] * y[j] for j in range(n)) for i in range(n))
        for t in range(m)
    ]
    cov = [
        [Kss[u][v] - sum(
            Ks[i][u] * sum(Inv[i][j] * Ks[j][v] for j in range(n))
            for i in range(n)) for v in range(m)]
        for u in range(m)
    ]
    return mean, cov


def lcg_normals(seed, n_rows, n_cols):
    """确定性 Box-Muller 标准正态数 (调用方提供给抽样入口的 Z)。"""
    rows = []
    for _ in range(n_rows):
        vals = []
        for _ in range(n_cols):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            u1 = (seed >> 8) / 8388608.0 + 1e-9
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            u2 = (seed >> 8) / 8388608.0
            vals.append(math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))
        rows.append(tuple(vals))
    return tuple(rows)


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("demo_iteration: 联合后验相关函数抽样 (完整协方差, Python 标准库)")
    print("=" * 72)

    X = [(0.0,), (1.0,), (2.0,), (3.0,), (4.0,)]
    y = [1.0, 0.0, -1.0, 0.5, -0.5]
    hp = Hyperparameters(alpha=1.2, lengthscale=1.1, noise=0.0)
    # 两个重复查询点 (下标 1,2), 一个训练点 (下标 3), 两个普通点
    Q = [(0.5,), (1.7,), (1.7,), (3.0,), (3.6,)]
    print(f"训练点 n={len(X)} (零噪声精确插值), 查询点 m={len(Q)} (含 1 组重复点)")
    print()

    # ---- 1. 满协方差联合后验 ----
    post = joint_posterior(X, y, Q, hp, allow_zero_noise=True)
    assert post.status == "ok", post.status
    print(f"[联合后验] status={post.status}, m={len(post.mean)}, 数值秩 rank={post.rank}"
          f" (重复点使 rank < m)")
    print(f"  训练点 x=3.0 处: mean={post.mean[3]:.12f} (应 = y=0.5), "
          f"方差={post.covariance[3][3]:g}")
    print(f"  重复坐标 1/2 的因子行逐位相同: {post.factor[1] == post.factor[2]}")
    print(f"  负特征值修正记录: {post.corrected_eigenvalues}")
    print()

    # ---- 2. 独立参考核对满协方差 ----
    m_ref, c_ref = ref_posterior(X, y, Q, hp.alpha, hp.lengthscale)
    err_m = max(abs(post.mean[i] - m_ref[i]) for i in range(len(Q)))
    err_c = max(
        abs(post.covariance[i][j] - c_ref[i][j])
        for i in range(len(Q)) for j in range(len(Q))
    )
    print(f"[独立核对] vs Gauss-Jordan 满后验: 均值最大误差={err_m:.2e}, "
          f"协方差最大误差={err_c:.2e}")
    print()

    # ---- 3. 调用方提供 Z, 抽取相关函数样本 ----
    N = 60_000
    Z = lcg_normals(20260926, N, post.rank)
    drawn = sample_posterior(X, y, Q, hp, Z, allow_zero_noise=True)
    assert drawn.status == "ok", drawn.status
    print(f"[相关抽样] {N} 条函数样本, Z 形状=({N}, {post.rank}); "
          f"样本=mean+Z F^T (共享满协方差因子)")
    print("  前 3 条样本 (重复列 1/2 逐位相等, 训练点列恒为后验均值):")
    for r in range(3):
        print("   ", "  ".join(f"{v:+.6f}" for v in drawn.samples[r]))
    dup_ok = all(row[1] == row[2] for row in drawn.samples)
    train_ok = all(row[3] == drawn.mean[3] for row in drawn.samples)
    emp_cov = [
        [sum((drawn.samples[r][i] - post.mean[i]) * (drawn.samples[r][j] - post.mean[j])
             for r in range(N)) / N for j in range(len(Q))]
        for i in range(len(Q))
    ]
    emp_err = max(abs(emp_cov[i][j] - post.covariance[i][j])
                  for i in range(len(Q)) for j in range(len(Q)))
    print(f"  重复列逐位相等: {dup_ok}; 训练点列恒为后验均值 {drawn.mean[3]:.12f}: {train_ok}")
    print(f"  经验协方差 vs 满后验协方差最大误差: {emp_err:.2e} (含非零交叉项)")
    print()

    # ---- 4. 固定观测值路径条件化 ----
    fixed = {1: 0.88}  # 固定重复点 (下标 1,2 为同一查询点) 中的一个
    Zc = lcg_normals(777, 3, drawn.rank - 1)
    cond = sample_posterior(X, y, Q, hp, Zc, fixed=fixed, allow_zero_noise=True)
    assert cond.status == "ok", cond.status
    print(f"[条件化] 固定 f(x=1.7)={fixed[1]} (重复坐标组 1/2): 条件秩={cond.rank}, "
          f"固定下标={cond.fixed_indices}")
    for r in cond.samples:
        assert r[1] == 0.88 and r[2] == 0.88
    print("  重复点 1/2 同步精确取固定值 0.88 (固定坐标精确保持):")
    for r in cond.samples:
        print("   ", "  ".join(f"{v:+.6f}" for v in r))
    print()

    bad = sample_posterior(
        X, y, Q, hp, ((),), fixed={1: 0.5, 2: 0.6}, allow_zero_noise=True
    )
    print(f"[拒绝边界] 给重复点指定不同值: status={bad.status!r}")
    print(f"  原因: {bad.incompatible_detail}")
    bad2 = sample_posterior(
        X, y, Q, hp, ((),), fixed={3: 0.99}, allow_zero_noise=True
    )
    print(f"[拒绝边界] 给零方差训练点指定不相容值: status={bad2.status!r}")
    print(f"  原因: {bad2.incompatible_detail[:96]}...")
    print()

    # ---- 5. 空查询 ----
    empty = sample_posterior(X, y, [], hp, (), allow_zero_noise=True)
    print(f"[空查询] status={empty.status}, samples={empty.samples}, rank={empty.rank}")
    print()
    print(f"演示完成, 实际耗时 {time.perf_counter() - t0:.2f} 秒 (上限 90 秒)。")


if __name__ == "__main__":
    main()
