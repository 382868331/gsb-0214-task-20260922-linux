#!/usr/bin/env python3
"""
demo.py - gprbf 固定输入演示 (确定性数据, 真实计算, 无网络/无第三方库)。

运行: python demo.py
内容:
  1. 正常结果: 12 个确定性 1D 样本上的精确 GP 回归, 打印 log 边际似然、
     对 log(a,l,s) 的解析梯度, 以及 5 个测试点的后验均值与 95% 置信区间;
  2. 边界触发 (拒绝): 重复输入 + 极小噪声且不给 jitter => Cholesky 失败,
     返回 status="not_pd" (绝不自动加未记录 jitter);
     随后给显式 jitter 恢复;
  3. 异常触发: NaN 输入被 ValueError 拒绝。
"""

import math
import time

from gprbf import Hyperparameters, fit_gp, log_marginal, predict

Z95 = 1.959963984540054  # 标准正态 0.975 分位, 95% 区间


def main():
    t0 = time.perf_counter()

    # --- 确定性数据: y = sin(x) + 0.15 x, 手写网格, 无随机数 ---
    X = [(-2.0 + 0.4 * i,) for i in range(12)]
    y = [math.sin(x[0]) + 0.15 * x[0] for x in X]
    hp = Hyperparameters(alpha=1.3, lengthscale=0.9, noise=0.2)

    print("=" * 68)
    print("gprbf demo: 精确 RBF 高斯过程回归 (Python 标准库)")
    print("=" * 68)
    print(f"训练样本 n={len(X)}, 维数 d=1")
    print(f"超参数 alpha={hp.alpha}, lengthscale={hp.lengthscale}, noise={hp.noise}")
    print()

    # --- 1. 正常结果: log 边际似然 + 解析梯度 ---
    res = log_marginal(X, y, hp)
    assert res.status == "ok", res.status
    ga, gl, gs = res.gradient
    print("[正常结果] log 边际似然与核参数解析梯度")
    print(f"  log p(y|X)      = {res.log_marginal:.10f}")
    print(f"  dL/d log alpha  = {ga: .10f}")
    print(f"  dL/d log length = {gl: .10f}")
    print(f"  dL/d log noise  = {gs: .10f}")
    print(f"  数值诊断: 对称性残差={res.symmetry_residual:.1e}, "
          f"Cholesky 残差={res.chol_residual:.1e}, 求解残差={res.solve_residual:.1e}")
    print()

    # --- 2. 后验预测与 95% 置信区间 (观测方差) ---
    X_test = [(-1.8,), (-0.9,), (0.0,), (0.9,), (1.8,)]
    pred = predict(X, y, X_test, hp)
    assert pred.status == "ok", pred.status
    print("[正常结果] 后验预测与 95% 置信区间 (mean +/- 1.96*sqrt(var_observed))")
    print(f"  {'x':>7} | {'mean':>10} | {'latent sd':>10} | {'obs sd':>9} | "
          f"{'95% CI lower':>12} ~ {'upper':>12}")
    for t, xt in enumerate(X_test):
        m = pred.mean[t]
        sd_l = math.sqrt(pred.latent_variance[t])
        sd_o = math.sqrt(pred.observed_variance[t])
        lo = m - Z95 * sd_o
        hi = m + Z95 * sd_o
        print(f"  {xt[0]:7.2f} | {m:10.6f} | {sd_l:10.6f} | {sd_o:9.6f} | "
              f"{lo:12.6f} ~ {hi:12.6f}")
    print(f"  预测求解残差 = {pred.solve_residual:.1e}")
    print()

    # --- 3. 边界: 重复输入 + 极小噪声, 无 jitter => not_pd 拒绝 ---
    print("[边界触发] 重复输入 + 极小噪声 (noise=1e-10), 未提供 jitter")
    Xd = [(0.0,), (0.0,), (1.0,), (1.0,)]
    yd = [0.5, 0.5, -0.5, -0.5]
    hp_d = Hyperparameters(alpha=1.0, lengthscale=1.0, noise=1e-10)
    bad = log_marginal(Xd, yd, hp_d)
    bad_p = predict(Xd, yd, [(0.5,)], hp_d)
    print(f"  log_marginal status = {bad.status!r} "
          f"(log_marginal={bad.log_marginal}, gradient={bad.gradient})")
    print(f"  predict      status = {bad_p.status!r} (mean={bad_p.mean})")
    print("  -> 协方差矩阵非正定, 模块拒绝给出结果, 未自动添加任何 jitter。")

    fixed = log_marginal(Xd, yd, hp_d, jitter=1e-6)
    print(f"  显式 jitter=1e-6 后 status = {fixed.status!r}, "
          f"log p(y|X) = {fixed.log_marginal:.8f} (jitter 已计入全部公式)")
    print()

    # --- 4. 异常边界: NaN 输入必须被 ValueError 拒绝 ---
    print("[异常触发] 训练数据含 NaN")
    try:
        log_marginal([(float("nan"),), (1.0,)], [0.0, 1.0], hp)
    except ValueError as exc:
        print(f"  ValueError 已捕获: {exc}")
    else:
        raise AssertionError("NaN 输入应当被拒绝")
    print()

    # --- 5. 不可变 fit 对象: 按稳定 ID 删除样本 (Cholesky 删除更新) ---
    print("[删除演示] fit_gp 构造不可变对象, delete(id) 生成新对象")
    ids = tuple(f"s{i}" for i in range(len(X)))
    fit = fit_gp(X, y, hp, jitter=1e-6, ids=ids)
    print(f"  原始对象: n={fit.n}, ids={fit.ids[:4]}..., factor_source={fit.factor_source}")
    fit_d = fit.delete("s4")  # 删除中部一条
    md = fit_d.marginal()
    # 独立全量重建对照
    keep = [i for i in range(len(X)) if i != 4]
    ref = log_marginal([X[i] for i in keep], [y[i] for i in keep], hp, jitter=1e-6)
    print(f"  删除 's4' 后: n={fit_d.n}, factor_source={fit_d.factor_source}, "
          f"downdate_residual={fit_d.downdate_residual:.1e}")
    print(f"  新 log p(y|X) = {md.log_marginal:.10f}  "
          f"(全量重建对照 {ref.log_marginal:.10f}, 差 {abs(md.log_marginal - ref.log_marginal):.1e})")
    pd_ = fit_d.predict(X_test)
    print(f"  删除后预测 x=0.0: mean={pd_.mean[2]:.6f}, "
          f"latent var={pd_.latent_variance[2]:.6f}, observed var={pd_.observed_variance[2]:.6f}")
    # 旧对象不受影响, 仍可查询
    print(f"  旧对象仍可查询: n={fit.n}, log p(y|X) = {fit.marginal().log_marginal:.10f}")
    # 连续删除直至只剩一条, 再删即先验
    fit_c = fit
    for sid in ids[1:]:
        fit_c = fit_c.delete(sid)
    print(f"  连续删除至 n={fit_c.n} (ids={fit_c.ids})")
    fit_empty = fit_c.delete(ids[0])
    pe = fit_empty.predict([(0.0,)])
    print(f"  再删最后一条: n={fit_empty.n}, 先验预测 mean={pe.mean[0]}, "
          f"latent var={pe.latent_variance[0]:.4f} (= alpha^2), "
          f"observed var={pe.observed_variance[0]:.6f}")
    try:
        fit_empty.delete("s0")
    except KeyError as exc:
        print(f"  未知 ID 拒绝: KeyError {exc}")
    print()

    print(f"演示完成, 实际耗时 {time.perf_counter() - t0:.2f} 秒。")


if __name__ == "__main__":
    main()
