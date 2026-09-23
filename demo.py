"""固定输入演示：RBF 高斯过程回归（离线、确定性、纯标准库）。

运行：``python demo.py``（约 8 秒内）。

内容：
1. 正常结果：在确定性生成的 1D 数据上拟合，打印 log 边际似然、对
   log(a,l,s) 的解析梯度（并用中心差分现场复核），再对一组查询点
   给出后验均值与 95% 观测置信区间；
2. 边界一：重复输入 + 噪声小到被浮点吞没 -> Cholesky 失败，状态为
   ``not_pd``（不抛异常、不偷偷加 jitter）；
3. 边界二：含 NaN 的输入 -> 抛出可定位的 GPInputError。
"""

from __future__ import annotations

import math
import random
import time

from gp_rbf import GPInputError, fit, predict


def main() -> None:
    t0 = time.perf_counter()

    # ---------- 1. 正常结果 ----------
    rng = random.Random(20260923)
    n = 24
    X = [[-4.0 + 8.0 * i / (n - 1)] for i in range(n)]
    y = [
        math.sin(1.2 * x[0]) + 0.15 * math.cos(0.7 * x[0])
        + rng.uniform(-0.03, 0.03)
        for x in X
    ]
    a, l, s = 1.0, 0.9, 0.2

    r = fit(X, y, a, l, s)
    assert r.status == "ok"

    print("=" * 68)
    print("正常结果：RBF-GP 回归（零均值先验）")
    print(f"训练样本 n={r.n}, 维度 d={r.d}, 参数 a={a} l={l} s={s}")
    print(f"log 边际似然 log p(y|X,a,l,s) = {r.log_marginal:.10f}")
    ga, gl, gs = r.grad_log
    print("解析梯度  (d/d log a, d/d log l, d/d log s) = "
          f"({ga:+.6f}, {gl:+.6f}, {gs:+.6f})")

    # 现场中心差分复核（步长 H=1e-3），不依赖测试目录
    H = 1e-3
    def lml_at(aa: float, ll: float, ss: float) -> float:
        rr = fit(X, y, aa, ll, ss)
        assert rr.status == "ok"
        return rr.log_marginal

    fd = (
        (lml_at(a * math.exp(H), l, s) - lml_at(a / math.exp(H), l, s)) / (2 * H),
        (lml_at(a, l * math.exp(H), s) - lml_at(a, l / math.exp(H), s)) / (2 * H),
        (lml_at(a, l, s * math.exp(H)) - lml_at(a, l, s / math.exp(H))) / (2 * H),
    )
    print("中心差分  (d/d log a, d/d log l, d/d log s) = "
          f"({fd[0]:+.6f}, {fd[1]:+.6f}, {fd[2]:+.6f})")
    print(f"两者最大偏差 = {max(abs(x - z) for x, z in zip(r.grad_log, fd)):.3e}")
    print(f"自检残差：对称性 {r.residuals.symmetry:.1e}, "
          f"Cholesky 重构 {r.residuals.chol_reconstruct:.1e}, "
          f"||C·alpha-y||∞ {r.residuals.chol_solve:.1e}")

    # 查询点：数据内部、边缘、远离数据
    Xs = [[0.0], [2.0], [4.0], [7.0]]
    p = predict(r, Xs)
    z = 1.96  # 95% 正态临界值
    print("-" * 68)
    print("后验预测（observed 方差含 s²，区间为 95% 置信区间）")
    print(f"{'x*':>6} | {'mean':>9} | {'latent sd':>10} | "
          f"{'95% CI (observed)':>28}")
    for q, xs in enumerate(Xs):
        sd_lat = math.sqrt(p.latent_var[q])
        half = z * math.sqrt(p.observed_var[q])
        print(f"{xs[0]:6.2f} | {p.mean[q]:9.5f} | {sd_lat:10.5f} | "
              f"[{p.mean[q]-half:9.5f}, {p.mean[q]+half:9.5f}]")
    print("说明：x*=7 远离训练数据，latent 标准差趋近先验 a=1.0，区间最宽。")

    # ---------- 2. 边界：Cholesky 失败 -> not_pd ----------
    print("=" * 68)
    print("边界一：完全重复输入且 s 小到 s² 被浮点吞没（s=1e-8）")
    r_bad = fit([[0.0], [0.0]], [0.5, 0.52], 1.0, 1.0, 1e-8)
    print(f"fit 状态 = {r_bad.status!r}（log_marginal={r_bad.log_marginal}, "
          f"grad={r_bad.grad_log}）")
    print("未抛异常、未自动加 jitter；显式给定 jitter=1e-12 后：")
    r_fix = fit([[0.0], [0.0]], [0.5, 0.52], 1.0, 1.0, 1e-8, jitter=1e-12)
    print(f"fit 状态 = {r_fix.status!r}，log 边际似然 = {r_fix.log_marginal:.6f}")

    # ---------- 3. 边界：非法输入被拒绝 ----------
    print("=" * 68)
    print("边界二：目标向量含 NaN —— 必须被拒绝并指出位置")
    try:
        fit([[0.0], [1.0]], [1.0, float("nan")], 1.0, 1.0, 0.2)
    except GPInputError as exc:
        print(f"捕获 GPInputError: {exc}")
    else:  # pragma: no cover
        raise AssertionError("NaN 输入未被拒绝")

    print("=" * 68)
    print(f"演示完成，用时 {time.perf_counter() - t0:.2f} 秒。")


if __name__ == "__main__":
    main()
