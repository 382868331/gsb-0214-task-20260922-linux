# gprbf — 精确 RBF 高斯过程回归（核参数解析梯度 + 一维导数观测联合预测）

仅使用 **Python 3.14.7 标准库**（`math`/`dataclasses`/`typing`/`unittest`）实现的可导入单文件库，
零第三方依赖，全程离线。不调用任何机器学习库，也不使用第三方同类实现替代核心算法。

核函数与训练协方差：

```
k(x, x') = alpha^2 * exp(-||x - x'||^2 / (2 * lengthscale^2))
K_y      = K(X, X) + (noise^2 + jitter) * I,   alpha / lengthscale / noise > 0
```

一维一阶导数观测（本次迭代新增，仅 d=1，记 r = x − x′，ℓ = lengthscale）：

```
Cov(f(x),  f'(x')) = ∂k/∂x'        = k · r / ℓ²
Cov(f'(x), f'(x')) = ∂²k/∂x∂x'     = k · (1/ℓ² − r²/ℓ⁴)
Var(f'(x))         = alpha² / ℓ²
联合协方差 C = [[K_ff + (s²+j)·I_n,   K_fd               ],
                [K_fdᵀ,                K_dd + (sd²+j)·I_m ]]
             (s = noise, sd = noise_deriv, j = jitter)
```

## 运行命令

```bash
python -m unittest discover -s tests -v   # 46 个测试, 实测 < 0.1 秒
python demo.py                             # 固定输入演示, 实测约 0.04 秒（上限 90 秒内）
```

无需安装任何依赖（`requirements.txt` 为空占位）；不需要网络、账号、密钥、数据库或 Docker。

## 文件

| 文件 | 说明 |
|---|---|
| `gprbf.py` | 本题库本体：校验、RBF 核、Cholesky、log 边际似然、解析梯度、后验预测、导数观测联合预测 |
| `tests/test_gprbf.py` | 纯函数值入口测试；**独立参考**（Gauss-Jordan 直接逆、独立核函数、独立 logdet、单点显式逆公式、中心差分）不导入被测核心来产生期望值 |
| `tests/test_gprbf_joint.py` | 导数观测联合预测测试；独立参考（独立核导数公式、参考核中心差分、稠密联合协方差 + Gauss-Jordan、手算 2×2 对角例） |
| `demo.py` | 确定性数据演示：正常结果（含 95% 置信区间）+ 联合预测对比 + 实际触发的 `not_pd` 边界与 `ValueError` 拒绝 |

## 接口

```python
from gprbf import Hyperparameters, log_marginal, predict, rbf_kernel

hp = Hyperparameters(alpha=1.3, lengthscale=0.9, noise=0.2)

r = log_marginal(X, y, hp, *, jitter=0.0)
# r.status == "ok":
#   r.log_marginal  = -1/2 y^T Ky^-1 y - 1/2 log|Ky| - n/2 log(2π)
#   r.gradient      = (dL/dlog alpha, dL/dlog lengthscale, dL/dlog noise)
#   r.alpha_vec     = Ky^-1 y;  r.K_inv = Ky 的显式逆（诊断/复用）
#   r.symmetry_residual / chol_residual / solve_residual 数值残差
#   r.effective_noise_variance = noise^2 + jitter
# r.status == "not_pd": Cholesky 失败, 似然/梯度/逆为 None
#
# 解析梯度: dL/dθ = 1/2 α^T (∂Ky/∂θ) α - 1/2 tr(Ky^-1 ∂Ky/∂θ)
#   ∂Ky/∂log α   = 2K
#   ∂Ky/∂log ℓ   = K ⊙ (D²/ℓ²)        （对角为 0）
#   ∂Ky/∂log s   = 2 s² I              （显式 jitter 为常数, 不产生梯度）

p = predict(X, y, X_test, hp, *, jitter=0.0)
# p.status == "ok":
#   p.mean[t]              = k_*^T Ky^-1 y
#   p.latent_variance[t]   = α² - k_*^T Ky^-1 k_*          （隐函数 f*）
#   p.observed_variance[t] = latent_variance + s² + jitter（新观测 y*）
#   p.raw_latent_variance  抹零前的原始值; p.clamped 记录被抹零下标
# p.status == "not_pd" / "negative_variance" 见下
```

### 一维导数观测联合预测（新增）

```python
from gprbf import predict_joint, rbf_kernel_df, rbf_kernel_dd

pj = predict_joint(X, y, Xd, yd, X_test, hp, *, noise_deriv, jitter=0.0)
# X, y    : n 个函数值观测 (噪声 hp.noise), 必须为一维, n >= 1
# Xd, yd  : m 个一阶导数观测 (噪声 noise_deriv), 一维; 可同时为空列表表示无导数观测
# noise_deriv: 导数观测噪声标准差, 必填关键字参数 (>0, 即使 m=0 也校验, 无默认值)
# 约束: n + m <= 80; X / Xd / X_test 均须一维 (多维 ValueError)
#
# pj.status == "ok":
#   pj.mean / latent_variance / observed_variance
#       测试点处 f(x*) 的后验均值 / 隐方差 / 观测方差 (+noise²+jitter)
#   pj.deriv_mean / deriv_latent_variance / deriv_observed_variance
#       测试点处 f'(x*) 的后验均值 / 隐方差 / 观测方差 (+noise_deriv²+jitter)
#   pj.raw_latent_variance / raw_deriv_latent_variance  抹零前原始值
#   pj.clamped / deriv_clamped  被抹零下标
#   pj.effective_noise_variance = noise²+jitter
#   pj.effective_deriv_noise_variance = noise_deriv²+jitter
# pj.status == "not_pd" / "negative_variance": 语义与 predict 相同
#
# 后验公式 (z = [y; yd], C 为联合协方差):
#   mean_f = k*ᵀC⁻¹z,  latent_f = α² − k*ᵀC⁻¹k*,   k*  = Cov(z, f(x*))
#   mean_d = kd*ᵀC⁻¹z, latent_d = α²/ℓ² − kd*ᵀC⁻¹kd*, kd* = Cov(z, f'(x*))

rbf_kernel_df(X_f, X_d, hp)  # [i][j] = Cov(f(X_f[i]), f'(X_d[j])), 公开可审计
rbf_kernel_dd(X_d1, X_d2, hp)  # [i][j] = Cov(f'(X_d1[i]), f'(X_d2[j])), 公开可审计
```

`Xd`/`yd` 同时为空时退化为纯函数值观测，函数后验与旧入口 `predict` 逐位一致（测试覆盖）。
旧入口 `log_marginal` / `predict` / `rbf_kernel` 签名与语义不变。

结果均为冻结 `dataclass`，字段以 `None`/元组区分成功与失败状态；函数无状态、不修改入参、
重复调用逐位一致。

## 输入上限与错误语义

- 训练样本 `1 ≤ n ≤ 80`；输入维数 `1 ≤ d ≤ 3`；预测点上限 10000（防护值，非题面要求）。
- 联合预测：导数观测 `0 ≤ m`，联合观测总数 `n + m ≤ 80`；导数观测仅支持一维
  （`X`/`Xd`/`X_test` 维数不为 1 时 `ValueError`，多维方向导数不在本迭代范围）。
- 所有数值输入必须是有限的 `int`/`float`：**NaN、Infinity 一律 `ValueError` 拒绝**，
  错误信息带具体下标（如 `X[3][1]`、`y[2]`、`Xd[1][0]`、`yd[0]`）便于定位。
- **`bool` 显式拒绝**（即使 Python 中 `bool` 是 `int` 子类）；普通整数坐标/目标值合法。
- `alpha/lengthscale/noise ≤ 0`、`noise_deriv ≤ 0`、`jitter < 0`、行列数不一致、
  维度不匹配：`ValueError`；类型不符（如传元组代替 `Hyperparameters`）：`TypeError`。
  不做任何静默纠正。
- 输入先整体校验、后分配计算，批次预测失败不产生部分结果。

## Cholesky 失败与 jitter 策略

- Cholesky 遇到根号项 `≤ 0`（含 NaN）即判定非正定，返回 `status="not_pd"`，
  **绝不自动添加未记录的 jitter**。
- 重复输入 + 噪声小到 `s²` 在对角加法中被浮点吸收时会真实触发该状态（demo 中演示）。
- 需要 jitter 时只能通过 `jitter=` 关键字**显式给定**（有限、`≥ 0`，默认严格为 0），
  且作为方差项进入**全部公式**：Cholesky、`α=Ky⁻¹y`、边际似然、梯度（jitter 对 log 超参
  是常数，不产生梯度项）、两种预测方差。`(s, jitter=j)` 与
  `(s'=sqrt(s²+j), jitter=0)` 的似然/预测一致、噪声梯度不同（测试覆盖）。
- 联合预测中 jitter 同样只作为方差项，同时加在函数观测块与导数观测块的对角上
  （`effective_noise_variance = s²+j`，`effective_deriv_noise_variance = sd²+j`）。

## 负微量方差处理

策略实现为公开纯函数 `classify_raw_variance(raw, alpha2) -> (value, kind)`，
`kind ∈ {"ok", "clamped", "negative"}`，可脱离预测单独审计；`predict` 逐点调用它：

`raw = α² − k_*ᵀKy⁻¹k_*` 在数学上必 ≥ 0，浮点上可能出现极小负值：

- `−1e-10·max(1, α²) ≤ raw < 0`：判为舍入负微量，latent 方差抹为 0，下标记入 `clamped`，
  `raw_latent_variance` 保留原始负值可审计；
- `raw < −1e-10·max(1, α²)`：状态为 `"negative_variance"`，**原样返回负值**，
  不静默纠正，交由调用方定位；
- `observed = latent + s² + jitter`，因此观测方差恒含完整噪声项。
- 联合预测中同一策略分别作用于函数方差（容差按 `max(1, α²)` 缩放）与导数方差
  （容差按先验导数方差 `max(1, α²/ℓ²)` 缩放），抹零下标分别记入
  `clamped` / `deriv_clamped`。

## 数值误差口径

测试对误差采用紧容差（n≤12 的双精度参考问题），不以宽容差掩盖边界错误：

| 量 | 测试容差 |
|---|---|
| log 边际似然 vs 直接逆参考 | `max(1e-12, 1e-11·|LML_ref|)` |
| `Ky⁻¹`、`Ky⁻¹y` vs Gauss-Jordan 直接逆 | 绝对误差 ≤ 1e-10 / 1e-11 |
| 后验均值/方差 vs 单点显式逆公式 | 绝对误差 ≤ 1e-10（均值 1e-11） |
| 解析梯度 vs 中心差分（步长 1e-6） | 绝对误差 ≤ 2e-5（中心差分截断+舍入本底） |
| `‖Ky·α−y‖∞`、`‖Ky−LLᵀ‖∞` 残差 | ≤ 1e-13（n=12 实测 ~2e-16） |
| 核矩阵对称 | 构造逐位对称，`symmetry_residual == 0.0` |
| 联合预测均值/方差 vs 独立稠密参考（Gauss-Jordan） | 均值 ≤ 1e-10，方差 ≤ 1e-9 |
| `rbf_kernel_df` vs 参考核中心差分（h=1e-6） | 绝对误差 ≤ 1e-8 |
| `rbf_kernel_dd` vs 参考核混合中心差分（h=1e-4） | 绝对误差 ≤ 1e-5（差分除以 4h² 放大舍入） |
| 导数后验均值 vs 参考均值函数中心差分 | 绝对误差 ≤ 1e-6（解析关系 d/dx* E[f*]=E[f'*]） |

误差来源：纯 Python 双精度浮点、`math.exp`/`math.sqrt`；Cholesky 为 O(n³) 精确分解路径，
无迭代逼近。

## 设计取舍

- 全部线性代数手写（Cholesky、前代/回代、逐列求解显式逆），保证零依赖与可审计性；
  牺牲大规模性能（80×80 仍为毫秒级）。
- 显式返回 `Ky⁻¹` 与多项残差，便于独立核对，而非只返回标量结果。
- 梯度经显式逆按公式直接计算（`tr(Ky⁻¹∂K)`），与 Cholesky 求解路径独立，
  中心差分是主要外部核对手段。
- 置信区间由调用方按 `mean ± z·sqrt(observed_variance)` 构造（demo 用 z=1.95996…，95%）。

## 已知限制

- 仅实现本题契约的数学子集：RBF 核、各向同性单长度尺度、零均值高斯似然、满批精确推断；
  不实现稀疏/ inducing point、ARD、非高斯似然、超参优化器，不宣称行业标准 GP 框架兼容。
- 导数观测仅限一维一阶：不实现多维方向导数、高阶导数；联合模型暂不提供
  log 边际似然与超参梯度（`log_marginal` 仍只接受纯函数值观测）。
- 不做增量分解或在线删样本：每次 `predict_joint` 调用对 (n+m) 阶联合协方差重新分解。
- 不做超参数优化；不存在“搜索预算”，因此无预算耗尽/无解之分，非正定情形以
  `"not_pd"` 状态显式表达（这是“矩阵被证明在此浮点表示下无法分解”，区别于输入非法）。
- 纯 Python O(n³)，面向 n≤80（联合预测 n+m≤80）；更大规模不适用。
- 预测方差的负微量阈值固定为相对 `max(1, 先验方差)` 的 1e-10
  （函数 `α²`，导数 `α²/ℓ²`）。
