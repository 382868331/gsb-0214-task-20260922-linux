# gprbf — 精确 RBF 高斯过程回归（核参数解析梯度 + 一维一阶导数联合预测）

仅使用 **Python 3.14.7 标准库**（`math`/`dataclasses`/`typing`/`unittest`）实现的可导入单文件库，
零第三方依赖，全程离线。不调用任何机器学习库，也不使用第三方同类实现替代核心算法。

核函数与训练协方差：

```
k(x, x') = alpha^2 * exp(-||x - x'||^2 / (2 * lengthscale^2))
K_y      = K(X, X) + (noise^2 + jitter) * I,   alpha / lengthscale / noise > 0
```

另支持**一维一阶导数观测**的联合 GP：导数观测带独立噪声 `derivative_noise`，
与函数观测拼成联合协方差做联合推断，预测点同时返回 f 与 f′ 的后验。
核的导数协方差（u = x − z，ℓ = lengthscale）：

```
Cov(f(x), f'(z))  = k(x,z) · u / ℓ²          （对第二点 z 求导）
Cov(f'(x), f(z))  = −k(x,z) · u / ℓ²         = −Cov(f(z), f'(x))
Cov(f'(x), f'(z)) = k(x,z) · (1/ℓ² − u²/ℓ⁴)  （对称; u=0 时先验 f′ 方差 = α²/ℓ²）
```

## 运行命令

```bash
python -m unittest discover -s tests -v   # 49 个测试, 实测 < 0.1 秒
python demo.py                             # 固定输入演示, 实测约 0.04 秒（上限 90 秒内）
```

无需安装任何依赖（`requirements.txt` 为空占位）；不需要网络、账号、密钥、数据库或 Docker。

## 文件

| 文件 | 说明 |
|---|---|
| `gprbf.py` | 本题库本体：校验、RBF 核、Cholesky、log 边际似然、解析梯度、后验预测、一维函数/导数核与联合预测 |
| `tests/test_gprbf.py` | 必要测试；**独立参考**（Gauss-Jordan 直接逆、独立核函数、独立 logdet、单点显式逆公式、独立导数核、中心差分）不导入被测核心来产生期望值 |
| `demo.py` | 确定性数据演示：正常结果（含 95% 置信区间）+ 实际触发的 `not_pd` 边界与 `ValueError` 拒绝；导数联合 GP 的正常/边界/非法输入三段 |

## 接口

```python
from gprbf import (
    Hyperparameters, log_marginal, predict, rbf_kernel,
    predict_with_derivatives, rbf_kernel_derivatives_1d,
)

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

# --- 一维一阶导数联合 GP (本次新增; 旧入口签名/行为不变) ---
dp = predict_with_derivatives(
    X, y, Xd, dy, X_test, hp,
    *, derivative_noise=sd, jitter=0.0,
)
# 观测: y_i = f(X_i) + N(0, noise^2+jitter);  dy_j = f'(Xd_j) + N(0, sd^2+jitter)
# 仅一维; nf, nd >= 1, nf+nd <= 80。按 [函数观测 | 导数观测] 拼联合协方差。
# dp.status == "ok":
#   dp.function_mean[t],  dp.function_latent_variance[t],
#   dp.function_observed_variance[t]  (= latent + noise^2 + jitter)
#   dp.derivative_mean[t], dp.derivative_latent_variance[t],
#   dp.derivative_observed_variance[t](= latent + sd^2 + jitter)
#   dp.raw_function_variance / dp.raw_derivative_variance 抹零前原值
#   dp.clamped_function / dp.clamped_derivative 被抹零下标
#   dp.symmetry_residual / dp.solve_residual 联合矩阵诊断量
#   dp.effective_function_noise_variance / effective_derivative_noise_variance
# dp.status == "not_pd" / "negative_variance": 预测字段为 None / 原样返回负值

K, G, H = rbf_kernel_derivatives_1d(X, Z, hp)   # 均要求一维
# K[i,j] = Cov(f(X_i), f(Z_j));  G[i,j] = Cov(f(X_i), f'(Z_j));
# H[i,j] = Cov(f'(X_i), f'(Z_j))
# 恒等式: G(X,Z) = -G(Z,X)^T (反对称, 对角 0); H(X,Z) = H(Z,X)^T
```

结果均为冻结 `dataclass`，字段以 `None`/元组区分成功与失败状态；函数无状态、不修改入参、
重复调用逐位一致。

## 输入上限与错误语义

- 训练样本 `1 ≤ n ≤ 80`；输入维数 `1 ≤ d ≤ 3`；预测点上限 10000（防护值，非题面要求）。
- 导数联合 GP 仅支持 **d = 1**：`X / Xd / X_test` 任一不是一维即 `ValueError`
  （信息含 `1D only` 或 `dimension`）；函数观测与导数观测各至少 1 个，
  **总数 nf + nd ≤ 80**。
- `derivative_noise` 与三种超参数同为有限 `int`/`float` 且 **> 0**；
  `bool` 拒绝，NaN/Infinity 拒绝；导数观测向量名 `dy`、点位名 `Xd`
  （错误信息带 `dy[2]`、`Xd[3][0]` 这类下标）。
- 所有数值输入必须是有限的 `int`/`float`：**NaN、Infinity 一律 `ValueError` 拒绝**，
  错误信息带具体下标（如 `X[3][1]`、`y[2]`）便于定位。
- **`bool` 显式拒绝**（即使 Python 中 `bool` 是 `int` 子类）；普通整数坐标/目标值合法。
- `alpha/lengthscale/noise/derivative_noise ≤ 0`、`jitter < 0`、行列数不一致、
  维度不匹配：`ValueError`；类型不符（如传元组代替 `Hyperparameters`）：`TypeError`。
  不做任何静默纠正。
- 输入先整体校验、后分配计算，批次预测失败不产生部分结果；
  联合函数无状态、不修改入参。

## Cholesky 失败与 jitter 策略

- Cholesky 遇到根号项 `≤ 0`（含 NaN）即判定非正定，返回 `status="not_pd"`，
  **绝不自动添加未记录的 jitter**。
- 重复输入 + 噪声小到 `s²` 在对角加法中被浮点吸收时会真实触发该状态（demo 中演示）；
  导数联合 GP 中重复导数点位 + 极小 `derivative_noise` 同样触发（demo 中演示）。
- 需要 jitter 时只能通过 `jitter=` 关键字**显式给定**（有限、`≥ 0`，默认严格为 0），
  且作为方差项进入**全部公式**：Cholesky、`α=Ky⁻¹y`、边际似然、梯度（jitter 对 log 超参
  是常数，不产生梯度项）、两种预测方差。`(s, jitter=j)` 与
  `(s'=sqrt(s²+j), jitter=0)` 的似然/预测一致、噪声梯度不同（测试覆盖）。
- 联合 GP 中同一个 jitter 同时进入函数块与导数块对角
  （`noise²+j` 与 `derivative_noise²+j`），见结果中的两个
  `effective_*_noise_variance` 字段（测试逐点核对）。

## 负微量方差处理

策略实现为公开纯函数 `classify_raw_variance(raw, alpha2) -> (value, kind)`，
`kind ∈ {"ok", "clamped", "negative"}`，可脱离预测单独审计；`predict` 与
`predict_with_derivatives` 都逐点（f 与 f′ 分量分别）调用它：

`raw = 先验方差 − k_*ᵀKy⁻¹k_*` 在数学上必 ≥ 0，浮点上可能出现极小负值。
纯函数入口先验方差为 `α²`；导数分量的先验方差为 `α²/ℓ²`（u=0 时的
Cov(f′,f′)），容差随该先验缩放：

- `−1e-10·max(1, 先验方差) ≤ raw < 0`：判为舍入负微量，latent 方差抹为 0，
  下标记入 `clamped` / `clamped_function` / `clamped_derivative`，
  `raw_*_variance` 保留原始负值可审计；
- `raw < −1e-10·max(1, 先验方差)`：状态为 `"negative_variance"`，**原样返回负值**，
  不静默纠正，交由调用方定位；
- `observed = latent + 对应噪声方差 + jitter`，因此观测方差恒含完整噪声项；
  函数观测与导数观测使用各自独立的噪声。

## 数值误差口径

测试对误差采用紧容差（n≤12 的双精度参考问题），不以宽容差掩盖边界错误：

| 量 | 测试容差 |
|---|---|
| log 边际似然 vs 直接逆参考 | `max(1e-12, 1e-11·|LML_ref|)` |
| `Ky⁻¹`、`Ky⁻¹y` vs Gauss-Jordan 直接逆 | 绝对误差 ≤ 1e-10 / 1e-11 |
| 后验均值/方差 vs 单点显式逆公式 | 绝对误差 ≤ 1e-10（均值 1e-11） |
| 联合后验 f/f′ 均值/方差 vs 独立 Gauss-Jordan 联合显式逆 | 1e-10（均值 1e-11） |
| 导数核 G/H vs 独立标量参考 / 中心差分 | 1e-14 / 1e-7（H 二阶差分 1e-6） |
| 后验函数均值斜率 d/dx·E[f*] = E[f′*] | 中心差分 ≤ 2e-6 |
| 解析梯度 vs 中心差分（步长 1e-6） | 绝对误差 ≤ 2e-5（中心差分截断+舍入本底） |
| `‖Ky·α−y‖∞`、`‖Ky−LLᵀ‖∞` 残差 | ≤ 1e-13（n=12 实测 ~2e-16） |
| 核矩阵 / 联合协方差对称 | 构造逐位对称，`symmetry_residual == 0.0` |
| G 反对称、H 对称、对角恒等式 G(x,x)=0、H(x,x)=α²/ℓ² | 逐位相等 |

期望值核对方式：独立 Gauss-Jordan 显式逆（联合分块另行手写）、独立标量核参考、
中心差分（一阶/二阶混合差分）、手算常量（x=1,z=0,ℓ=α=1 时 G=e^−1/2、H=0）、
解析恒等式（后验均值的导数 = 导数后验均值）、以及
`derivative_noise→∞` 时联合结果退化回旧纯函数入口 `predict`（交叉入口核对）。

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
- 导数观测支持仅限**一维、一阶**导数：不支持多维方向导数、二阶及以上导数、
  导数观测的在线增删（每次调用为独立满批推断）。
- 联合 GP 不另提供 log 边际似然/超参梯度入口；`log_marginal` 仍只处理纯函数值观测。
- 不做超参数优化；不存在“搜索预算”，因此无预算耗尽/无解之分，非正定情形以
  `"not_pd"` 状态显式表达（这是“矩阵被证明在此浮点表示下无法分解”，区别于输入非法）。
- 纯 Python O(n³)，面向总观测数 ≤ 80；更大规模不适用。
- 预测方差的负微量阈值固定为相对 `max(1, 先验方差)` 的 1e-10。
