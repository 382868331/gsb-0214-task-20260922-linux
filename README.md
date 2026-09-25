# gprbf — 精确 RBF 高斯过程回归（核参数解析梯度）

仅使用 **Python 3.14.7 标准库**（`math`/`dataclasses`/`typing`/`unittest`）实现的可导入单文件库，
零第三方依赖，全程离线。不调用任何机器学习库，也不使用第三方同类实现替代核心算法。

核函数与训练协方差：

```
k(x, x') = alpha^2 * exp(-||x - x'||^2 / (2 * lengthscale^2))
K_y      = K(X, X) + (noise^2 + jitter) * I,   alpha / lengthscale / noise > 0
```

## 运行命令

```bash
python -m unittest discover -s tests -v   # 36 个测试, 实测 < 1 秒
python demo.py                             # 固定输入演示, 实测约 0.04 秒（上限 90 秒内）
```

无需安装任何依赖（`requirements.txt` 为空占位）；不需要网络、账号、密钥、数据库或 Docker。

## 文件

| 文件 | 说明 |
|---|---|
| `gprbf.py` | 本题库本体：校验、RBF 核、Cholesky、log 边际似然、解析梯度、后验预测 |
| `tests/test_gprbf.py` | 必要测试；**独立参考**（Gauss-Jordan 直接逆、独立核函数、独立 logdet、单点显式逆公式、中心差分）不导入被测核心来产生期望值 |
| `tests/test_gpfit_delete.py` | GPFit 删除测试：重复坐标有噪声、删首/中/尾、连续删除、未知 ID、删空后先验；与全量重建对照 |
| `demo.py` | 确定性数据演示：正常结果（含 95% 置信区间）+ 实际触发的 `not_pd` 边界与 `ValueError` 拒绝 |

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

结果均为冻结 `dataclass`，字段以 `None`/元组区分成功与失败状态；函数无状态、不修改入参、
重复调用逐位一致。

### 不可变拟合对象 GPFit（按稳定 ID 删除样本）

```python
from gprbf import fit_gp

fit = fit_gp(X, y, hp, *, jitter=0.0, ids=None)   # ids 缺省为 (0..n-1)
fit.n / fit.ids / fit.X / fit.y / fit.hyperparameters / fit.jitter
m = fit.marginal()        # 与 log_marginal 相同的 MarginalResult
p = fit.predict(X_test)   # 与 predict 相同的 Prediction

fit2 = fit.delete("s4")   # 按稳定样本 ID 删除一条, 返回新对象; fit 本身不变
```

- **不可变**：`fit_gp` 返回的 `GPFit` 不暴露任何修改操作；`delete()` 生成新对象，
  旧对象保持原样、可继续查询（测试覆盖）。
- **稳定 ID**：`ids` 为 `int/float/str`（拒绝 bool、NaN/Inf、重复值），删除后剩余样本
  顺序压缩但 ID 不变、**已删 ID 不复用**（对新对象而言即未知 ID）；未知 ID 抛 `KeyError`。
- **核参数与显式 jitter 不变**：`delete` 不重新估计超参，也不改变 jitter。
- **Cholesky 删除更新**：删除第 k 行/列通过 Givens 旋转对 Cholesky 因子做 O(n²)
  局部更新（`_chol_delete_row`），不做全量重新分解；更新因子用
  `max|K_new − L Lᵀ| ≤ 1e-8·max(1, α²+var)` 验收，**失败时明确回退全量分解**，
  并记录在新对象诊断中：`factor_source ∈ {"full","downdate","refactor"}` 与
  `downdate_residual`（父对象 `not_pd` 无因子可用时也走 `refactor`）。
- **删除后所有量使用新样本数 n**：`marginal()` 的似然/解析梯度与
  `predict()` 的 latent/observed 方差都基于删除后的样本集。
- **删到最后一条**：返回空拟合对象（n=0），`marginal()` 为 `log p(∅)=0`、零梯度，
  `predict()` 退化为先验：均值 0、latent 方差 α²、observed 方差 α²+s²+jitter，
  不产生空矩阵索引错误。

## 输入上限与错误语义

- 训练样本 `1 ≤ n ≤ 80`；输入维数 `1 ≤ d ≤ 3`；预测点上限 10000（防护值，非题面要求）。
- 所有数值输入必须是有限的 `int`/`float`：**NaN、Infinity 一律 `ValueError` 拒绝**，
  错误信息带具体下标（如 `X[3][1]`、`y[2]`）便于定位。
- **`bool` 显式拒绝**（即使 Python 中 `bool` 是 `int` 子类）；普通整数坐标/目标值合法。
- `alpha/lengthscale/noise ≤ 0`、`jitter < 0`、行列数不一致、维度不匹配：`ValueError`；
  类型不符（如传元组代替 `Hyperparameters`）：`TypeError`。不做任何静默纠正。
- 输入先整体校验、后分配计算，批次预测失败不产生部分结果。

## Cholesky 失败与 jitter 策略

- Cholesky 遇到根号项 `≤ 0`（含 NaN）即判定非正定，返回 `status="not_pd"`，
  **绝不自动添加未记录的 jitter**。
- 重复输入 + 噪声小到 `s²` 在对角加法中被浮点吸收时会真实触发该状态（demo 中演示）。
- 需要 jitter 时只能通过 `jitter=` 关键字**显式给定**（有限、`≥ 0`，默认严格为 0），
  且作为方差项进入**全部公式**：Cholesky、`α=Ky⁻¹y`、边际似然、梯度（jitter 对 log 超参
  是常数，不产生梯度项）、两种预测方差。`(s, jitter=j)` 与
  `(s'=sqrt(s²+j), jitter=0)` 的似然/预测一致、噪声梯度不同（测试覆盖）。

## 负微量方差处理

策略实现为公开纯函数 `classify_raw_variance(raw, alpha2) -> (value, kind)`，
`kind ∈ {"ok", "clamped", "negative"}`，可脱离预测单独审计；`predict` 逐点调用它：

`raw = α² − k_*ᵀKy⁻¹k_*` 在数学上必 ≥ 0，浮点上可能出现极小负值：

- `−1e-10·max(1, α²) ≤ raw < 0`：判为舍入负微量，latent 方差抹为 0，下标记入 `clamped`，
  `raw_latent_variance` 保留原始负值可审计；
- `raw < −1e-10·max(1, α²)`：状态为 `"negative_variance"`，**原样返回负值**，
  不静默纠正，交由调用方定位；
- `observed = latent + s² + jitter`，因此观测方差恒含完整噪声项。

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
- 不做超参数优化；不存在“搜索预算”，因此无预算耗尽/无解之分，非正定情形以
  `"not_pd"` 状态显式表达（这是“矩阵被证明在此浮点表示下无法分解”，区别于输入非法）。
- 纯 Python O(n³)，面向 n≤80；更大规模不适用。
- 预测方差的负微量阈值固定为相对 `max(1, α²)` 的 1e-10。
- `GPFit` 仅支持删除，不支持插入/更新样本；删除更新验收容差固定为
  `1e-8·max(1, α²+var)`，超过即回退全量分解并记录。
