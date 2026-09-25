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
python -m unittest discover -s tests -v   # 33 个测试, 实测 < 1 秒
python demo.py                             # 固定输入演示, 实测 < 1 秒（90 秒预算内）
```

无需安装任何依赖（`requirements.txt` 为空占位）；不需要网络、账号、密钥、数据库或 Docker。

## 文件

| 文件 | 说明 |
|---|---|
| `gprbf.py` | 本题库本体：校验、RBF 核、Cholesky、log 边际似然、解析梯度、后验预测、`GPFit` 不可变删除 |
| `tests/test_gprbf.py` | 回归测试；**独立参考**（Gauss-Jordan 直接逆、独立核函数、独立 logdet、单点显式逆公式、中心差分）不导入被测核心来产生期望值 |
| `tests/test_gpfit_delete.py` | `GPFit` 删除测试：重复坐标有噪声、删首/中/尾、连续删除至空、未知 ID、回退诊断；对照独立直接逆参考与模块级全量重建 |
| `demo.py` | 确定性数据演示：正常结果（含 95% 置信区间）+ 实际触发的 `not_pd` 边界与 `ValueError` 拒绝 + `GPFit` 连续删除与先验预测 |

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

## GPFit：不可变拟合对象与样本删除

```python
from gprbf import GPFit, Hyperparameters

fit = GPFit.fit(X, y, hp, *, jitter=0.0, sample_ids=None)  # 唯一全量 Cholesky 入口
# sample_ids 缺省为 (0..n-1); 给定时必须可哈希且互不重复, 与样本终身绑定

fit2 = fit.delete(sample_id)   # 返回新对象; fit 完全不变、可继续查询
fit2.log_marginal()            # 契约同模块级函数, 使用剩余样本数 n-1
fit2.predict(X_test)           # latent / observed 两种方差同上
fit2.sample_ids                # 剩余稳定 ID (顺序压缩, ID 不复用)
fit2.diagnostics               # 每次删除的 DeletionRecord 元组
```

- **删除更新**：`delete` 用 Cholesky 行删除（删行后 Givens 旋转恢复下三角，O(n^2)）得到新因子，**不把每次删除当全量 fit**；随后用 `max|Ky' - L'L'^T|` 做残差验收（阈值 `DOWNDATE_RESIDUAL_RTOL = 1e-8`，相对 `max(1, max diag Ky)`）。验收失败或分解本身失败时**明确回退全量 Cholesky**，`DeletionRecord.method` 记录 `"cholesky_downdate"` / `"full_refit"` / `"trivial_empty"` 及两侧残差。
- **核参数与显式 jitter 不变**：删除只改训练集，`alpha/lengthscale/noise/jitter` 原样保留；更新后的边际似然、解析梯度、latent/observed 预测全部按新样本数计算。
- **不可变**：对象无任何修改方法；`delete` 失败（未知 ID 抛 `KeyError`）不留部分变更。
- **删到最后一点**（n=0）：`log_marginal()` 返回空数据似然 `0.0` 与零梯度；`predict()` 返回**先验预测**（均值 0，隐方差 `alpha^2`，观测方差 `alpha^2 + noise^2 + jitter`），不产生空矩阵索引错误。
- 初始拟合即 `not_pd` 时状态保留在对象上，后续查询返回 `status="not_pd"`。

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
- `GPFit` 只支持删除样本，不支持新增/更新样本或超参（那会需要不同的更新结构）；
  不支持导数观测、参数优化与隐式 jitter。删除更新为 O(n²)，但查询接口
  （`log_marginal`/`predict`）仍按各自 O(n³)/O(n²m) 公式现场计算诊断量。
