# gprbf — 精确 RBF 高斯过程回归（核参数解析梯度）

仅使用 **Python 3.14.7 标准库**（`math`/`dataclasses`/`typing`/`unittest`）实现的可导入单文件库，
零第三方依赖，全程离线。不调用任何机器学习库，也不使用第三方同类实现替代核心算法。

**增量能力（本次）**：多个查询点的**联合后验满协方差**与调用方给定标准正态数的
**相关函数样本**（`joint_posterior` / `sample_posterior`），支持重复查询点（秩亏损但保留完全
相关性）、训练点零噪声精确插值、空查询，以及固定观测值的路径条件化。逐点 `predict` 均值接口
与旧行为完全不变。

核函数与训练协方差：

```
k(x, x') = alpha^2 * exp(-||x - x'||^2 / (2 * lengthscale^2))
K_y      = K(X, X) + (noise^2 + jitter) * I,   alpha / lengthscale > 0, noise >= 0
# 常规 log_marginal/predict 仍要求 noise > 0;
# 零噪声 (训练点精确插值) 仅由 joint_posterior/sample_posterior 在
# allow_zero_noise=True 时显式启用。
```

## 运行命令

```bash
python -m unittest discover -s tests -v   # 基线回归 + 联合抽样测试, 实测约 3 秒
python demo.py                             # 固定输入演示, 实测约 0.04 秒（上限 8 秒内）
python demo_iteration.py                   # 联合相关抽样真实离线演示, 实测 < 1 秒（上限 90 秒）
```

无需安装任何依赖（`requirements.txt` 为空占位）；不需要网络、账号、密钥、数据库或 Docker。

### 隔离 venv 准备（可选，仅用标准库）

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip   # 不需要安装任何第三方包; requirements.txt 为空
python -m unittest discover -s tests -v
python demo_iteration.py
deactivate
```

不使用 venv 也可直接运行；全程不读取任何随机种子配置或环境变量，标准正态数由调用方显式传入。

## 文件

| 文件 | 说明 |
|---|---|
| `gprbf.py` | 本题库本体：校验、RBF 核、Cholesky、log 边际似然、解析梯度、后验预测 |
| `tests/test_gprbf.py` | 基线必要测试；**独立参考**（Gauss-Jordan 直接逆、独立核函数、独立 logdet、单点显式逆公式、中心差分）不导入被测核心来产生期望值 |
| `tests/test_joint_samples.py` | 增量测试：满协方差独立核对、重复点完全相关、零噪声训练点、空查询、固定值条件化、经验协方差核对 |
| `demo.py` | 确定性数据演示：正常结果（含 95% 置信区间）+ 实际触发的 `not_pd` 边界与 `ValueError` 拒绝 |
| `demo_iteration.py` | 联合抽样演示：相关函数样本、重复点/零噪声/空查询、固定值条件化、不相容拒绝、独立 Gauss-Jordan 核对 |

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

### 联合后验与相关函数样本（增量）

```python
from gprbf import joint_posterior, sample_posterior, Hyperparameters

hp0 = Hyperparameters(alpha=1.2, lengthscale=1.1, noise=0.0)
Q = [(0.5,), (1.7,), (1.7,), (3.0,)]   # 含两个重复查询点, 最后一个是训练点

post = joint_posterior(X, y, Q, hp0, allow_zero_noise=True)
# status=ok 时: mean=联合后验均值; covariance=完整后验协方差 C;
# factor=相关结构因子 F 满足 F F^T = C (重复点因子行逐位相同);
# rank=数值秩 (重复查询点使 rank < m, 零特征方向保留而非解除);
# eigenvalues/corrected_eigenvalues 为特征分解审计信息。
# status=not_pd: 训练协方差无法 Cholesky (不自动加 jitter/噪声);
# status=not_psd: 后验有超过容差的负特征值, 明确拒绝。

# Z 由调用方提供, 形状 (n_draws, post.rank) 的标准正态数; 本库不读随机种子。
s = sample_posterior(X, y, Q, hp0, Z, allow_zero_noise=True)
# status=ok 时 samples=(n_draws, m), 每条样本 = mean + Z F^T;
# 重复查询坐标逐位相等 (完全相关, 不是逐点独立抽样);
# 零方差坐标 (零噪声训练点) 恒为后验均值。
# Z 行列数不符 -> status=bad_shape; Z 含 NaN/Inf -> ValueError。

# 固定观测值的路径条件化 (满高斯条件公式):
s2 = sample_posterior(X, y, Q, hp0, Z2, fixed={0: 0.88},
                      allow_zero_noise=True)
# 条件系统只在非零协方差子空间用截断谱伪逆求解:
#   mean_cond = mean + C_{,I} C_I^+ (v - mean_I)
#   C_cond    = C - C_{,I} C_I^+ C_{I,}
# 不相容的零方差观测 (给重复点两个不同值, 或给零方差训练点矛盾值)
# -> status=incompatible_fixed 明确拒绝, 不返回样本;
# 相容固定坐标逐位精确保持, 其重复副本同步;
# Z 列数须等于条件后秩 s2.rank。

# 空查询: m=0, rank=0, Z 为 (n_draws, 0), samples 为长度 n_draws 的空元组。
empty = sample_posterior(X, y, [], hp0, (), allow_zero_noise=True)
```

半正定分解口径（`psd_factor` / `jacobi_eigen` 公开可审计）：先对称化，再用循环
Jacobi 特征分解；特征值小于 `-1e-10*max(1,λmax)` 判为明显非半正定并拒绝；容差内负微量抹零
并在 `corrected_eigenvalues` 记录（下标, 原始值）；不超过 `1e-12*max(1,λmax)` 的为零特征方向，
保留秩亏损（重复点完全相关性由此体现），**绝不通过加观测噪声或 jitter 掩盖奇异性**。

结果均为冻结 `dataclass`，字段以 `None`/元组区分成功与失败状态；函数无状态、不修改入参、
重复调用逐位一致。

## 输入上限与错误语义

- 训练样本 `1 ≤ n ≤ 80`；输入维数 `1 ≤ d ≤ 3`；逐点 `predict` 预测点上限 10000，
  联合后验 `joint_posterior`/`sample_posterior` 查询点上限 80（满协方差 m×m、O(m³) 分解）。
- 联合接口的 `X_test` 允许为空（m=0）；逐点 `predict` 仍要求至少一个查询点。
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
- 联合抽样只覆盖隐函数 f* 的高斯后验（可在调用方自行加观测噪声）；样本所需标准正态数
  完全由调用方按 `rank` 提供，本库不内置 RNG、不读种子配置。条件化仅支持“固定坐标=给定值”
  的路径（线性高斯条件），不实现一般线性/不等式观测、多保真或在线更新。
- 秩判定阈值固定为 `1e-12*max(1,λmax)`，负特征值拒绝阈值固定为
  `1e-10*max(1,λmax)`；零噪声相容观测的零空间容差固定为相对 1e-9。
