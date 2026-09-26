# gprbf — 精确 RBF 高斯过程回归（核参数解析梯度）

仅使用 **Python 3.14.7 标准库**（`math`/`dataclasses`/`typing`/`unittest`）实现的可导入单文件库，
零第三方依赖，全程离线。不调用任何机器学习库，也不使用第三方同类实现替代核心算法。

核函数与训练协方差：

```
k(x, x') = alpha^2 * exp(-||x - x'||^2 / (2 * lengthscale^2))
K_y      = K(X, X) + (noise^2 + jitter) * I,
# 旧边际似然/逐点预测接口: alpha / lengthscale / noise > 0
# 联合抽样接口允许 noise = 0, 用于零噪声训练点精确插值
```

## 环境准备（隔离 venv，可选但推荐）

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python --version  # 应显示 Python 3.14.7
```

不激活 venv 时也可直接使用系统 Python 3.14.7。模块只依赖标准库，不需要 `pip install`。

## 运行命令

```bash
python -m unittest discover -s tests -v
python demo.py
python demo_iteration.py                  # 联合抽样增量演示, 固定输入且 < 90 秒
```

无需安装依赖（`requirements.txt` 为空占位）；不需要网络、账号、密钥、数据库或 Docker。
所有演示均真实计算确定性输入，不读取随机种子配置、不调用 RNG、不使用 sleep 或预录结果。

## 文件

| 文件 | 说明 |
|---|---|
| `gprbf.py` | 题库本体：RBF 核、Cholesky、log 边际似然、解析梯度、后验预测、奇异 PSD 联合抽样 |
| `tests/test_gprbf.py` | 基线回归；独立参考不调用被测核心产生期望值 |
| `tests/test_joint_samples.py` | 联合抽样增量：重复查询、零噪声训练点、空查询、固定值条件化、PSD 秩判断 |
| `demo.py` | 确定性数据演示：正常结果（含 95% 置信区间）+ 实际触发的 `not_pd` 边界与 `ValueError` 拒绝 |
| `demo_iteration.py` | 联合抽样演示：完整协方差、重复点完全相关、零噪声训练点、固定值、拒绝与空查询 |

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

### 联合相关函数抽样

```python
from gprbf import Hyperparameters, sample_posterior

hp0 = Hyperparameters(alpha=1.2, lengthscale=0.8, noise=0.0)
r = sample_posterior(
    X, y, X_test, hp0, standard_normal,
    jitter=0.0,
    fixed_values=None,
)
```

- `standard_normal` 由调用方提供，形状为 `样本数 × 自由查询点数`；一行对应一次完整函数抽样。
- 抽样使用 `f = mean + C z`，其中 `C C^T` 是一次性计算的完整后验协方差；重复查询点共享同一随机方向，不逐点独立抽样。
- 联合抽样入口允许 `noise=0.0`，以支持训练点零噪声精确插值；旧 `log_marginal`/`predict` 仍要求 `noise>0`，接口和状态语义不变。
- 空 `X_test` 合法：每行 `standard_normal` 也必须为空行，返回相应数量的空样本。
- `fixed_values={index: value}` 可在已得后验上再条件化；重复坐标必须给同一值，不相容零方差观测返回 `status="incompatible_observation"`。
- `SampleResult.samples` 为 `(样本, 查询点)` 元组；`covariance` 是固定值条件化后的完整协方差，固定坐标行列为零，样本固定坐标逐位等于输入。
- 重复点使协方差半正定而非正定：内部使用带秩判断的对称特征分解，允许零特征方向；小负特征值只在相对容差内置零并记录于 `corrected_eigenvalues`。
- 明显负特征值返回 `status="not_psd"`；不会自动加入观测噪声或 jitter 掩盖问题。
- 联合抽样显式持有 `m×m` 协方差，查询点上限为 `MAX_JOINT_SAMPLE_POINTS = 80`；旧逐点 `predict` 仍保留 10000 点上限。

结果均为冻结 `dataclass`，字段以 `None`/元组区分成功与失败状态；函数无状态、不修改入参、
重复调用逐位一致。

## 输入上限与错误语义

- 训练样本 `1 ≤ n ≤ 80`；输入维数 `1 ≤ d ≤ 3`；预测点上限 10000（防护值，非题面要求）。
- 所有数值输入必须是有限的 `int`/`float`：**NaN、Infinity 一律 `ValueError` 拒绝**，
  错误信息带具体下标（如 `X[3][1]`、`y[2]`）便于定位。
- **`bool` 显式拒绝**（即使 Python 中 `bool` 是 `int` 子类）；普通整数坐标/目标值合法。
- 旧 `log_marginal`/`predict` 中 `alpha/lengthscale/noise ≤ 0` 会拒绝；`sample_posterior` 允许 `noise=0` 但仍拒绝负噪声；
- `jitter < 0`、行列数不一致、维度不匹配：`ValueError`；
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
