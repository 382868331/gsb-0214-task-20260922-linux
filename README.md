# gp_rbf —— 精确 RBF 高斯过程回归（核参数解析梯度）

零依赖、离线的教学/工程内核模块：对 RBF 高斯过程回归提供**精确**（非稀疏、
非近似推断）的 log 边际似然、对 `log(a, l, s)` 的**解析梯度**，以及
latent / observed 两种后验预测方差。全部线性代数（Cholesky、三角求解、
矩阵求逆）在 `gp_rbf/_linalg.py` 中以 Python 标准库从零实现，**不调用任何
机器学习库或第三方同类实现**。

## 运行环境

* Python **3.14.7**（仅使用标准库 `math`、`dataclasses`、`typing`；3.10+ 亦可运行）
* Linux 原生，装好 Python 后**无需安装任何依赖**，无网络/账号/密钥/数据库/云 API
* 核心、测试、演示全部离线、确定性

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（24 个）
python demo.py                            # 固定输入演示，约 0.1 秒完成（预算 8 秒内）
```

## 数学定义

零均值 GP 先验，核函数

```
k(x, x') = a^2 * exp( -||x - x'||^2 / (2 l^2) ),   a > 0, l > 0
```

训练协方差

```
C = K(X, X) + s^2 I,                                s > 0
```

若显式给定 `jitter >= 0`，实际使用 `C = K + (s^2 + jitter) I`。
jitter **只来自函数参数**：Cholesky 失败时模块**绝不**自动添加任何
未记录的 jitter。

* log 边际似然：`L = -1/2 y'C⁻¹y - 1/2 log|C| - n/2 log(2π)`，经 Cholesky 计算。
* 解析梯度（令 `G = ααᵀ - C⁻¹`，`α = C⁻¹y`）：
  * `dL/d log a = tr(G K)`（因 `∂K/∂log a = 2K`）
  * `dL/d log l = (1/2l²) Σᵢⱼ Gᵢⱼ Kᵢⱼ ‖xᵢ-xⱼ‖²`
  * `dL/d log s = s² tr(G)`（因 `∂C/∂log s = 2s²I`）
  * jitter 是固定常数，计入 `C`、似然与全部预测公式，但不产生超参导数。
* 预测（`k* = k(X, x*)`）：
  * 均值 `m = k*ᵀ α`
  * latent 方差（隐函数值）：`a² - k*ᵀ C⁻¹ k*`
  * observed 方差（含新观测噪声）：latent 方差 `+ s²`（**不加** jitter）

## 公开接口

```python
from gp_rbf import fit, predict, predict_one, GPInputError

r = fit(X, y, a, l, s, jitter=0.0)        # -> FitResult
if r.status == "ok":                       # 或 r.ok
    r.log_marginal                         # float
    r.grad_log                             # (dL/d log a, dL/d log l, dL/d log s)
    r.residuals.symmetry                   # max|K-K^T|（构造对称，恒为 0.0）
    r.residuals.chol_reconstruct           # ||C - LL^T||_F
    r.residuals.chol_solve                 # ||C·alpha - y||_inf
    p = predict(r, Xstar)                  # -> Prediction
    p.mean, p.latent_var, p.observed_var   # 各为 list[float]，逐查询点
    p.clamped                              # 被钳制的负微量记录 tuple
else:                                      # r.status == "not_pd"
    r.log_marginal is r.grad_log is None   # 不回退、不加 jitter、不编造数值

m, lv, ov = predict_one(r, x)              # 单点便捷接口
```

数据结构均为不可变 `dataclass`（`Hyperparams / Residuals / FitResult /
Prediction / ClampedVariance`）。`fit` 是纯函数，`predict` 只读拟合结果，
不存在有状态的部分变更。

## 输入上限与错误语义

* 训练样本 `1 <= n <= 80`；特征维度 `1 <= d <= 3`；查询点数 `1..80` 且维度与训练一致。
* 所有数值输入拒绝 `NaN`/`Infinity`（抛 `GPInputError`）。
* 标量超参 `a/l/s` 必须严格为正，`jitter >= 0`；**`bool` 被显式拒绝**
  （`True/False` 不是合法数值参数），数据矩阵中的 bool 同样拒绝。
* 错误信息带定位前缀，如 `y[2]: 拒绝 NaN/Infinity`、`X[1]: 行长度不一致`。
* 非法输入在任何计算开始前抛出，不做静默纠正。

### `not_pd` 语义（与“预算耗尽”严格区分）

本模块不含任何迭代搜索/优化器，因此不存在“预算耗尽”。Cholesky 中一旦出现
非正主元（含重复输入且噪声被浮点吞没等情形），即**确定性地证明**当前 `C`
非正定，返回 `status="not_pd"`（不抛异常）；对该结果调用 `predict` 会抛
`RuntimeError`。恢复手段只有调用方显式增大 `s` 或传入 `jitter` 后重新 `fit`。

### 负预测微量处理

latent 方差数学上非负，浮点下可能得到 `-1e-10 <= raw < 0` 的负微量：
钳为 `0.0` 并在 `Prediction.clamped` 记录 `(index, raw, kind)`；若
`raw < -1e-10`（见 `gp_rbf.NEG_VAR_TOL`）则抛 `RuntimeError`，不掩盖
真正的错误。observed 方差同理（其上还加了 `s²`，正常不会为负）。

## 误差口径（IEEE-754 double，不用宽容差掩盖边界）

测试中的期望值**不**来自被测核心：`tests/ref_gp.py` 是独立的 Gauss-Jordan
带部分选主元直接求逆参考（不导入 `gp_rbf`），另有 n=1 闭式标量公式与
中心差分。

* **条件数温和（κ ≲ 100）**：Cholesky 路径与直接逆参考的 log 边际似然、
  预测均值/方差一致到 **1e-12 量级**（测试 places=9~13）；
  `chol_reconstruct ≲ 2e-15`、`chol_solve ≲ 5e-16`（见 demo 自检残差）。
* **解析梯度**：与中心差分（H=1e-3）偏差约 `5e-7`，即二阶中心差分的
  `O(H²)` 截断量级；H 按 √10 缩小时偏差严格按 H² 下降（已验证
  H=3e-3→1e-4 误差 1e-5→1e-8），证明解析值是精确导数而非巧合吻合。
* **病态矩阵（κ≈1e5~1e10，如完全重复输入 + 极小噪声）**：两条独立求逆
  路径之间的差异约为 `κ·eps_mach`（1e-10~1e-6），这是双精度浮点的误差
  传播下界；此类配置在测试中只按该量级核对状态与相对误差，精确到 1e-13
  的公式核对在 κ≈51 的同结构配置上进行。
* 负方差钳制容差 `1e-10` 是绝对量，仅针对“在 0 附近”的方差。

## 设计取舍

* 显式 `C⁻¹` 仅用于梯度；似然与预测走 Cholesky 三角求解（O(n³) 一次分解，
  O(n²) 每查询点），数值稳定性优于显式求逆。
* 核矩阵对称构造、逐元素独立计算，不做对角提升/截断，故 `K-Kᵀ` 逐位为 0；
  对称残差仍照常返回。
* 提供梯度但**不内置超参优化器**（题目只要求梯度）；调用方可自行用梯度做
  L-BFGS 等。无搜索即无“预算耗尽”这一结果类别。
* 不实现均值函数、ARD、其他核或稀疏/近似 GP——仅按本题数学子集实现，
  不宣称行业标准 GP 框架兼容性。

## 已知限制

* 规模上限 80 样本 / 3 维是接口边界；稠密 O(n³) 纯 Python 实现不为更大规模设计。
* 仅零均值先验、各向同性单长度尺度 RBF、高斯似然。
* 梯度只对 `(log a, log l, log s)`；jitter 无梯度（按定义为常数）。
* 置信区间演示使用正态临界值 1.96 的解析形式（GP 后验即高斯，精确）。
* 极端病态（κ 接近 1/eps≈4.5e15）下双精度本身不可靠；此时应增大 `s` 或
  显式给定 jitter，模块不会代用户做这种建模决定。

## 仓库结构

```
gp_rbf/__init__.py    公开接口
gp_rbf/model.py       fit/predict、解析梯度、残差与负微量处理
gp_rbf/_kernel.py     RBF 核
gp_rbf/_linalg.py     Cholesky/三角求解/求逆（纯标准库）
gp_rbf/_validate.py   输入校验与 GPInputError
tests/test_gp.py      24 个必要测试
tests/ref_gp.py       独立直接逆参考（不导入 gp_rbf）
demo.py               固定输入演示
```
