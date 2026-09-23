"""gp_rbf：精确 RBF 高斯过程回归（纯 Python 标准库）。

公开接口
--------

* :func:`fit`            —— Cholesky 拟合、log 边际似然与解析超参梯度
* :func:`predict`        —— 批量后验预测（latent / observed 两种方差）
* :func:`predict_one`    —— 单点预测
* :class:`Hyperparams`   —— 超参 (a, l, s, jitter)
* :class:`FitResult`     —— 拟合结果（含 status、梯度、数值残差）
* :class:`Prediction`    —— 预测结果
* :class:`GPInputError`  —— 非法输入异常
* :data:`NEG_VAR_TOL`    —— 负预测方差钳制容差
* :data:`MAX_SAMPLES` / :data:`MAX_DIMS` —— 80 样本、3 维
"""

from ._kernel import rbf_scalar
from ._validate import GPInputError, MAX_DIMS, MAX_SAMPLES
from .model import (
    NEG_VAR_TOL,
    ClampedVariance,
    FitResult,
    Hyperparams,
    Prediction,
    Residuals,
    STATUS_NOT_PD,
    STATUS_OK,
    fit,
    predict,
    predict_one,
)

__all__ = [
    "fit",
    "predict",
    "predict_one",
    "Hyperparams",
    "FitResult",
    "Prediction",
    "ClampedVariance",
    "Residuals",
    "GPInputError",
    "STATUS_OK",
    "STATUS_NOT_PD",
    "NEG_VAR_TOL",
    "MAX_SAMPLES",
    "MAX_DIMS",
    "rbf_scalar",
]
