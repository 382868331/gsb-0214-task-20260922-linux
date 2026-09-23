"""输入校验：统一的可定位错误语义。"""

from __future__ import annotations

import math
from typing import Any, List

MAX_SAMPLES = 80
MAX_DIMS = 3


class GPInputError(ValueError):
    """非法输入（形状、取值、边界）。错误信息以参数名为前缀，可定位。"""


def _ensure_finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GPInputError(f"{where}: 必须是实数（int/float），收到 {type(value).__name__}")
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise GPInputError(f"{where}: 拒绝 NaN/Infinity")
    return f


def ensure_positive(value: Any, name: str) -> float:
    """校验正标量；bool 被显式拒绝（True/False 不是合法数值参数）。"""
    f = _ensure_finite(value, name)
    if f <= 0.0:
        raise GPInputError(f"{name}: 必须 > 0，收到 {value!r}")
    return f


def ensure_nonneg(value: Any, name: str) -> float:
    """校验非负标量；bool 被显式拒绝。"""
    f = _ensure_finite(value, name)
    if f < 0.0:
        raise GPInputError(f"{name}: 必须 >= 0，收到 {value!r}")
    return f


def check_matrix(
    X: Any,
    name: str,
    *,
    max_rows: int = MAX_SAMPLES,
    max_cols: int = MAX_DIMS,
    min_rows: int = 1,
    min_cols: int = 1,
) -> List[List[float]]:
    """校验二维实数矩阵，返回深拷贝的 float 列表。

    - 必须是非空 list/序列的序列，行长度一致；
    - 每个元素必须是 int/float 且有限，bool 拒绝；
    - 行数 1..80，列数 1..3（默认契约）。
    """
    if not isinstance(X, (list, tuple)) or isinstance(X, (str, bytes)):
        raise GPInputError(f"{name}: 必须是行的序列（list[list]），收到 {type(X).__name__}")
    if not (min_rows <= len(X) <= max_rows):
        raise GPInputError(
            f"{name}: 行数必须在 [{min_rows}, {max_rows}]，收到 {len(X)}"
        )
    width: int
    rows: List[List[float]] = []
    for i, row in enumerate(X):
        if not isinstance(row, (list, tuple)) or isinstance(row, (str, bytes)):
            raise GPInputError(f"{name}[{i}]: 必须是行序列，收到 {type(row).__name__}")
        if i == 0:
            width = len(row)
            if not (min_cols <= width <= max_cols):
                raise GPInputError(
                    f"{name}[{i}]: 列数必须在 [{min_cols}, {max_cols}]，收到 {width}"
                )
        elif len(row) != width:
            raise GPInputError(
                f"{name}[{i}]: 行长度不一致，期望 {width}，收到 {len(row)}"
            )
        out_row: List[float] = []
        for j, v in enumerate(row):
            out_row.append(_ensure_finite(v, f"{name}[{i}][{j}]"))
        rows.append(out_row)
    return rows


def check_vector(
    y: Any, name: str, length: int
) -> List[float]:
    """校验长度为 ``length`` 的有限实数向量，返回 float 列表。"""
    if not isinstance(y, (list, tuple)) or isinstance(y, (str, bytes)):
        raise GPInputError(f"{name}: 必须是序列，收到 {type(y).__name__}")
    if len(y) != length:
        raise GPInputError(f"{name}: 长度必须为 {length}，收到 {len(y)}")
    out: List[float] = []
    for i, v in enumerate(y):
        out.append(_ensure_finite(v, f"{name}[{i}]"))
    return out
