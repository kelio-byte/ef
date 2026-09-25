"""用途：定义训练与采样使用的时间桥接系数调度器。
输入：连续时间张量 t。
输出：桥接系数 κ(t) 及其导数。
"""

from abc import ABC, abstractmethod

from torch import Tensor


class KappaScheduler(ABC):
    """作用：规定桥接调度器的正向、导数和反函数接口。输入：时间或 κ 张量。输出：对应调度值。"""

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        """作用：计算桥接系数。输入：时间张量。输出：κ(t) 张量。"""
        ...

    @abstractmethod
    def derivative(self, t: Tensor) -> Tensor:
        """作用：计算桥接系数导数。输入：时间张量。输出：κ'(t) 张量。"""
        ...

    @abstractmethod
    def inverse(self, kappa: Tensor) -> Tensor:
        """作用：反解桥接时间。输入：κ 值。输出：满足 κ(t)=κ 的时间张量。"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """作用：返回调度器名称。输入：无。输出：名称字符串。"""
        ...


class CubicScheduler(KappaScheduler):
    """作用：实现 κ(t)=t³ 的时间调度。输入：连续时间。输出：桥接系数及相关变换。"""

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        """作用：初始化三次调度器。输入：兼容保留的 a、b 参数。输出：调度器实例。"""
        super().__init__()
        self.a = a
        self.b = b

    def __call__(self, t: Tensor) -> Tensor:
        """作用：计算三次桥接系数。输入：时间张量。输出：t 的三次方。"""
        return t**3

    def derivative(self, t: Tensor) -> Tensor:
        """作用：计算三次桥接系数的导数。输入：时间张量。输出：3t²。"""
        return 3 * t**2

    def inverse(self, kappa: Tensor) -> Tensor:
        """作用：反解三次桥接时间。输入：κ 张量。输出：κ 的立方根。"""
        return kappa ** (1.0 / 3.0)

    @property
    def name(self) -> str:
        """作用：标识调度器类型。输入：无。输出：字符串 `cubic`。"""
        return "cubic"
