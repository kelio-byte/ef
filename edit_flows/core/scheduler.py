from abc import ABC, abstractmethod

from torch import Tensor


class KappaScheduler(ABC):
    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor: ...

    @abstractmethod
    def derivative(self, t: Tensor) -> Tensor: ...

    @abstractmethod
    def inverse(self, kappa: Tensor) -> Tensor:
        """Inverse mapping: given kappa ∈ [0,1], return t such that kappa(t) = kappa."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def __call__(self, t: Tensor) -> Tensor:
        return t**3

    def derivative(self, t: Tensor) -> Tensor:
        return 3 * t**2

    def inverse(self, kappa: Tensor) -> Tensor:
        return kappa ** (1.0 / 3.0)

    @property
    def name(self) -> str:
        return "cubic"
