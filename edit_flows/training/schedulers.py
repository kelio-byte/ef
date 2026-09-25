"""用途：按 Noam 规则更新优化器学习率。
输入：优化器、模型维度、预热步数和缩放系数。
输出：每个训练步对应的学习率。
"""

class NoamScheduler:
    """作用：管理 Noam 学习率进度。输入：优化器及预热配置。输出：可逐步更新和恢复的调度器。"""

    def __init__(
        self,
        optimizer,
        d_model: int,
        warmup_steps: int = 8000,
        factor: float = 1.0,
    ):
        """作用：初始化 Noam 调度状态。输入：优化器、模型维度、预热步数和缩放系数。输出：调度器实例。"""
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def step(self):
        """作用：推进一步并更新优化器学习率。输入：无。输出：当前学习率。"""
        self._step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self, step: int = None) -> float:
        """作用：按 Noam 公式计算学习率。输入：可选训练步数。输出：学习率浮点数。"""
        step = self._step if step is None else step
        if step <= 0:
            return 0.0
        scale = self.d_model ** (-0.5)
        warmup = step * (self.warmup_steps ** (-1.5))
        decay = step ** (-0.5)
        return self.factor * scale * min(decay, warmup)

    def state_dict(self) -> dict:
        """作用：导出调度器恢复状态。输入：无。输出：包含当前步数的字典。"""
        return {"_step": self._step}

    def load_state_dict(self, state: dict):
        """作用：恢复调度器进度。输入：此前导出的状态字典。输出：原地更新调度器。"""
        self._step = int(state.get("_step", 0))
