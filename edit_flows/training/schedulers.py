class NoamScheduler:
    def __init__(
        self,
        optimizer,
        d_model: int,
        warmup_steps: int = 8000,
        factor: float = 1.0,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        step = self._step
        scale = self.d_model ** (-0.5)
        warmup = step * (self.warmup_steps ** (-1.5))
        decay = step ** (-0.5)
        return self.factor * scale * min(decay, warmup)

    def state_dict(self) -> dict:
        return {"_step": self._step}

    def load_state_dict(self, state: dict):
        self._step = state["_step"]
