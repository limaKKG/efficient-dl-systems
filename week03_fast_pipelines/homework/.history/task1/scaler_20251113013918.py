import torch 

class ManualLossScaler:
    def __init__(self, init_scale: float = 1024.0, 
                        growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, 
                 growth_interval: int = 200,
                 dynamic: bool = False) -> None:
        self.scale = float(init_scale)
        self.dynamic = bool(dynamic)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self._good_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    @staticmethod
    def _has_inf_or_nan(model: torch.nn.Module) -> bool:
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad
                if torch.isnan(g).any() or torch.isinf(g).any():
                    return True
        return False

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        inv = 1.0 / self.scale
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    with torch.no_grad():
                        p.grad.mul_(inv)

    def update(self, found_inf: bool) -> None:
        if not self.dynamic:
            return
        if found_inf:
            self.scale = max(self.scale * self.backoff_factor, 1.0)
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps % self.growth_interval == 0:
                self.scale *= self.growth_factor