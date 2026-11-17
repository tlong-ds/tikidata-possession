"""loss.near_change adds H/A/D cross-entropy within +-steps of label transitions.
69% of stage-2 team errors fall within 1 s of a true change.
"""

from . import LossStep, register


@register
class NearChange(LossStep):
    name = "near_change"
    defaults = {"weight": 0.5, "steps": 12}          # 12 steps = 1 s (stride 2 at 25 fps).

    def loss(self, out, batch):
        from src.core.train_switch import near_change_loss
        return near_change_loss(out["had"], batch["y"], batch["w"], int(self.steps))
