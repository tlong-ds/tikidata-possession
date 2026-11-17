"""Multitask loss (stage-2 plan section 4): L = L_HAD + w_touch*(L_touch + L_team) + w_ball*L_ball.

    L_HAD: H/A/D cross-entropy with slight smoothing near uncertain change boundaries.
    L_touch: pos_weight BCE (~1% touch frames), using temporally blurred targets.
    L_team: touching-team cross-entropy near unambiguous touches only.
    L_ball: Huber on scaled true ball positions, encouraging physical understanding.
All losses use frame weight w (0 excludes out-of-half, padding or rejected pseudo-labelled intervals).
"""

import torch
import torch.nn.functional as F

LOSS_DEFAULTS = {
    "w_touch": 0.5,
    "w_ball": 0.2,
    "smooth_near": 0.1,       # Soften H/A/D labels near transitions.
    "touch_pos_weight": 10.0,
    "huber_delta": 0.05,      # ~2,6 m theo x sau khi chia thang 52,5
}


def _wmean(v, w):
    return (v * w).sum() / w.sum().clamp(min=1e-6)


def compute_losses(out, batch, cfg=None):
    c = dict(LOSS_DEFAULTS, **(cfg or {}))
    w = batch["w"]
    y = batch["y"].long()
    valid = (y >= 0).float() * w

    logp = F.log_softmax(out["had"].float(), dim=-1)
    tgt = F.one_hot(y.clamp(min=0), 3).float()
    eps = c["smooth_near"] * batch["near"].float()
    tgt = tgt * (1 - eps.unsqueeze(-1)) + eps.unsqueeze(-1) / 3.0
    l_had = _wmean(-(tgt * logp).sum(-1), valid)

    pw = torch.tensor(c["touch_pos_weight"], device=w.device)
    l_touch = _wmean(F.binary_cross_entropy_with_logits(out["touch"].float(), batch["touch"],
                                                        pos_weight=pw, reduction="none"), w)

    team = batch["team"].long()
    tmask = (team >= 0).float() * w
    l_team = _wmean(F.cross_entropy(out["team"].float().reshape(-1, 2), team.clamp(min=0).reshape(-1),
                                    reduction="none").reshape(team.shape), tmask)

    bmask = batch["ball_ok"].float() * w
    l_ball = _wmean(F.huber_loss(out["ball"].float(), batch["ball"], reduction="none",
                                 delta=c["huber_delta"]).sum(-1), bmask)

    total = l_had + c["w_touch"] * (l_touch + l_team) + c["w_ball"] * l_ball
    return total, {"had": l_had.item(), "touch": l_touch.item(), "team": l_team.item(),
                   "ball": l_ball.item(), "total": total.item()}
