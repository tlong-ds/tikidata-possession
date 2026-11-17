"""Targets and loss for the stage-3 ownership-switch head.

Targets are built from y AFTER augmentation (team swap, time reversal), matching the inputs:
    change_k[t] = y[t] == k and y[t-1] != k and y[t-1] valid, k = H (column 0), A (column 1)
Include restarts after D (D -> H, D -> A): deadball_switch_gate=True also needs restart-team evidence.

Blur each change with an asymmetric Gaussian: wider BEFORE (sigma_early), narrower AFTER
(sigma_late), since late predictions cost more than early ones (1 s late gives spellF1 ~0.65).
Units are window STEPS: stride 2 -> one step = two frames = 0.08 s.
"""

import torch
import torch.nn.functional as F

H, A = 0, 1


def change_impulses(y):
    """y [B,W] (int, -1 ignored) -> float [B,W,2], 1 at the first step of a new H/A owner.

    Window step 0 has no predecessor, so it is never a change and is excluded from the loss.
    """
    y = y.long()
    prev = torch.full_like(y, -1)
    prev[:, 1:] = y[:, :-1]
    out = []
    for k in (H, A):
        out.append(((y == k) & (prev != k) & (prev >= 0)).float())
    return torch.stack(out, -1)


def asym_kernel(sigma_early=2.0, sigma_late=1.0, radius=None, device=None):
    """1D kernel [2r+1] for F.conv1d correlation: index r+m (m>0) is m steps BEFORE a change;
    r-m is m steps AFTER. Peak = 1.
    """
    if radius is None:
        radius = int(max(1, round(3 * max(sigma_early, sigma_late))))
    m = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    s = torch.where(m > 0, torch.tensor(float(sigma_early)), torch.tensor(float(sigma_late)))
    s = s.clamp(min=1e-3)
    return torch.exp(-0.5 * (m / s) ** 2)


def switch_target(y, sigma_early=2.0, sigma_late=1.0):
    """y [B,W] -> soft [B,W,2] targets in [0,1]; sum nearby changes and clip at 1."""
    imp = change_impulses(y)                                   # [B,W,2]
    B, W, _ = imp.shape
    k = asym_kernel(sigma_early, sigma_late, device=imp.device)
    r = (len(k) - 1) // 2
    x = imp.permute(0, 2, 1).reshape(B * 2, 1, W)
    out = F.conv1d(x, k.view(1, 1, -1), padding=r)
    return out.reshape(B, 2, W).permute(0, 2, 1).clamp(0.0, 1.0)


def switch_mask(y, w):
    """Switch loss weights [B,W]: w, excluding step 0 and invalid current / previous y."""
    y = y.long()
    ok = (y >= 0).float()
    ok[:, 1:] = ok[:, 1:] * (y[:, :-1] >= 0).float()
    ok[:, 0] = 0.0
    return ok * w.float()


def switch_loss(logits, target, mask, pos_weight=20.0):
    """BCE with pos_weight (changes occupy ~1.4% of steps), weighted mean over [B,W,2]."""
    pw = torch.tensor(float(pos_weight), device=logits.device)
    l = F.binary_cross_entropy_with_logits(logits.float(), target, pos_weight=pw, reduction="none")
    m = mask.unsqueeze(-1).expand_as(l)
    return (l * m).sum() / m.sum().clamp(min=1e-6)


def near_change(y, steps):
    """bool [B,W]: within +-steps of any y transition (H<->A, D<->live)."""
    y = y.long()
    ch = torch.zeros_like(y, dtype=torch.bool)
    ch[:, 1:] = (y[:, 1:] != y[:, :-1]) & (y[:, 1:] >= 0) & (y[:, :-1] >= 0)
    if steps <= 0:
        return ch
    x = ch.float().unsqueeze(1)
    return F.max_pool1d(x, 2 * steps + 1, stride=1, padding=steps).squeeze(1) > 0
