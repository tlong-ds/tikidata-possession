"""Stage-2 training data (plan section 2), using shared team data (src/nap2/shared.py).

    prepare_file(path): degraded View + labels -> stage-1 B_hat / K -> features
    ChunkSampler: sample, shuffle and batch W-step chunks wholly within a half, using stride
    augment(batch): per-sample team swap (H<->A, x flip) and vertical flip on GPU

Each sample is a long chunk (default 128 steps x 2 frames = 10.2 s), predicting EVERY step
rather than only the center of a 3 s window. The BiGRU still reads past and future; overlapping
inference retains central predictions, at roughly W-fold lower cost.

Degradation and View construction belong to pipeline.feed (pipeline/FEED.md); match splits use
pipeline.split. Block 2 is rerun during preparation, so changing stage-1 block 2 needs no data rebuild.
"""

import numpy as np
import torch

from src.nap1.ball import estimate_ball
from src.nap2 import shared
from src.nap2.shared import runs_of
from src.nap2.features import build_features, targets_from_arrays

TARGET_KEYS = ("y", "touch", "team", "ball", "ball_ok", "near", "w")


def prepare_view(V, targets=None, ball_params=None, feat_cfg=None):
    """src.nap1.view.View -> numpy-array dict ready for chunk sampling / inference."""
    Bhat, K = estimate_ball(V, ball_params if ball_params is not None else shared.nap1_params())
    f = build_features(V, Bhat, feat_cfg)
    d = {"xp": f["xp"].astype(np.float16), "xf": f["xf"], "half": np.asarray(V.meta["half"]),
         "K_rule": K.astype(np.float32)}
    if targets is not None:
        d.update(targets)
    return d


def prepare_file(path, max_frames=None, with_targets=True, ball_params=None, feat_cfg=None):
    """Shared data file -> prepare_view output plus name, touch_true and baseline when available."""
    return prepare_raw(shared.load_file(path, max_frames), with_targets, ball_params, feat_cfg)


def prepare_fresh(name, epoch, run=0, max_frames=None):
    """NEW degraded training match (shared.fresh_train) -> the same format as prepare_file."""
    return prepare_raw(shared.fresh_train(name, epoch, run, max_frames))


def prepare_item(item, max_frames=None):
    """item: file path or ("fresh", match, epoch, run); subprocess-compatible."""
    if isinstance(item, (tuple, list)) and item and item[0] == "fresh":
        return prepare_fresh(item[1], item[2], item[3], max_frames)
    return prepare_file(item, max_frames)


def prepare_raw(raw, with_targets=True, ball_params=None, feat_cfg=None):
    V = shared.as_view(raw)
    tg = None
    if with_targets and "y" in raw:
        tg = targets_from_arrays(raw["y"], raw.get("touch_true"), raw["ball_xy"], raw["half"], V.meta["fps"])
    d = prepare_view(V, tg, ball_params, feat_cfg)
    d["name"] = raw["name"]
    for k in ("touch_true", "baseline"):              # Baseline: smooth50 is already in the dev file.
        if k in raw:
            d[k] = raw[k]
    if "y" in raw:
        d["y_raw"] = raw["y"]                          # Original labels for pipeline.score_export scoring.
    return d


class ChunkSampler:
    def __init__(self, matches, chunk=128, stride=2, oversample=2):
        self.matches = matches
        self.chunk, self.stride = chunk, stride
        self.span = (chunk - 1) * stride + 1
        self.tile = max(1, chunk * stride // max(1, oversample))   # oversample=2: half-overlapping chunks.

    def _items(self, rng):
        items = []
        tile = self.tile
        for mi, d in enumerate(self.matches):
            for a, b, h in runs_of(d["half"]):
                if h <= 0 or b - a < self.span:
                    continue
                off = int(rng.integers(0, max(1, min(tile, b - a - self.span + 1))))
                items += [(mi, s) for s in range(a + off, b - self.span + 1, tile)]
        return items

    def n_batches(self, batch_size):
        n = len(self._items(np.random.default_rng(0)))
        return max(1, -(-n // batch_size))

    def batches(self, rng, batch_size, max_chunks=None):
        items = self._items(rng)
        rng.shuffle(items)
        if max_chunks:
            items = items[:max_chunks]
        for i in range(0, len(items), batch_size):
            yield self.collate(items[i:i + batch_size])

    def collate(self, items):
        steps = np.arange(self.chunk) * self.stride
        out = {k: [] for k in ("xp", "xf") + TARGET_KEYS}
        for mi, s in items:
            d = self.matches[mi]
            idx = s + steps
            for k in out:
                out[k].append(d[k][idx])
        b = {k: torch.from_numpy(np.stack(v)) for k, v in out.items()}
        b["xp"] = b["xp"].float()
        return b


def augment(batch, gen, p_swap=0.5, p_flip=0.5):
    """Per-sample team swap and vertical flip; gen is a CPU torch.Generator for reproducibility."""
    xp, xf = batch["xp"].clone(), batch["xf"].clone()
    B, dev = xp.shape[0], xp.device
    swap = (torch.rand(B, generator=gen) < p_swap).to(dev)
    flip = (torch.rand(B, generator=gen) < p_flip).to(dev)
    sx = torch.where(swap, -1.0, 1.0).view(B, 1, 1)
    sy = torch.where(flip, -1.0, 1.0).view(B, 1, 1)
    for col in (0, 2, 6):                       # x, vx, dx
        xp[..., col] *= sx
    for col in (1, 3, 7):                       # y, vy, dy
        xp[..., col] *= sy
    pres = xp[..., 5]
    sw = swap.view(B, 1, 1)
    xp[..., 4] = torch.where(sw, pres * (1 - xp[..., 4]), xp[..., 4])
    sx2, sy2 = sx.view(B, 1), sy.view(B, 1)
    xf[..., 0] *= sx2
    xf[..., 4] *= sx2
    xf[..., 1] *= sy2
    xf[..., 5] *= sy2
    out = dict(batch, xp=xp, xf=xf)
    sw2 = swap.view(B, 1)
    y = batch["y"]
    out["y"] = torch.where(sw2 & (y >= 0) & (y <= 1), 1 - y, y)
    team = batch["team"]
    out["team"] = torch.where(sw2 & (team >= 0), 1 - team, team)
    ball = batch["ball"].clone()
    ball[..., 0] *= sx2
    ball[..., 1] *= sy2
    out["ball"] = ball
    return out
