"""Stage-2 network (plan section 3): player set -> team pooling -> temporal network -> heads.

    PlayerEncoder: shared MLP phi (9 -> 64 -> 64) per player, masked team mean pooling, then
                   concatenate group 1, group 2 and counts. Teams use the feature team column,
                   not slots: slot permutations preserve outputs; team swap flips the team column.
                   Separate module for stage-3 reuse (3a ball prediction, branch B fine-tuning).
    Temporal: bidirectional GRU (default) or dilated TCN, reading past and future.
    Heads: had [.,3], touch [.], team [.,2], ball [.,2].
"""

import torch
import torch.nn as nn

from src.nap2.features import F_GROUP, F_PRESENT, N_FRAME_FEAT, N_PLAYER_FEAT

MODEL_DEFAULTS = {
    "d_phi": 64,
    "d_model": 128,
    "temporal": "bigru",     # "bigru" | "tcn"
    "gru_hidden": 64,        # Per direction; output 2*64 = 128.
    "gru_layers": 2,
    "tcn_levels": 5,         # Dilations 1,2,4,8,16 -> receptive field of 63 steps.
    "dropout": 0.1,
}


class PlayerEncoder(nn.Module):
    def __init__(self, d_in=N_PLAYER_FEAT, d_phi=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, d_phi), nn.ReLU(), nn.Linear(d_phi, d_phi), nn.ReLU())
        self.d_out = 2 * d_phi + 2

    def forward(self, xp):
        """xp [B,W,46,F] -> [B,W,2*d_phi+2]"""
        pres = xp[..., F_PRESENT]
        grp = xp[..., F_GROUP]
        h = self.phi(xp)
        out = []
        counts = []
        for w in (pres * (1 - grp), pres * grp):
            n = w.sum(-1, keepdim=True)
            out.append((h * w.unsqueeze(-1)).sum(-2) / n.clamp(min=1.0))
            counts.append(n / 11.0)
        return torch.cat(out + counts, dim=-1)


class TCN(nn.Module):
    def __init__(self, d, levels, dropout):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(levels):
            dil = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(d, d, 3, padding=dil, dilation=dil), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(d, d, 1)))

    def forward(self, z):                      # [B,W,d]
        h = z.transpose(1, 2)
        for blk in self.blocks:
            h = h + blk(h)
        return h.transpose(1, 2)


class Nap2Net(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        c = dict(MODEL_DEFAULTS, **(cfg or {}))
        self.cfg = c
        self.encoder = PlayerEncoder(N_PLAYER_FEAT, c["d_phi"])
        d = c["d_model"]
        self.proj = nn.Sequential(nn.Linear(self.encoder.d_out + N_FRAME_FEAT, d), nn.ReLU(),
                                  nn.Dropout(c["dropout"]))
        if c["temporal"] == "bigru":
            self.temporal = nn.GRU(d, c["gru_hidden"], num_layers=c["gru_layers"], batch_first=True,
                                   bidirectional=True, dropout=c["dropout"] if c["gru_layers"] > 1 else 0.0)
            d_t = 2 * c["gru_hidden"]
        elif c["temporal"] == "tcn":
            self.temporal = TCN(d, c["tcn_levels"], c["dropout"])
            d_t = d
        else:
            raise ValueError(c["temporal"])
        self.had = nn.Linear(d_t, 3)
        self.touch = nn.Linear(d_t, 1)
        self.team = nn.Linear(d_t, 2)
        self.ball = nn.Linear(d_t, 2)

    def forward(self, xp, xf):
        """xp [B,W,46,9], xf [B,W,7] -> dict had [B,W,3], touch [B,W], team [B,W,2], ball [B,W,2]"""
        z = self.proj(torch.cat([self.encoder(xp), xf], dim=-1))
        if isinstance(self.temporal, nn.GRU):
            z, _ = self.temporal(z)
        else:
            z = self.temporal(z)
        return {"had": self.had(z), "touch": self.touch(z).squeeze(-1), "team": self.team(z),
                "ball": self.ball(z)}


def build_model(cfg=None):
    return Nap2Net(cfg)
