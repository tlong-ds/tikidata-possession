"""Stage-3 switchnet = stage-2 network + extra inputs (src.core.feats) + team-switch head.

Retain src.nap2.model.Nap2Net attribute names (encoder, proj, temporal, had, touch, team, ball)
so init_from_nap2 can load stage-2 checkpoints. Widened input layers copy existing columns and
zero new ones, initially reproducing stage-2 outputs EXACTLY; the switch head learns from scratch.
No changes to src/nap2/.
"""

import torch
import torch.nn as nn

from src.nap2.features import F_GROUP, F_PRESENT, N_FRAME_FEAT, N_PLAYER_FEAT
from src.nap2.model import TCN, Nap2Net, PlayerEncoder

from .feats import N_FRAME_FEAT3, N_PLAYER_FEAT3

NAP3_DEFAULTS = {"n_player_feat": N_PLAYER_FEAT3, "n_frame_feat": N_FRAME_FEAT3, "player_attn": False, "attn_heads": 4,
                 "par_tcn": 0, "par_player": False}
NAP3_MODEL_KEYS = ("player_attn", "attn_heads", "par_tcn", "par_player")      # Stage-3 configuration inherited from the initialization checkpoint.
N_BALL_IN = 4                                         # xf[..., :4] = ball_x, ball_y, ball_seen, ball_sigma (stage 2).


class PlayerAttn(nn.Module):
    """Masked self-attention among a frame's players, with an always-present ball token so no frame
    is fully masked. Add outputs residually through zero-initialized `out`: initially unchanged.
    """

    def __init__(self, d, heads=4):
        super().__init__()
        self.ball = nn.Linear(N_BALL_IN, d)
        self.norm = nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, pres, ball):
        """h [B,W,S,d], pres [B,W,S] (>0 = present), ball [B,W,4] -> h [B,W,S,d]."""
        B, W, S, d = h.shape
        tok = torch.cat([h, self.ball(ball).unsqueeze(-2)], dim=-2).reshape(B * W, S + 1, d)
        skip = torch.cat([pres <= 0, torch.zeros(B, W, 1, dtype=torch.bool, device=h.device)], dim=-1)
        q = self.norm(tok)
        a, _ = self.mha(q, q, q, key_padding_mask=skip.reshape(B * W, S + 1), need_weights=False)
        return h + self.out(a[:, :S]).reshape(B, W, S, d)


class AttnPlayerEncoder(PlayerEncoder):
    """Stage-2 PlayerEncoder + PlayerAttn between encoder.phi and team pooling. Preserve encoder.phi.*
    names and d_out for init_by_names / freeze._input_mask; new block: encoder.attn.*.
    """

    def __init__(self, d_in, d_phi=64, heads=4):
        super().__init__(d_in, d_phi)
        self.attn = PlayerAttn(d_phi, heads)

    def forward(self, xp, ball):
        pres = xp[..., F_PRESENT]
        grp = xp[..., F_GROUP]
        h = self.attn(self.phi(xp), pres, ball)
        out, counts = [], []
        for w in (pres * (1 - grp), pres * grp):
            n = w.sum(-1, keepdim=True)
            out.append((h * w.unsqueeze(-1)).sum(-2) / n.clamp(min=1.0))
            counts.append(n / 11.0)
        return torch.cat(out + counts, dim=-1)


class Nap3Net(Nap2Net):
    def __init__(self, cfg=None):
        super().__init__(dict(NAP3_DEFAULTS, **(cfg or {})))
        c = self.cfg                                   # Already merged with stage-2 MODEL_DEFAULTS.
        npf, nff = int(c["n_player_feat"]), int(c["n_frame_feat"])
        self.player_attn = bool(c.get("player_attn"))
        if self.player_attn:
            self.encoder = AttnPlayerEncoder(npf, c["d_phi"], int(c.get("attn_heads", 4)))
        elif npf != N_PLAYER_FEAT:
            self.encoder = PlayerEncoder(npf, c["d_phi"])
        if npf != N_PLAYER_FEAT or nff != N_FRAME_FEAT:
            self.proj = nn.Sequential(nn.Linear(self.encoder.d_out + nff, c["d_model"]), nn.ReLU(),
                                      nn.Dropout(c["dropout"]))
        self.switch = nn.Linear(self.had.in_features, 2)
        d_t = self.had.in_features
        # Parallel temporal branch, added residually through a zero-initialized output layer (initially the old network):
        #   par_tcn = k: k-layer dilated TCN (1, 2, ..., 2^(k-1)) on z, with a fixed multiscale context.
        #   par_player: one-layer BiGRU on player encoder outputs only, without ball or frame features.
        #               Learn ownership cues from formation, robust to smoothed or occluded balls.
        self.par_tcn = int(c.get("par_tcn") or 0)
        self.par_player = bool(c.get("par_player"))
        if self.par_tcn:
            self.ptcn = TCN(c["d_model"], self.par_tcn, c["dropout"])
            self.ptcn_out = nn.Linear(c["d_model"], d_t)
            nn.init.zeros_(self.ptcn_out.weight)
            nn.init.zeros_(self.ptcn_out.bias)
        if self.par_player:
            self.pgru = nn.GRU(self.encoder.d_out, c["gru_hidden"], num_layers=1, batch_first=True, bidirectional=True)
            self.pgru_out = nn.Linear(2 * c["gru_hidden"], d_t)
            nn.init.zeros_(self.pgru_out.weight)
            nn.init.zeros_(self.pgru_out.bias)

    def forward(self, xp, xf):
        """xp [B,W,46,11], xf [B,W,21] -> had [B,W,3], touch [B,W], team [B,W,2], ball [B,W,2],
        switch [B,W,2] (switch-to-H and switch-to-A logits).
        """
        enc = self.encoder(xp, xf[..., :N_BALL_IN]) if self.player_attn else self.encoder(xp)
        z0 = self.proj(torch.cat([enc, xf], dim=-1))
        if isinstance(self.temporal, nn.GRU):
            z, _ = self.temporal(z0)
        else:
            z = self.temporal(z0)
        if self.par_tcn:
            z = z + self.ptcn_out(self.ptcn(z0))
        if self.par_player:
            z = z + self.pgru_out(self.pgru(enc)[0])
        return {"had": self.had(z), "touch": self.touch(z).squeeze(-1), "team": self.team(z),
                "ball": self.ball(z), "switch": self.switch(z)}


def build_model(cfg=None):
    return Nap3Net(cfg)


def init_by_names(model, state, src_xp, src_xf, dst_xp, dst_xf):
    """Load a state_dict into Nap3Net, aligning input columns by NAME: copy shared columns,
    zero new ones, discard source-only columns. Input layers: encoder.phi.0 (xp) and proj.0 ([encoder | xf]).

    Zero new columns preserve the source checkpoint's initial outputs. Return dict: widened
    (realigned input-layer keys), missing (absent source keys, e.g. switch.* from stage 2),
    new_xp / new_xf (new columns), dropped_xp / dropped_xf (unused source columns).
    """
    src_xp, src_xf, dst_xp, dst_xf = list(src_xp), list(src_xf), list(dst_xp), list(dst_xf)
    own = model.state_dict()
    d_enc = model.encoder.d_out
    widened, new = [], {}
    for k, v in state.items():
        if k not in own:
            continue
        tgt = own[k]
        if k == "encoder.phi.0.weight" and v.dim() == 2 and v.shape[1] == len(src_xp):
            w = torch.zeros_like(tgt)
            for j, n in enumerate(dst_xp):
                if n in src_xp:
                    w[:, j] = v[:, src_xp.index(n)]
            new[k] = w
            if src_xp != dst_xp:
                widened.append(k)
            continue
        if k == "proj.0.weight" and v.dim() == 2 and v.shape[1] == d_enc + len(src_xf):
            w = torch.zeros_like(tgt)
            w[:, :d_enc] = v[:, :d_enc]
            for j, n in enumerate(dst_xf):
                if n in src_xf:
                    w[:, d_enc + j] = v[:, d_enc + src_xf.index(n)]
            new[k] = w
            if src_xf != dst_xf:
                widened.append(k)
            continue
        if tgt.shape == v.shape:
            new[k] = v
            continue
        raise ValueError("không nạp được %s: %s -> %s" % (k, tuple(v.shape), tuple(tgt.shape)))
    missing, unexpected = model.load_state_dict(new, strict=False)
    if unexpected:
        raise ValueError("khoá lạ: %s" % unexpected)
    return {"widened": widened, "missing": list(missing),
            "new_xp": [n for n in dst_xp if n not in src_xp], "new_xf": [n for n in dst_xf if n not in src_xf],
            "dropped_xp": [n for n in src_xp if n not in dst_xp], "dropped_xf": [n for n in src_xf if n not in dst_xf]}


def source_names(state, ck=None):
    """Checkpoint input names from its saved layout; infer legacy layouts from column counts:
    9/7 = stage 2, 11/21 = stage 3 without steps.
    """
    from .feats import NAP2_XF, NAP2_XP, XF_NAMES, XP_NAMES
    lay = (ck or {}).get("layout")
    if lay and lay.get("xp_names"):
        return list(lay["xp_names"]), list(lay["xf_names"])
    n = state["encoder.phi.0.weight"].shape[1]
    if n == len(NAP2_XP):
        return list(NAP2_XP), list(NAP2_XF)
    if n == len(XP_NAMES):
        return list(XP_NAMES), list(XF_NAMES)
    raise ValueError("checkpoint có %d cột xp nhưng không lưu layout" % n)


def init_from_nap2(model, state):
    """Load stage-2 state_dict into Nap3Net with new columns zeroed; return widened / missing (switch.*)."""
    from .feats import NAP2_XF, NAP2_XP, XF_NAMES, XP_NAMES
    info = init_by_names(model, state, NAP2_XP, NAP2_XF, XP_NAMES[:model.cfg["n_player_feat"]],
                         XF_NAMES[:model.cfg["n_frame_feat"]])
    return {"widened": info["widened"], "missing": info["missing"]}
