"""Freeze stage-3 blocks when fine-tuning an existing checkpoint.

Nap3Net blocks: encoder, proj, temporal, had, touch, team, ball, switch. Freeze entries are
parameter-name prefixes (e.g. "encoder", "encoder.phi.2", "temporal") or named groups:
    backbone = encoder, proj, temporal        heads = had, touch, team, ball, switch
freeze_old_inputs = true trains ONLY NEW feature columns in encoder.phi.0 and proj.0; old columns,
encoder-to-proj weights and biases are gradient-masked, even if their block is frozen.
Frozen parameters are excluded from AdamW; masked layers use weight_decay = 0 to prevent drift.
"""

import torch

BLOCKS = ("encoder", "proj", "temporal", "had", "touch", "team", "ball", "switch")
PRESETS = {"backbone": ("encoder", "proj", "temporal"), "heads": ("had", "touch", "team", "ball", "switch"),
           "all": BLOCKS}
INPUT_LAYERS = ("encoder.phi.0.", "proj.0.")


def expand(names, model=None):
    out = []
    for n in names or []:
        out += list(PRESETS.get(n, (n,)))
    if model is not None:
        pnames = [k for k, _ in model.named_parameters()]
        bad = [f for f in out if not any(k == f or k.startswith(f + ".") for k in pnames)]
        if bad:
            raise ValueError("freeze: không có khối %s (khối: %s, nhóm: %s)" % (bad, ", ".join(BLOCKS),
                                                                                 ", ".join(PRESETS)))
    return list(dict.fromkeys(out))


def _input_mask(name, p, new_xp_idx, new_xf_idx, d_enc):
    """1 on new columns and 0 elsewhere, for input-layer weights only."""
    m = torch.zeros_like(p)
    if name == "encoder.phi.0.weight":
        for j in new_xp_idx:
            m[:, j] = 1.0
    elif name == "proj.0.weight":
        for j in new_xf_idx:
            m[:, d_enc + j] = 1.0
    return m


def apply(model, freeze=(), freeze_old_inputs=False, new_xp_idx=(), new_xf_idx=(), lr=1e-3, wd=1e-2):
    """Set requires_grad and hooks from the configuration; return (AdamW param_groups, info)."""
    for h in getattr(model, "_freeze_hooks", []):
        h.remove()
    model._freeze_hooks = []
    frozen = expand(freeze, model)
    d_enc = model.encoder.d_out
    decay, masked = [], []
    n_train = n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if freeze_old_inputs and name.startswith(INPUT_LAYERS):
            if name.endswith(".bias"):
                p.requires_grad_(False)
                continue
            m = _input_mask(name, p, new_xp_idx, new_xf_idx, d_enc)
            if not bool(m.any()):
                p.requires_grad_(False)
                continue
            p.requires_grad_(True)
            model._freeze_hooks.append(p.register_hook(lambda g, m=m: g * m))
            masked.append(p)
            n_train += int(m.sum().item())
            continue
        on = not any(name == f or name.startswith(f + ".") for f in frozen)
        p.requires_grad_(on)
        if on:
            decay.append(p)
            n_train += p.numel()
    groups = [{"params": decay, "lr": lr, "weight_decay": wd}]
    if masked:
        groups.append({"params": masked, "lr": lr, "weight_decay": 0.0})
    info = {"freeze": frozen, "freeze_old_inputs": bool(freeze_old_inputs), "trainable": n_train, "total": n_total}
    return [g for g in groups if g["params"]], info
