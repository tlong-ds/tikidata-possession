"""decode.wide expands the stage-4 tuning grid beyond the baseline's selected boundaries
(default grid: minimum scale 0.05, maximum window 3, minimum gate_scale 0.5, lambda_emit 1.0,
floor 1e-6). Only the grid changes, not E/S/C/K; inference only needs the new decode_switch.json.
"""

from . import DecodeStep, register

INT_KEYS = ("window",)


@register
class DecodeWide(DecodeStep):
    name = "wide"
    defaults = {"scale": [0.01, 0.02, 0.05], "lambda_emit": [0.3, 0.5, 1.0], "window": [3, 8, 15],
                "gate_scale": [0.1, 0.25, 0.5], "floor": [1e-10, 1e-8, 1e-6]}

    def grid(self, grid):
        g = dict(grid)
        for k, v in self.params.items():
            vals = list(v) if isinstance(v, (list, tuple)) else [v]
            if vals:
                g[k] = [int(x) if k in INT_KEYS else float(x) for x in vals]
        return g
