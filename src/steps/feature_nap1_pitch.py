"""feature.nap1_pitch: stage-1 pitch control at the ball, P(group 1 control) - 0.5.
Negate on team swap (team_negate_xf).
"""

import numpy as np

from . import FeatureStep, register


@register
class Nap1Pitch(FeatureStep):
    name = "nap1_pitch"
    xf_names = ("p_home_minus_half",)
    team_negate_xf = ("p_home_minus_half",)

    def compute(self, V, Bhat, raw):
        from src.nap1.pitch_control import pitch_control
        from src.nap2 import shared
        half = np.asarray(V.meta["half"])
        p1 = np.nan_to_num(pitch_control(V, Bhat, shared.nap1_params()), nan=0.5)
        return {"xf": np.where(half > 0, p1 - 0.5, 0.0)[:, None]}
