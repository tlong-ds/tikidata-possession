"""feature.nap1_dead: stage-1 dead-ball probability (src.nap1.deadball.dead_prob) as one xf column.
Team- and pitch-direction-independent; no column-pair declarations needed.
"""

import numpy as np

from . import FeatureStep, register


@register
class Nap1Dead(FeatureStep):
    name = "nap1_dead"
    xf_names = ("p_dead",)

    def compute(self, V, Bhat, raw):
        from src.nap1.deadball import dead_prob
        from src.nap2 import shared
        half = np.asarray(V.meta["half"])
        d = np.nan_to_num(dead_prob(V, Bhat, shared.nap1_params()), nan=0.5)
        return {"xf": np.where(half > 0, d, 0.0)[:, None]}
