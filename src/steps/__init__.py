"""Stage-3 plug-in improvement steps: one src/steps/<type>_<name>.py per step.

Each improvement adds a step, a test and exps/<name>.toml without editing the core, allowing
parallel work. Enable a step with:

    [[step]]
    use = "feature.nap1_dead"      # <type>.<name>
    weight = 0.5                   # other parameters -> self.params

Types:
    feature: append xf [T,k] frame and/or xp [T,46,k] player inputs
    loss: add a training-loss term
    decode: adjust (E,S,C,K) before stage 4 and/or expand its tuning grid
    blend: export another source's frame probabilities for blending before stage 4

Feature columns follow existing XP_NAMES / XF_NAMES in TOML order, named <step>.<column>.
Checkpoint initialization aligns columns by NAME and zeros new ones, preserving initial outputs.
"""

import importlib
import pkgutil

REGISTRY = {}
KINDS = ("feature", "loss", "decode", "blend")


class Step:
    kind = None
    name = None
    defaults = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError("%s: tham số lạ %s (có: %s)" % (self.key(), sorted(unknown), sorted(self.defaults)))
        self.params = dict(self.defaults, **params)

    @classmethod
    def key(cls):
        return "%s.%s" % (cls.kind, cls.name)

    def spec(self):
        return {"use": self.key(), **self.params}

    def __getattr__(self, k):
        p = self.__dict__.get("params", {})
        if k in p:
            return p[k]
        raise AttributeError(k)


class FeatureStep(Step):
    """compute(V, Bhat, raw) -> {"xf": float [T,len(xf_names)], "xp": float [T,46,len(xp_names)]}.

    Declare columns by name for augmentation transforms:
      team_pairs_xf        swap H/A columns on team swap
      team_complement_xf   x -> 1-x on team swap (home probability)
      team_negate_xf       x -> -x on team swap (attack-direction signed quantity)
      flip_negate_xf/_xp   negate on vertical pitch flip (y-axis quantity)
      reverse_pairs_xf/_xp swap past/future on time reversal; reverse_negate_xp negates
    Return zeros outside halves (half <= 0).
    """
    kind = "feature"
    xf_names = ()
    xp_names = ()
    team_pairs_xf = ()
    team_complement_xf = ()
    team_negate_xf = ()
    flip_negate_xf = ()
    flip_negate_xp = ()
    reverse_pairs_xf = ()
    reverse_pairs_xp = ()
    reverse_negate_xp = ()

    def compute(self, V, Bhat, raw):
        raise NotImplementedError

    def preprocess(self, raw):
        """raw load_file / fresh_train dict -> raw, BEFORE View construction for train/dev/competition.
        Default: unchanged. Preprocessing-only steps declare zero columns and return {} from compute.
        """
        return raw


class LossStep(Step):
    """loss(out, batch) -> scalar tensor, added to the loss with coefficient params['weight']."""
    kind = "loss"
    defaults = {"weight": 1.0}

    def loss(self, out, batch):
        raise NotImplementedError


class DecodeStep(Step):
    """adjust(E,S,C,K,d) -> (E,S,C,K) before stage 4; grid(grid) -> expanded SwitchParams grid."""
    kind = "decode"

    def adjust(self, E, S, C, K, d):
        return E, S, C, K

    def grid(self, grid):
        return grid


class BlendStep(Step):
    """export(files, out_dir, workers) writes src.core.blend frame probabilities per data file.
    grid() returns a weight-grid component for src.core.blend.tune, e.g.
    {"team": {"nap1": [0, 0.1]}, "dead": {...}}.
    """
    kind = "blend"

    def export(self, files, out_dir, workers=1):
        raise NotImplementedError

    def grid(self):
        return {}


def register(cls):
    if cls.kind not in KINDS or not cls.name:
        raise ValueError("step cần kind trong %s và name" % (KINDS,))
    k = cls.key()
    if k in REGISTRY and REGISTRY[k] is not cls:
        raise ValueError("trùng tên step %s" % k)
    REGISTRY[k] = cls
    return cls


_DISCOVERED = False


def discover():
    """Import all src/steps/ modules; each registers its own steps."""
    global _DISCOVERED
    if not _DISCOVERED:
        for m in pkgutil.iter_modules(__path__):
            if not m.name.startswith("_"):
                importlib.import_module("%s.%s" % (__name__, m.name))
        _DISCOVERED = True
    return dict(REGISTRY)


def make(spec):
    """{"use": "<type>.<name>", ...parameters} -> step object."""
    discover()
    spec = dict(spec)
    use = spec.pop("use", None)
    if use not in REGISTRY:
        raise KeyError("không có step %r (có: %s)" % (use, ", ".join(sorted(REGISTRY))))
    return REGISTRY[use](**spec)


def make_all(specs, kind=None):
    steps = [make(s) for s in (specs or [])]
    return [s for s in steps if kind is None or s.kind == kind]


# ---------------------------------------------------------------------------------------------- Layout

class Layout:
    """Input-column names and positions: existing stage-3 columns, then feature-step columns in order."""

    def __init__(self, feature_specs=()):
        from src.core import feats
        self.steps = make_all(feature_specs, "feature")
        bad = [s.spec()["use"] for s in make_all(feature_specs) if s.kind != "feature"]
        if bad:
            raise ValueError("Layout chỉ nhận feature step, không nhận %s" % bad)
        self.xp_names = list(feats.XP_NAMES)
        self.xf_names = list(feats.XF_NAMES)
        self.team_pairs_xf = [tuple(p) for p in feats.TEAM_PAIRS_XF]
        self.reverse_pairs_xf = [tuple(p) for p in feats.REVERSE_PAIRS_XF]
        self.reverse_pairs_xp = [tuple(p) for p in feats.REVERSE_PAIRS_XP]
        self.reverse_negate_xp = list(feats.REVERSE_NEGATE_XP)
        self.team_complement_xf, self.team_negate_xf = [], []
        self.flip_negate_xf, self.flip_negate_xp = [], []
        for s in self.steps:
            xf0, xp0 = len(self.xf_names), len(self.xp_names)
            fi = {n: xf0 + i for i, n in enumerate(s.xf_names)}
            pi = {n: xp0 + i for i, n in enumerate(s.xp_names)}
            self.xf_names += ["%s.%s" % (s.name, n) for n in s.xf_names]
            self.xp_names += ["%s.%s" % (s.name, n) for n in s.xp_names]
            self.team_pairs_xf += [(fi[a], fi[b]) for a, b in s.team_pairs_xf]
            self.team_complement_xf += [fi[a] for a in s.team_complement_xf]
            self.team_negate_xf += [fi[a] for a in s.team_negate_xf]
            self.flip_negate_xf += [fi[a] for a in s.flip_negate_xf]
            self.flip_negate_xp += [pi[a] for a in s.flip_negate_xp]
            self.reverse_pairs_xf += [(fi[a], fi[b]) for a, b in s.reverse_pairs_xf]
            self.reverse_pairs_xp += [(pi[a], pi[b]) for a, b in s.reverse_pairs_xp]
            self.reverse_negate_xp += [pi[a] for a in s.reverse_negate_xp]
        for names in (self.xp_names, self.xf_names):
            if len(set(names)) != len(names):
                raise ValueError("trùng tên cột: %s" % names)

    @property
    def n_xp(self):
        return len(self.xp_names)

    @property
    def n_xf(self):
        return len(self.xf_names)

    def specs(self):
        return [s.spec() for s in self.steps]

    def to_dict(self):
        return {"features": self.specs(), "xp_names": self.xp_names, "xf_names": self.xf_names}

    @classmethod
    def from_dict(cls, d):
        lay = cls((d or {}).get("features", []))
        if d and d.get("xf_names") and d["xf_names"] != lay.xf_names:
            raise ValueError("cột xf của checkpoint khác cột dựng lại từ step hiện tại: %s vs %s"
                             % (d["xf_names"], lay.xf_names))
        return lay

    def preprocess(self, raw):
        """Apply every feature-step preprocess in TOML order."""
        for s in self.steps:
            raw = s.preprocess(raw)
        return raw

    def extra(self, V, Bhat, raw):
        """Feature-step columns: {"xp": [T,46,k], "xf": [T,k]} float32; k may be zero."""
        import numpy as np
        T, S = V.P.shape[:2]
        xp, xf = [np.zeros((T, S, 0), np.float32)], [np.zeros((T, 0), np.float32)]
        for s in self.steps:
            out = s.compute(V, Bhat, raw)
            a = np.asarray(out.get("xf", np.zeros((T, 0))), np.float32).reshape(T, -1)
            b = np.asarray(out.get("xp", np.zeros((T, S, 0))), np.float32).reshape(T, S, -1)
            if a.shape[1] != len(s.xf_names) or b.shape[2] != len(s.xp_names):
                raise ValueError("%s trả %d cột xf / %d cột xp, khai báo %d / %d" % (
                    s.key(), a.shape[1], b.shape[2], len(s.xf_names), len(s.xp_names)))
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError("%s trả NaN/inf" % s.key())
            xf.append(a)
            xp.append(b)
        return {"xp": np.concatenate(xp, -1), "xf": np.concatenate(xf, -1)}
