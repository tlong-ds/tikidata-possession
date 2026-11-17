"""Convert EXTERNAL public matches to the pipeline cache format (see pipeline/schema.py)."""


def ensure_external(names, log=print, workers=1):
    """Ensure cache/external/<name>.npz exists for external matches (Metrica downloads automatically;
    PFF / SkillCorner require local raw data or an existing cache, e.g. from a private Kaggle dataset).
    Returns {name: path}."""
    from .registry import source_of
    groups = {}
    for n in names:
        groups.setdefault(source_of(n), []).append(n)
    out = {}
    for src, ns in groups.items():
        if src == "metrica":
            from .metrica import ensure_cache
            out.update(ensure_cache(ns, log=log))
        elif src == "pff":
            from .pff import ensure_cache
            out.update(ensure_cache(ns, workers=workers, log=log))
        else:
            from .skillcorner import ensure_cache
            out.update(ensure_cache(ns, workers=workers, log=log))
    return out


def expand(names):
    """Expand source aliases in TOML: "PFF", "SKILLCORNER", "METRICA" -> every match from that source."""
    from . import registry
    groups = {"METRICA": registry.METRICA, "PFF": registry.PFF, "SKILLCORNER": registry.SKILLCORNER}
    out = []
    for n in names:
        for m in groups.get(n, [n]):
            if m not in out:
                out.append(m)
    return out
