"""Names of external matches available for training (lightweight imports for pipeline.split).

Each source has a converter in pipeline/external/ writing DFL-compatible cache/external/<name>.npz.
"""

METRICA = ["METRICA-SG1", "METRICA-SG2", "METRICA-SG3"]

# PFF FC FIFA World Cup 2022 (64 matches, broadcast tracking + events). Access through PFF's form.
PFF_IDS = [3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828,
           3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845,
           3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 10502, 10503, 10504,
           10505, 10506, 10507, 10508, 10509, 10510, 10511, 10512, 10513, 10514, 10515, 10516, 10517]
PFF = ["PFF-%d" % i for i in PFF_IDS]

# SkillCorner opendata (20 A-League 2024/25 matches, 10 fps broadcast tracking, MIT).
SKILLCORNER_IDS = [1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691, 1996435, 1996436, 2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461]
SKILLCORNER = ["SKC-%d" % i for i in SKILLCORNER_IDS]

ALL = METRICA + PFF + SKILLCORNER


def source_of(name):
    """Return 'metrica' | 'pff' | 'skillcorner' for an external match name."""
    if name in METRICA:
        return "metrica"
    if name in PFF:
        return "pff"
    if name in SKILLCORNER:
        return "skillcorner"
    raise ValueError("không phải trận ngoài đã đăng ký: %r" % (name,))
