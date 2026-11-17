"""pipeline.external.metrica: read Metrica formats, derive event-based H/A/D labels and
produce the DFL cache contract. Uses small synthetic data only, never downloaded files.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from pipeline.external import metrica as mt
from pipeline.quality import restart_frames
from pipeline.schema import load_match_file, save_match, validate_match
from pipeline.view import resolve_geometry

H, A, D = 0, 1, 2


def table(rows):
    """rows (team, type, subtype, period, start, end) -> read_events_*-compatible event table."""
    return mt._event_table([{"team": {"H": 1, "A": 2}.get(t, 0), "type": ty, "subtype": st, "period": p,
                             "start_frame": s, "end_frame": e, "start_x": .5, "start_y": .5,
                             "end_x": .5, "end_y": .5} for t, ty, st, p, s, e in rows])


ROWS = [
    ("H", "SET PIECE", "KICK OFF", 1, 10, 10),
    ("H", "PASS", "", 1, 10, 20),
    ("H", "BALL LOST", "INTERCEPTION", 1, 30, 40),
    ("A", "CHALLENGE", "AERIAL-WON", 1, 35, 35),       # Challenge: no ownership change.
    ("A", "RECOVERY", "INTERCEPTION", 1, 40, 40),
    ("A", "BALL OUT", "", 1, 50, 60),                  # Dead from the end frame, when the ball crosses the line.
    ("H", "SET PIECE", "THROW IN", 1, 100, 100),
    ("H", "PASS", "", 1, 100, 110),
    ("H", "CHALLENGE", "GROUND-FAULT-WON", 1, 130, 130),
    ("H", "FAULT RECEIVED", "", 1, 130, 130),          # Dead from the start frame.
    ("H", "SET PIECE", "FREE KICK", 1, 160, 160),
    ("H", "PASS", "", 1, 160, 170),
    ("A", "SET PIECE", "THROW IN", 1, 220, 220),       # Restart without a preceding stoppage event: dead from 170.
    ("A", "SHOT", "OFF TARGET-OUT", 1, 230, 240),
    ("H", "RECOVERY", "", 1, 300, 300),                # Keeper picks up a dead ball; do not restart play.
    ("H", "PASS", "GOAL KICK", 1, 310, 330),           # Goal kick = restart.
    ("H", "BALL LOST", "END HALF", 1, 340, 340),
    # Half 2 lacks KICK OFF; the first PASS opens the half.
    ("A", "PASS", "", 2, 410, 420),
    ("A", "PASS", "OFFSIDE", 2, 420, 450),              # Offside: dead from the end frame.
    ("H", "PASS", "", 2, 480, 490),                     # Missing SET PIECE: restart play.
]


def period_of(T=600, split=400):
    p = np.ones(T, np.int8)
    p[split:] = 2
    return p


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.y, self.half, self.info = mt.event_labels(table(ROWS), period_of())

    def test_tokens(self):
        self.assertEqual(mt._sub_tokens("HEAD-ON TARGET-GOAL"), ("HEAD", "ON TARGET", "GOAL"))
        self.assertEqual(mt._sub_tokens(float("nan")), ())
        r, stop, poss, _ = mt._classify("PASS", {"GOAL KICK"})
        self.assertTrue(r and poss and stop is None)
        self.assertEqual(mt._classify("SHOT", {"ON TARGET", "GOAL"})[1], "end")
        self.assertIsNone(mt._classify("SHOT", {"ON TARGET", "SAVED"})[1])

    def test_half_bounds(self):
        h = self.half
        self.assertTrue((h[:10] == 0).all() and h[10] == 1)
        self.assertTrue((h[10:340 + 50] == 1).all() and (h[390:410] == 0).all())
        self.assertTrue((self.y[h == 0] == -1).all() and (self.y[h > 0] >= 0).all())
        self.assertEqual(h[410], 2)

    def test_first_half_states(self):
        y = self.y
        expect = [(10, H), (39, H), (40, A), (59, A), (60, D), (99, D), (100, H), (129, H), (130, D),
                  (159, D), (160, H), (169, H), (170, D), (219, D), (220, A), (239, A), (240, D),
                  (300, D), (309, D), (310, H), (339, H), (340, D), (389, D)]
        for f, s in expect:
            self.assertEqual(int(y[f]), s, "khung %d" % f)

    def test_second_half_states(self):
        y = self.y
        # After the final ball frame 490 (pass end), D until half end at 490 +2 s.
        for f, s in [(410, A), (449, A), (450, D), (479, D), (480, H), (489, H), (490, D), (539, D)]:
            self.assertEqual(int(y[f]), s, "khung %d" % f)
        self.assertTrue((self.half[410:540] == 2).all() and (self.half[540:] == 0).all())

    def test_counts(self):
        i = self.info
        self.assertEqual(i["implicit_stops"], 1)
        self.assertEqual(i["implicit_restarts"], 1)
        self.assertEqual(i["restarts"], 5)
        self.assertEqual(i["switches"], 1)


class TestTouchAndOrientation(unittest.TestCase):
    def test_touch(self):
        ev = table([("H", "PASS", "", 1, 5, 9), ("A", "CHALLENGE", "GROUND-WON", 1, 5, 5),
                    ("H", "CHALLENGE", "GROUND-LOST", 1, 7, 7), ("A", "CARRY", "", 1, 8, 9),
                    ("A", "BALL OUT", "", 1, 9, 9)])
        tt = mt.touch_labels(ev, 12)
        self.assertEqual(int(tt[5]), 3)
        self.assertTrue((np.delete(tt, 5) == 0).all())

    def test_left_group(self):
        T = 60
        P = np.full((T, 46, 3), np.nan, np.float32)
        half = np.r_[np.ones(30, np.int8), 2 * np.ones(30, np.int8)]
        P[:30, :11, 0], P[:30, 24:35, 0] = .7, .3           # Half 1: away on the left.
        P[30:, :11, 0], P[30:, 24:35, 0] = .3, .7
        self.assertEqual(mt.left_group(P, half), {"firstHalf": 2, "secondHalf": 1})


def write_csv_game(d, game, T=600):
    """Tiny Metrica CSV match: two players per team + ball; away starts on the left in half 1."""
    os.makedirs(os.path.join(d, game))
    rng = np.random.default_rng(0)
    period = period_of(T)
    for team, x0 in (("Home", .7), ("Away", .3)):
        n = 2
        lines = [",,," + ",".join(x for _ in range(n) for x in (team, "")) + ",,",
                 ",,," + ",".join(x for k in range(n) for x in ("%d" % (k + 1), "")) + ",,",
                 "Period,Frame,Time [s]," + ",".join(x for k in range(n) for x in ("Player%d" % (k + 1), ""))
                 + ",Ball,"]
        for f in range(T):
            side = x0 if period[f] == 1 else 1 - x0
            xs = [side + rng.normal(0, .01) for _ in range(n)]
            vals = []
            for x in xs:
                vals += ["%.5f" % x, "%.5f" % rng.uniform(.2, .8)]
            ball = ["NaN", "NaN"] if f % 7 == 0 else ["%.5f" % rng.uniform(0, 1)] * 2
            lines.append("%d,%d,%.2f," % (period[f], f + 1, (f + 1) / 25) + ",".join(vals + ball))
        with open(os.path.join(d, game, "%s_RawTrackingData_%s_Team.csv" % (game, team)), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    hdr = "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y"
    out = [hdr]
    for t, ty, st, p, s, e in ROWS:
        out.append("%s,%s,%s,%d,%d,0,%d,0,P1,,0.5,0.5,NaN,NaN" % (
            {"H": "Home", "A": "Away"}[t], ty, st, p, s + 1, e + 1))
    with open(os.path.join(d, game, game + "_RawEventsData.csv"), "w") as fh:
        fh.write("\n".join(out) + "\n")


def write_epts_game(d, T=600):
    """Tiny Sample_Game_3: metadata.xml (two players/team, one home substitution), tracking.txt, events.json."""
    g = os.path.join(d, "Sample_Game_3")
    os.makedirs(g)
    players = [("P1", "TA"), ("P2", "TA"), ("P3", "TA"), ("P4", "TB"), ("P5", "TB")]
    pl_xml = "".join('<Player id="%s" teamId="%s"><Name>%s</Name></Player>' % (p, t, p) for p, t in players)
    ch_xml = "".join('<PlayerChannel id="c%s" playerId="%s"/>' % (p, p) for p, _ in players)

    def spec(s, e, ids):
        return ('<DataFormatSpecification endFrame="%d" separator=":" startFrame="%d">' % (e, s)
                + '<StringRegister name="frameCount"/><SplitRegister separator=";">'
                + "".join('<SplitRegister separator=","><PlayerChannelRef playerChannelId="c%s"/>'
                          '<PlayerChannelRef playerChannelId="c%s_y"/></SplitRegister>' % (p, p) for p in ids)
                + '</SplitRegister><SplitRegister separator=","><BallChannelRef channelId="x"/></SplitRegister>'
                "</DataFormatSpecification>")

    params = "".join("<ProviderParameter><Name>%s</Name><Value>%d</Value></ProviderParameter>" % kv for kv in
                     (("first_half_start", 1), ("first_half_end", 400), ("second_half_start", 401),
                      ("second_half_end", T)))
    xml = ("<main><Metadata><GlobalConfig><ProviderGlobalParameters>%s</ProviderGlobalParameters></GlobalConfig>"
           '<Sessions><Session><MatchParameters><Score idLocalTeam="TA" idVisitingTeam="TB"/></MatchParameters>'
           "</Session></Sessions><Players>%s</Players></Metadata><DataFormatSpecifications>%s%s"
           "</DataFormatSpecifications><Devices><Device><Sensors><Sensor><Channels>%s</Channels></Sensor>"
           "</Sensors></Device></Devices></main>") % (
        params, pl_xml, spec(1, 450, ["P1", "P2", "P4", "P5"]), spec(451, T, ["P1", "P3", "P4", "P5"]), ch_xml)
    with open(os.path.join(g, "Sample_Game_3_metadata.xml"), "w") as fh:
        fh.write(xml)
    lines = []
    for f in range(T):
        side = .3 if f < 400 else .7                        # Home on the left in half 1.
        xy = [(side, .4), (side, .6), (1 - side, .4), (1 - side, .6)]
        lines.append("%d:%s:%s" % (f + 1, ";".join("%.5f,%.5f" % p for p in xy), "0.5,0.5" if f % 5 else "NaN,NaN"))
    with open(os.path.join(g, "Sample_Game_3_tracking.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    data = []
    for t, ty, st, p, s, e in ROWS:
        sub = [{"name": x} for x in st.split("-") if x]
        data.append({"team": {"id": {"H": "TA", "A": "TB"}[t]}, "type": {"name": ty},
                     "subtypes": sub[0] if len(sub) == 1 else (sub or None), "period": p,
                     "start": {"frame": s + 1, "x": .5, "y": .5}, "end": {"frame": e + 1, "x": None, "y": None}})
    with open(os.path.join(g, "Sample_Game_3_events.json"), "w") as fh:
        json.dump({"data": data, "metadata": {}}, fh)


class TestReaders(unittest.TestCase):
    def check_match(self, m, left):
        self.assertEqual(validate_match(m), [])
        self.assertEqual(m.source, "dfl")
        self.assertEqual(m.meta["provider"], "metrica")
        self.assertEqual(m.meta["left_group"], left)
        y_ref, half_ref, _ = mt.event_labels(table(ROWS), period_of())
        np.testing.assert_array_equal(m.extras["y_true"], y_ref)
        np.testing.assert_array_equal(m.half, half_ref)
        self.assertTrue(np.isnan(m.P[..., 2]).all() and np.isnan(m.B[:, 2]).all())
        g = resolve_geometry(m)
        want = {"1": 1 if left["firstHalf"] == 1 else -1, "2": 1 if left["secondHalf"] == 1 else -1}
        self.assertEqual(g["attack_sign"], want)
        rs = restart_frames(m)
        self.assertEqual(sorted(rs.tolist()), [10, 100, 160, 220, 310])

    def test_csv_game_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            write_csv_game(d, "Sample_Game_1")
            m = mt.load_game(d, "Sample_Game_1")
            self.assertEqual(m.name, "METRICA-SG1")
            self.assertTrue(np.isfinite(m.P[:, [0, 1, 24, 25], 0]).all())
            self.assertTrue(np.isnan(m.P[:, 2:24, 0]).all())
            self.check_match(m, {"firstHalf": 2, "secondHalf": 1})
            path = save_match(m, os.path.join(d, "out", m.name + ".npz"))
            back = load_match_file(path)
            np.testing.assert_array_equal(back.extras["y_true"], m.extras["y_true"])
            np.testing.assert_array_equal(back.extras["touch_true"], m.extras["touch_true"])
            self.assertEqual(back.meta["left_group"], m.meta["left_group"])

    def test_epts_game(self):
        with tempfile.TemporaryDirectory() as d:
            write_epts_game(d)
            m = mt.load_game(d, "Sample_Game_3")
            self.assertEqual(m.name, "METRICA-SG3")
            self.check_match(m, {"firstHalf": 1, "secondHalf": 2})
            # P2 leaves at frame 450 and P3 enters; first-fit reuses slot 1.
            self.assertTrue(np.isfinite(m.P[:450, 1, 0]).all() and np.isfinite(m.P[450:, 1, 0]).all())
            self.assertTrue(np.isnan(m.P[:, 2, 0]).all())
            self.assertTrue(np.isfinite(m.P[:, 24:26, 0]).all())
            self.assertEqual(m.meta["ball_processing"]["kind"], "dfl_like_ball")
            self.assertTrue(np.isfinite(m.B[m.half > 0, :2]).all())     # Default DFL-style ball is fully visible.
            raw = mt.load_game(d, "Sample_Game_3", ball="raw")
            self.assertEqual(raw.meta["ball_processing"], {"kind": "raw"})
            self.assertTrue(np.isnan(raw.B[::5, 0]).all())                 # Keep the reader's original ball unchanged.
            np.testing.assert_array_equal(raw.extras["y_true"], m.extras["y_true"])


class TestDflLikeBall(unittest.TestCase):
    """dfl_like_ball: hold the previous live position during dead ball, interpolate live gaps, round, add deterministic noise."""

    def setUp(self):
        T = 200
        self.half = np.r_[np.zeros(5), np.ones(95), np.full(95, 2), np.zeros(5)].astype(np.int8)
        y = np.full(T, H)
        y[self.half == 0] = -1
        y[40:60] = D                       # Dead-ball interval within half 1.
        y[100:110] = D                     # Half 2 starts dead, without a preceding live frame.
        y[150:170] = A
        self.y = y
        t = np.arange(T)
        B = np.full((T, 3), np.nan, np.float32)
        B[:, 0] = 0.2 + 0.003 * t
        B[:, 1] = 0.5 + 0.001 * t
        B[[20, 21, 22, 130], :2] = np.nan  # Missing ball during live play.
        B[40:60, :2] = np.nan              # Missing ball during dead ball, as in Metrica.
        B[105, :2] = [0.9, 0.9]            # Visible ball within a dead interval must still be held stationary.
        self.B = B

    def run_(self, **kw):
        kw = dict({"sig": 0.0, "noise_m": 0.0, "noise_s": 0.0, "quant": 0.0}, **kw)
        return mt.dfl_like_ball(self.B, self.half, self.y, **kw)

    def test_dead_hold_live_interp(self):
        out = self.run_()
        inh = self.half > 0
        self.assertTrue(np.isfinite(out[inh, :2]).all())
        np.testing.assert_allclose(out[40:60, :2], np.repeat(out[39:40, :2], 20, 0))
        np.testing.assert_allclose(out[100:110, :2], np.repeat(out[110:111, :2], 10, 0))
        t = np.arange(len(self.B))
        np.testing.assert_allclose(out[[20, 21, 22, 130], 0], 0.2 + 0.003 * t[[20, 21, 22, 130]], atol=1e-6)
        np.testing.assert_allclose(out[60:100, :2], self.B[60:100, :2])     # Visible live frames remain unchanged.
        np.testing.assert_array_equal(out[~inh, :2], self.B[~inh, :2])     # Outside halves: unchanged.
        self.assertTrue(np.isnan(out[:, 2]).all())                       # Leave the speed column untouched.

    def test_quant_and_noise(self):
        out = self.run_(quant=0.01)
        xy = out[self.half > 0, :2].astype(np.float64) * np.array(mt.PITCH)
        self.assertLess(np.abs(xy * 100 - np.round(xy * 100)).max(), 1e-3)
        a = self.run_(noise_m=0.03, noise_s=0.04, seed=3)
        b = self.run_(noise_m=0.03, noise_s=0.04, seed=3)
        c = self.run_(noise_m=0.03, noise_s=0.04, seed=4)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        np.testing.assert_allclose(a[40:60, :2], np.repeat(a[39:40, :2], 20, 0))  # Noise must not disrupt stationary dead-ball positions.
        d = (a[60:100, :2] - self.run_()[60:100, :2]) * np.array(mt.PITCH)
        self.assertLess(np.abs(d).max(), 0.2)

    def test_bad_ball_mode(self):
        with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as d:
            write_epts_game(d)
            mt.load_game(d, "Sample_Game_3", ball="smooth")


class TestEnsureCache(unittest.TestCase):
    """ensure_cache: verify raw SHA-256 before conversion; write cache/external, reuse next time, reconvert on version changes."""

    def test_convert_skip_and_sha(self):
        import hashlib
        from pipeline.split import EXTERNAL_TRAIN
        self.assertTrue(set(mt.GAMES.values()) <= set(EXTERNAL_TRAIN))
        with tempfile.TemporaryDirectory() as d:
            raw, cache = os.path.join(d, "data"), os.path.join(d, "cache")
            os.makedirs(raw)
            write_epts_game(raw)
            exp = {}
            for f in sorted(os.listdir(os.path.join(raw, "Sample_Game_3"))):
                rel = "Sample_Game_3/" + f
                exp[rel] = hashlib.sha256(open(os.path.join(raw, rel), "rb").read()).hexdigest()
            self.assertEqual(mt.verify_raw(raw, exp), [])
            logs = []
            with mock_env(METRICA_RAW=raw):
                out = mt.ensure_cache(["METRICA-SG3"], cache_dir=cache, fetch=False, expected=exp, log=logs.append)
            p = out["METRICA-SG3"]
            self.assertEqual(p, os.path.join(cache, "external", "METRICA-SG3.npz"))
            m = load_match_file(p)
            self.assertEqual(m.meta["converter_version"], mt.CONVERTER_VERSION)
            self.assertEqual(m.meta["ball_processing"]["kind"], "dfl_like_ball")
            self.assertEqual(len(logs), 1)
            mt.ensure_cache(["METRICA-SG3"], cache_dir=cache, raw_dir="/không/có", log=logs.append)
            self.assertEqual(len(logs), 1)                          # Existing cache with matching version: skip reconversion.
            bad = dict(exp, **{next(iter(exp)): "0" * 64})
            self.assertTrue(mt.verify_raw(raw, bad))
            os.remove(p)
            with mock_env(METRICA_RAW=raw), self.assertRaises(FileNotFoundError):
                mt.ensure_cache(["METRICA-SG3"], cache_dir=cache, fetch=False, expected=bad)
            with self.assertRaises(ValueError):
                mt.ensure_cache(["DFL-MAT-J03WN1"], cache_dir=cache, fetch=False)


class mock_env:
    def __init__(self, **kw):
        self.kw, self.old = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
