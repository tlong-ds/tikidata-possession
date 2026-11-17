import numpy as np
import pytest

from src.core import kickoff

XML = """<file><ALL_INSTANCES>
<instance><ID>1</ID><start>4.0</start><end>9.0</end><code>SET PIECES</code>
 <label><group>Type</group><text>Kick Off (Start)</text></label><label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>2</ID><start>30.0</start><end>35.0</end><code>SET PIECES</code>
 <label><group>Type</group><text>Kick Off (After Goal)</text></label><label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>3</ID><start>60.0</start><end>65.0</end><code>Set Pieces</code>
 <label><group>Type</group><text>kick off (start)</text></label><label><group>Half</group><text>2nd Half</text></label></instance>
<instance><ID>4</ID><start>70.0</start><end>75.0</end><code>BUILD UP</code>
 <label><group>Type</group><text>Kick Off (Start)</text></label><label><group>Half</group><text>2nd Half</text></label></instance>
</ALL_INSTANCES></file>"""


def test_windows_only_start_kickoffs(tmp_path):
    p = tmp_path / "st.xml"
    p.write_text(XML)
    assert kickoff.kickoff_windows(str(p), 10) == {1: (40, 90), 2: (600, 650)}


def test_duplicate_half_rejected(tmp_path):
    p = tmp_path / "st.xml"
    p.write_text(XML.replace("Kick Off (After Goal)", "Kick Off (Start)"))
    with pytest.raises(ValueError):
        kickoff.kickoff_windows(str(p), 10)


def _ball(T):
    B = np.full((T, 3), np.nan, np.float32)
    B[:, 2] = 0
    return B


def test_locate_nearest_centre_and_fallback():
    B = _ball(100)
    B[10:20, :2] = [[5 - i * 0.5, 1.0] for i in range(10)]
    B[10:20, 2] = 1
    assert kickoff.locate(B, (5, 30)) == 19          # x=0.5 is nearest the center.
    assert kickoff.locate(B, (40, 60)) == 50         # No ball -> window midpoint.


def test_apply_sets_pre_kickoff_graded_frames_to_D():
    T, fps = 300, 10
    half = np.zeros(T, np.int8)
    half[20:120] = 1
    half[150:290] = 2
    graded = half > 0
    graded[285:290] = False
    codes = np.full(T, 3, np.int8)
    codes[graded] = 0
    B = _ball(T)
    B[21, :] = [0.1, 0.0, 1]                         # Half 1 kicks off immediately (gap <2 s): unchanged.
    B[200, :] = [0.2, 0.1, 1]                        # Half 2 kicks off at 200.
    B[195, :] = [3.0, 0.0, 1]
    out, info = kickoff.apply(codes, half, graded, B, {1: (15, 40), 2: (190, 215)}, fps)
    assert (out[20:120] == 0).all()
    assert (out[150:200] == kickoff.CODE_D).all()
    assert (out[200:285] == 0).all()
    assert (out[:20] == 3).all() and (out[120:150] == 3).all()
    assert info["1"]["changed_to_D"] == 0 and info["2"] == {
        "half_first": 150, "kickoff": 200, "window": [190, 215], "changed_to_D": 50}
    assert (codes[150:200] == 0).all()               # Do not mutate the input array.
