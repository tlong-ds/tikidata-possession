import tempfile
import unittest
from pathlib import Path
from pipeline.release_views import provider_overrides

class ProviderOrientationTests(unittest.TestCase):
    def xml(self,a='left_to_right',b='right_to_left',bad_slot=False):
        teams=''.join(f'<Team id="{i}"><ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half</Name><Value>{d}</Value></ProviderParameter></ProviderTeamsParameters></Team>' for i,d in [('a',a),('b',b)])
        players=''.join(f'<Player teamId="{("b" if bad_slot and i==0 else "a" if i<24 else "b")}"/>' for i in range(46))
        return '<main><Teams>'+teams+'</Teams><Players>'+players+'</Players></main>'
    def parse(self,xml):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'metadata.xml';p.write_text(xml);return provider_overrides(p)
    def test_both_declared_directions_and_halftime_reversal(self):
        for a,b,sign in [('left_to_right','right_to_left',1),('right_to_left','left_to_right',-1)]:
            o=self.parse(self.xml(a,b));self.assertEqual(o['1']['attack_sign'],sign);self.assertEqual(o['2']['attack_sign'],-sign)
            self.assertEqual(len(o['1']['metadata_sha256']),64)
    def test_conflict_missing_direction_and_wrong_slots_rejected(self):
        for xml in (self.xml(b='left_to_right'),self.xml(a=''),self.xml(bad_slot=True)):
            with self.assertRaises(ValueError):self.parse(xml)
