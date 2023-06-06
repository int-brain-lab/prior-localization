import unittest
from pathlib import Path
import numpy as np

from one.api import ONE
from prior_localization.prepare_data import prepare_ephys


class TestEphysInput(unittest.TestCase):

    def setUp(self) -> None:
        self.one = ONE()
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        _, self.probe_names = self.one.eid2pid(self.eid)
        self.qc = 1
        self.subject = self.one.eid2ref(self.eid)['subject']
        self.fixtures_dir = Path(__file__).parent.joinpath('fixtures')
