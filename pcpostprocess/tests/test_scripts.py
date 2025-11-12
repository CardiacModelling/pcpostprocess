#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest

from pcpostprocess.scripts.run_herg_qc import run as run_herg_qc
from pcpostprocess.scripts.summarise_herg_export import run as run_summarise


store_output = False


class TestScripts(unittest.TestCase):
    """
    Tests the scripts bundled with pcpostprocess.
    """

    @classmethod
    def setUpClass(self):
        if store_output:  # pragma: no cover
            self.temp_dir = None
            self.plot_dir = 'test_output'
            os.makedirs(self.plot_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.plot_dir = self.temp_dir.name

    @classmethod
    def tearDownClass(self):
        if self.temp_dir:
            self.temp_dir.cleanup()

    def test_run_herg_qc_and_summarise_herg_export(self):
        # Test run_herg_qc_, then summarise_herg_export

        data = os.path.join('test_data', '13112023_MW2_FF')
        d1 = os.path.join(self.plot_dir, 'run_herg_qc')
        d2 = os.path.join(self.plot_dir, 'summarise_herg_export')

        # Test run herg qc
        erev = -90.71
        qc_map = {'staircaseramp (2)_2kHz': 'staircaseramp'}
        write_map = {'staircaseramp2': 'staircaseramp2'}
        run_herg_qc(
            data, d1, qc_map, ('A03', 'A20', 'D16'),
            write_traces=True, write_map=write_map,
            save_id='13112023_MW2', reversal_potential=erev)

        with open(os.path.join(d1, 'passed_wells.txt'), 'r') as f:
            self.assertEqual(f.read().strip(), 'A03')

        # Test summarise herg export
        run_summarise(d1, d2, '13112023_MW2', reversal_potential=erev)


if __name__ == "__main__":  # pragma: no cover
    if '-store' in sys.argv:
        sys.argv.remove('-store')
        store_output = True
    else:
        print('Add -store to store output')

    unittest.main()

