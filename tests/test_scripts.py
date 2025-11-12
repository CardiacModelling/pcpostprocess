#!/usr/bin/env python3
import os
import tempfile
import unittest

from pcpostprocess.scripts.run_herg_qc import run as run_herg_qc
from pcpostprocess.scripts.summarise_herg_export import run as run_summarise

store_output = False


class TestScripts(unittest.TestCase):
    """
    Tests the scripts bundled with pcpostprocess.
    """

    def test_run_herg_qc_and_summarise_herg_export(self):
        # Test run_herg_qc_, then summarise_herg_export

        data = os.path.join('tests', 'test_data', '13112023_MW2_FF')
        with tempfile.TemporaryDirectory() as d:
            if store_output:
                d = 'test_output'
            d1 = os.path.join(d, 'run_herg_qc')
            d2 = os.path.join(d, 'summarise_herg_export')

            # Test run herg qc
            erev = -90.71
            qc_map = {'staircaseramp (2)_2kHz': 'staircaseramp'}
            write_map = {'staircaseramp2': 'staircaseramp2'}
            run_herg_qc(
                data, d1, qc_map, ('A03', 'A20', 'D16'),
                write_traces=True, write_map=write_map,
                save_id='13112023_MW2', reversal_potential=erev)

            # Test summarise herg export
            run_summarise(d1, d2, '13112023_MW2', reversal_potential=erev)


if __name__ == '__main__':
    store_output=True
    unittest.main()

