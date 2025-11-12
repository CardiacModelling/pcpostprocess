#!/usr/bin/env python3
import os
import unittest
import sys
import tempfile

from syncropatch_export.trace import Trace

from pcpostprocess import infer_reversal, leak_correct
from pcpostprocess.detect_ramp_bounds import detect_ramp_bounds


store_output = False


class TestInferReversal(unittest.TestCase):
    """
    Tests the `infer_reversal` method.
    """
    @classmethod
    def setUpClass(self):
        if store_output:  # pragma: no cover
            self.temp_dir = None
            self.plot_dir = os.path.join('test_output', 'infer_reversal')
            os.makedirs(self.plot_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.plot_dir = self.temp_dir.name

    @classmethod
    def tearDownClass(self):
        if self.temp_dir:
            self.temp_dir.cleanup()

    def test_infer_reversal(self):
        # Test infer_reversal_potential, including plot

        # Load test data
        data = os.path.join('test_data', '13112023_MW2_FF',
                            'staircaseramp (2)_2kHz_15.01.07')
        trace = Trace(data, 'staircaseramp (2)_2kHz_15.01.07.json')

        # Get times and voltages
        times = trace.get_times()
        voltages = trace.get_voltage()

        # Get protocol and leak ramp indices
        protocol = trace.get_voltage_protocol().get_all_sections()
        leak_indices = detect_ramp_bounds(times, protocol, ramp_index=0)

        # Load current for one well, one sweep, and leak correct
        well, sweep = 'A03', 0
        current = trace.get_trace_sweeps(sweeps=[sweep])[well][0]
        _, leak = leak_correct.fit_linear_leak(
            current, voltages, times, *leak_indices)
        current -= leak

        # Estimate reversal potential
        fpath = os.path.join(self.plot_dir, f'{well}-{sweep}.png')
        erev = infer_reversal.infer_reversal_potential(
            current, times, protocol, voltages, fpath, -89.57184)
        self.assertAlmostEqual(erev, -89.57184, 5)


if __name__ == "__main__":
    if '-store' in sys.argv:
        store_output = True
        sys.argv.remove('-store')
    else:
        print('Add -store to store output')

    unittest.main()
