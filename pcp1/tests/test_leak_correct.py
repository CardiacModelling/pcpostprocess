#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest

from syncropatch_export.trace import Trace

from pcpostprocess import leak_correct
from pcpostprocess.detect_ramp_bounds import detect_ramp_bounds


store_output = False


class TestLeakCorrect(unittest.TestCase):
    """
    Test the leak correction methods
    """

    @classmethod
    def setUpClass(self):
        if store_output:  # pragma: no cover
            self.temp_dir = None
            self.plot_dir = os.path.join('test_output', 'leak_correct')
            os.makedirs(self.plot_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.plot_dir = self.temp_dir.name

        test_data_dir = os.path.join(
            'test_data', '13112023_MW2_FF', 'staircaseramp (2)_2kHz_15.01.07')
        json_file = 'staircaseramp (2)_2kHz_15.01.07.json'

        self.trace = Trace(test_data_dir, json_file)

        # get currents and QC from trace object
        self.currents = self.trace.get_all_traces(leakcorrect=False)
        self.currents['times'] = self.trace.get_times()
        self.currents['voltages'] = self.trace.get_voltage()

        self.QC = self.trace.get_onboard_QC_values()

        # Find first times ahead of these times
        voltage_protocol = self.trace.get_voltage_protocol().get_all_sections()
        times = self.currents['times'].flatten()
        self.ramp_bound_indices = detect_ramp_bounds(times, voltage_protocol, ramp_index=0)

    @classmethod
    def tearDownClass(self):
        if self.temp_dir:
            self.temp_dir.cleanup()

    def test_fit_linear_leak(self):
        # Test fit_linear_leak, and plotting
        well, sweep = 'A01', 0

        current = self.trace.get_trace_sweeps(sweeps=[sweep])[well][0]
        voltage = self.trace.get_voltage()
        time = self.trace.get_times()
        fname = os.path.join(self.plot_dir, f'{well}-{sweep}.png')

        leak_correct.fit_linear_leak(
            current, voltage, time, *self.ramp_bound_indices, save_fname=fname)

    def test_get_leak_correct(self):
        # Test get_leak_corrected
        well, sweep = 'A01', 0

        trace = self.trace
        current = self.currents[well][sweep]
        voltage = trace.get_voltage()
        time = trace.get_times()

        x = leak_correct.get_leak_corrected(
            current, voltage, time, *self.ramp_bound_indices)
        self.assertEqual(x.shape, (30784,))


if __name__ == "__main__":  # pragma: no cover
    if '-store' in sys.argv:
        sys.argv.remove('-store')
        store_output = True
    else:
        print('Add -store to store output')

    unittest.main()

