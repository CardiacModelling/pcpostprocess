#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest

import matplotlib.pyplot as plt
from syncropatch_export.trace import Trace

from pcpostprocess.detect_ramp_bounds import detect_ramp_bounds
from pcpostprocess.subtraction_plots import do_subtraction_plot


store_output = False


class TestSubtractionPlots(unittest.TestCase):
    """
    Tests the subtraction_plots module.
    """

    @classmethod
    def setUpClass(self):
        if store_output:  # pragma: no cover
            self.temp_dir = None
            self.plot_dir = os.path.join('test_output', 'subtraction_plots')
            os.makedirs(self.plot_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.plot_dir = self.temp_dir.name

        # TODO: Only one test, why are we doing this?
        dir_before = os.path.join(
            'test_data', '13112023_MW2_FF', 'staircaseramp (2)_2kHz_15.01.07')
        dir_after = os.path.join(
            'test_data', '13112023_MW2_FF', 'staircaseramp (2)_2kHz_15.11.33')

        json_file_before = 'staircaseramp (2)_2kHz_15.01.07.json'
        json_file_after = 'staircaseramp (2)_2kHz_15.11.33.json'

        self.before_trace = Trace(dir_before, json_file_before)
        self.after_trace = Trace(dir_after, json_file_after)

    @classmethod
    def tearDownClass(self):
        if self.temp_dir:
            self.temp_dir.cleanup()

    def test_do_subtraction_plot(self):
        # Tests do_subtraction_plot

        fig = plt.figure(figsize=(16, 18), layout='constrained')
        times = self.before_trace.get_times()

        well = 'A01'
        before_current = self.before_trace.get_trace_sweeps()[well]
        after_current = self.after_trace.get_trace_sweeps()[well]
        voltage_protocol = self.before_trace.get_voltage_protocol()
        ramp_bounds = detect_ramp_bounds(
            times, voltage_protocol.get_all_sections())

        sweeps = [0, 1]
        voltages = self.before_trace.get_voltage()
        do_subtraction_plot(fig, times, sweeps, before_current, after_current,
                            voltages, ramp_bounds, well=well)

        fig.savefig(os.path.join(self.plot_dir, f'{well}.png'))


if __name__ == "__main__":
    if '-store' in sys.argv:
        sys.argv.remove('-store')
        store_output = True
    else:
        print('Add -store to store output')

    unittest.main()

