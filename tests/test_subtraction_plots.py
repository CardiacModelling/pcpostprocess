#!/usr/bin/env python3
import os
import unittest

import matplotlib.pyplot as plt
from syncropatch_export.trace import Trace

from pcpostprocess.detect_ramp_bounds import detect_ramp_bounds
from pcpostprocess.subtraction_plots import do_subtraction_plot


class TestSubtractionPlots(unittest.TestCase):
    def setUp(self):
        test_data_dir_before = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                            "staircaseramp (2)_2kHz_15.01.07")

        test_data_dir_after = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                           "staircaseramp (2)_2kHz_15.11.33")

        json_file_before = "staircaseramp (2)_2kHz_15.01.07.json"
        json_file_after = "staircaseramp (2)_2kHz_15.11.33.json"

        self.output_dir = os.path.join('test_output', 'test_trace_class')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Use identical traces for purpose of the test
        self.before_trace = Trace(test_data_dir_before, json_file_before)
        self.after_trace = Trace(test_data_dir_after, json_file_after)

    def test_do_subtraction_plot(self):
        fig = plt.figure(figsize=(16, 18), layout='constrained')
        times = self.before_trace.get_times()

        well = 'A01'
        before_current = self.before_trace.get_trace_sweeps()[well]
        after_current = self.after_trace.get_trace_sweeps()[well]

        voltage_protocol = self.before_trace.get_voltage_protocol()

        ramp_bounds = detect_ramp_bounds(times,
                                         voltage_protocol.get_all_sections())

        sweeps = [0, 1]
        voltages = self.before_trace.get_voltage()
        do_subtraction_plot(fig, times, sweeps, before_current, after_current,
                            voltages, ramp_bounds, well=well)

        fig.savefig(os.path.join(self.output_dir, f"subtraction_plot_{well}"))


if __name__ == "__main__":
    pass
