import os
import unittest

import matplotlib.pyplot as plt

from syncropatch_export.trace import Trace

from pcpostprocess.subtraction_plots import do_subtraction_plot
from pcpostprocess.detect_ramp_bounds import detect_ramp_bounds


class TestSubtractionPlots(unittest.TestCase):
    def setUp(self):
        test_data_dir = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                     "staircaseramp (2)_2kHz_15.01.07")
        json_file = "staircaseramp (2)_2kHz_15.01.07.json"

        self.output_dir = os.path.join('test_output', 'test_trace_class')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.ramp_bounds = [1700, 2500]

        # Use identical traces for purpose of the test
        self.before_trace = Trace(test_data_dir, json_file)
        self.after_trace = Trace(test_data_dir, json_file)

    def test_do_subtraction_plot(self):
        fig = plt.figure(layout='constrained')
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

if __name__ == "__main__":
    pass
