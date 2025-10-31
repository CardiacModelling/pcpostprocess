#!/usr/bin/env python3
import os
import unittest

from syncropatch_export.trace import Trace

from pcpostprocess import leak_correct
from pcpostprocess import directory_builder
from pcpostprocess.detect_ramp_bounds import detect_ramp_bounds


class TestLeakCorrect(unittest.TestCase):
    def setUp(self):
        test_data_dir = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                     "staircaseramp (2)_2kHz_15.01.07")
        json_file = "staircaseramp (2)_2kHz_15.01.07.json"

        self.output_dir = directory_builder.setup_output_directory("test_output",
                                                        self.__class__.__name__)

        self.test_trace = Trace(test_data_dir, json_file)

        # get currents and QC from trace object
        self.currents = self.test_trace.get_all_traces(leakcorrect=False)
        self.currents['times'] = self.test_trace.get_times()
        self.currents['voltages'] = self.test_trace.get_voltage()

        self.QC = self.test_trace.get_onboard_QC_values()

        # Find first times ahead of these times
        voltage_protocol = self.test_trace.get_voltage_protocol().get_all_sections()
        times = self.currents['times'].flatten()
        self.ramp_bound_indices = detect_ramp_bounds(times, voltage_protocol, ramp_no=0)

    def test_plot_leak_fit(self):
        well = 'A01'
        sweep = 0

        voltage = self.test_trace.get_voltage()
        times = self.test_trace.get_times()

        current = self.test_trace.get_trace_sweeps(sweeps=[sweep])[well][0, :]

        leak_correct.fit_linear_leak(current, voltage, times,
                                     *self.ramp_bound_indices,
                                     output_dir=self.output_dir,
                                     save_fname=f"{well}_sweep{sweep}_leak_correction")

    def test_get_leak_correct(self):
        trace = self.test_trace
        currents = self.currents
        well = 'A01'
        sweep = 0
        voltage = trace.get_voltage()
        times = trace.get_times()

        current = currents[well][sweep, :]
        x = leak_correct.get_leak_corrected(current, voltage, times,
                                            *self.ramp_bound_indices)
        self.assertEqual(x.shape, (30784,))


if __name__ == "__main__":
    pass
