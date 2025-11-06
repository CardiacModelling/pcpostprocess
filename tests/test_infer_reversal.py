#!/usr/bin/env python3
import os
import unittest

from syncropatch_export.trace import Trace

from pcpostprocess import infer_reversal, leak_correct


class TestInferReversal(unittest.TestCase):
    def setUp(self):
        test_data_dir = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                     "staircaseramp (2)_2kHz_15.01.07")
        json_file = "staircaseramp (2)_2kHz_15.01.07.json"

        self.output_dir = os.path.join('test_output', self.__class__.__name__)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.ramp_bounds = [1700, 2500]
        self.test_trace = Trace(test_data_dir, json_file)

        # get currents and QC from trace object
        self.currents = self.test_trace.get_all_traces(leakcorrect=False)
        self.currents['times'] = self.test_trace.get_times()
        self.currents['voltages'] = self.test_trace.get_voltage()

        self.protocol_desc = self.test_trace.get_voltage_protocol().get_all_sections()
        self.voltages = self.test_trace.get_voltage()

        self.correct_Erev = -90.06155612421054001970333047211170196533203125

    def test_plot_leak_fit(self):
        well = 'A03'
        sweep = 0

        voltage = self.test_trace.get_voltage()
        times = self.test_trace.get_times()

        current = self.test_trace.get_trace_sweeps(sweeps=[sweep])[well][0, :]
        params, Ileak = leak_correct.fit_linear_leak(current, voltage, times,
                                                     *self.ramp_bounds,
                                                     output_dir=self.output_dir,
                                                     save_fname=f"{well}_sweep{sweep}_leak_correction")

        I_corrected = current - Ileak

        E_rev = infer_reversal.infer_reversal_potential(
            I_corrected, times, self.protocol_desc,
            self.voltages,
            output_path=os.path.join(self.output_dir,
                                     f"{well}_staircase"),
            known_Erev=self.correct_Erev)
        self.assertEqual(E_rev, self.correct_Erev)


if __name__ == "__main__":
    pass
