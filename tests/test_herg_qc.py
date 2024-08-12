import logging
import os
import string
import unittest

import numpy as np
from syncropatch_export.trace import Trace

from pcpostprocess.hergQC import hERGQC


class TestHergQC(unittest.TestCase):

    def setUp(self):
        filepath = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                'staircaseramp (2)_2kHz_15.01.07')

        self.all_wells = [
            lab + str(i).zfill(2) for lab in string.ascii_uppercase[:16]
            for i in range(1, 25)]

        filepath2 = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                 'staircaseramp (2)_2kHz_15.11.33')

        json_file = "staircaseramp (2)_2kHz_15.01.07"
        json_file2 = "staircaseramp (2)_2kHz_15.11.33"

        self.output_dir = os.path.join('test_output', 'test_herg_qc')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.test_trace_before = Trace(filepath, json_file)
        self.test_trace_after = Trace(filepath2, json_file2)

        self.voltage = self.test_trace_before.get_voltage()
        self.times = self.test_trace_after.get_times()

        # Calculate sampling rate in (use kHz)
        self.sampling_rate = int(1.0 / (self.times[1] - self.times[0]))

    def test_qc1(self):
        def passed(result):
            return all([x for x, _ in result])

        plot_dir = os.path.join(self.output_dir, "test_qc1")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        hergqc = hERGQC(sampling_rate=self.sampling_rate,
                        plot_dir=plot_dir,
                        voltage=self.voltage)

        # qc1 checks that rseal, cm, rseries are within range
        rseal_lo, rseal_hi = 1e8, 1e12
        rseal_mid = (rseal_lo + rseal_hi) / 2

        cm_lo, cm_hi = 1e-12, 1e-10
        cm_mid = (cm_lo + cm_hi) / 2

        rseries_lo, rseries_hi = 1e6, 2.5e7
        rseries_mid = (rseries_lo + rseries_hi) / 2

        rseal_tol = 0.1
        cm_tol = 1e-13
        rseries_tol = 0.1

        test_matrix = [
            [(rseal_lo, cm_lo, rseries_lo), True],
            [(rseal_mid, cm_mid, rseries_mid), True],
            [(rseal_hi, cm_hi, rseries_hi), True],
            [(rseal_lo - rseal_tol, cm_lo, rseries_lo), False],
            [(rseal_lo, cm_lo - cm_tol, rseries_lo), False],
            [(rseal_lo, cm_lo, rseries_lo - rseries_tol), False],
            [(rseal_hi + rseal_tol, cm_hi, rseries_hi), False],
            [(rseal_hi, cm_hi + cm_tol, rseries_hi), False],
            [(rseal_hi, cm_hi, rseries_hi + rseries_tol), False],
            [(np.inf, cm_mid, rseries_mid), False],
            [(rseal_mid, np.inf, rseries_mid), False],
            [(rseal_mid, cm_mid, np.inf), False],
            [(None, cm_mid, rseries_mid), False],
            [(rseal_mid, None, rseries_mid), False],
            [(rseal_mid, cm_mid, None), False],
            [(None, None, None), False],
            [(0, 0, 0), False],
        ]

        for (rseal, cm, rseries), expected in test_matrix:
            self.assertEqual(
                passed(hergqc.qc1(rseal, cm, rseries)),
                expected,
                f"({rseal}, {cm}, {rseries})",
            )

    def test_qc2(self):
        pass

    def test_qc3(self):
        pass

    def test_qc4(self):
        pass

    def test_qc5(self):
        pass

    def test_qc6(self):
        pass

    def test_run_qc(self):
        self.assertTrue(np.all(np.isfinite(self.voltage)))
        self.assertTrue(np.all(np.isfinite(self.times)))

        plot_dir = os.path.join(self.output_dir,
                                'test_run_qc')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        hergqc = hERGQC(sampling_rate=self.sampling_rate,
                        plot_dir=plot_dir,
                        voltage=self.voltage)

        sweeps = [0, 1]
        before = self.test_trace_before.get_trace_sweeps(sweeps)
        after = self.test_trace_after.get_trace_sweeps(sweeps)
        qc_vals_before = self.test_trace_before.get_onboard_QC_values(sweeps=sweeps)
        qc_vals_after = self.test_trace_after.get_onboard_QC_values(sweeps=sweeps)

        # Spot check a few wells
        # We could check all of the wells but it's time consuming

        test_wells = ['A01', 'A02', 'A03', 'A04', 'A05', 'D01']

        voltage_protocol = self.test_trace_before.get_voltage_protocol()

        for well in test_wells:
            with self.subTest(well):
                # Take values from the first sweep only
                qc_vals_before_well = np.array(qc_vals_before[well])[0, :]
                qc_vals_after_well = np.array(qc_vals_after[well])[0, :]

                before_well = np.array(before[well])
                after_well = np.array(after[well])

                #  Assume that there are no discontinuities at the start or end of ramps
                voltage_steps = [tstart
                                 for tstart, tend, vstart, vend in
                                 voltage_protocol.get_all_sections() if vend == vstart]

                passed, qcs = hergqc.run_qc(voltage_steps,
                                            self.times, before_well, after_well,
                                            qc_vals_before_well,
                                            qc_vals_after_well, n_sweeps=2)

                logging.debug(well, passed)

                trace = ""
                for label, results in qcs.items():
                    if any([x == False for x, _ in results]):
                        trace += f"{label}: {results}\n"
                self.assertTrue(passed, trace)
