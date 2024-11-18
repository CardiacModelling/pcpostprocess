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

        # TODO: Test on select data

    def test_qc2(self):
        plot_dir = os.path.join(self.output_dir, "test_qc2")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        hergqc = hERGQC(sampling_rate=self.sampling_rate,
                        plot_dir=plot_dir,
                        voltage=self.voltage)

        # qc2 checks that raw and subtracted SNR are above a minimum threshold
        recording = np.asarray([0, 1] * 500 + [0, 10] * 500)  # snr = 70.75
        result = hergqc.qc2(recording)
        self.assertTrue(result[0], f"({result[1]})")

        recording = np.asarray([0, 1] * 500 + [0, 6.03125] * 500)  # snr = 25.02
        result = hergqc.qc2(recording)
        self.assertTrue(result[0], f"({result[1]})")

        recording = np.asarray([0, 1] * 500 + [0, 6.015625] * 500)  # snr = 24.88
        result = hergqc.qc2(recording)
        self.assertFalse(result[0], f"({result[1]})")

        recording = np.asarray([0, 1] * 1000)  # snr = 1.0
        result = hergqc.qc2(recording)
        self.assertFalse(result[0], f"({result[1]})")

        # TODO: Test on select data

    def test_qc3(self):
        plot_dir = os.path.join(self.output_dir, "test_qc3")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        hergqc = hERGQC(sampling_rate=self.sampling_rate,
                        plot_dir=plot_dir,
                        voltage=self.voltage)

        # qc3 checks that rmsd of two sweeps are similar
        test_matrix = [
            (0, True),
            (1, True),
            (2, True),
            (3, True),
            (4, False),
            (5, False),
            (6, False),
        ]

        for i, expected in test_matrix:
            recording0 = np.asarray([0, 1] * 1000)
            recording1 = recording0 + i
            result = hergqc.qc3(recording0, recording1)
            self.assertEqual(result[0], expected, f"({i}: {result[1]})")

        # TODO: Test on select data

    def test_qc4(self):
        def passed(result):
            return all([x for x, _ in result])

        plot_dir = os.path.join(self.output_dir, "test_qc1")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        hergqc = hERGQC(
            sampling_rate=self.sampling_rate, plot_dir=plot_dir, voltage=self.voltage
        )

        # qc4 checks that rseal, cm, rseries are similar before/after E-4031 change
        test_matrix = [
            [([1, 1], [1, 1], [1, 1]), True],
            [([1, 2], [1, 2], [1, 2]), True],
            [([1, 3], [1, 3], [1, 3]), True],
            [([1, 4], [1, 4], [1, 4]), False],
            [([1, 5], [1, 5], [1, 5]), False],
        ]

        for (rseals, cms, rseriess), expected in test_matrix:
            self.assertEqual(
                passed(hergqc.qc4(rseals, cms, rseriess)),
                expected,
                f"({rseals}, {cms}, {rseriess})",
            )

        # TODO: Test on select data

    def test_qc5(self):
        plot_dir = os.path.join(self.output_dir, "test_qc5")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        hergqc = hERGQC(
            sampling_rate=self.sampling_rate, plot_dir=plot_dir, voltage=self.voltage
        )

        # qc5 checks that the maximum current during the second half of the 
        # staircase changes by at least 75% of the raw trace after E-4031 addition
        test_matrix = [
            (-2.0, True),
            (-1.0, True),
            (-0.75, True),
            (-0.5, False),
            (0.0, False),
            (0.5, False),
            (0.75, False),
            (1.0, False),
        ]

        for i, expected in test_matrix:
            recording0 = np.asarray([0, 1] * 1000)
            recording1 = recording0 + i
            result = hergqc.qc5(recording0, recording1)
            self.assertEqual(result[0], expected, f"({i}: {result[1]})")

        # TODO: Test on select data

    def test_qc5_1(self):
        plot_dir = os.path.join(self.output_dir, "test_qc5")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        hergqc = hERGQC(
            sampling_rate=self.sampling_rate, plot_dir=plot_dir, voltage=self.voltage
        )

        # qc5_1 checks that the RMSD to zero of staircase protocol changes 
        # by at least 50% of the raw trace after E-4031 addition.
        test_matrix = [
            (0.1, True),
            (0.2, True),
            (0.3, True),
            (0.4, True),
            (0.49, True),
            (0.5, True),
            (0.51, False),
            (0.6, False),
            (0.7, False),
            (0.8, False),
            (0.9, False),
            (1.0, False),
        ]

        for i, expected in test_matrix:
            recording0 = np.asarray([0, 1] * 1000)
            recording1 = recording0 * i
            result = hergqc.qc5_1(recording0, recording1)
            self.assertEqual(result[0], expected, f"({i}: {result[1]})")

        # TODO: Test on select data

    def test_qc6(self):
        # TODO: Test on select data
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

        test_wells = ['A01', 'A02', 'A03']

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

                QC = hergqc.run_qc(voltage_steps,
                                   self.times, before_well, after_well,
                                   qc_vals_before_well,
                                   qc_vals_after_well, n_sweeps=2)

                logging.debug(well, QC.all_passed())

                trace = ""
                for label, result in QC.items():
                    if not QC.qc_passed(label):
                        trace += f"{label}: {result}\n"
                print(f"Testing Well, {well}")
                self.assertTrue(QC.all_passed(), trace)
