import copy
import logging
import os
import string
import unittest

import numpy as np
from syncropatch_export.trace import Trace

from pcpostprocess.hergQC import NOISE_LEN, hERGQC


def all_passed(result):
    return all([x for x, _ in result])


class TestHergQC(unittest.TestCase):

    def setUp(self):
        base_path = os.path.join("tests", "test_data", "13112023_MW2_FF")

        label_before = "staircaseramp (2)_2kHz_15.01.07"
        label_after = "staircaseramp (2)_2kHz_15.11.33"

        path_before = os.path.join(base_path, label_before)
        path_after = os.path.join(base_path, label_after)

        trace_before = Trace(path_before, json_file=label_before)
        trace_after = Trace(path_after, json_file=label_after)

        sweeps = [0, 1]
        self.trace_sweeps_before = trace_before.get_trace_sweeps(sweeps)
        self.trace_sweeps_after = trace_after.get_trace_sweeps(sweeps)

        self.qc_vals_before = trace_before.get_onboard_QC_values(sweeps=sweeps)
        self.qc_vals_after = trace_after.get_onboard_QC_values(sweeps=sweeps)

        self.voltage = trace_before.get_voltage()
        self.times = trace_after.get_times()
        self.n_sweeps = 2

        self.all_wells = [
            row + str(i).zfill(2)
            for row in string.ascii_uppercase[:16]
            for i in range(1, 25)
        ]

        sampling_rate = int(1.0 / (self.times[1] - self.times[0]))  # in kHz

        #  Assume that there are no discontinuities at the start or end of ramps
        voltage_protocol = trace_before.get_voltage_protocol()
        self.voltage_steps = [
            tstart
            for tstart, _, vstart, vend in voltage_protocol.get_all_sections()
            if vend == vstart
        ]

        plot_dir = "test_output"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        self.hergqc = hERGQC(
            sampling_rate=sampling_rate,
            plot_dir=plot_dir,
            voltage=self.voltage,
        )

    def test_qc_inputs(self):
        self.assertTrue(np.all(np.isfinite(self.voltage)))
        self.assertTrue(np.all(np.isfinite(self.times)))

    def test_qc1(self):

        hergqc = copy.deepcopy(self.hergqc)
        plot_dir = os.path.join(hergqc.plot_dir, "test_qc1")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        hergqc.plot_dir = plot_dir

        # qc1 checks that rseal, cm, rseries are within range
        rseal_lo, rseal_hi = 1e8, 1e12
        rseal_mid = (rseal_lo + rseal_hi) / 2
        rseal_tol = 0.1

        cm_lo, cm_hi = 1e-12, 1e-10
        cm_mid = (cm_lo + cm_hi) / 2
        cm_tol = 1e-13

        rseries_lo, rseries_hi = 1e6, 2.5e7
        rseries_mid = (rseries_lo + rseries_hi) / 2
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
                all_passed(hergqc.qc1(rseal, cm, rseries)),
                expected,
                f"QC1: {rseal}, {cm}, {rseries}",
            )

        # Test on data
        failed_wells_before = [
            'A10', 'A12', 'A13', 'A16', 'A19', 'A20', 'A24', 'B05', 'B07', 'B11',
            'B12', 'B13', 'B15', 'B19', 'B21', 'B23', 'C02', 'C04', 'C07', 'C09',
            'C11', 'C12', 'C14', 'C18', 'C19', 'C20', 'C23', 'D02', 'D03', 'D05',
            'D09', 'D10', 'D14', 'D15', 'D16', 'D19', 'D21', 'E04', 'E06', 'E07',
            'E10', 'E11', 'E14', 'E15', 'E16', 'E17', 'E19', 'E22', 'E23', 'F01',
            'F03', 'F04', 'F05', 'F07', 'F09', 'F12', 'F13', 'F14', 'F15', 'F18',
            'F19', 'F20', 'F21', 'F24', 'G06', 'G09', 'G12', 'G13', 'G14', 'G16',
            'G17', 'G20', 'G21', 'G23', 'G24', 'H01', 'H02', 'H03', 'H06', 'H07',
            'H08', 'H10', 'H11', 'H13', 'H14', 'H15', 'H16', 'H19', 'H21', 'H23',
            'H24', 'I01', 'I04', 'I05', 'I06', 'I07', 'I08', 'I10', 'I11', 'I12',
            'I16', 'I17', 'I21', 'J07', 'J10', 'J16', 'J17', 'J19', 'J21', 'J24',
            'K02', 'K03', 'K06', 'K07', 'K11', 'K12', 'K16', 'K17', 'K20', 'K22',
            'K24', 'L01', 'L02', 'L04', 'L05', 'L08', 'L10', 'L11', 'L13', 'L16',
            'L17', 'L18', 'L24', 'M01', 'M04', 'M06', 'M09', 'M12', 'M14', 'M15',
            'M16', 'M19', 'M20', 'M21', 'N03', 'N04', 'N06', 'N08', 'N11', 'N13',
            'N14', 'N16', 'N18', 'N21', 'N24', 'O01', 'O02', 'O03', 'O05', 'O07',
            'O10', 'O11', 'O17', 'O19', 'O22', 'O24', 'P01', 'P03', 'P06', 'P07',
            'P08', 'P09', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P21',
            'P22', 'P24'
        ]

        for well in self.all_wells:
            qc_vals_before = np.array(self.qc_vals_before[well])[0, :]
            ex_pass_before = well not in failed_wells_before
            self.assertEqual(
                all_passed(hergqc.qc1(*qc_vals_before)),
                ex_pass_before,
                f"QC1: {well} (before) {qc_vals_before}",
            )

        failed_wells_after = [
            'A06', 'A08', 'A10', 'A12', 'A13', 'A16', 'A19', 'A20', 'A24', 'B02',
            'B05', 'B07', 'B11', 'B12', 'B13', 'B15', 'B19', 'B21', 'B23', 'C01',
            'C02', 'C04', 'C07', 'C09', 'C11', 'C12', 'C14', 'C18', 'C20', 'C22',
            'C23', 'D03', 'D05', 'D09', 'D10', 'D14', 'D15', 'D19', 'D21', 'E01',
            'E03', 'E04', 'E06', 'E07', 'E09', 'E10', 'E11', 'E14', 'E15', 'E16',
            'E17', 'E19', 'E20', 'E22', 'E23', 'E24', 'F01', 'F03', 'F04', 'F07',
            'F09', 'F12', 'F13', 'F14', 'F15', 'F16', 'F18', 'F19', 'F20', 'F21',
            'F24', 'G06', 'G08', 'G09', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17',
            'G20', 'G21', 'G23', 'G24', 'H01', 'H02', 'H03', 'H04', 'H06', 'H07',
            'H08', 'H09', 'H10', 'H11', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18',
            'H19', 'H20', 'H21', 'H23', 'H24', 'I01', 'I04', 'I05', 'I06', 'I07',
            'I08', 'I10', 'I12', 'I16', 'I17', 'I20', 'I21', 'J03', 'J07', 'J10',
            'J14', 'J16', 'J17', 'J19', 'J21', 'J23', 'J24', 'K02', 'K03', 'K06',
            'K07', 'K14', 'K16', 'K20', 'K22', 'K23', 'K24', 'L01', 'L02', 'L03',
            'L04', 'L05', 'L08', 'L10', 'L11', 'L12', 'L13', 'L16', 'L17', 'L18',
            'L24', 'M01', 'M02', 'M04', 'M05', 'M06', 'M08', 'M09', 'M12', 'M13',
            'M14', 'M15', 'M19', 'M20', 'M21', 'N04', 'N06', 'N08', 'N11', 'N13',
            'N14', 'N16', 'N18', 'N19', 'N21', 'N24', 'O01', 'O02', 'O03', 'O05',
            'O07', 'O08', 'O10', 'O11', 'O14', 'O15', 'O17', 'O19', 'O22', 'O24',
            'P01', 'P03', 'P06', 'P07', 'P08', 'P09', 'P11', 'P12', 'P13', 'P14',
            'P15', 'P16', 'P17', 'P18', 'P19', 'P21', 'P22'
        ]

        for well in self.all_wells:
            qc_vals_after = np.array(self.qc_vals_after[well])[0, :]
            ex_pass_after = well not in failed_wells_after
            self.assertEqual(
                all_passed(hergqc.qc1(*qc_vals_after)),
                ex_pass_after,
                f"QC1: {well} (after) {qc_vals_after}",
            )

    def test_qc2(self):
        hergqc = copy.deepcopy(self.hergqc)
        plot_dir = os.path.join(hergqc.plot_dir, "test_qc2")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        hergqc.plot_dir = plot_dir

        # qc2 checks that raw and subtracted SNR are above a minimum threshold
        test_matrix = [
            (10, 8082.1, True),
            (1, 74.0, True),
            (0.601, 25.1, True),
            (0.6, 25.0, False),
            (0.5, 16.8, False),
            (0.1, 0.5, False),
        ]

        for i, ex_snr, ex_pass in test_matrix:
            recording = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [i] * 500)
            pass_, snr = hergqc.qc2(recording)
            self.assertAlmostEqual(
                snr, ex_snr, 1, f"QC2: ({i}) {snr} != {ex_snr}")
            self.assertEqual(pass_, ex_pass, f"QC2: ({i}) {pass_} != {ex_pass}")

        # Test on data
        failed_wells_raw = ["P16"]
        failed_wells_subtracted = [
            "B09", "C11", "H19", "H24", "K22", "O16", "P16"
        ]

        for well in self.all_wells:
            before = np.array(self.trace_sweeps_before[well])
            after = np.array(self.trace_sweeps_after[well])

            raw = []
            subtracted = []
            for i in range(self.n_sweeps):
                raw.append(hergqc.qc2(before[i]))
                subtracted.append(hergqc.qc2(before[i] - after[i]))

            ex_pass_raw = well not in failed_wells_raw
            self.assertEqual(
                all_passed(raw),
                ex_pass_raw,
                f"QC2: {well} (raw) {raw}",
            )

            ex_pass_subtracted = well not in failed_wells_subtracted
            self.assertEqual(
                all_passed(subtracted),
                ex_pass_subtracted,
                f"QC2: {well} (subtracted) {subtracted}",
            )

    def test_qc3(self):
        hergqc = copy.deepcopy(self.hergqc)
        plot_dir = os.path.join(hergqc.plot_dir, "test_qc3")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        hergqc.plot_dir = plot_dir

        # qc3 checks that rmsd of two sweeps are similar

        # Test with same noise, different signal
        test_matrix = [
            (-10, False),
            (-9, False),
            (-8, False),  # rmsdc - rmsd = -0.6761186497920804
            (-7, True),  # rmsdc - rmsd = 0.25355095037585507
            (0, True),  # rmsdc - rmsd = 6.761238263598085
            (8, True),  # rmsdc - rmsd = 0.6761272774054383
            (9, False),  # rmsdc - rmsd = -0.08451158778363332
            (10, False),
        ]

        recording1 = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [40] * 500)
        for i, expected in test_matrix:
            recording2 = np.asarray(
                [0, 0.1] * (NOISE_LEN // 2) + [40 + i] * 500)
            result = hergqc.qc3(recording1, recording2)
            self.assertEqual(result[0], expected, f"({i}: {result[1]})")

        # Test with same signal, different noise
        # TODO: Find failing example
        test_matrix = [
            (10, True),
            (100, True),
            (1000, True),
        ]

        recording1 = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [40] * 500)
        for i, expected in test_matrix:
            recording2 = np.asarray(
                [0, 0.1 * i] * (NOISE_LEN // 2) + [40] * 500)
            result = hergqc.qc3(recording1, recording2)
            self.assertEqual(result[0], expected, f"({i}: {result[1]})")

        # TODO: Test on select data

    def test_qc4(self):
        hergqc = copy.deepcopy(self.hergqc)
        plot_dir = os.path.join(hergqc.plot_dir, "test_qc4")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        hergqc.plot_dir = plot_dir

        # qc4 checks that rseal, cm, rseries are similar before/after E-4031 change
        r_lo, r_hi = 1e6, 3e7
        c_lo, c_hi = 1e-12, 1e-10

        # Test rseals
        cms = [c_lo, c_lo]
        rseriess = [r_lo, r_lo]

        test_matrix = [
            (1.1, True),
            (3.0, True),
            (3.5, False),
            (5.0, False),
        ]

        for i, expected in test_matrix:
            rseals = [r_lo, i * r_lo]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                expected,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

            rseals = [r_hi, i * r_hi]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                expected,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

        # Test cms
        rseals = [r_lo, r_lo]
        rseriess = [r_lo, r_lo]

        test_matrix = [
            (1.1, True),
            (3.0, True),
            (3.5, False),
            (5.0, False),
        ]

        for i, expected in test_matrix:
            cms = [c_lo, i * c_lo]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                expected,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

            cms = [c_hi, i * c_hi]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                expected,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

        # Test rseriess
        cms = [c_lo, c_lo]
        rseals = [r_lo, r_lo]

        test_matrix = [
            (1.1, True),
            (3.0, True),
            (3.5, False),
            (5.0, False),
        ]

        for i, expected in test_matrix:
            rseriess = [r_lo, i * r_lo]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                expected,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

            rseriess = [r_hi, i * r_hi]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                expected,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

        # TODO: Test on select data

    def test_qc5(self):
        hergqc = copy.deepcopy(self.hergqc)
        plot_dir = os.path.join(hergqc.plot_dir, "test_qc5")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        hergqc.plot_dir = plot_dir

        # qc5 checks that the maximum current during the second half of the
        # staircase changes by at least 75% of the raw trace after E-4031 addition
        test_matrix = [
            (-1.0, True),
            (0.1, True),
            (0.2, True),  # max_diffc - max_diff = -0.5
            (0.25, True),  # max_diffc - max_diff = 0.0
            (0.3, False),  # max_diffc - max_diff = 0.5
            (0.5, False),
            (1.0, False),
        ]

        recording1 = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [10] * 500)
        for i, expected in test_matrix:
            recording2 = np.asarray(
                [0, 0.1] * (NOISE_LEN // 2) + [10 * i] * 500)
            result = hergqc.qc5(recording1, recording2)
            self.assertEqual(result[0], expected, f"({i}: {result[1]})")

        # TODO: Test on select data

    def test_qc5_1(self):
        hergqc = copy.deepcopy(self.hergqc)
        plot_dir = os.path.join(hergqc.plot_dir, "test_qc5_1")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        hergqc.plot_dir = plot_dir

        # qc5_1 checks that the RMSD to zero of staircase protocol changes
        # by at least 50% of the raw trace after E-4031 addition.
        test_matrix = [
            (-1.0, False),  # rmsd0_diffc - rmsd0_diff = 4.2
            (-0.5, False),  # rmsd0_diffc - rmsd0_diff = 0.0001
            (-0.4, True),  # rmsd0_diffc - rmsd0_diff = -0.8
            (-0.1, True),  # rmsd0_diffc - rmsd0_diff = -3.4
            (0.1, True),  # rmsd0_diffc - rmsd0_diff = -3.4
            (0.4, True),  # rmsd0_diffc - rmsd0_diff = -0.8
            (0.5, False),  # rmsd0_diffc - rmsd0_diff = 0.0001
            (1.0, False),  # rmsd0_diffc - rmsd0_diff = 4.2
        ]

        recording1 = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [10] * 500)
        for i, expected in test_matrix:
            recording2 = np.asarray(
                [0, 0.1] * (NOISE_LEN // 2) + [10 * i] * 500)
            result = hergqc.qc5_1(recording1, recording2)
            self.assertEqual(result[0], expected, f"({i}: {result[1]})")

        # TODO: Test on select data

    def test_qc6(self):
        hergqc = copy.deepcopy(self.hergqc)
        plot_dir = os.path.join(hergqc.plot_dir, "test_qc6")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        hergqc.plot_dir = plot_dir

        # qc6 checks that the first step up to +40 mV, before the staircase, in
        # the subtracted trace is bigger than -2 x estimated noise level.
        test_matrix = [
            (-100, False),  # valc - val = 9.9
            (-10, False),  # valc - val = 0.9
            (-2, False),  # valc - val = 0.1
            (-1, True),  # valc - val = 0
            (1, True),  # valc - val = -0.2
            (2, True),  # valc - val = -0.3
            (10, True),  # valc - val = -1.1
            (100, True),  # valc - val = -10.1
        ]
        for i, expected in test_matrix:
            recording = np.asarray(
                [0, 0.1] * (NOISE_LEN // 2) + [0.1 * i] * 500)
            result = hergqc.qc6(recording, win=[NOISE_LEN, -1])
            self.assertEqual(result[0], expected, f"({i}: {result[1]})")

        # TODO: Test on select data

    def test_run_qc(self):
        # Spot check a few wells; could check all, but it's time consuming.

        hergqc = copy.deepcopy(self.hergqc)
        plot_dir = os.path.join(hergqc.plot_dir, "test_run_qc")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        hergqc.plot_dir = plot_dir

        test_matrix = [
            ("A01", True),
            ("A02", True),
            ("A03", True),
            ("A04", False),
            ("A05", False),
            ("D01", False),
        ]

        for well, expected in test_matrix:
            with self.subTest(well):
                # Take values from the first sweep only
                before = np.array(self.trace_sweeps_before[well])
                after = np.array(self.trace_sweeps_after[well])

                qc_vals_before = np.array(self.qc_vals_before[well])[0, :]
                qc_vals_after = np.array(self.qc_vals_after[well])[0, :]

                QC = hergqc.run_qc(
                    voltage_steps=self.voltage_steps,
                    times=self.times,
                    before=before,
                    after=after,
                    qc_vals_before=qc_vals_before,
                    qc_vals_after=qc_vals_after,
                    n_sweeps=self.n_sweeps,
                )

                logging.debug(well, QC.all_passed())

                trace = ""
                for label, result in QC.items():
                    if not QC.qc_passed(label):
                        trace += f"{well} {label}: {result}\n"

                self.assertEqual(QC.all_passed(), expected, trace)
