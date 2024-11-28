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
        os.makedirs(plot_dir, exist_ok=True)

        self.hergqc = hERGQC(
            sampling_rate=sampling_rate,
            plot_dir=plot_dir,
            voltage=self.voltage,
        )

    def clone_herg_qc(self, plot_dir):
        hergqc = copy.deepcopy(self.hergqc)
        plot_dir = os.path.join(hergqc.plot_dir, plot_dir)
        os.makedirs(plot_dir, exist_ok=True)
        hergqc.plot_dir = plot_dir
        return hergqc

    def test_qc_inputs(self):
        self.assertTrue(np.all(np.isfinite(self.voltage)))
        self.assertTrue(np.all(np.isfinite(self.times)))

    def test_qc1(self):
        hergqc = self.clone_herg_qc("test_qc1")

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

        for (rseal, cm, rseries), ex_pass in test_matrix:
            self.assertEqual(
                all_passed(hergqc.qc1(rseal, cm, rseries)),
                ex_pass,
                f"QC1: {rseal}, {cm}, {rseries}",
            )

        # Test on data - values before
        failed_wells_rseal_before = [
            'A10', 'A12', 'A13', 'A16', 'A19', 'A20', 'B05', 'B07', 'B11', 'B12',
            'B13', 'B15', 'B19', 'B21', 'B23', 'C02', 'C04', 'C07', 'C09', 'C11',
            'C12', 'C14', 'C18', 'C19', 'C20', 'D02', 'D03', 'D05', 'D09', 'D10',
            'D14', 'D15', 'D19', 'D21', 'E04', 'E07', 'E10', 'E11', 'E14', 'E15',
            'E16', 'E17', 'E22', 'E23', 'F01', 'F03', 'F04', 'F05', 'F07', 'F09',
            'F12', 'F13', 'F14', 'F15', 'F18', 'F19', 'F20', 'F21', 'F24', 'G06',
            'G09', 'G12', 'G13', 'G14', 'G16', 'G20', 'G23', 'G24', 'H01', 'H02',
            'H03', 'H06', 'H07', 'H08', 'H10', 'H11', 'H13', 'H14', 'H15', 'H16',
            'H19', 'H21', 'H23', 'H24', 'I01', 'I04', 'I05', 'I07', 'I08', 'I10',
            'I11', 'I12', 'I16', 'I17', 'I21', 'J07', 'J10', 'J16', 'J17', 'J19',
            'J21', 'J24', 'K02', 'K03', 'K06', 'K07', 'K11', 'K12', 'K16', 'K17',
            'K20', 'K22', 'K24', 'L01', 'L02', 'L04', 'L05', 'L08', 'L10', 'L11',
            'L13', 'L16', 'L17', 'L18', 'L24', 'M01', 'M04', 'M06', 'M09', 'M12',
            'M16', 'M19', 'M21', 'N03', 'N04', 'N06', 'N08', 'N11', 'N13', 'N14',
            'N16', 'N18', 'N21', 'N24', 'O01', 'O02', 'O03', 'O05', 'O07', 'O10',
            'O11', 'O17', 'O19', 'O22', 'O24', 'P01', 'P03', 'P06', 'P07', 'P08',
            'P09', 'P12', 'P13', 'P14', 'P15', 'P17', 'P18', 'P21', 'P22', 'P24'
        ]

        failed_wells_cm_before = [
            'A12', 'A13', 'A16', 'A19', 'B07', 'B11', 'B13', 'B15', 'B21', 'B23',
            'C02', 'C04', 'C07', 'C11', 'C12', 'C14', 'C18', 'C20', 'D03', 'D10',
            'D14', 'E04', 'E07', 'E10', 'E15', 'E16', 'E17', 'E22', 'E23', 'F01',
            'F03', 'F04', 'F07', 'F12', 'F14', 'F15', 'F18', 'F19', 'F20', 'F21',
            'F24', 'G09', 'G12', 'G13', 'G16', 'G20', 'G23', 'G24', 'H01', 'H03',
            'H06', 'H07', 'H10', 'H15', 'H19', 'H21', 'H23', 'H24', 'I04', 'I05',
            'I07', 'I10', 'I12', 'I16', 'I17', 'I21', 'J07', 'J16', 'J19', 'J21',
            'K02', 'K16', 'K22', 'L01', 'L02', 'L04', 'L05', 'L08', 'L10', 'L11',
            'L13', 'L17', 'L18', 'M01', 'M04', 'M12', 'M19', 'M21', 'N06', 'N08',
            'N11', 'N14', 'N18', 'N21', 'N24', 'O01', 'O03', 'O07', 'O10', 'O17',
            'O19', 'O22', 'O24', 'P01', 'P06', 'P07', 'P08', 'P12', 'P13', 'P14',
            'P15', 'P16', 'P18', 'P21', 'P22'
        ]

        failed_wells_rseries_before = [
            'A12', 'A13', 'A16', 'A19', 'A24', 'B07', 'B11', 'B13', 'B15', 'B21',
            'B23', 'C02', 'C04', 'C07', 'C11', 'C12', 'C14', 'C18', 'C20', 'C23',
            'D03', 'D09', 'D10', 'D14', 'D15', 'D16', 'E04', 'E06', 'E07', 'E10',
            'E15', 'E16', 'E17', 'E19', 'E22', 'E23', 'F01', 'F03', 'F04', 'F07',
            'F12', 'F14', 'F15', 'F18', 'F19', 'F20', 'F21', 'F24', 'G09', 'G12',
            'G13', 'G16', 'G17', 'G20', 'G21', 'G23', 'G24', 'H01', 'H03', 'H07',
            'H10', 'H15', 'H19', 'H21', 'H23', 'H24', 'I04', 'I05', 'I06', 'I07',
            'I10', 'I12', 'I16', 'I17', 'I21', 'J07', 'J16', 'J19', 'J21', 'K02',
            'K16', 'K22', 'L01', 'L02', 'L04', 'L05', 'L08', 'L10', 'L11', 'L13',
            'L17', 'L18', 'M01', 'M04', 'M12', 'M14', 'M15', 'M19', 'M20', 'M21',
            'N06', 'N08', 'N11', 'N14', 'N18', 'N21', 'N24', 'O01', 'O03', 'O07',
            'O10', 'O17', 'O19', 'O22', 'O24', 'P01', 'P06', 'P07', 'P08', 'P12',
            'P13', 'P14', 'P15', 'P16', 'P18', 'P21', 'P22'
        ]

        for well in self.all_wells:
            qc_vals_before = np.array(self.qc_vals_before[well])[0, :]
            result = hergqc.qc1(*qc_vals_before)

            pass_rseal_before, rseal_before = result[0]
            ex_pass_rseal_before = well not in failed_wells_rseal_before
            self.assertEqual(
                pass_rseal_before,
                ex_pass_rseal_before,
                f"QC1: {well} (rseal before) {rseal_before}",
            )

            pass_cm_before, cm_before = result[1]
            ex_pass_cm_before = well not in failed_wells_cm_before
            self.assertEqual(
                pass_cm_before,
                ex_pass_cm_before,
                f"QC1: {well} (cm before) {cm_before}",
            )

            pass_rseries_before, rseries_before = result[2]
            ex_pass_rseries_before = well not in failed_wells_rseries_before
            self.assertEqual(
                pass_rseries_before,
                ex_pass_rseries_before,
                f"QC1: {well} (rseries before) {rseries_before}",
            )

        # Test on data - values after
        failed_wells_rseal_after = [
            'A10', 'A12', 'A13', 'A16', 'A19', 'A20', 'A24', 'B02', 'B05', 'B07',
            'B11', 'B12', 'B13', 'B15', 'B21', 'B23', 'C02', 'C04', 'C07', 'C09',
            'C11', 'C12', 'C14', 'C18', 'C20', 'C22', 'D03', 'D05', 'D10', 'D14',
            'D19', 'D21', 'E04', 'E07', 'E10', 'E11', 'E14', 'E15', 'E16', 'E17',
            'E22', 'E23', 'F01', 'F03', 'F04', 'F07', 'F09', 'F12', 'F13', 'F14',
            'F15', 'F18', 'F19', 'F20', 'F21', 'F24', 'G06', 'G08', 'G09', 'G12',
            'G13', 'G15', 'G16', 'G20', 'G23', 'G24', 'H01', 'H03', 'H06', 'H07',
            'H08', 'H09', 'H10', 'H11', 'H14', 'H15', 'H16', 'H17', 'H18', 'H19',
            'H21', 'H23', 'H24', 'I04', 'I05', 'I06', 'I07', 'I08', 'I10', 'I12',
            'I16', 'I17', 'I21', 'J07', 'J10', 'J16', 'J17', 'J19', 'J21', 'J23',
            'J24', 'K02', 'K03', 'K07', 'K14', 'K16', 'K20', 'K22', 'K23', 'K24',
            'L01', 'L02', 'L04', 'L05', 'L08', 'L10', 'L11', 'L13', 'L17', 'L18',
            'L24', 'M01', 'M04', 'M06', 'M09', 'M12', 'M19', 'M21', 'N04', 'N06',
            'N08', 'N11', 'N13', 'N14', 'N16', 'N18', 'N21', 'N24', 'O01', 'O02',
            'O03', 'O07', 'O08', 'O10', 'O11', 'O17', 'O19', 'O22', 'O24', 'P01',
            'P06', 'P07', 'P08', 'P09', 'P12', 'P13', 'P14', 'P15', 'P17', 'P18',
            'P21', 'P22'
        ]

        failed_wells_cm_after = [
            'A12', 'A13', 'A19', 'A20', 'B07', 'B11', 'B13', 'B15', 'B19', 'B21',
            'B23', 'C02', 'C04', 'C11', 'C12', 'C14', 'C18', 'C20', 'D10', 'D14',
            'E03', 'E09', 'E10', 'E15', 'E16', 'E17', 'E19', 'E22', 'E23', 'F01',
            'F03', 'F04', 'F07', 'F12', 'F14', 'F15', 'F18', 'F19', 'F20', 'F21',
            'F24', 'G06', 'G09', 'G12', 'G13', 'G16', 'G20', 'G23', 'G24', 'H01',
            'H03', 'H07', 'H09', 'H10', 'H15', 'H19', 'H21', 'H23', 'H24', 'I04',
            'I05', 'I07', 'I10', 'I12', 'I16', 'I17', 'I21', 'J07', 'J16', 'J19',
            'J21', 'K02', 'K16', 'K22', 'K23', 'L01', 'L02', 'L04', 'L05', 'L08',
            'L10', 'L11', 'L13', 'L16', 'L17', 'L18', 'M01', 'M04', 'M12', 'M15',
            'M19', 'M21', 'N06', 'N08', 'N11', 'N14', 'N18', 'N19', 'N21', 'N24',
            'O01', 'O03', 'O07', 'O10', 'O15', 'O17', 'O19', 'O22', 'O24', 'P01',
            'P06', 'P08', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P21',
            'P22'
        ]

        failed_wells_rseries_after = [
            'A06', 'A08', 'A12', 'A13', 'A19', 'A20', 'A24', 'B07', 'B11', 'B13',
            'B15', 'B19', 'B21', 'B23', 'C01', 'C02', 'C04', 'C09', 'C11', 'C12',
            'C14', 'C18', 'C20', 'C22', 'C23', 'D09', 'D10', 'D14', 'D15', 'D19',
            'E01', 'E03', 'E04', 'E06', 'E09', 'E10', 'E15', 'E16', 'E17', 'E19',
            'E20', 'E22', 'E23', 'E24', 'F01', 'F03', 'F04', 'F07', 'F12', 'F14',
            'F15', 'F16', 'F18', 'F19', 'F20', 'F21', 'F24', 'G06', 'G09', 'G12',
            'G13', 'G14', 'G16', 'G17', 'G20', 'G21', 'G23', 'G24', 'H01', 'H02',
            'H03', 'H04', 'H07', 'H09', 'H10', 'H13', 'H14', 'H15', 'H19', 'H20',
            'H21', 'H23', 'H24', 'I01', 'I04', 'I05', 'I07', 'I10', 'I12', 'I16',
            'I17', 'I20', 'I21', 'J03', 'J07', 'J14', 'J16', 'J19', 'J21', 'K02',
            'K06', 'K16', 'K22', 'K23', 'K24', 'L01', 'L02', 'L03', 'L04', 'L05',
            'L08', 'L10', 'L11', 'L12', 'L13', 'L16', 'L17', 'L18', 'M01', 'M02',
            'M04', 'M05', 'M08', 'M09', 'M12', 'M13', 'M14', 'M15', 'M19', 'M20',
            'M21', 'N06', 'N08', 'N11', 'N14', 'N18', 'N19', 'N21', 'N24', 'O01',
            'O03', 'O05', 'O07', 'O10', 'O11', 'O14', 'O15', 'O17', 'O19', 'O22',
            'O24', 'P01', 'P03', 'P06', 'P08', 'P11', 'P12', 'P13', 'P14', 'P15',
            'P16', 'P17', 'P18', 'P19', 'P21', 'P22'
        ]

        for well in self.all_wells:
            qc_vals_after = np.array(self.qc_vals_after[well])[0, :]
            result = hergqc.qc1(*qc_vals_after)

            pass_rseal_after, rseal_after = result[0]
            ex_pass_rseal_after = well not in failed_wells_rseal_after
            self.assertEqual(
                pass_rseal_after,
                ex_pass_rseal_after,
                f"QC1: {well} (rseal after) {rseal_after}",
            )

            pass_cm_after, cm_after = result[1]
            ex_pass_cm_after = well not in failed_wells_cm_after
            self.assertEqual(
                pass_cm_after,
                ex_pass_cm_after,
                f"QC1: {well} (cm after) {cm_after}",
            )

            pass_rseries_after, rseries_after = result[2]
            ex_pass_rseries_after = well not in failed_wells_rseries_after
            self.assertEqual(
                pass_rseries_after,
                ex_pass_rseries_after,
                f"QC1: {well} (rseries after) {rseries_after}",
            )

    def test_qc2(self):
        hergqc = self.clone_herg_qc("test_qc2")

        # qc2 checks that raw and subtracted SNR are above a minimum threshold
        test_matrix = [
            (10, True, 8082.1),
            (1, True, 74.0),
            (0.601, True, 25.1),
            (0.6, False, 25.0),
            (0.599, False, 24.9),
            (0.5, False, 16.8),
            (0.1, False, 0.5),
        ]

        for i, ex_pass, ex_snr in test_matrix:
            recording = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [i] * 500)
            pass_, snr = hergqc.qc2(recording)
            self.assertAlmostEqual(
                snr, ex_snr, 1, f"QC2: ({i}) {snr} != {ex_snr}"
            )
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
        hergqc = self.clone_herg_qc("test_qc3")

        # qc3 checks that rmsd of two sweeps are similar

        # Test with same noise, different signal
        test_matrix = [
            (-10, False, -2.5),
            (-9, False, -1.6),
            (-8, False, -0.7),
            (-7, True, 0.3),
            (0, True, 6.8),
            (8, True, 0.68),
            (9, False, -0.08),
            (10, False, -0.8),
        ]

        recording1 = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [40] * 500)
        for i, ex_pass, ex_d_rmsd in test_matrix:
            recording2 = np.asarray(
                [0, 0.1] * (NOISE_LEN // 2) + [40 + i] * 500
            )
            pass_, d_rmsd = hergqc.qc3(recording1, recording2)
            self.assertAlmostEqual(
                d_rmsd,
                ex_d_rmsd,
                1,
                f"QC3: ({i}) {d_rmsd} != {ex_d_rmsd}",
            )
            self.assertEqual(pass_, ex_pass, f"QC3: ({i}) {pass_} != {ex_pass}")

        # Test with same signal, different noise
        test_matrix = [
            (-100, True, 11.3),
            (-10, True, 6.3),
            (-1, True, 6.7),
            (0, True, 6.7),
            (1, True, 6.8),
            (10, True, 6.4),
            (100, True, 11.4),
        ]

        recording1 = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [40] * 500)
        for i, ex_pass, ex_d_rmsd in test_matrix:
            recording2 = np.asarray(
                [0, 0.1 * i] * (NOISE_LEN // 2) + [40] * 500
            )
            pass_, d_rmsd = hergqc.qc3(recording1, recording2)
            self.assertAlmostEqual(
                d_rmsd,
                ex_d_rmsd,
                1,
                f"QC3: ({i}) {d_rmsd} != {ex_d_rmsd}",
            )
            self.assertEqual(pass_, ex_pass, f"QC3: ({i}) {pass_} != {ex_pass}")

        # Test on data
        failed_wells_raw = [
            'A21', 'B05', 'B10', 'C19', 'E09', 'E19', 'F22', 'F23', 'I06', 'K23',
            'L09', 'M05', 'M06', 'M10', 'N12', 'N17', 'O13', 'O15', 'P11'
        ]

        failed_wells_E4031 = ['A05', 'A07', 'C19', 'E19', 'J16']

        failed_wells_subtracted = [
            'A05', 'A20', 'A21', 'A24', 'B05', 'B07', 'B10', 'B15', 'B21', 'B23',
            'C04', 'C12', 'C14', 'C17', 'C18', 'C19', 'C20', 'D21', 'E04', 'E09',
            'E10', 'E15', 'E16', 'E17', 'E18', 'E19', 'E23', 'F04', 'F06', 'F07',
            'F12', 'F20', 'F21', 'G08', 'G09', 'G10', 'G12', 'G13', 'G16', 'G20',
            'G23', 'G24', 'H03', 'H07', 'H15', 'H19', 'H21', 'H24', 'I04', 'I05',
            'I12', 'I17', 'I21', 'J07', 'J16', 'J19', 'K02', 'K06', 'K22', 'K23',
            'L01', 'L05', 'L08', 'L09', 'L10', 'L11', 'L13', 'L24', 'M01', 'M02',
            'M04', 'M05', 'M06', 'M10', 'M12', 'M21', 'N04', 'N06', 'N08', 'N11',
            'N12', 'N17', 'N20', 'N24', 'O01', 'O03', 'O07', 'O10', 'O13', 'O15',
            'O19', 'O22', 'P01', 'P03', 'P06', 'P08', 'P12', 'P15', 'P17', 'P18'
        ]

        for well in self.all_wells:
            before = np.array(self.trace_sweeps_before[well])
            after = np.array(self.trace_sweeps_after[well])

            pass_raw, d_rmsd_raw = hergqc.qc3(before[0, :], before[1, :])
            ex_pass_raw = well not in failed_wells_raw
            self.assertEqual(
                pass_raw,
                ex_pass_raw,
                f"QC3: {well} (raw) {d_rmsd_raw}",
            )

            pass_E4031, d_rmsd_E4031 = hergqc.qc3(after[0, :], after[1, :])
            ex_pass_E4031 = well not in failed_wells_E4031
            self.assertEqual(
                pass_E4031,
                ex_pass_E4031,
                f"QC3: {well} (E4031) {d_rmsd_E4031}",
            )

            pass_subtracted, d_rmsd_subtracted = hergqc.qc3(
                before[0, :] - after[0, :],
                before[1, :] - after[1, :],
            )
            ex_pass_subtracted = well not in failed_wells_subtracted
            self.assertEqual(
                pass_subtracted,
                ex_pass_subtracted,
                f"QC3: {well} (subtracted) {d_rmsd_subtracted}",
            )

    def test_qc4(self):
        hergqc = self.clone_herg_qc("test_qc4")

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

        for i, ex_pass in test_matrix:
            rseals = [r_lo, i * r_lo]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                ex_pass,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

            rseals = [r_hi, i * r_hi]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                ex_pass,
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

        for i, ex_pass in test_matrix:
            cms = [c_lo, i * c_lo]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                ex_pass,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

            cms = [c_hi, i * c_hi]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                ex_pass,
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

        for i, ex_pass in test_matrix:
            rseriess = [r_lo, i * r_lo]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                ex_pass,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

            rseriess = [r_hi, i * r_hi]
            self.assertEqual(
                all_passed(hergqc.qc4(rseals, cms, rseriess)),
                ex_pass,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

        # Test on data
        failed_wells_rseals = [
            'A04', 'A05', 'A07', 'A16', 'A21', 'A23', 'B02', 'B04', 'B11', 'B16',
            'C10', 'C19', 'C22', 'C23', 'D03', 'D23', 'E01', 'E02', 'E03', 'E07',
            'F23', 'H01', 'H09', 'H17', 'I06', 'I11', 'J11', 'K01', 'K09', 'K12',
            'K14', 'K23', 'M05', 'M10', 'N02', 'N09', 'N17', 'O08', 'O14', 'P16',
            'P24'
        ]

        failed_wells_cms = [
            'A12', 'A13', 'A16', 'A19', 'A20', 'B07', 'B11', 'B13', 'B15', 'B19',
            'B21', 'B23', 'C02', 'C04', 'C07', 'C11', 'C12', 'C14', 'C18', 'D03',
            'D10', 'D14', 'E03', 'E04', 'E07', 'E09', 'E10', 'E15', 'E16', 'E17',
            'E19', 'E22', 'E23', 'F01', 'F03', 'F04', 'F07', 'F12', 'F14', 'F15',
            'F18', 'F19', 'F20', 'F21', 'F24', 'G06', 'G09', 'G12', 'G13', 'G16',
            'G20', 'G23', 'G24', 'H01', 'H03', 'H06', 'H07', 'H09', 'H10', 'H15',
            'H19', 'H21', 'H23', 'H24', 'I04', 'I05', 'I07', 'I10', 'I12', 'I16',
            'I17', 'I21', 'J07', 'J16', 'J19', 'J21', 'K02', 'K16', 'K22', 'K23',
            'L01', 'L02', 'L05', 'L08', 'L10', 'L11', 'L13', 'L16', 'L17', 'L18',
            'M01', 'M04', 'M12', 'M15', 'M19', 'M21', 'N06', 'N08', 'N11', 'N14',
            'N18', 'N19', 'N21', 'N24', 'O01', 'O03', 'O07', 'O10', 'O15', 'O17',
            'O19', 'O22', 'O24', 'P01', 'P06', 'P07', 'P08', 'P12', 'P13', 'P14',
            'P15', 'P16', 'P17', 'P18', 'P21', 'P22'
        ]

        failed_wells_rseriess = [
            'A12', 'A13', 'A16', 'A19', 'A20', 'B07', 'B11', 'B13', 'B15', 'B19',
            'B21', 'B23', 'C02', 'C04', 'C07', 'C11', 'C12', 'C14', 'C18', 'C22',
            'D03', 'D10', 'D14', 'E01', 'E03', 'E04', 'E07', 'E09', 'E10', 'E15',
            'E16', 'E17', 'E22', 'E23', 'F01', 'F03', 'F04', 'F07', 'F12', 'F14',
            'F15', 'F18', 'F19', 'F20', 'F21', 'F24', 'G06', 'G09', 'G12', 'G13',
            'G14', 'G16', 'G20', 'G23', 'G24', 'H01', 'H03', 'H04', 'H07', 'H09',
            'H10', 'H13', 'H15', 'H19', 'H21', 'H23', 'H24', 'I04', 'I05', 'I06',
            'I07', 'I10', 'I12', 'I16', 'I17', 'I21', 'J07', 'J14', 'J16', 'J19',
            'J21', 'K02', 'K16', 'K22', 'K23', 'L01', 'L02', 'L05', 'L08', 'L10',
            'L11', 'L12', 'L13', 'L16', 'L17', 'L18', 'M01', 'M02', 'M04', 'M05',
            'M12', 'M13', 'M15', 'M19', 'M20', 'M21', 'N06', 'N08', 'N11', 'N14',
            'N18', 'N19', 'N21', 'N24', 'O01', 'O03', 'O07', 'O10', 'O14', 'O15',
            'O17', 'O19', 'O22', 'O24', 'P01', 'P06', 'P07', 'P08', 'P12', 'P13',
            'P14', 'P15', 'P16', 'P17', 'P18', 'P21', 'P22'
        ]

        for well in self.all_wells:
            qc_vals_before = np.array(self.qc_vals_before[well])[0, :]
            qc_vals_after = np.array(self.qc_vals_after[well])[0, :]

            rseals = [qc_vals_before[0], qc_vals_after[0]]
            cms = [qc_vals_before[1], qc_vals_after[1]]
            rseriess = [qc_vals_before[2], qc_vals_after[2]]
            result = hergqc.qc4(rseals, cms, rseriess)

            pass_rseals, d_rseal = result[0]
            ex_pass_rseals = well not in failed_wells_rseals
            self.assertEqual(
                pass_rseals,
                ex_pass_rseals,
                f"QC4: {well} (rseals) {d_rseal} {rseals}",
            )

            pass_cms, d_cm = result[1]
            ex_pass_cms = well not in failed_wells_cms
            self.assertEqual(
                pass_cms,
                ex_pass_cms,
                f"QC4: {well} (cms) {d_cm} {cms}",
            )

            pass_rseriess, d_rseries = result[2]
            ex_pass_rseriess = well not in failed_wells_rseriess
            self.assertEqual(
                pass_rseriess,
                ex_pass_rseriess,
                f"QC4: {well} (rseriess) {d_rseries} {rseriess}",
            )

    def test_qc5(self):
        hergqc = self.clone_herg_qc("test_qc5")

        # qc5 checks that the maximum current during the second half of the
        # staircase changes by at least 75% of the raw trace after E-4031 addition
        test_matrix = [
            (-1.0, True, -12.5),
            (0.1, True, -1.5),
            (0.2, True, -0.5),
            (0.24, True, -0.1),
            (0.25, True, 0),
            (0.26, False, 0.1),
            (0.3, False, 0.5),
            (0.5, False, 2.5),
            (1.0, False, 7.5),
        ]

        recording1 = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [10] * 500)
        for i, ex_pass, ex_d_max_diff in test_matrix:
            recording2 = np.asarray(
                [0, 0.1] * (NOISE_LEN // 2) + [10 * i] * 500
            )
            pass_, d_max_diff = hergqc.qc5(recording1, recording2)
            self.assertAlmostEqual(
                d_max_diff,
                ex_d_max_diff,
                1,
                f"QC5: ({i}) {d_max_diff} != {ex_d_max_diff}",
            )
            self.assertEqual(pass_, ex_pass, f"QC5: ({i}) {pass_} != {ex_pass}")

        # Test on data
        failed_wells = [
            'A10', 'A12', 'A13', 'A15', 'A19', 'A20', 'A24', 'B05', 'B07', 'B09',
            'B11', 'B12', 'B13', 'B15', 'B18', 'B19', 'B21', 'B23', 'C02', 'C04',
            'C05', 'C07', 'C08', 'C09', 'C11', 'C12', 'C14', 'C17', 'C18', 'C19',
            'C20', 'C21', 'C22', 'C24', 'D02', 'D05', 'D09', 'D10', 'D11', 'D12',
            'D14', 'D17', 'D18', 'D19', 'D21', 'E04', 'E10', 'E11', 'E13', 'E14',
            'E15', 'E16', 'E17', 'E18', 'E19', 'E22', 'E23', 'F01', 'F03', 'F04',
            'F06', 'F07', 'F09', 'F10', 'F12', 'F13', 'F14', 'F15', 'F16', 'F18',
            'F19', 'F20', 'F21', 'F22', 'F24', 'G03', 'G06', 'G08', 'G09', 'G12',
            'G13', 'G14', 'G15', 'G16', 'G18', 'G19', 'G20', 'G23', 'G24', 'H01',
            'H02', 'H03', 'H04', 'H06', 'H07', 'H08', 'H10', 'H11', 'H14', 'H15',
            'H16', 'H18', 'H19', 'H21', 'H23', 'H24', 'I01', 'I04', 'I05', 'I06',
            'I07', 'I08', 'I10', 'I11', 'I12', 'I13', 'I14', 'I16', 'I17', 'I18',
            'I21', 'J07', 'J10', 'J12', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20',
            'J21', 'J23', 'J24', 'K02', 'K03', 'K06', 'K07', 'K11', 'K12', 'K13',
            'K14', 'K16', 'K18', 'K20', 'K22', 'K23', 'K24', 'L01', 'L02', 'L03',
            'L04', 'L05', 'L07', 'L08', 'L10', 'L11', 'L12', 'L13', 'L16', 'L17',
            'L18', 'L24', 'M01', 'M02', 'M03', 'M04', 'M06', 'M08', 'M09', 'M11',
            'M12', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M21', 'N04', 'N06',
            'N07', 'N08', 'N11', 'N13', 'N14', 'N16', 'N18', 'N20', 'N21', 'N22',
            'N24', 'O01', 'O02', 'O03', 'O07', 'O08', 'O10', 'O12', 'O15', 'O16',
            'O17', 'O18', 'O19', 'O20', 'O21', 'O22', 'O24', 'P01', 'P03', 'P05',
            'P06', 'P07', 'P08', 'P09', 'P10', 'P12', 'P13', 'P14', 'P15', 'P17',
            'P18', 'P20', 'P21', 'P22', 'P24'
        ]

        for well in self.all_wells:
            before = np.array(self.trace_sweeps_before[well])
            after = np.array(self.trace_sweeps_after[well])

            pass_, d_max_diff = hergqc.qc5(
                before[0, :], after[0, :], hergqc.qc5_win
            )
            ex_pass = well not in failed_wells
            self.assertEqual(
                pass_,
                ex_pass,
                f"QC5: {well} {d_max_diff}",
            )

    def test_qc5_1(self):
        hergqc = self.clone_herg_qc("test_qc5_1")

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
        for i, ex_pass in test_matrix:
            recording2 = np.asarray(
                [0, 0.1] * (NOISE_LEN // 2) + [10 * i] * 500)
            result = hergqc.qc5_1(recording1, recording2)
            self.assertEqual(result[0], ex_pass, f"({i}: {result[1]})")

        # TODO: Test on select data

    def test_qc6(self):
        hergqc = self.clone_herg_qc("test_qc6")

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
        for i, ex_pass in test_matrix:
            recording = np.asarray(
                [0, 0.1] * (NOISE_LEN // 2) + [0.1 * i] * 500)
            result = hergqc.qc6(recording, win=[NOISE_LEN, -1])
            self.assertEqual(result[0], ex_pass, f"({i}: {result[1]})")

        # TODO: Test on select data

    def test_run_qc(self):
        # Spot check a few wells; could check all, but it's time consuming.
        hergqc = self.clone_herg_qc("test_run_qc")

        test_matrix = [
            ("A01", True),
            ("A02", True),
            ("A03", True),
            ("A04", False),
            ("A05", False),
            ("D01", False),
        ]

        for well, ex_pass in test_matrix:
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

                self.assertEqual(QC.all_passed(), ex_pass, trace)
