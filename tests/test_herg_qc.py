import copy
import logging
import os
import unittest

import numpy as np
from syncropatch_export.trace import Trace

from pcpostprocess.hergQC import NOISE_LEN, hERGQC


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
        def passed(result):
            return all([x for x, _ in result])

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
                passed(hergqc.qc1(rseal, cm, rseries)),
                expected,
                f"QC1: {rseal}, {cm}, {rseries}",
            )

        # Test on data
        test_wells_before = {
            'A01': True, 'A02': True, 'A03': True, 'A04': True, 'A05': True,
            'A06': True, 'A07': True, 'A08': True, 'A09': True, 'A10': False,
            'A11': True, 'A12': False, 'A13': False, 'A14': True, 'A15': True,
            'A16': False, 'A17': True, 'A18': True, 'A19': False, 'A20': False,
            'A21': True, 'A22': True, 'A23': True, 'A24': False, 'B01': True,
            'B02': True, 'B03': True, 'B04': True, 'B05': False, 'B06': True,
            'B07': False, 'B08': True, 'B09': True, 'B10': True, 'B11': False,
            'B12': False, 'B13': False, 'B14': True, 'B15': False, 'B16': True,
            'B17': True, 'B18': True, 'B19': False, 'B20': True, 'B21': False,
            'B22': True, 'B23': False, 'B24': True, 'C01': True, 'C02': False,
            'C03': True, 'C04': False, 'C05': True, 'C06': True, 'C07': False,
            'C08': True, 'C09': False, 'C10': True, 'C11': False, 'C12': False,
            'C13': True, 'C14': False, 'C15': True, 'C16': True, 'C17': True,
            'C18': False, 'C19': False, 'C20': False, 'C21': True, 'C22': True,
            'C23': False, 'C24': True, 'D01': True, 'D02': False, 'D03': False,
            'D04': True, 'D05': False, 'D06': True, 'D07': True, 'D08': True,
            'D09': False, 'D10': False, 'D11': True, 'D12': True, 'D13': True,
            'D14': False, 'D15': False, 'D16': False, 'D17': True, 'D18': True,
            'D19': False, 'D20': True, 'D21': False, 'D22': True, 'D23': True,
            'D24': True, 'E01': True, 'E02': True, 'E03': True, 'E04': False,
            'E05': True, 'E06': False, 'E07': False, 'E08': True, 'E09': True,
            'E10': False, 'E11': False, 'E12': True, 'E13': True, 'E14': False,
            'E15': False, 'E16': False, 'E17': False, 'E18': True, 'E19': False,
            'E20': True, 'E21': True, 'E22': False, 'E23': False, 'E24': True,
            'F01': False, 'F02': True, 'F03': False, 'F04': False, 'F05': False,
            'F06': True, 'F07': False, 'F08': True, 'F09': False, 'F10': True,
            'F11': True, 'F12': False, 'F13': False, 'F14': False, 'F15': False,
            'F16': True, 'F17': True, 'F18': False, 'F19': False, 'F20': False,
            'F21': False, 'F22': True, 'F23': True, 'F24': False, 'G01': True,
            'G02': True, 'G03': True, 'G04': True, 'G05': True, 'G06': False,
            'G07': True, 'G08': True, 'G09': False, 'G10': True, 'G11': True,
            'G12': False, 'G13': False, 'G14': False, 'G15': True, 'G16': False,
            'G17': False, 'G18': True, 'G19': True, 'G20': False, 'G21': False,
            'G22': True, 'G23': False, 'G24': False, 'H01': False, 'H02': False,
            'H03': False, 'H04': True, 'H05': True, 'H06': False, 'H07': False,
            'H08': False, 'H09': True, 'H10': False, 'H11': False, 'H12': True,
            'H13': False, 'H14': False, 'H15': False, 'H16': False, 'H17': True,
            'H18': True, 'H19': False, 'H20': True, 'H21': False, 'H22': True,
            'H23': False, 'H24': False, 'I01': False, 'I02': True, 'I03': True,
            'I04': False, 'I05': False, 'I06': False, 'I07': False, 'I08': False,
            'I09': True, 'I10': False, 'I11': False, 'I12': False, 'I13': True,
            'I14': True, 'I15': True, 'I16': False, 'I17': False, 'I18': True,
            'I19': True, 'I20': True, 'I21': False, 'I22': True, 'I23': True,
            'I24': True, 'J01': True, 'J02': True, 'J03': True, 'J04': True,
            'J05': True, 'J06': True, 'J07': False, 'J08': True, 'J09': True,
            'J10': False, 'J11': True, 'J12': True, 'J13': True, 'J14': True,
            'J15': True, 'J16': False, 'J17': False, 'J18': True, 'J19': False,
            'J20': True, 'J21': False, 'J22': True, 'J23': True, 'J24': False,
            'K01': True, 'K02': False, 'K03': False, 'K04': True, 'K05': True,
            'K06': False, 'K07': False, 'K08': True, 'K09': True, 'K10': True,
            'K11': False, 'K12': False, 'K13': True, 'K14': True, 'K15': True,
            'K16': False, 'K17': False, 'K18': True, 'K19': True, 'K20': False,
            'K21': True, 'K22': False, 'K23': True, 'K24': False, 'L01': False,
            'L02': False, 'L03': True, 'L04': False, 'L05': False, 'L06': True,
            'L07': True, 'L08': False, 'L09': True, 'L10': False, 'L11': False,
            'L12': True, 'L13': False, 'L14': True, 'L15': True, 'L16': False,
            'L17': False, 'L18': False, 'L19': True, 'L20': True, 'L21': True,
            'L22': True, 'L23': True, 'L24': False, 'M01': False, 'M02': True,
            'M03': True, 'M04': False, 'M05': True, 'M06': False, 'M07': True,
            'M08': True, 'M09': False, 'M10': True, 'M11': True, 'M12': False,
            'M13': True, 'M14': False, 'M15': False, 'M16': False, 'M17': True,
            'M18': True, 'M19': False, 'M20': False, 'M21': False, 'M22': True,
            'M23': True, 'M24': True, 'N01': True, 'N02': True, 'N03': False,
            'N04': False, 'N05': True, 'N06': False, 'N07': True, 'N08': False,
            'N09': True, 'N10': True, 'N11': False, 'N12': True, 'N13': False,
            'N14': False, 'N15': True, 'N16': False, 'N17': True, 'N18': False,
            'N19': True, 'N20': True, 'N21': False, 'N22': True, 'N23': True,
            'N24': False, 'O01': False, 'O02': False, 'O03': False, 'O04': True,
            'O05': False, 'O06': True, 'O07': False, 'O08': True, 'O09': True,
            'O10': False, 'O11': False, 'O12': True, 'O13': True, 'O14': True,
            'O15': True, 'O16': True, 'O17': False, 'O18': True, 'O19': False,
            'O20': True, 'O21': True, 'O22': False, 'O23': True, 'O24': False,
            'P01': False, 'P02': True, 'P03': False, 'P04': True, 'P05': True,
            'P06': False, 'P07': False, 'P08': False, 'P09': False, 'P10': True,
            'P11': True, 'P12': False, 'P13': False, 'P14': False, 'P15': False,
            'P16': False, 'P17': False, 'P18': False, 'P19': True, 'P20': True,
            'P21': False, 'P22': False, 'P23': True, 'P24': False
        }

        for well in test_wells_before:
            qc_vals_before = np.array(self.qc_vals_before[well])[0, :]
            self.assertEqual(
                passed(hergqc.qc1(*qc_vals_before)),
                test_wells_before[well],
                f"QC1: {well} (before) {qc_vals_before}",
            )

        test_wells_after = {
            'A01': True, 'A02': True, 'A03': True, 'A04': True, 'A05': True,
            'A06': False, 'A07': True, 'A08': False, 'A09': True, 'A10': False,
            'A11': True, 'A12': False, 'A13': False, 'A14': True, 'A15': True,
            'A16': False, 'A17': True, 'A18': True, 'A19': False, 'A20': False,
            'A21': True, 'A22': True, 'A23': True, 'A24': False, 'B01': True,
            'B02': False, 'B03': True, 'B04': True, 'B05': False, 'B06': True,
            'B07': False, 'B08': True, 'B09': True, 'B10': True, 'B11': False,
            'B12': False, 'B13': False, 'B14': True, 'B15': False, 'B16': True,
            'B17': True, 'B18': True, 'B19': False, 'B20': True, 'B21': False,
            'B22': True, 'B23': False, 'B24': True, 'C01': False, 'C02': False,
            'C03': True, 'C04': False, 'C05': True, 'C06': True, 'C07': False,
            'C08': True, 'C09': False, 'C10': True, 'C11': False, 'C12': False,
            'C13': True, 'C14': False, 'C15': True, 'C16': True, 'C17': True,
            'C18': False, 'C19': True, 'C20': False, 'C21': True, 'C22': False,
            'C23': False, 'C24': True, 'D01': True, 'D02': True, 'D03': False,
            'D04': True, 'D05': False, 'D06': True, 'D07': True, 'D08': True,
            'D09': False, 'D10': False, 'D11': True, 'D12': True, 'D13': True,
            'D14': False, 'D15': False, 'D16': True, 'D17': True, 'D18': True,
            'D19': False, 'D20': True, 'D21': False, 'D22': True, 'D23': True,
            'D24': True, 'E01': False, 'E02': True, 'E03': False, 'E04': False,
            'E05': True, 'E06': False, 'E07': False, 'E08': True, 'E09': False,
            'E10': False, 'E11': False, 'E12': True, 'E13': True, 'E14': False,
            'E15': False, 'E16': False, 'E17': False, 'E18': True, 'E19': False,
            'E20': False, 'E21': True, 'E22': False, 'E23': False, 'E24': False,
            'F01': False, 'F02': True, 'F03': False, 'F04': False, 'F05': True,
            'F06': True, 'F07': False, 'F08': True, 'F09': False, 'F10': True,
            'F11': True, 'F12': False, 'F13': False, 'F14': False, 'F15': False,
            'F16': False, 'F17': True, 'F18': False, 'F19': False, 'F20': False,
            'F21': False, 'F22': True, 'F23': True, 'F24': False, 'G01': True,
            'G02': True, 'G03': True, 'G04': True, 'G05': True, 'G06': False,
            'G07': True, 'G08': False, 'G09': False, 'G10': True, 'G11': True,
            'G12': False, 'G13': False, 'G14': False, 'G15': False, 'G16': False,
            'G17': False, 'G18': True, 'G19': True, 'G20': False, 'G21': False,
            'G22': True, 'G23': False, 'G24': False, 'H01': False, 'H02': False,
            'H03': False, 'H04': False, 'H05': True, 'H06': False, 'H07': False,
            'H08': False, 'H09': False, 'H10': False, 'H11': False, 'H12': True,
            'H13': False, 'H14': False, 'H15': False, 'H16': False, 'H17': False,
            'H18': False, 'H19': False, 'H20': False, 'H21': False, 'H22': True,
            'H23': False, 'H24': False, 'I01': False, 'I02': True, 'I03': True,
            'I04': False, 'I05': False, 'I06': False, 'I07': False, 'I08': False,
            'I09': True, 'I10': False, 'I11': True, 'I12': False, 'I13': True,
            'I14': True, 'I15': True, 'I16': False, 'I17': False, 'I18': True,
            'I19': True, 'I20': False, 'I21': False, 'I22': True, 'I23': True,
            'I24': True, 'J01': True, 'J02': True, 'J03': False, 'J04': True,
            'J05': True, 'J06': True, 'J07': False, 'J08': True, 'J09': True,
            'J10': False, 'J11': True, 'J12': True, 'J13': True, 'J14': False,
            'J15': True, 'J16': False, 'J17': False, 'J18': True, 'J19': False,
            'J20': True, 'J21': False, 'J22': True, 'J23': False, 'J24': False,
            'K01': True, 'K02': False, 'K03': False, 'K04': True, 'K05': True,
            'K06': False, 'K07': False, 'K08': True, 'K09': True, 'K10': True,
            'K11': True, 'K12': True, 'K13': True, 'K14': False, 'K15': True,
            'K16': False, 'K17': True, 'K18': True, 'K19': True, 'K20': False,
            'K21': True, 'K22': False, 'K23': False, 'K24': False, 'L01': False,
            'L02': False, 'L03': False, 'L04': False, 'L05': False, 'L06': True,
            'L07': True, 'L08': False, 'L09': True, 'L10': False, 'L11': False,
            'L12': False, 'L13': False, 'L14': True, 'L15': True, 'L16': False,
            'L17': False, 'L18': False, 'L19': True, 'L20': True, 'L21': True,
            'L22': True, 'L23': True, 'L24': False, 'M01': False, 'M02': False,
            'M03': True, 'M04': False, 'M05': False, 'M06': False, 'M07': True,
            'M08': False, 'M09': False, 'M10': True, 'M11': True, 'M12': False,
            'M13': False, 'M14': False, 'M15': False, 'M16': True, 'M17': True,
            'M18': True, 'M19': False, 'M20': False, 'M21': False, 'M22': True,
            'M23': True, 'M24': True, 'N01': True, 'N02': True, 'N03': True,
            'N04': False, 'N05': True, 'N06': False, 'N07': True, 'N08': False,
            'N09': True, 'N10': True, 'N11': False, 'N12': True, 'N13': False,
            'N14': False, 'N15': True, 'N16': False, 'N17': True, 'N18': False,
            'N19': False, 'N20': True, 'N21': False, 'N22': True, 'N23': True,
            'N24': False, 'O01': False, 'O02': False, 'O03': False, 'O04': True,
            'O05': False, 'O06': True, 'O07': False, 'O08': False, 'O09': True,
            'O10': False, 'O11': False, 'O12': True, 'O13': True, 'O14': False,
            'O15': False, 'O16': True, 'O17': False, 'O18': True, 'O19': False,
            'O20': True, 'O21': True, 'O22': False, 'O23': True, 'O24': False,
            'P01': False, 'P02': True, 'P03': False, 'P04': True, 'P05': True,
            'P06': False, 'P07': False, 'P08': False, 'P09': False, 'P10': True,
            'P11': False, 'P12': False, 'P13': False, 'P14': False, 'P15': False,
            'P16': False, 'P17': False, 'P18': False, 'P19': False, 'P20': True,
            'P21': False, 'P22': False, 'P23': True, 'P24': True
        }

        for well in test_wells_after:
            qc_vals_after = np.array(self.qc_vals_after[well])[0, :]
            self.assertEqual(
                passed(hergqc.qc1(*qc_vals_after)),
                test_wells_after[well],
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
            (10, True),  # snr = 8082
            (1, True),  # snr = 74
            (0.601, True),  # snr = 25.07
            (0.6, False),  # snr = 24.98
            (0.5, False),  # snr = 17
            (0.1, False),  # snr = 0.5
        ]

        for i, expected in test_matrix:
            recording = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [i] * 500)
            result = hergqc.qc2(recording)
            self.assertEqual(result[0], expected, f"({i}: {result[1]})")

        # TODO: Test on select data

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
            recording2 = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [40 + i] * 500)
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
            recording2 = np.asarray([0, 0.1 * i] * (NOISE_LEN // 2) + [40] * 500)
            result = hergqc.qc3(recording1, recording2)
            self.assertEqual(result[0], expected, f"({i}: {result[1]})")

        # TODO: Test on select data

    def test_qc4(self):
        def passed(result):
            return all([x for x, _ in result])

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
                passed(hergqc.qc4(rseals, cms, rseriess)),
                expected,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

            rseals = [r_hi, i * r_hi]
            self.assertEqual(
                passed(hergqc.qc4(rseals, cms, rseriess)),
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
                passed(hergqc.qc4(rseals, cms, rseriess)),
                expected,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

            cms = [c_hi, i * c_hi]
            self.assertEqual(
                passed(hergqc.qc4(rseals, cms, rseriess)),
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
                passed(hergqc.qc4(rseals, cms, rseriess)),
                expected,
                f"({i}: {rseals}, {cms}, {rseriess})",
            )

            rseriess = [r_hi, i * r_hi]
            self.assertEqual(
                passed(hergqc.qc4(rseals, cms, rseriess)),
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
            recording2 = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [10 * i] * 500)
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
            recording2 = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [10 * i] * 500)
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
            recording = np.asarray([0, 0.1] * (NOISE_LEN // 2) + [0.1 * i] * 500)
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
                    n_sweeps=2,
                )

                logging.debug(well, QC.all_passed())

                trace = ""
                for label, result in QC.items():
                    if not QC.qc_passed(label):
                        trace += f"{well} {label}: {result}\n"

                self.assertEqual(QC.all_passed(), expected, trace)
