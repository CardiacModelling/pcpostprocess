import logging
import os

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

NOISE_LEN = 200

class QCDict:

    labels = [
        "qc1.rseal",
        "qc1.cm",
        "qc1.rseries",
        "qc2.raw",
        "qc2.subtracted",
        "qc3.raw",
        "qc3.E4031",
        "qc3.subtracted",
        "qc4.rseal",
        "qc4.cm",
        "qc4.rseries",
        "qc5.staircase",
        "qc5.1.staircase",
        "qc6.subtracted",
        "qc6.1.subtracted",
        "qc6.2.subtracted",
    ]

    def __init__(self):
        self._dict = OrderedDict([(label, [(False, None)]) for label in QCDict.labels])

    def __str__(self):
        return self._dict.__str__()

    def __repr__(self):
        return self._dict.__repr__()

    def __getitem__(self, key):
        return self._dict.__getitem__(key)

    def __setitem__(self, key, value):
        if key not in QCDict.labels:
            raise KeyError(f"Invalid QC key: {key}")
        self._dict.__setitem__(key, value)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def qc_passed(self, label):
        """Return whether a single QC passed."""
        return all([x for x, _ in self._dict[label]])

    def passed_list(self):
        """Return a list of booleans indicating whether each QC passed."""
        return [self.qc_passed(label) for label in QCDict.labels]

    def all_passed(self):
        """Return whether all QC passed."""
        return all(self.passed_list())


class hERGQC:

    def __init__(self, sampling_rate=5, plot_dir=None, voltage=np.array([]),
                 n_sweeps=None, removal_time=5):
        # TODO docstring

        self._plot_dir = plot_dir

        self._n_qc = 16

        self.removal_time = removal_time

        self.voltage = np.array(voltage)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.sampling_rate = sampling_rate

        # Define all thresholds

        # qc1
        self.rsealc = [1e8, 1e12]  # in Ohm, converted from [0.1, 1000] GOhm
        self.cmc = [1e-12, 1e-10]  # in F, converted from [1, 100] pF
        self.rseriesc = [1e6, 2.5e7]  # in Ohm, converted from [1, 25] MOhm
        # qc2
        self.snrc = 25
        # qc3
        self.rmsd0c = 0.2
        # qc4
        self.rsealsc = 0.5
        self.cmsc = 0.5
        self.rseriessc = 0.5
        # qc5
        self.max_diffc = 0.75
        # self.qc5_win = [3275, 5750]  # indices with hERG screening peak!
        # indices where hERG could peak (for different temperatures)
        self.qc5_win = np.array([8550 + 400, 10950 + 400]) * sampling_rate
        # qc5_1
        self.rmsd0_diffc = 0.5
        # qc6
        self.negative_tolc = -2
        ''' # These are for `whole` (just staircase) protocol at 5kHz
        self.qc6_win = [3000, 7000]  # indices for first +40 mV
        self.qc6_1_win = [35250, 37250]  # indices for second +40 mV
        self.qc6_2_win = [40250, 42250]  # indices for third +40 mV
        '''
        # These are for `staircaseramp` protocol

        # Firstly, indices for 1st +40 mV
        self.qc6_win = np.array([1000, 1800]) * sampling_rate
        # indices for 2nd +40 mV
        self.qc6_1_win = np.array([7450, 7850]) * sampling_rate
        # indices for 3rd +40 mV
        self.qc6_2_win = np.array([8450, 8850]) * sampling_rate

        # Ensure these indices are integers
        self.qc5_win = self.qc5_win.astype(int)
        self.qc6_win = self.qc6_win.astype(int)
        self.qc6_1_win = self.qc6_1_win.astype(int)
        self.qc6_2_win = self.qc6_2_win.astype(int)

        self._debug = True

    @property
    def plot_dir(self):
        return self._plot_dir

    @plot_dir.setter
    def plot_dir(self, path):
        self._plot_dir = path

    def set_trace(self, before, after, qc_vals_before, qc_vals_after, n_sweeps):
        self._before = before
        self._qc_vals_before = qc_vals_before
        self._after = after
        self._qc_vals_after = qc_vals_after

    def set_debug(self, debug):
        self._debug = debug

    def run_qc(self, voltage_steps, times,
               before, after, qc_vals_before,
               qc_vals_after, n_sweeps=None):
        """Run each QC criteria on a single (before-trace, after-trace) pair.

        @param voltage_steps is a list of times at which there are discontinuities in Vcmd
        @param times is the array of observation times
        @param before is the before-drug current trace
        @param after is the post-drug current trace
        @param qc_vals_before is an array of values for each pre-drug sweep where each row is (Rseal, Cm, Rseries)
        @param qc_vals_after is an array of values for each post-drug sweep where each row is (Rseal, Cm, Rseries)
        @n_sweeps is the number of sweeps to be considered
        """

        if not n_sweeps:
            n_sweeps = len(before)

        before = np.array(before)
        after = np.array(after)

        before = self.filter_capacitive_spikes(before, times, voltage_steps)
        after = self.filter_capacitive_spikes(after, times, voltage_steps)

        QC = QCDict()

        if len(before) == 0 or len(after) == 0:
            return QC

        if (None in qc_vals_before) or (None in qc_vals_after):
            return QC

        qc1_1 = self.qc1(*qc_vals_before)
        qc1_2 = self.qc1(*qc_vals_after)

        QC['qc1.rseal'] = [qc1_1[0], qc1_2[0]]
        QC['qc1.cm'] = [qc1_1[1], qc1_2[1]]
        QC['qc1.rseries'] = [qc1_1[2], qc1_2[2]]

        QC['qc2.raw'] = []
        QC['qc2.subtracted'] = []
        for i in range(n_sweeps):
            qc2_1 = self.qc2(before[i])
            qc2_2 = self.qc2(before[i] - after[i])

            QC['qc2.raw'].append(qc2_1)
            QC['qc2.subtracted'].append(qc2_2)

        qc3_1 = self.qc3(before[0, :], before[1, :])
        qc3_2 = self.qc3(after[0, :], after[1, :])
        qc3_3 = self.qc3(before[0, :] - after[0, :],
                         before[1, :] - after[1, :])

        QC['qc3.raw'] = [qc3_1]
        QC['qc3.E4031'] = [qc3_2]
        QC['qc3.subtracted'] = [qc3_3]

        rseals = [qc_vals_before[0], qc_vals_after[0]]
        cms = [qc_vals_before[1], qc_vals_after[1]]
        rseriess = [qc_vals_before[2], qc_vals_after[2]]
        qc4 = self.qc4(rseals, cms, rseriess)

        QC['qc4.rseal'] = [qc4[0]]
        QC['qc4.cm'] = [qc4[1]]
        QC['qc4.rseries'] = [qc4[2]]

        # indices where hERG peaks
        qc5 = self.qc5(before[0, :], after[0, :],
                       self.qc5_win)

        qc5_1 = self.qc5_1(before[0, :], after[0, :], label='1')

        QC['qc5.staircase'] = [qc5]
        QC['qc5.1.staircase'] = [qc5_1]

        # Ensure thats the windows are correct by checking the voltage trace
        assert np.all(
            np.abs(self.voltage[self.qc6_win[0]: self.qc6_win[1]] - 40.0))\
            < 1e-8
        assert np.all(
            np.abs(self.voltage[self.qc6_1_win[0]: self.qc6_1_win[1]] - 40.0))\
            < 1e-8
        assert np.all(
            np.abs(self.voltage[self.qc6_2_win[0]: self.qc6_2_win[1]] - 40.0))\
            < 1e-8

        QC['qc6.subtracted'] = []
        QC['qc6.1.subtracted'] = []
        QC['qc6.2.subtracted'] = []
        for i in range(before.shape[0]):
            qc6 = self.qc6((before[i, :] - after[i, :]), self.qc6_win, label="0")
            qc6_1 = self.qc6((before[i, :] - after[i, :]), self.qc6_1_win, label="1")
            qc6_2 = self.qc6((before[i, :] - after[i, :]), self.qc6_2_win, label="2")

            QC['qc6.subtracted'].append(qc6)
            QC['qc6.1.subtracted'].append(qc6_1)
            QC['qc6.2.subtracted'].append(qc6_2)

        if self.plot_dir and self._debug:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.subplots()
            ax.plot(times, (before - after).T, label='subtracted')
            ax.plot(times, (before).T, label='before')
            ax.plot(times, (after).T, label='after')
            for l1, l2, l3, l4 in zip(self.qc5_win, self.qc6_win,
                                      self.qc6_1_win, self.qc6_2_win):
                plt.axvline(times[l1], c='#7f7f7f', label='qc5')
                plt.axvline(times[l2], c='#ff7f0e', ls='--', label='qc6')
                plt.axvline(times[l3], c='#2ca02c', ls='-.', label='qc6_1')
                plt.axvline(times[l4], c='#9467bd', ls=':', label='qc6_2')
            plt.xlabel('Time index (sample)')
            plt.ylabel('Current [pA]')

            # https://stackoverflow.com/a/13589144
            # fix legend labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys())

            fig.savefig(os.path.join(self.plot_dir, 'qc_debug.png'))
            plt.close(fig)

        return QC

    def qc1(self, rseal, cm, rseries):
        # Check R_seal, C_m, R_series within desired range
        if (
            rseal is None
            or rseal < self.rsealc[0]
            or rseal > self.rsealc[1]
            or not np.isfinite(rseal)
        ):
            self.logger.debug(f"rseal: {rseal}")
            qc11 = False
        else:
            qc11 = True

        if (
            cm is None
            or cm < self.cmc[0]
            or cm > self.cmc[1]
            or not np.isfinite(cm)
        ):
            self.logger.debug(f"cm: {cm}")
            qc12 = False
        else:
            qc12 = True

        if (
            rseries is None
            or rseries < self.rseriesc[0]
            or rseries > self.rseriesc[1]
            or not np.isfinite(rseries)
        ):
            self.logger.debug(f"rseries: {rseries}")
            qc13 = False
        else:
            qc13 = True

        return [(qc11, rseal), (qc12, cm), (qc13, rseries)]

    def qc2(self, recording, method=3):
        # Check SNR is good
        if method == 1:
            # Not sure if this is good...
            snr = scipy.stats.signaltonoise(recording)
        elif method == 2:
            noise = np.std(recording[:NOISE_LEN])
            snr = (np.max(recording) - np.min(recording) - 2 * noise) / noise
        elif method == 3:
            noise = np.std(recording[:NOISE_LEN])
            snr = (np.std(recording) / noise) ** 2

        if snr < self.snrc or not np.isfinite(snr):
            self.logger.debug(f"snr: {snr}")
            result = False
        else:
            result = True

        return (result, snr)

    def qc3(self, recording1, recording2, method=3):
        # Check 2 sweeps similar
        if method == 1:
            rmsdc = 2  # A/F * F
        elif method == 2:
            noise_1 = np.std(recording1[:NOISE_LEN])
            peak_1 = (np.max(recording1) - noise_1)
            noise_2 = np.std(recording2[:NOISE_LEN])
            peak_2 = (np.max(recording2) - noise_2)
            rmsdc = max(np.mean([peak_1, peak_2]) * 0.1,
                        np.mean([noise_1, noise_2]) * 5)
        elif method == 3:
            noise_1 = np.std(recording1[:NOISE_LEN])
            noise_2 = np.std(recording2[:NOISE_LEN])
            rmsd0_1 = np.sqrt(np.mean((recording1) ** 2))
            rmsd0_2 = np.sqrt(np.mean((recording2) ** 2))
            rmsdc = max(np.mean([rmsd0_1, rmsd0_2]) * self.rmsd0c,
                        np.mean([noise_1, noise_2]) * 6)
        rmsd = np.sqrt(np.mean((recording1 - recording2) ** 2))
        if rmsd > rmsdc or not (np.isfinite(rmsd) and np.isfinite(rmsdc)):
            self.logger.debug(f"rmsd: {rmsd}, rmsdc: {rmsdc}")
            result = False
        else:
            result = True

        return (result, rmsdc - rmsd)

    def qc4(self, rseals, cms, rseriess):
        # Check R_seal, C_m, R_series stability
        # Require std/mean < x%
        qc41 = True
        qc42 = True
        qc43 = True

        if None in list(rseals):
            qc41 = False
            d_rseal = None
        else:
            d_rseal = np.std(rseals) / np.mean(rseals)
            if d_rseal > self.rsealsc or not (
                    np.isfinite(np.mean(rseals)) and np.isfinite(np.std(rseals))):
                self.logger.debug(f"d_rseal:  {d_rseal}")
                qc41 = False

        if None in list(cms):
            qc42 = False
            d_cm = None
        else:
            d_cm = np.std(cms) / np.mean(cms)
            if d_cm > self.cmsc or not (
                    np.isfinite(np.mean(cms)) and np.isfinite(np.std(cms))):
                self.logger.debug(f"d_cm: {d_cm}")
                qc42 = False

        if None in list(rseriess):
            qc43 = False
            d_rseries = None
        else:
            d_rseries = np.std(rseriess) / np.mean(rseriess)
            if d_rseries > self.rseriessc or not (
                    np.isfinite(np.mean(rseriess))
                    and np.isfinite(np.std(rseriess))):
                self.logger.debug(f"d_rseries: {d_rseries}")
                qc43 = False

        return [(qc41, d_rseal), (qc42, d_cm), (qc43, d_rseries)]

    def qc5(self, recording1, recording2, win=None, label=''):
        # Check pharma peak value drops after E-4031 application
        # Require subtracted peak > 70% of the original peak
        if win is not None:
            i, f = win
        else:
            i, f = 0, None

        if self.plot_dir and self._debug:
            if win is not None:
                plt.axvspan(win[0], win[1], color='grey', alpha=.1)
            plt.plot(recording1, label='recording1')
            plt.plot(recording2, label='recording2')
            plt.savefig(os.path.join(self.plot_dir, f"qc5_{label}"))
            plt.clf()

        wherepeak = np.argmax(recording1[i:f])
        max_diff = recording1[i:f][wherepeak] - recording2[i:f][wherepeak]
        max_diffc = self.max_diffc * recording1[i:f][wherepeak]

        if (max_diff < max_diffc) or not (np.isfinite(max_diff)
                                          and np.isfinite(max_diffc)):
            self.logger.debug(f"max_diff:  {max_diff}, max_diffc: {max_diffc}")
            result = False
        else:
            result = True

        return (result, max_diffc - max_diff)

    def qc5_1(self, recording1, recording2, win=None, label=''):
        # Check RMSD_0 drops after E-4031 application
        # Require RMSD_0 (after E-4031 / before) diff > 50% of RMSD_0 before
        if win is not None:
            i, f = win
        else:
            i, f = 0, -1

        if self.plot_dir and self._debug:
            if win is not None:
                plt.axvspan(win[0], win[1], color='grey', alpha=.1)
            fig = plt.figure()
            ax = fig.subplots()
            ax.plot(recording1, label='recording1')
            ax.plot(recording2, label='recording2')
            fig.savefig(os.path.join(self.plot_dir, f"qc5_{label}"))
            plt.close(fig)

        rmsd0_diff = np.sqrt(np.mean(recording1[i:f] ** 2)) \
            - np.sqrt(np.mean(recording2[i:f] ** 2))

        rmsd0_diffc = self.rmsd0_diffc *\
            np.sqrt(np.mean(recording1[i:f] ** 2))

        if (rmsd0_diff < rmsd0_diffc) or not (np.isfinite(rmsd0_diff)
                                              and np.isfinite(rmsd0_diffc)):
            self.logger.debug(f"rmsd0_diff: {rmsd0_diff}, rmsd0c: {rmsd0_diffc}")
            result = False
        else:
            result = True

        return (result, rmsd0_diffc - rmsd0_diff)

    def qc6(self, recording1, win=None, label=''):
        # Check subtracted staircase +40mV step up is non negative
        if win is not None:
            i, f = win
        else:
            i, f = 0, -1

        if self.plot_dir and self._debug:
            if win is not None:
                plt.axvspan(win[0], win[1], color='grey', alpha=.1)
            plt.plot(recording1, label='recording1')
            plt.savefig(os.path.join(self.plot_dir, f"qc6_{label}"))
            plt.clf()

        val = np.mean(recording1[i:f])

        # valc = -0.005 * np.abs(np.sqrt(np.mean((recording1) ** 2)))  # or just 0
        valc = self.negative_tolc * np.std(recording1[:NOISE_LEN])
        if (val < valc) or not (np.isfinite(val)
                                and np.isfinite(valc)):
            self.logger.debug(f"qc6_{label} val:  {val}, valc: {valc}")
            result = False
        else:
            result = True

        return (result, valc - val)

    def filter_capacitive_spikes(self, current, times, voltage_step_times):
        """
        Set values to 0 where they lie less that self.removal time after a change in voltage

        @param current: The observed current
        @param times: the times at which the current was observed
        @param voltage_step_times: the times at which there are discontinuities in Vcmd
        """

        for tstart, tend in zip(voltage_step_times,
                                np.append(voltage_step_times[1:], np.inf)):
            win_end = tstart + self.removal_time
            win_end = min(tend, win_end)
            i_start = np.argmax(times >= tstart)
            i_end = np.argmax(times > win_end)

            if i_end == 0:
                break
            if len(current.shape) == 2:
                current[:, i_start:i_end] = 0
            elif len(current.shape) == 1:
                current[i_start:i_end] = 0
            else:
                raise ValueError("Current must have 1 or 2 dimensions")

        return current
