import logging
import os
from collections import OrderedDict, UserDict

import matplotlib.pyplot as plt
import numpy as np


class QCDict(UserDict):
    """
    Stores the results from QC checks.

    Each entry is a ``label -> [(bool, value),...]`` mapping.
    The bool in each tuple indicates whether the QC passed,
    and the value is the result that was checked (e.g. the SNR value).
    The list can contain multiple tuples if the QC checks multiple values, as
    QC1 does, or if the checks are run multiple times e.g. once per sweep.
    """

    labels = (
        'qc1.rseal',        # [before, after]
        'qc1.cm',           # [before, after]
        'qc1.rseries',      # [before, after]
        'qc2.raw',          # [sweep[0], sweep[1], ...]
        'qc2.subtracted',   # [sweep[0], sweep[1], ...]
        'qc3.raw',
        'qc3.E4031',
        'qc3.subtracted',
        'qc4.rseal',
        'qc4.cm',
        'qc4.rseries',
        'qc5.staircase',
        'qc5.1.staircase',
        'qc6.subtracted',
        'qc6.1.subtracted',
        'qc6.2.subtracted',
    )

    def __init__(self):
        super().__init__([(label, [(False, None)]) for label in QCDict.labels])

    def __setitem__(self, key, value):
        if key not in QCDict.labels:
            raise KeyError(f'Invalid QC key: {key}')
        super().__setitem__(key, value)

    def qc_passed(self, label):
        """Return whether a single QC passed."""
        return all(x for x, _ in self[label])

    def passed_list(self):
        """Return a list of booleans indicating whether each QC passed."""
        return [all(x[0] for x in tuples) for tuples in self.values()]

    def all_passed(self):
        """Return whether all QC passed."""
        return all(all(x[0] for x in tuples) for tuples in self.values())


class hERGQC:
    """
    Performs hERG staircase quality control similar to Lei 2019.

    @param voltage: A voltage trace, sampled at ``sampling_rate``.
    @param sampling_rate: The number of samples per time unit.
    @param removal_time: Number of time units to remove after each step in the protocol
    @param noise_len: Number of initial samples during which the signal is flat, to use for noise estimates.
    @param plot_dir: An optional directory to store plots in
    """
    def __init__(self, voltage, sampling_rate=5, removal_time=5, noise_len=200,
                 plot_dir=None):

        self.voltage = np.array(voltage)
        self.sampling_rate = sampling_rate
        self.removal_time = removal_time
        self.noise_len = int(noise_len)

        # Passing in a plot dir enables debug mode
        self._plot_dir = plot_dir
        self.logger = logging.getLogger(__name__)
        if self._plot_dir is not None:
            self.logger.setLevel(logging.DEBUG)
            # https://github.com/CardiacModelling/pcpostprocess/issues/42
        self._plot_dir = plot_dir

        # Define all thresholds

        # TODO: These should be args? Or perhaps this is good so that this QC
        # class can be extended?
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

    def run_qc(self, voltage_steps, times, before, after, qc_vals_before,
               qc_vals_after, n_sweeps=None):
        """Run each QC criteria on a single (before-trace, after-trace) pair.

        @param voltage_steps is a list of times at which there are discontinuities in Vcmd
        @param times is the array of observation times
        @param before are the before-drug current traces, in an array with
               sweeps on the first index and values on the second.
        @param after is the post-drug current traces, in an array with sweeps
               on the first index and values on the second.
        @param qc_vals_before is a sequence (Rseal, Cm, Rseries)
        @param qc_vals_after is a sequence (Rseal, Cm, Rseries)
        @n_sweeps is the number of sweeps to be run QC on.
        @returns A :class:`QCDict` with the test results.
        """
        # TODO: Why doesn't each sweep have its own "qc_vals" ?

        if not n_sweeps:
            n_sweeps = len(before)

        before = self.filter_capacitive_spikes(np.array(before), times, voltage_steps)
        after = self.filter_capacitive_spikes(np.array(after), times, voltage_steps)

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
        qc3_3 = self.qc3(before[0, :] - after[0, :], before[1, :] - after[1, :])

        QC['qc3.raw'] = [qc3_1]
        QC['qc3.E4031'] = [qc3_2]   # Change to 'control' and 'blocker' ?
        QC['qc3.subtracted'] = [qc3_3]

        rseals = [qc_vals_before[0], qc_vals_after[0]]
        cms = [qc_vals_before[1], qc_vals_after[1]]
        rseriess = [qc_vals_before[2], qc_vals_after[2]]
        qc4 = self.qc4(rseals, cms, rseriess)

        QC['qc4.rseal'] = [qc4[0]]
        QC['qc4.cm'] = [qc4[1]]
        QC['qc4.rseries'] = [qc4[2]]

        # indices where hERG peaks
        qc5 = self.qc5(before[0, :], after[0, :], self.qc5_win)
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
            qc6 = self.qc6(
                before[i, :] - after[i, :], self.qc6_win, label='0')
            qc6_1 = self.qc6(
                before[i, :] - after[i, :], self.qc6_1_win, label='1')
            qc6_2 = self.qc6(
                before[i, :] - after[i, :], self.qc6_2_win, label='2')

            QC['qc6.subtracted'].append(qc6)
            QC['qc6.1.subtracted'].append(qc6_1)
            QC['qc6.2.subtracted'].append(qc6_2)

        if self._plot_dir is not None:
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

            fig.savefig(os.path.join(self._plot_dir, 'qc_debug.png'))
            plt.close(fig)

        return QC

    def qc1(self, rseal, cm, rseries):
        """
        Checks that the given ``rseal``, ``cm`` and ``rseries`` are within the
        desired range.

        @param rseal A scalar indicating a seal resistance in Ohm
        @param cm A scalar cell capacitance in Farad
        @param rseries A scalar series resistance in Ohm
        @returns A tuple ``((qc11_passed, rseal), (qc12_passed, cm), (qc13_passed, rseries))``.
        """
        # TODO Is there any reason these are public, other than that it was
        # convenient for testing?
        # TODO Could just be 1 method called 3 times
        # TODO Should boundaries be arguments?

        # Check R_seal, C_m, R_series within desired range
        qc11 = not (rseal is None
                    or rseal < self.rsealc[0]
                    or rseal > self.rsealc[1]
                    or not np.isfinite(rseal))
        qc12 = not (cm is None
                    or cm < self.cmc[0]
                    or cm > self.cmc[1]
                    or not np.isfinite(cm))
        qc13 = not (rseries is None
                    or rseries < self.rseriesc[0]
                    or rseries > self.rseriesc[1]
                    or not np.isfinite(rseries))

        if not qc11:
            self.logger.debug(f'rseal: {rseal}')
        if not qc12:
            self.logger.debug(f'cm: {cm}')
        if not qc13:
            self.logger.debug(f'rseries: {rseries}')

        return [(qc11, rseal), (qc12, cm), (qc13, rseries)]

    def qc2(self, recording):
        """
        Checks that the signal-to-noise ratio is below a certain threshold.

        Here the signal-to-noise ratio is defined as
        ``(std(recording) / std(noise))``, where ``noise`` is the initial part
        of ``recording``.

        @param recording A 1-d numpy array containing recorded currents.
        @returns a tuple ``(passed, signal-to-noise-ratio)``.
        """
        snr = (np.std(recording) / np.std(recording[:self.noise_len])) ** 2
        passed = snr >= self.snrc and np.isfinite(snr)
        if not passed:
            self.logger.debug(f'snr: {snr}')
        return (passed, snr)

    def qc3(self, recording1, recording2):
        """
        Checks that ``recording1`` and ``recording2`` are similar, using a
        measure based on root-mean-square-distance.

        Particularly, it checks whether
        ``RMSD(recording1, recording2) < threshold``,
        where ``threshold`` is determined as the maximum of
        ``w * (RMSD(recording1, 0) + RMSD(recording2, 0))`` and
        ``6 * (std(noise1) + std(noise2))``.

        @param recording1 A 1-d numpy array containing recorded currents
        @param recording2 The same, but for a repeat measurement
        @returns a tuple ``(passed, RMSD - threshold)``
        """
        def rmsd(x, y=0):
            return np.sqrt(np.mean((x - y)**2))

        t1 = self.rmsd0c * np.mean((rmsd(recording1), rmsd(recording2)))
        t2 = 6 * np.mean((
            np.std(recording1[:self.noise_len]),
            np.std(recording2[:self.noise_len])))
        # TODO: This second cut-off is not documented in the paper
        t = max(t1, t2)
        r = rmsd(recording1, recording2)
        passed = r <= t and np.isfinite(t) and np.isfinite(r)
        if not passed:
            self.logger.debug(f'rmsd: {r}, rmsdc: {t}')

        # TODO: Is this the best thing to return? Why not t _and_ r?
        return (passed, t - r)

    def qc4(self, rseals, cms, rseriess):
        """
        Checks that Rseal, Cm, and Rseries values remain steady during an
        experiment.

        For each quantity, it checks if ``std(x) / mean(x) <= cutoff``.

        @param rseals A list of Rseal values, for repeated experiments, in Ohm.
        @param cms A list of Cm values, in Farad.
        @param rseriess A list of Rseries values, in Ohm
        @returns a tuple ``((qc41_passed, ex41), (qc42_passed, ex42), (qc43_passed, ex43))``
                 where ``exY`` is the amount by which the criterion exceeded the threshold.
        """
        # TODO: Run this on values for all sweeps, not 1 before and after ?
        #       Or, if using only 2, write as just |a-b|/(a+b)

        # TODO: This uses numpy's standard `std` call without the bias
        #       correction. Do we want to use the bias corrected one instead?
        #       matters for above, as becomes 2|a-b|/(a+b) !

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
                self.logger.debug(f'd_rseal:  {d_rseal}')
                qc41 = False

        if None in list(cms):
            qc42 = False
            d_cm = None
        else:
            d_cm = np.std(cms) / np.mean(cms)
            if d_cm > self.cmsc or not (
                    np.isfinite(np.mean(cms)) and np.isfinite(np.std(cms))):
                self.logger.debug(f'd_cm: {d_cm}')
                qc42 = False

        if None in list(rseriess):
            qc43 = False
            d_rseries = None
        else:
            d_rseries = np.std(rseriess) / np.mean(rseriess)
            if d_rseries > self.rseriessc or not (
                    np.isfinite(np.mean(rseriess))
                    and np.isfinite(np.std(rseriess))):
                self.logger.debug(f'd_rseries: {d_rseries}')
                qc43 = False

        return [(qc41, d_rseal), (qc42, d_cm), (qc43, d_rseries)]

    def qc5(self, recording1, recording2, win, label=None):
        """
        Checks whether the peak current in ``recording1`` during a given window
        has been blocked in ``recording2``.

        First, find an ``i`` such that ``recording1[i]`` is the largest
        (positive) current (if a window is given, only this window is
        searched). Then, checks whether
        ``(recording1[i] - recording2[i]) / recording1[i] >= 0.75`` (in other
        words whether at least 75% of the peak current has been blocked).

        @param recording1 A staircase before blocker application
        @param recording2 A staircase with a strong blocker, e.g. E-4031
        @param win A tuple ``(i, j)`` such that ``recording[i:j]`` is the window to search in
        @param label An optional label for a plot
        @returns a tuple ``(passed, 0.75 * recording1[i] - (recording1[i] - recording2[i]))``.
        """
        if self._plot_dir is not None and label is not None:
            plt.axvspan(win[0], win[1], color='grey', alpha=.1)
            plt.plot(recording1, label='recording1')
            plt.plot(recording2, label='recording2')
            plt.savefig(os.path.join(self._plot_dir, f'qc5_{label}'))
            plt.clf()

        i, f = win
        wherepeak = np.argmax(recording1[i:f])
        max_diff = recording1[i:f][wherepeak] - recording2[i:f][wherepeak]
        max_diffc = self.max_diffc * recording1[i:f][wherepeak]

        if (max_diff < max_diffc) or not (np.isfinite(max_diff)
                                          and np.isfinite(max_diffc)):
            self.logger.debug(f'max_diff:  {max_diff}, max_diffc: {max_diffc}')
            result = False
        else:
            result = True

        # TODO: More sensible to return max_diff / recording1[i:f][wherepeak] here?
        return (result, max_diffc - max_diff)

    def qc5_1(self, recording1, recording2, label=None):
        """
        Checks whether root-mean-squared current in ``recording1`` has been
        blocked in ``recording2``,

        The test passes if
        ``(RMSD(recording2, 0) - RMSD(recording1, 0)) / RMSD(recording1, 0) <= 0.5``

        @param recording1 A staircase before blocker application
        @param recording2 A staircase with a strong blocker, e.g. E-4031
        @param label An optional label for a plot
        @returns a tuple ``(passed, 0.5 * recording1[i] - (recording1[i] - recording2[i]))``.
        """
        # TODO: Just rephrase this as RMSD(I2) / RMSD(I1) <= 0.5

        if self._plot_dir is not None and label is not None:
            fig = plt.figure()
            ax = fig.subplots()
            ax.plot(recording1, label='recording1')
            ax.plot(recording2, label='recording2')
            fig.savefig(os.path.join(self._plot_dir, f'qc5_{label}'))
            plt.close(fig)

        def rmsd(x, y=0):
            return np.sqrt(np.mean((x - y)**2))

        rmsd0_diff = rmsd(recording1) - rmsd(recording2)
        rmsd0_diffc = self.rmsd0_diffc * rmsd(recording1)

        if (rmsd0_diff < rmsd0_diffc) or not (np.isfinite(rmsd0_diff)
                                              and np.isfinite(rmsd0_diffc)):
            self.logger.debug(
                f'rmsd0_diff: {rmsd0_diff}, rmsd0c: {rmsd0_diffc}')
            result = False
        else:
            result = True

        # TODO Just return rmsd(I2) / RMSD(I1)
        return (result, rmsd0_diffc - rmsd0_diff)

    def qc6(self, recording1, win, label=None):
        """
        Checks that the current in a particular window is non-negative.

        Instead of comparing the whole current to 0, we compare the mean
        current to -2 times the noise level (estimed from the start of the
        trace).

        @param recording1 The full current trace
        @param win A tuple ``(i, j)`` such that ``recording[i:j]`` is the window to check
        @param label A label for an optional plot
        @returns a tuple ``(passed, -2 * noise -
        """
        # Check subtracted staircase +40mV step up is non negative

        if self._plot_dir is not None and label is not None:
            if win is not None:
                plt.axvspan(win[0], win[1], color='grey', alpha=.1)
            plt.plot(recording1, label='recording1')
            plt.savefig(os.path.join(self._plot_dir, f'qc6_{label}'))
            plt.clf()

        i, f = win
        val = np.mean(recording1[i:f])
        valc = self.negative_tolc * np.std(recording1[:self.noise_len])
        if (val < valc) or not (np.isfinite(val)
                                and np.isfinite(valc)):
            self.logger.debug(f'qc6_{label} val:  {val}, valc: {valc}')
            result = False
        else:
            result = True

        # TODO Return val / valc here?
        return (result, valc - val)

    def filter_capacitive_spikes(self, current, times, voltage_step_times):
        """
        Set currents to 0 where they lie less than ``removal_time`` after a change in voltage.

        @param current: The observed current, as a 1 or 2-dimensional array.
                        If a 2-dimensional array is used, repeats must be
                        on the first axis, and time series values on the 2nd.
        @param times: the times at which the current was observed
        @param voltage_step_times: the times at which there are discontinuities in Vcmd
        @returns the ``current`` with some samples set to 0
        """
        voltage_step_ends = np.append(voltage_step_times[1:], np.inf)
        for tstart, tend in zip(voltage_step_times, voltage_step_ends):
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
                raise ValueError('Current array must be 1 or 2-dimensional')

        return current
