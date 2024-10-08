import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly


def infer_reversal_potential(current, times, voltage_segments, voltages,
                             ax=None, output_path=None, plot=None,
                             known_Erev=None, figsize=(5, 3)):

    if output_path:
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if (ax or output_path) and plot is not False:
        plot = True

    # Find indices of observations during the reversal ramp
    ramps = [line for line in voltage_segments if line[2] != line[3]]

    # Assume the last ramp is the reversal ramp (convert to ms)
    tstart, tend = np.array(ramps)[-1, :2]

    istart = np.argmax(times > tstart)
    iend = np.argmax(times > tend)

    times = times[istart:iend]
    current = current[istart:iend]
    voltages = voltages[istart:iend]

    try:
        fitted_poly = poly.Polynomial.fit(voltages, current, 4)
    except ValueError as exc:
        logging.warning(str(exc))
        return np.nan

    try:
        roots = np.unique([np.real(root) for root in fitted_poly.roots()
                           if root > np.min(voltages) and root < np.max(voltages)])
    except np.linalg.LinAlgError as exc:
        logging.warning(str(exc))
        return np.nan

    # Take the last root (greatest voltage). This should be the first time that
    # the current crosses 0 and where the ion-channel kinetics are too slow to
    # play a role

    if len(roots) == 0:
        return np.nan

    if plot:
        created_fig = False
        if ax is None and output_path is not None:

            created_fig = True
            fig = plt.figure(figsize=figsize)
            ax = fig.subplots()

        ax.set_xlabel('$V$ (mV)')
        ax.set_ylabel('$I$ (nA)')

        # Now plot current vs voltage
        ax.plot(voltages, current, 'x', markersize=2, color='grey', alpha=.5)
        ax.axvline(roots[-1], linestyle='--', color='grey', label=r'$E_\mathrm{obs}$')
        if known_Erev:
            ax.axvline(known_Erev, linestyle='--', color='orange',
                       label="Calculated $E_{Kr}$")
        ax.axhline(0, linestyle='--', color='grey')
        ax.plot(*fitted_poly.linspace())
        ax.legend()

        if output_path is not None:
            fig = ax.figure
            fig.savefig(output_path)

        if created_fig:
            plt.close(fig)

    return roots[-1]
