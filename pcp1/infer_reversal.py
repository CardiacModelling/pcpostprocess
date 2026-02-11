import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

from .detect_ramp_bounds import detect_ramp_bounds


def infer_reversal_potential(current, times, voltage_segments, voltages,
                             output_path=None, known_Erev=None,
                             figsize=(5, 3)):
    """
    Infers a reversal potential in a time series, based on a reversal ramp.

    The data is denoised by fitting a 4-th order polynomial through the ramp
    data, from which a reversal potential is then detected. If no polynomial
    can be fit or the resulting zero-crossing is outside of
    ``min(voltages), max(voltages)``, then ``np.nan`` is returned.

    @param current: The currents that make up a time series with ``times``
    @param times: The sampled times
    @param voltage_segments: A list of tuples (tstart, tend, vstart, vend)
    describing voltage steps or ramps. It is assumed the final ramp is the
    reversal ramp.
    @param voltages: The sampled voltages
    @param output_path: An optional path to store a plot at
    @param known_Erev: A known reversal potential to include in the plot
    @param figsize: A size for the plot.

    @return: The inferred reversal potential
    """

    # Get ramp bounds, assuming final ramp is the reversal ramp
    istart, iend = detect_ramp_bounds(times, voltage_segments, -1)

    current = current[istart:iend]
    voltages = voltages[istart:iend]

    # Fit a 4-th order polynomial
    try:
        fitted_poly = poly.Polynomial.fit(voltages, current, 4)
    except ValueError as exc:
        logging.warning(str(exc))
        return np.nan

    # Try extracting the polynomial's roots, accepting only ones that are
    # within the range of sampled voltages (so not using ramp info here!)
    try:
        vmin, vmax = np.min(voltages), np.max(voltages)
        roots = np.unique([np.real(root) for root in fitted_poly.roots()
                           if root > vmin and root < vmax])
    except np.linalg.LinAlgError as exc:
        logging.warning(str(exc))
        return np.nan

    # Take the last root (greatest voltage). This should be the first time that
    # the current crosses 0 and where the ion-channel kinetics are too slow to
    # play a role

    if len(roots) == 0:
        return np.nan
    erev = roots[-1]

    # Optional plot
    if output_path is not None:

        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
        ax.set_xlabel('$V$ (mV)')  # Assuming mV here
        ax.set_ylabel('$I$ (nA)')

        # Now plot current vs voltage
        ax.plot(voltages, current, 'x', markersize=2, color='grey', alpha=.5)
        ax.axvline(erev, linestyle='--', color='grey', label=r'$E_\mathrm{obs}$')
        if known_Erev:
            ax.axvline(known_Erev, linestyle='--', color='orange',
                       label="Calculated $E_{Kr}$")
        ax.axhline(0, linestyle='--', color='grey')
        ax.plot(*fitted_poly.linspace())
        ax.legend()

        fig.savefig(output_path)
        plt.close(fig)

    return erev
