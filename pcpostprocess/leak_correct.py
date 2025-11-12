#
# Leak correction methods
#
import numpy as np
from matplotlib import pyplot as plt


def linear_reg(V, I):
    """
    Performs linear regression on ``V`` and ``I``, returning coefficients
    ``b_0, b_1`` such that ``I`` if fit by ``b_0 + b_1 * V``.
    """

    # number of observations/points
    n = np.size(V)
    # TODO Test that I has same size, nice error if not

    # mean of V and I vector
    m_V = np.mean(V)
    m_I = np.mean(I)

    # calculate cross-deviation and deviation about V
    SS_VI = np.sum(I * V) - n * m_I * m_V
    SS_VV = np.sum(V * V) - n * m_V * m_V

    # calculating regression coefficients
    b_1 = SS_VI / SS_VV
    b_0 = m_I - b_1 * m_V

    # return intercept, gradient
    return b_0, b_1


def get_leak_corrected(current, voltages, times, ramp_start_index,
                       ramp_end_index, *args, **kwargs):
    """
    Leak correct all data in a trace by subtracting a linear current derived
    from a leak ramp.

    @param current: the observed currents taken from the entire sweep
    @param voltages: the voltages at each timepoint; has the same shape as current
    @param ramp_start_index: the index of the observation where the leak ramp begins
    @ramp_end_index: the index of the observation where the leak ramp ends

    Any extra arguments will be passed to ``fit_linear_leak``, allowing figure
    creation.

    @return: A leak correct trace with the same shape as current
    """

    (b0, b1), I_leak = fit_linear_leak(current, voltages, times, ramp_start_index,
                                       ramp_end_index, *args, **kwargs)

    return current - I_leak


def fit_linear_leak(current, voltage, times, ramp_start_index, ramp_end_index,
                    save_fname=None, figsize=(5.54, 7)):
    """
    Fits linear leak to a leak ramp, returning

    @param current: the observed currents taken from the entire sweep
    @param voltage: the voltages at each timepoint; has the same shape as current
    @param ramp_start_index: the index of the observation where the leak ramp begins
    @param ramp_end_index: the index of the observation where the leak ramp ends
    @param save_fname: if set, a debugging figure will be made and stored with this name
    @param figsize: if ``save_fname`` is set, the figure size.

    @return: the linear regression parameters obtained from fitting the leak
    ramp as a tuple ``(intercept, slope)``, and the linear leak current as a
    numpy array.
    """

    # TODO: This should not be handled here
    if len(current) == 0:
        return (np.nan, np.nan), np.full(times.shape, np.nan)

    # TODO: This should definitely not be handled here! Raise an error if this
    # needs to be done
    current = current.flatten()

    # Convert to mV for convinience
    # TODO: This is apparently no longer done?
    V = voltage
    # TODO: Remove these unecessary renames?
    I_obs = current  # pA
    # TODO: Make the user give half-open intervals instead of using the +1 here!
    b_0, b_1 = linear_reg(V[ramp_start_index:ramp_end_index+1],
                          I_obs[ramp_start_index:ramp_end_index+1])
    I_leak = b_1 * V + b_0

    # TODO: Should this not be an error instead?
    if not np.all(np.isfinite(I_leak)) or not np.all(np.isfinite(I_obs)):
        return (np.nan, np.nan), np.full(times.shape, np.nan)

    if save_fname:
        # fit to leak ramp
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        (ax1, ax3), (ax2, ax4) = fig.subplots(2, 2)

        for ax in (ax1, ax2, ax3, ax4):
            ax.spines[['top', 'right']].set_visible(False)

        time_range = (0, times.max() / 5)

        #  Current vs time
        ax1.set_title(r'\textbf{a}', loc='left')
        ax1.set_xlabel(r'$t$ (ms)')
        ax1.set_ylabel(r'$I_\mathrm{obs}$ (pA)')
        ax1.set_xticklabels([])
        ax1.set_xlim(*time_range)

        # Voltage vs time
        ax2.set_title(r'\textbf{b}', loc='left')
        ax2.set_xlabel(r'$t$ (ms)')
        ax2.set_ylabel(r'$V_\mathrm{cmd}$ (mV)')
        ax2.set_xlim(*time_range)

        # Current vs voltage
        ax3.set_title(r'\textbf{c}', loc='left')
        ax3.set_xlabel(r'$V_\mathrm{cmd}$ (mV)')
        ax3.set_ylabel(r'$I_\mathrm{obs}$ (pA)')

        ax4.set_xlabel(r'$t$ (ms)')
        ax4.set_ylabel(r'current (pA)')
        ax4.set_title(r'\textbf{d}', loc='left')

        start_t = times[ramp_start_index]
        end_t = times[ramp_end_index]

        ax1.axvspan(start_t, end_t, alpha=.5, color='grey')
        ax1.plot(times, I_obs)
        # ax1.axvline(start_t, linestyle='--', color='k', alpha=0.5)
        # ax1.axvline(end_t, linestyle='--', color='k', alpha=0.5)

        ax2.axvspan(start_t, end_t, alpha=.5, color='grey')
        ax2.plot(times, V)

        # ax2.axvline(start_t, linestyle='--', color='k', alpha=0.5)
        # ax2.axvline(end_t, linestyle='--', color='k', alpha=0.5)

        ax3.plot(V[ramp_start_index:ramp_end_index+1],
                 I_obs[ramp_start_index:ramp_end_index+1], 'x')
        ax3.plot(V[ramp_start_index:ramp_end_index+1],
                 I_leak[ramp_start_index:ramp_end_index+1], '--')

        ax4.plot(times, I_obs, label=r'$I_\mathrm{obs}$')
        ax4.plot(times, I_leak, linestyle='--', label=r'$I_\mathrm{L}$')
        ax4.plot(times, I_obs - I_leak,
                 alpha=0.5, label=r'$I_\mathrm{obs} - I_\mathrm{L}$')
        ax4.legend(frameon=False)

        if save_fname is not None:
            fig.savefig(save_fname)
            plt.close(fig)

    return (b_0, b_1), I_leak
