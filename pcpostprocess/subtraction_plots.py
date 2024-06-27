import logging

import numpy as np
from . leak_correct import fit_linear_leak
from matplotlib.gridspec import GridSpec


def setup_subtraction_grid(fig, nsweeps):
    # Use 5 x 2 grid when there are 2 sweeps
    gs = GridSpec(6, nsweeps, figure=fig)

    # plot protocol at the top
    protocol_axs = [fig.add_subplot(gs[0, i]) for i in range(nsweeps)]

    # Plot before drug traces
    before_axs = [fig.add_subplot(gs[1, i]) for i in range(nsweeps)]

    # Plot after traces
    after_axs = [fig.add_subplot(gs[2, i]) for i in range(nsweeps)]

    # Leak corrected traces
    corrected_axs = [fig.add_subplot(gs[3, i]) for i in range(nsweeps)]

    # Subtracted traces on one axis
    subtracted_ax = fig.add_subplot(gs[4, :])

    # Long axis for protocol on the bottom (full width)
    long_protocol_ax = fig.add_subplot(gs[5, :])

    return protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, long_protocol_ax


def do_subtraction_plot(fig, times, sweeps, before_currents, after_currents,
                        sub_df, voltages, ramp_bounds, well=None, protocol=None):

    #  Filter dataframe to relevant entries
    if well in sub_df.columns:
        sub_df = sub_df[sub_df.well == well]
    if protocol in sub_df.columns:
        sub_df = sub_df[sub_df.protocol == protocol]

    sweeps = list(sorted(sub_df.sweep.unique()))
    nsweeps = len(sweeps)
    sub_df = sub_df.set_index('sweep')

    if len(sub_df.index) == 0:
        logging.debug("do_subtraction_plot received empty dataframe")
        return

    axs = setup_subtraction_grid(fig, nsweeps)
    protocol_axs, before_axs, after_axs, corrected_axs, \
        subtracted_ax, long_protocol_ax = axs

    for ax in protocol_axs:
        ax.plot(times, voltages, color='black')
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$V_\mathrm{command}$ (mV)')

    # Compute and store leak currents
    before_leak_currents = np.full((voltages.shape[0], nsweeps),
                                   np.nan)
    after_leak_currents = np.full((voltages.shape[0], nsweeps),
                                  np.nan)
    for i, sweep in enumerate(sweeps):

        assert sub_df.loc[sweep] == 1

        gleak, Eleak = sub_df.loc[sweep][['gleak_before', 'E_leak_before']].values.astype(np.float64)
        before_leak_currents[i, :] = gleak * (voltages - Eleak)

        gleak, Eleak = sub_df.loc[sweep][['gleak_after', 'E_leak_after']].values.astype(np.float64)
        after_leak_currents[i, :] = gleak * (voltages - Eleak)

    for i, (sweep, ax) in enumerate(zip(sweeps, before_axs)):
        gleak, Eleak = sub_df.loc[sweep][['gleak_before', 'E_leak_before']]
        ax.plot(times, before_currents[i, :], label=f"pre-drug raw, sweep {sweep}")
        ax.plot(times, before_leak_currents[i, :],
                label=r'$I_\mathrm{leak}$.' f"g={gleak:1E}, E={Eleak:.1e}")
        # ax.legend()

        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'pre-drug trace')
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        # ax.tick_params(axis='y', rotation=90)

    for i, (sweep, ax) in enumerate(zip(sweeps, after_axs)):
        gleak, Eleak = sub_df.loc[sweep][['gleak_after', 'E_leak_after']]
        ax.plot(times, after_currents[i, :], label=f"post-drug raw, sweep {sweep}")
        ax.plot(times, after_leak_currents[i, :],
                label=r"$I_\mathrm{leak}$." f"g={gleak:1E}, E={Eleak:.1e}")
        # ax.legend()
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_xlabel('$t$ (s)')
        ax.set_ylabel(r'post-drug trace')
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        # ax.tick_params(axis='y', rotation=90)

    for i, (sweep, ax) in enumerate(zip(sweeps, corrected_axs)):
        corrected_currents = before_currents[i, :] - before_leak_currents[i, :]
        corrected_after_currents = after_currents[i, :] - after_leak_currents[i, :]
        ax.plot(times, corrected_currents,
                label=f"leak corrected before drug trace, sweep {sweep}")
        ax.plot(times, corrected_after_currents,
                label=f"leak corrected after drug trace, sweep {sweep}")
        ax.set_xlabel(r'$t$ (s)')
        ax.set_ylabel(r'leak corrected traces')
        # ax.tick_params(axis='y', rotation=90)
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax = subtracted_ax
    for i, sweep in enumerate(sweeps):
        before_trace = before_currents[i, :].flatten()
        after_trace = after_currents[i, :].flatten()
        before_params, before_leak = fit_linear_leak(before_trace,
                                                     well, sweep,
                                                     ramp_bounds)
        after_params, after_leak = fit_linear_leak(after_trace,
                                                   well, sweep,
                                                   ramp_bounds)

        subtracted_currents = before_currents[i, :] - before_leak_currents[i, :] - \
            (after_currents[i, :] - after_leak_currents[i, :])
        ax.plot(times, subtracted_currents, label=f"sweep {sweep}")

    ax.set_ylabel(r'$I_\mathrm{obs} - I_\mathrm{l}$ (mV)')
    ax.set_xlabel('$t$ (s)')

    long_protocol_ax.plot(times, voltages, color='black')
    long_protocol_ax.set_xlabel('time (s)')
    long_protocol_ax.set_ylabel(r'$V_\mathrm{command}$ (mV)')
    long_protocol_ax.tick_params(axis='y', rotation=90)

