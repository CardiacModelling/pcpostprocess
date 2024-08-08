import numpy as np
from matplotlib.gridspec import GridSpec

from .leak_correct import fit_linear_leak


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

    for ax, cap in zip(list(protocol_axs) + list(before_axs) + list(after_axs) + list(corrected_axs) + [subtracted_ax] + [long_protocol_ax], 'abcdefghijklm'):
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(cap, loc='left', fontweight='bold')

    return protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, long_protocol_ax


def do_subtraction_plot(fig, times, sweeps, before_currents, after_currents,
                        voltages, ramp_bounds, well=None, protocol=None):

    nsweeps = before_currents.shape[0]
    sweeps = list(range(nsweeps))

    before_currents = before_currents
    after_currents = after_currents

    axs = setup_subtraction_grid(fig, nsweeps)
    protocol_axs, before_axs, after_axs, corrected_axs, \
        subtracted_ax, long_protocol_ax = axs

    for ax in protocol_axs:
        ax.plot(times*1e-3, voltages, color='black')
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$V_\mathrm{cmd}$ (mV)')

    all_leak_params_before = []
    all_leak_params_after = []
    for i in range(len(sweeps)):
        before_params, _ = fit_linear_leak(before_currents, voltages, times,
                                           *ramp_bounds)
        all_leak_params_before.append(before_params)

        after_params, _ = fit_linear_leak(before_currents, voltages, times,
                                          *ramp_bounds)
        all_leak_params_after.append(after_params)

    # Compute and store leak currents
    before_leak_currents = np.full((nsweeps, voltages.shape[0]),
                                   np.nan)
    after_leak_currents = np.full((nsweeps, voltages.shape[0]),
                                  np.nan)
    for i, sweep in enumerate(sweeps):

        b0, b1 = all_leak_params_before[i]
        gleak = b1
        Eleak = -b1/b0
        before_leak_currents[i, :] = gleak * (voltages - Eleak)

        b0, b1 = all_leak_params_after[i]
        gleak = b1
        Eleak = -b1/b0

        after_leak_currents[i, :] = gleak * (voltages - Eleak)

    for i, (sweep, ax) in enumerate(zip(sweeps, before_axs)):
        gleak, Eleak = all_leak_params_before[i]
        ax.plot(times*1e-3, before_currents[i, :], label=f"pre-drug raw, sweep {sweep}")
        ax.plot(times*1e-3, before_leak_currents[i, :],
                label=r'$I_\mathrm{L}$.' f"g={gleak:1E}, E={Eleak:.1e}")
        # ax.legend()

        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'pre-drug trace')
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        # ax.tick_params(axis='y', rotation=90)

    for i, (sweep, ax) in enumerate(zip(sweeps, after_axs)):
        gleak, Eleak = all_leak_params_before[i]
        ax.plot(times*1e-3, after_currents[i, :], label=f"post-drug raw, sweep {sweep}")
        ax.plot(times*1e-3, after_leak_currents[i, :],
                label=r"$I_\mathrm{L}$." f"g={gleak:1E}, E={Eleak:.1e}")
        # ax.legend()
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_xlabel('$t$ (s)')
        ax.set_ylabel(r'post-drug trace')
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        # ax.tick_params(axis='y', rotation=90)

    for i, (sweep, ax) in enumerate(zip(sweeps, corrected_axs)):
        corrected_before_currents = before_currents[i, :] - before_leak_currents[i, :]
        corrected_after_currents = after_currents[i, :] - after_leak_currents[i, :]
        ax.plot(times*1e-3, corrected_before_currents,
                label=f"leak-corrected pre-drug trace, sweep {sweep}")
        ax.plot(times*1e-3, corrected_after_currents,
                label=f"leak-corrected post-drug trace, sweep {sweep}")
        ax.set_xlabel(r'$t$ (s)')
        ax.set_ylabel(r'leak-corrected traces')
        # ax.tick_params(axis='y', rotation=90)
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax = subtracted_ax
    for i, sweep in enumerate(sweeps):
        before_trace = before_currents[i, :].flatten()
        after_trace = after_currents[i, :].flatten()
        before_params, before_leak = fit_linear_leak(before_trace, voltages, times,
                                                     *ramp_bounds)
        after_params, after_leak = fit_linear_leak(after_trace, voltages, times,
                                                   *ramp_bounds)

        subtracted_currents = before_currents[i, :] - before_leak_currents[i, :] - \
            (after_currents[i, :] - after_leak_currents[i, :])
        ax.plot(times*1e-3, subtracted_currents, label=f"sweep {sweep}", alpha=.5)

        #  Cycle to next colour
        ax.plot([np.nan], [np.nan], label=f"sweep {sweep}", alpha=.5)

    ax.set_ylabel(r'$I_\mathrm{obs} - I_\mathrm{L}$ (mV)')
    ax.set_xlabel('$t$ (s)')

    long_protocol_ax.plot(times*1e-3, voltages, color='black')
    long_protocol_ax.set_xlabel('time (s)')
    long_protocol_ax.set_ylabel(r'$V_\mathrm{cmd}$ (mV)')
    long_protocol_ax.tick_params(axis='y', rotation=90)

