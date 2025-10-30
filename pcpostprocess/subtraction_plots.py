import numpy as np
from matplotlib.gridspec import GridSpec

from .leak_correct import fit_linear_leak


def setup_subtraction_grid(fig, nsweeps):
    # Use 6 x 2 grid when there are 2 sweeps
    gs = GridSpec(6, nsweeps, figure=fig, height_ratios=[0.7, 2, 2, 2, 0.7, 3])
    # gs.subplots(sharey='row')
    # plot protocol at the top
    protocol_axs = [fig.add_subplot(gs[0, i]) for i in range(nsweeps)]

    # Plot before drug traces
    before_axs = [fig.add_subplot(gs[1, i]) for i in range(nsweeps)]

    # Plot after traces
    after_axs = [fig.add_subplot(gs[2, i]) for i in range(nsweeps)]

    # Leak corrected traces
    corrected_axs = [fig.add_subplot(gs[3, i]) for i in range(nsweeps)]

    # Long axis for protocol on the bottom (full width)
    long_protocol_ax = fig.add_subplot(gs[4, :])

    # Subtracted traces on one axis
    subtracted_ax = fig.add_subplot(gs[5, :])

    for ax, cap in zip(list(protocol_axs) + list(before_axs)
                       + list(after_axs) + list(corrected_axs)
                       + [long_protocol_ax] + [subtracted_ax],
                       'abcdefghijklm'):
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(cap, loc='left', fontweight='bold')
        if cap != 'a':
            ax.sharex(protocol_axs[0])

    return protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, long_protocol_ax


def do_subtraction_plot(fig, times, sweeps, before_currents, after_currents,
                        voltages, ramp_bounds, well=None, protocol=None):

    nsweeps = before_currents.shape[0]
    sweeps = list(range(nsweeps))

    axs = setup_subtraction_grid(fig, nsweeps)
    protocol_axs, before_axs, after_axs, corrected_axs, \
        subtracted_ax, long_protocol_ax = axs

    for i, ax in enumerate(protocol_axs):
        ax.plot(times*1e-3, voltages, color='black')
        ax.set_title(f'Well {well}, sweep {sweeps[i]}', fontweight='bold')
        ax.tick_params(axis='x', labelbottom=False)
        # ax.set_xlabel('time (s)')
    protocol_axs[0].set_ylabel(r'$V_\mathrm{cmd}$ (mV)', fontsize=16)

    all_leak_params_before = []
    all_leak_params_after = []

    alpha_of_zero = 0.2
    style_of_zero = '-'
    range_of_zero = [times[0]*1e-3, times[-1]*1e-3]

    # Compute and store leak currents
    before_leak_currents = np.full((nsweeps, voltages.shape[0]),
                                   np.nan)
    after_leak_currents = np.full((nsweeps, voltages.shape[0]),
                                  np.nan)

    for i in range(len(sweeps)):
        before_params, before_leak_current = fit_linear_leak(before_currents, voltages, times,
                                                             *ramp_bounds)
        before_leak_currents[i, :] = before_leak_current
        all_leak_params_before.append(before_params)

        after_params, after_leak_current = fit_linear_leak(after_currents, voltages, times,
                                                           *ramp_bounds)
        all_leak_params_after.append(after_params)
        after_leak_currents[i, :] = after_leak_current

    for i, (sweep, ax) in enumerate(zip(sweeps, before_axs)):
        b0, b1 = all_leak_params_before[i]
        gleak = b1
        Eleak = -b0/b1

        ax.plot(times*1e-3, before_currents[i, :], label="Pre-drug raw")
        ax.plot(times*1e-3, before_leak_currents[i, :],
                label=f"Fitted leak g={gleak:7.5g}, E={Eleak:7.4g} mV")
        ax.plot(range_of_zero, [0, 0], color='black', linestyle=style_of_zero, alpha=alpha_of_zero)
        ax.legend()
        ax.tick_params(axis='x', labelbottom=False)

        # ax.set_xlabel('time (s)')
    before_axs[0].set_ylabel(r'Pre-drug trace', fontsize=16)
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    # ax.tick_params(axis='y', rotation=90)

    for i, (sweep, ax) in enumerate(zip(sweeps, after_axs)):
        b0, b1 = all_leak_params_before[i]
        gleak = b1
        Eleak = -b0/b1

        ax.plot(times*1e-3, after_currents[i, :], label="Post-drug raw")
        ax.plot(times*1e-3, after_leak_currents[i, :],
                label=f"Fitted leak g={gleak:7.5g}, E={Eleak:7.4g} mV")
        ax.plot(range_of_zero, [0, 0], color='black', linestyle=style_of_zero, alpha=alpha_of_zero)
        ax.legend()
        ax.tick_params(axis='x', labelbottom=False)

    after_axs[0].set_ylabel(r'Post-drug trace', fontsize=16)

    for i, (sweep, ax) in enumerate(zip(sweeps, corrected_axs)):
        corrected_before_currents = before_currents[i, :] - before_leak_currents[i, :]
        corrected_after_currents = after_currents[i, :] - after_leak_currents[i, :]
        ax.plot(times*1e-3, corrected_before_currents,
                label="Leak-corrected pre-drug trace")
        ax.plot(times*1e-3, corrected_after_currents,
                label="Leak-corrected post-drug trace")
        ax.plot(range_of_zero, [0, 0], color='black', linestyle=style_of_zero, alpha=alpha_of_zero)
        ax.set_xlabel(r'Time (s)')
        ax.legend()
    corrected_axs[0].set_ylabel(r'Leak-corrected traces', fontsize=16)
    # ax.tick_params(axis='y', rotation=90)
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    for i, sweep in enumerate(sweeps):
        subtracted_currents = before_currents[i, :] - before_leak_currents[i, :] - \
            (after_currents[i, :] - after_leak_currents[i, :])

        subtracted_ax.plot(times*1e-3, subtracted_currents, label=f"sweep {sweep}", alpha=.5)

        #  Cycle to next colour
        # subtracted_ax.plot([np.nan], [np.nan], label=f"sweep {sweep}", alpha=.5)

    subtracted_ax.legend()
    subtracted_ax.plot(range_of_zero, [0, 0], color='black',
                       linestyle=style_of_zero, alpha=alpha_of_zero)
    subtracted_ax.set_ylabel('Final subtracted traces', fontsize=16)
    subtracted_ax.set_xlabel('Time (s)', fontsize=16)

    long_protocol_ax.plot(times*1e-3, voltages, color='black')
    # long_protocol_ax.set_xlabel('time (s)')
    long_protocol_ax.set_ylabel(r'$V_\mathrm{cmd}$ (mV)', fontsize=16)
    long_protocol_ax.tick_params(axis='x', labelbottom=False)

    fig.align_ylabels()

