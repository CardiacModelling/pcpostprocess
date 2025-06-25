import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from syncropatch_export.trace import Trace

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

    for ax, cap in zip(list(protocol_axs) + list(before_axs)
                       + list(after_axs) + list(corrected_axs)
                       + [subtracted_ax] + [long_protocol_ax],
                       'abcdefghijklm'):
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(cap, loc='left', fontweight='bold')

    return protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, long_protocol_ax


def do_subtraction_plot(fig, times, sweeps, before_currents, after_currents,
                        voltages, ramp_bounds, well=None, protocol=None):

    nsweeps = before_currents.shape[0]
    sweeps = list(range(nsweeps))

    axs = setup_subtraction_grid(fig, nsweeps)
    protocol_axs, before_axs, after_axs, corrected_axs, \
        subtracted_ax, long_protocol_ax = axs
    first = True
    for ax in protocol_axs:
        ax.plot(times*1e-3, voltages, color='black')
        ax.set_xticklabels([])

        if first:
            ax.set_ylabel(r'$V_\mathrm{cmd}$ (mV)')
            first = False

    all_leak_params_before = []
    all_leak_params_after = []
    for i in range(len(sweeps)):
        before_params, _ = fit_linear_leak(before_currents[i, :], voltages, times,
                                           *ramp_bounds)
        all_leak_params_before.append(before_params)

        after_params, _ = fit_linear_leak(after_currents[i, :], voltages, times,
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
        Eleak = -b0/b1
        before_leak_currents[i, :] = gleak * (voltages - Eleak)

        b0, b1 = all_leak_params_after[i]
        gleak = b1
        Eleak = -b0/b1

        after_leak_currents[i, :] = gleak * (voltages - Eleak)

    first = True
    for i, (sweep, ax) in enumerate(zip(sweeps, before_axs)):
        b0, b1 = all_leak_params_before[i]
        ax.plot(times*1e-3, before_currents[i, :], label=f"pre-drug raw, sweep {sweep}")
        ax.plot(times*1e-3, before_leak_currents[i, :],
                label=r'$I_\mathrm{L}$.' f"g={b1:1E}, E={-b0/b1:.1e}")
        ax.set_xticklabels([])

        if first:
            ax.set_ylabel(r'pre-drug trace')
            first = False
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        # ax.tick_params(axis='y', rotation=90)

    first = True
    for i, (sweep, ax) in enumerate(zip(sweeps, after_axs)):
        b0, b1 = all_leak_params_after[i]
        ax.plot(times*1e-3, after_currents[i, :], label=f"post-drug raw, sweep {sweep}")
        ax.plot(times*1e-3, after_leak_currents[i, :],
                label=r"$I_\mathrm{L}$." f"g={b1:1E}, E={-b0/b1:.1e}")
        ax.set_xticklabels([])
        if first:
            ax.set_ylabel(r'post-drug trace')
            first = False
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        # ax.tick_params(axis='y', rotation=90)

    first = True
    for i, (sweep, ax) in enumerate(zip(sweeps, corrected_axs)):
        corrected_before_currents = before_currents[i, :] - before_leak_currents[i, :]
        corrected_after_currents = after_currents[i, :] - after_leak_currents[i, :]
        corrb, _ = pearsonr(corrected_before_currents, voltages)
        ax.plot(times*1e-3, corrected_before_currents,
                label=f"leak-corrected pre-drug trace, sweep {sweep}, PC={corrb:.2f}")
        corra, _ = pearsonr(corrected_after_currents, voltages)
        ax.plot(times*1e-3, corrected_after_currents,
                label=f"leak-corrected post-drug trace, sweep {sweep}, PC={corra:.2f}")
        ax.set_xlabel('time (s)')
        if first:
            ax.set_ylabel(r'leak-corrected traces')
            first = False

        # sortedy = sorted(corrected_after_currents+corrected_before_currents)
        # ax.set_ylim(sortedy[60]*1.1, sortedy[-60]*1.1)
        ax.legend(bbox_to_anchor=(1.05, 1-0.5*i), loc='upper left')
        # ax.tick_params(axis='y', rotation=90)
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax = subtracted_ax
    ax.axhline(0, linestyle='--', color='lightgrey')
    sweep_list = []
    pcs = []
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
        corrs, _ = pearsonr(subtracted_currents, voltages)
        sweep_list += [sweep]
        pcs += [corrs]
        #  Cycle to next colour
        ax.plot([np.nan], [np.nan], label=f"sweep {sweep}", alpha=.5)
    # sortedy = sorted(subtracted_currents)
    # ax.set_ylim(sortedy[30]*1.1, sortedy[-30]*1.1)
    ax.set_ylabel(r'$I_\mathrm{obs} - I_\mathrm{L}$ (mV)')
    ax.legend(bbox_to_anchor=(1.05, 0.8), loc='upper left')
    ax.set_xticklabels([])

    long_protocol_ax.plot(times*1e-3, voltages, color='black')
    long_protocol_ax.set_xlabel('time (s)')
    long_protocol_ax.set_ylabel(r'$V_\mathrm{cmd}$ (mV)')
    long_protocol_ax.tick_params(axis='y', rotation=90)
    fig.tight_layout()

    corr_dict = {'sweeps': sweeps, 'pcs': pcs}
    return corr_dict


def linear_reg(V, I_obs):
    # number of observations/points
    n = np.size(V)

    # mean of V and I vector
    m_V = np.mean(V)
    m_I = np.mean(I_obs)

    # calculating cross-deviation and deviation about V
    SS_VI = np.sum(I_obs*V) - n*m_I*m_V
    SS_VV = np.sum(V*V) - n*m_V*m_V

    # calculating regression coefficients
    b_1 = SS_VI / SS_VV
    b_0 = m_I - b_1*m_V

    # return intercept, gradient
    return b_0, b_1


def regenerate_subtraction_plots(data_path='.', save_dir='.', processed_path=None,
                                 protocols_in=None, passed_only=False):
    '''
    Generate subtraction plots of all sweeps of all experiments in a directory
    '''
    data_dir = os.listdir(data_path)
    passed_wells = None
    passed = ''
    if 'passed_wells.txt' in data_dir:
        return None
    else:
        data_dir = [x for x in data_dir if os.path.isdir(os.path.join(data_path, x))]
        fig = plt.figure(figsize=[15, 24], layout='constrained')
        exp_list = []
        protocol_list = []
        well_list = []
        sweep_list = []
        corr_list = []
        passed_list = []

        if protocols_in is None:
            protocols_in = ['staircaseramp', 'staircaseramp (2)', 'ProtocolChonStaircaseRamp',
                            'staircaseramp_2kHz_fixed_ramp', 'staircaseramp (2)_2kHz',
                            'staircase-ramp', 'Staircase_hERG']
        for exp in data_dir:
            exp_files = os.listdir(os.path.join(data_path, exp))
            exp_files = [x for x in exp_files if any([y in x for y in protocols_in])]
            if not exp_files:
                continue
            protocols = set(['_'.join(x.split('_')[:-1]) for x in exp_files])
            if processed_path:
                with open(processed_path+'/'+exp+'/passed_wells.txt', 'r') as file:
                    passed_wells = file.read()
                passed_wells = [x for x in passed_wells.split('\n') if x]
                if passed_only:
                    wells = passed_wells
                else:
                    wells = [row + str(i).zfill(2) for row in string.ascii_uppercase[:16] for i in range(1, 25)]
            else:
                wells = [row + str(i).zfill(2) for row in string.ascii_uppercase[:16] for i in range(1, 25)]
            for prot in protocols:
                time_strs = [x.split('_')[-1] for x in exp_files if prot+'_'+x.split('_')[-1] == x]
                time_strs = sorted(time_strs)
                if len(time_strs) == 2:
                    time_strs = [time_strs]
                elif len(time_strs) == 4:
                    time_strs = [[time_strs[0], time_strs[2]], [time_strs[1], time_strs[3]]]
                for it, time_str in enumerate(time_strs):
                    filepath_before = os.path.join(data_path, exp,
                                                   f"{prot}_{time_str[0]}")
                    json_file_before = f"{prot}_{time_str[0]}"
                    before_trace = Trace(filepath_before, json_file_before)
                    filepath_after = os.path.join(data_path, exp,
                                                  f"{prot}_{time_str[1]}")
                    json_file_after = f"{prot}_{time_str[1]}"
                    after_trace = Trace(filepath_after, json_file_after)
                    # traces = {z:[x for x in os.listdir(data_path+'/'+exp+'/traces')
                    # if x.endswith('.csv') and all([y in x for y in [z+'-','subtracted']])]
                    # for z in protocols}
                    times = before_trace.get_times()
                    voltages = before_trace.get_voltage()
                    voltage_protocol = before_trace.get_voltage_protocol()
                    protocol_desc = voltage_protocol.get_all_sections()
                    ramp_bounds = detect_ramp_bounds(times, protocol_desc)
                    before_current_all = before_trace.get_trace_sweeps()
                    after_current_all = after_trace.get_trace_sweeps()

                    # Convert everything to nA...
                    before_current_all = {key: value * 1e-3 for key, value in before_current_all.items()}
                    after_current_all = {key: value * 1e-3 for key, value in after_current_all.items()}
                    for well in wells:
                        sweeps = before_current_all[well].shape[0]
                        before_current = before_current_all[well]
                        after_current = after_current_all[well]
                        sweep_dict = do_subtraction_plot(fig, times, sweeps, before_current, after_current,
                                                         voltages, ramp_bounds, well=None, protocol=None)
                        exp_list += [exp]*len(sweep_dict['sweeps'])
                        protocol_list += [prot]*len(sweep_dict['sweeps'])
                        well_list += [well]*len(sweep_dict['sweeps'])
                        sweep_list += sweep_dict['sweeps']
                        corr_list += sweep_dict['pcs']
                        if passed_wells:
                            if well in passed_wells:
                                passed = 'passed'
                            else:
                                passed = 'failed'
                            passed_list += [passed]*len(sweep_dict['sweeps'])
                        # fig.savefig(os.path.join(save_dir,
                        #  f"{exp}-{prot}-{well}-sweep{it}-subtraction-{passed}"))
                        fig.clf()
        if passed_wells:
            outdf = pd.DataFrame.from_dict({'exp': exp_list, 'protocol': protocol_list,
                                            'well': well_list, 'sweep': sweep_list, 'pc': corr_list,
                                            'passed': passed_list})
        else:
            outdf = pd.DataFrame.from_dict({'exp': exp_list, 'protocol': protocol_list,
                                            'well': well_list, 'sweep': sweep_list, 'pc': corr_list})
        outdf.to_csv(os.path.join(save_dir, 'subtraction_results.csv'))


def detect_ramp_bounds(times, voltage_sections, ramp_no=0):
    """
    Extract the the times at the start and end of the nth ramp in the protocol.

    @param times: np.array containing the time at which each sample was taken
    @param voltage_sections 2d np.array where each row describes a segment of the protocol: (tstart, tend, vstart, end)
    @param ramp_no: the index of the ramp to select. Defaults to 0 - the first ramp

    @returns tstart, tend: the start and end times for the ramp_no+1^nth ramp
    """

    ramps = [(tstart, tend, vstart, vend) for tstart, tend, vstart, vend
             in voltage_sections if vstart != vend]
    try:
        ramp = ramps[ramp_no]
    except IndexError:
        print(f"Requested {ramp_no+1}th ramp (ramp_no={ramp_no}),"
              " but there are only {len(ramps)} ramps")

    tstart, tend = ramp[:2]

    ramp_bounds = [np.argmax(times > tstart), np.argmax(times > tend)]
    return ramp_bounds
