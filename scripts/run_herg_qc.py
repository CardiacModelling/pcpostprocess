import argparse
import datetime
import importlib.util
import json
import logging
import multiprocessing
import os
import string
import subprocess
import sys

import cycler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
import scipy
from syncropatch_export.trace import Trace
from syncropatch_export.voltage_protocols import VoltageProtocol

from pcpostprocess.hergQC import hERGQC
from pcpostprocess.infer_reversal import infer_reversal_potential
from pcpostprocess.leak_correct import fit_linear_leak, get_leak_corrected
from pcpostprocess.subtraction_plots import do_subtraction_plot

matplotlib.use('Agg')
plt.rcParams["axes.formatter.use_mathtext"] = True

pool_kws = {'maxtasksperchild': 1}

color_cycle = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_cycle)

all_wells = [row + str(i).zfill(2) for row in string.ascii_uppercase[:16]
             for i in range(1, 25)]


def get_git_revision_hash() -> str:
    #  Requires git to be installed
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument('-c', '--no_cpus', default=1, type=int)
    parser.add_argument('--output_dir')
    parser.add_argument('-w', '--wells', nargs='+')
    parser.add_argument('--protocols', nargs='+')
    parser.add_argument('--reversal_spread_threshold', type=float, default=10)
    parser.add_argument('--export_failed', action='store_true')
    parser.add_argument('--selection_file')
    parser.add_argument('--subtracted_only', action='store_true')
    parser.add_argument('--figsize', nargs=2, type=int, default=[5, 8])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_level', default='INFO')
    parser.add_argument('--Erev', default=-90.71, type=float)
    parser.add_argument('--output_traces', action='store_true',
                        help="When true output raw and processed traces as .csv files")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    if args.output_dir is None:
        args.output_dir = os.path.join('output', 'hergqc')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'info.txt'), 'w') as description_fout:
        git_hash = get_git_revision_hash()
        datetimestr = str(datetime.datetime.now())
        description_fout.write(f"Date: {datetimestr}\n")
        description_fout.write(f"Commit {git_hash}\n")
        command = " ".join(sys.argv)
        description_fout.write(f"Command: {command}\n")

    spec = importlib.util.spec_from_file_location(
        'export_config',
        os.path.join(args.data_directory,
                     'export_config.py'))

    if args.wells is None:
        args.wells = all_wells
        wells = args.wells

    else:
        wells = args.wells

    # Import and exec config file
    global export_config
    export_config = importlib.util.module_from_spec(spec)

    sys.modules['export_config'] = export_config
    spec.loader.exec_module(export_config)

    export_config.savedir = args.output_dir

    args.saveID = export_config.saveID
    args.savedir = export_config.savedir
    args.D2S = export_config.D2S
    args.D2SQC = export_config.D2S_QC

    protocols_regex = \
        r'^([a-z|A-Z|_|0-9| |\-|\(|\)]+)_([0-9][0-9]\.[0-9][0-9]\.[0-9][0-9])$'

    protocols_regex = re.compile(protocols_regex)

    res_dict = {}
    for dirname in os.listdir(args.data_directory):
        dirname = os.path.basename(dirname)
        match = protocols_regex.match(dirname)

        if match is None:
            continue

        protocol_name = match.group(1)

        if protocol_name not in export_config.D2S\
           and protocol_name not in export_config.D2S_QC:
            continue

        # map name to new name using export_config
        # savename = export_config.D2S[protocol_name]
        time = match.group(2)

        if protocol_name not in res_dict:
            res_dict[protocol_name] = []

        res_dict[protocol_name].append(time)

    readnames, savenames, times_list = [], [], []

    combined_dict = {**export_config.D2S, **export_config.D2S_QC}

    # Select QC protocols and times
    for protocol in res_dict:
        if protocol not in export_config.D2S_QC:
            continue

        times = sorted(res_dict[protocol])

        savename = export_config.D2S_QC[protocol]

        if len(times) == 2:
            savenames.append(savename)
            readnames.append(protocol)
            times_list.append(times)

        elif len(times) == 4:
            savenames.append(savename)
            readnames.append(protocol)
            times_list.append([times[0], times[2]])

            # Make seperate savename for protocol repeat
            savename = combined_dict[protocol] + '_2'
            assert savename not in export_config.D2S.values()
            savenames.append(savename)
            times_list.append([times[1], times[3]])
            readnames.append(protocol)

    with multiprocessing.Pool(min(args.no_cpus, len(readnames)),
                              **pool_kws) as pool:

        pool_argument_list = zip(readnames, savenames, times_list,
                                 [args for i in readnames])
        well_selections, qc_dfs = \
            list(zip(*pool.starmap(run_qc_for_protocol, pool_argument_list)))

    qc_df = pd.concat(qc_dfs, ignore_index=True)

    # Do QC which requires both repeats
    # qc3.bookend check very first and very last staircases are similar
    protocol, savename = list(export_config.D2S_QC.items())[0]
    times = sorted(res_dict[protocol])
    if len(times) == 4:
        qc3_bookend_dict = qc3_bookend(protocol, savename,
                                       times, args)
    else:
        qc3_bookend_dict = {well: True for well in qc_df.well.unique()}

    qc_df['qc3.bookend'] = [qc3_bookend_dict[well] for well in qc_df.well]

    savedir = args.output_dir
    saveID = export_config.saveID

    if not os.path.exists(os.path.join(args.output_dir, savedir)):
        os.makedirs(os.path.join(args.output_dir, savedir))

    #  qc_df will be updated and saved again, but it's useful to save them here for debugging
    # Write qc_df to file
    qc_df.to_csv(os.path.join(savedir, 'QC-%s.csv' % saveID))

    # Write data to JSON file
    qc_df.to_json(os.path.join(savedir, 'QC-%s.json' % saveID),
                  orient='records')

    # Overwrite old files
    for protocol in list(export_config.D2S_QC.values()):
        fname = os.path.join(savedir, 'selected-%s-%s.txt' % (saveID, protocol))
        with open(fname, 'w') as fout:
            pass

    overall_selection = []
    for well in qc_df.well.unique():
        failed = False
        for well_selection, protocol in zip(well_selections,
                                            list(savenames)):

            logging.debug(f"{well_selection} selected from protocol {protocol}")
            fname = os.path.join(savedir, 'selected-%s-%s.txt' %
                                 (saveID, protocol))
            if well not in well_selection:
                failed = True
            else:
                with open(fname, 'a') as fout:
                    fout.write(well)
                    fout.write('\n')

        # well in every selection
        if not failed:
            overall_selection.append(well)

    selectedfile = os.path.join(savedir, 'selected-%s.txt' % saveID)
    with open(selectedfile, 'w') as fout:
        for well in overall_selection:
            fout.write(well)
            fout.write('\n')

    logfile = os.path.join(savedir, 'table-%s.txt' % saveID)
    with open(logfile, 'a') as f:
        f.write('\\end{table}\n')

    # Export all protocols
    savenames, readnames, times_list = [], [], []
    for protocol in res_dict:

        if args.protocols:
            if savename not in args.protocols:
                continue

        # Sort into chronological order
        times = sorted(res_dict[protocol])
        savename = combined_dict[protocol]

        readnames.append(protocol)

        if len(times) == 2:
            savenames.append(savename)
            times_list.append(times)

        elif len(times) == 4:
            savenames.append(savename)
            times_list.append(times[::2])

            # Make seperate savename for protocol repeat
            savename = combined_dict[protocol] + '_2'
            assert savename not in combined_dict.values()
            savenames.append(savename)
            times_list.append(times[1::2])
            readnames.append(protocol)

    wells_to_export = wells if args.export_failed else overall_selection

    logging.info(f"exporting wells {wells}")

    no_protocols = len(res_dict)

    args_list = list(zip(readnames, savenames, times_list, [wells_to_export] *
                         len(savenames),
                         [args for i in readnames]))

    with multiprocessing.Pool(min(args.no_cpus, no_protocols),
                              **pool_kws) as pool:
        dfs = list(pool.starmap(extract_protocol, args_list))

    extract_df = pd.concat(dfs, ignore_index=True)
    extract_df['selected'] = extract_df['well'].isin(overall_selection)

    logging.info(f"extract_df: {extract_df}")

    qc_erev_spread = {}
    erev_spreads = {}
    passed_qc_dict = {}
    for well in extract_df.well.unique():
        logging.info(f"Checking QC for well {well}")
        # Select only this well
        sub_df = extract_df[extract_df.well == well]
        sub_qc_df = qc_df[qc_df.well == well]

        passed_qc3_bookend = np.all(sub_qc_df['qc3.bookend'].values)
        logging.info(f"passed_QC3_bookend_all {passed_qc3_bookend}")
        passed_QC_Erev_all = np.all(sub_df['QC.Erev'].values)
        passed_QC1_all = np.all(sub_df.QC1.values)
        logging.info(f"passed_QC1_all {passed_QC1_all}")

        passed_QC4_all = np.all(sub_df.QC4.values)
        logging.info(f"passed_QC4_all {passed_QC4_all}")
        passed_QC6_all = np.all(sub_df.QC6.values)
        logging.info(f"passed_QC6_all {passed_QC1_all}")

        E_revs = sub_df['E_rev'].values.flatten().astype(np.float64)
        E_rev_spread = E_revs.max() - E_revs.min()
        # QC Erev spread: check spread in reversal potential isn't too large
        passed_QC_Erev_spread = E_rev_spread <= args.reversal_spread_threshold
        logging.info(f"passed_QC_Erev_spread {passed_QC_Erev_spread}")

        qc_erev_spread[well] = passed_QC_Erev_spread
        erev_spreads[well] = E_rev_spread

        passed_QC_Erev_all = np.all(sub_df['QC.Erev'].values)
        logging.info(f"passed_QC_Erev_all {passed_QC_Erev_all}")

        was_selected = np.all(sub_df['selected'].values)

        passed_qc = passed_qc3_bookend and was_selected\
            and passed_QC_Erev_all and passed_QC6_all\
            and passed_QC_Erev_spread and passed_QC1_all\
            and passed_QC4_all

        passed_qc_dict[well] = passed_qc

    extract_df['passed QC'] = [passed_qc_dict[well] for well in extract_df.well]
    extract_df['QC.Erev.spread'] = [qc_erev_spread[well] for well in extract_df.well]
    extract_df['Erev_spread'] = [erev_spreads[well] for well in extract_df.well]

    chrono_dict = {times[0]: prot for prot, times in zip(savenames, times_list)}

    with open(os.path.join(args.output_dir, 'chrono.txt'), 'w') as fout:
        for key in sorted(chrono_dict):
            val = chrono_dict[key]
            #  Output order of protocols
            fout.write(val)
            fout.write('\n')

    #  Update qc_df
    update_cols = []
    for index, vals in qc_df.iterrows():
        append_dict = {}

        well = vals['well']

        sub_df = extract_df[(extract_df.well == well)]

        append_dict['QC.Erev.all_protocols'] =\
            np.all(sub_df['QC.Erev'])

        append_dict['QC.Erev.spread'] =\
            np.all(sub_df['QC.Erev.spread'])

        append_dict['QC1.all_protocols'] =\
            np.all(sub_df['QC1'])

        append_dict['QC4.all_protocols'] =\
            np.all(sub_df['QC4'])

        append_dict['QC6.all_protocols'] =\
            np.all(sub_df['QC6'])

        update_cols.append(append_dict)

    for key in append_dict:
        qc_df[key] = [row[key] for row in update_cols]

    qc_styled_df = create_qc_table(qc_df)
    logging.info(qc_styled_df)
    qc_styled_df.to_excel(os.path.join(args.output_dir, 'qc_table.xlsx'))
    qc_styled_df.to_latex(os.path.join(args.output_dir, 'qc_table.tex'))

    # Save in csv format
    qc_df.to_csv(os.path.join(savedir, 'QC-%s.csv' % saveID))

    # Write data to JSON file
    qc_df.to_json(os.path.join(savedir, 'QC-%s.json' % saveID),
                  orient='records')

    #  Load only QC vals. TODO use a new variabile name to avoid confusion
    qc_vals_df = extract_df[['well', 'sweep', 'protocol', 'Rseal', 'Cm', 'Rseries']].copy()
    qc_vals_df['drug'] = 'before'
    qc_vals_df.to_csv(os.path.join(args.output_dir, 'qc_vals_df.csv'))

    extract_df.to_csv(os.path.join(args.output_dir, 'subtraction_qc.csv'))

    with open(os.path.join(args.output_dir, 'passed_wells.txt'), 'w') as fout:
        for well, passed in passed_qc_dict.items():
            if passed:
                fout.write(well)
                fout.write('\n')


def create_qc_table(qc_df):
    if len(qc_df.index) == 0:
        return None

    if 'Unnamed: 0' in qc_df:
        qc_df = qc_df.drop('Unnamed: 0', axis='columns')

    qc_criteria = list(qc_df.drop(['protocol', 'well'], axis='columns').columns)

    def agg_func(x):
        x = x.values.flatten().astype(bool)
        return bool(np.all(x))

    qc_df[qc_criteria] = qc_df[qc_criteria].astype(bool)

    qc_df['protocol'] = ['staircaseramp1_2' if p == 'staircaseramp2' else p
                         for p in qc_df.protocol]

    print(qc_df.protocol.unique())

    fails_dict = {}
    no_wells = 384

    dfs = []
    protocol_headings = ['staircaseramp1', 'staircaseramp1_2', 'all']
    for protocol in protocol_headings:
        fails_dict = {}
        for crit in sorted(qc_criteria) + ['all']:
            if protocol != 'all':
                sub_df = qc_df[qc_df.protocol == protocol].copy()
            else:
                sub_df = qc_df.copy()

            agg_dict = {crit: agg_func for crit in qc_criteria}
            if crit != 'all':
                col = sub_df.groupby('well').agg(agg_dict).reset_index()[crit]
                vals = col.values.flatten()
                n_passed = vals.sum()
            else:
                excluded = [crit for crit in qc_criteria
                            if 'all' in crit or 'spread' in crit or 'bookend' in crit]
                if protocol == 'all':
                    excluded = []
                crit_included = [crit for crit in qc_criteria if crit not in excluded]

                col = sub_df.groupby('well').agg(agg_dict).reset_index()
                n_passed = np.sum(np.all(col[crit_included].values, axis=1).flatten())

            crit = re.sub('_', r'\_', crit)
            fails_dict[crit] = (crit, no_wells - n_passed)

        new_df = pd.DataFrame.from_dict(fails_dict, orient='index',
                                        columns=['crit', 'wells failing'])
        new_df['protocol'] = protocol
        new_df.set_index('crit')
        dfs.append(new_df)

    ret_df = pd.concat(dfs, ignore_index=True)

    ret_df['wells failing'] = ret_df['wells failing'].astype(int)

    ret_df['protocol'] = pd.Categorical(ret_df['protocol'],
                                        categories=protocol_headings,
                                        ordered=True)

    return ret_df


def extract_protocol(readname, savename, time_strs, selected_wells, args):
    logging.info(f"extracting {savename}")
    savedir = args.output_dir
    saveID = args.saveID

    traces_dir = os.path.join(savedir, 'traces')

    if not os.path.exists(traces_dir):
        try:
            os.makedirs(traces_dir)
        except FileExistsError:
            pass

    row_dict = {}

    subtraction_plots_dir = os.path.join(savedir, 'subtraction_plots')

    if not os.path.isdir(subtraction_plots_dir):
        try:
            os.makedirs(subtraction_plots_dir)
        except FileExistsError:
            pass

    logging.info(f"Exporting {readname} as {savename}")

    filepath_before = os.path.join(args.data_directory,
                                   f"{readname}_{time_strs[0]}")
    filepath_after = os.path.join(args.data_directory,
                                  f"{readname}_{time_strs[1]}")
    json_file_before = f"{readname}_{time_strs[0]}"
    json_file_after = f"{readname}_{time_strs[1]}"
    before_trace = Trace(filepath_before,
                         json_file_before)
    after_trace = Trace(filepath_after,
                        json_file_after)

    voltage_protocol = before_trace.get_voltage_protocol()
    times = before_trace.get_times()
    voltages = before_trace.get_voltage()

    #  Find start of leak section
    desc = voltage_protocol.get_all_sections()
    ramp_bounds = detect_ramp_bounds(times, desc)
    tstart, tend = ramp_bounds

    nsweeps_before = before_trace.NofSweeps = 2
    nsweeps_after = after_trace.NofSweeps = 2

    assert nsweeps_before == nsweeps_after

    # Time points
    times_before = before_trace.get_times()
    times_after = after_trace.get_times()

    try:
        assert all(np.abs(times_before - times_after) < 1e-8)
    except Exception as exc:
        logging.warning(f"Exception thrown when handling {savename}: ", str(exc))
        return

    header = "\"current\""

    qc_before = before_trace.get_onboard_QC_values()
    qc_after = after_trace.get_onboard_QC_values()
    qc_vals_all = before_trace.get_onboard_QC_values()

    for i_well, well in enumerate(selected_wells):  # Go through all wells
        if i_well % 24 == 0:
            logging.info('row ' + well[0])

        if args.selection_file:
            if well not in selected_wells:
                continue

        if None in qc_before[well] or None in qc_after[well]:
            continue

        if args.output_traces:
            # Save 'before drug' trace as .csv
            for sweep in range(nsweeps_before):
                out = before_trace.get_trace_sweeps([sweep])[well][0]
                save_fname = os.path.join(traces_dir, f"{saveID}-{savename}-"
                                          f"{well}-before-sweep{sweep}.csv")

                np.savetxt(save_fname, out, delimiter=',',
                           header=header)

        if args.output_traces:
            # Save 'after drug' trace as .csv
            for sweep in range(nsweeps_after):
                save_fname = os.path.join(traces_dir, f"{saveID}-{savename}-"
                                          f"{well}-after-sweep{sweep}.csv")
                out = after_trace.get_trace_sweeps([sweep])[well][0]
                if len(out) > 0:
                    np.savetxt(save_fname, out,
                               delimiter=',', comments='', header=header)

    voltage_before = before_trace.get_voltage()
    voltage_after = after_trace.get_voltage()

    assert len(voltage_before) == len(voltage_after)
    assert len(voltage_before) == len(times_before)
    assert len(voltage_after) == len(times_after)
    voltage = voltage_before

    voltage_df = pd.DataFrame(np.vstack((times_before.flatten(),
                                         voltage.flatten())).T,
                              columns=['time', 'voltage'])

    if not os.path.exists(os.path.join(traces_dir,
                                       f"{saveID}-{savename}-voltages.csv")):
        voltage_df.to_csv(os.path.join(traces_dir,
                                       f"{saveID}-{savename}-voltages.csv"))

    np.savetxt(os.path.join(traces_dir, f"{saveID}-{savename}-times.csv"),
               times_before)

    # plot subtraction
    fig = plt.figure(figsize=args.figsize, layout='constrained')

    reversal_plot_dir = os.path.join(savedir, 'reversal_plots')

    rows = []

    before_leak_current_dict = {}
    after_leak_current_dict = {}

    for well in selected_wells:
        before_current = before_trace.get_trace_sweeps()[well]
        after_current = after_trace.get_trace_sweeps()[well]

        before_leak_currents = []
        after_leak_currents = []

        out_dir = os.path.join(savedir,
                               f"{saveID}-{savename}-leak_fit-before")

        for sweep in range(before_current.shape[0]):
            row_dict = {
                'well': well,
                'sweep': sweep,
                'protocol': savename
            }

            qc_vals = qc_vals_all[well][sweep]
            if qc_vals is None:
                continue
            if len(qc_vals) == 0:
                continue

            row_dict['Rseal'] = qc_vals[0]
            row_dict['Cm'] = qc_vals[1]
            row_dict['Rseries'] = qc_vals[2]

            before_params, before_leak = fit_linear_leak(before_current[sweep, :],
                                                         voltages, times,
                                                         *ramp_bounds,
                                                         output_dir=out_dir,
                                                         save_fname=f"{well}_sweep{sweep}.png"
                                                         )

            before_leak_currents.append(before_leak)

            out_dir = os.path.join(savedir,
                                   f"{saveID}-{savename}-leak_fit-after")
            # Convert linear regression parameters into conductance and reversal
            row_dict['gleak_before'] = before_params[1]
            row_dict['E_leak_before'] = -before_params[0] / before_params[1]

            after_params, after_leak = fit_linear_leak(after_current[sweep, :],
                                                       voltages, times,
                                                       *ramp_bounds,
                                                       save_fname=f"{well}_sweep{sweep}.png",
                                                       output_dir=out_dir)

            after_leak_currents.append(after_leak)

            # Convert linear regression parameters into conductance and reversal
            row_dict['gleak_after'] = after_params[1]
            row_dict['E_leak_after'] = -after_params[0] / after_params[1]

            subtracted_trace = before_current[sweep, :] - before_leak\
                - (after_current[sweep, :] - after_leak)
            after_corrected = after_current[sweep, :] - after_leak
            before_corrected = before_current[sweep, :] - before_leak

            E_rev_before = infer_reversal_potential(before_corrected, times,
                                                    desc, voltages, plot=True,
                                                    output_path=os.path.join(reversal_plot_dir,
                                                                             f"{well}_{savename}_sweep{sweep}_before"),
                                                    known_Erev=args.Erev)

            E_rev_after = infer_reversal_potential(after_corrected, times,
                                                   desc, voltages,
                                                   plot=True,
                                                   output_path=os.path.join(reversal_plot_dir,
                                                                            f"{well}_{savename}_sweep{sweep}_after"),
                                                   known_Erev=args.Erev)

            E_rev = infer_reversal_potential(subtracted_trace, times, desc,
                                             voltages, plot=True,
                                             output_path=os.path.join(reversal_plot_dir,
                                                                      f"{well}_{savename}_sweep{sweep}_subtracted"),
                                             known_Erev=args.Erev)

            row_dict['R_leftover'] =\
                np.sqrt(np.sum((after_corrected)**2)/(np.sum(before_corrected**2)))

            row_dict['QC.R_leftover'] = row_dict['R_leftover'] < 0.5

            row_dict['E_rev'] = E_rev
            row_dict['E_rev_before'] = E_rev_before
            row_dict['E_rev_after'] = E_rev_after

            row_dict['QC.Erev'] = E_rev < -50 and E_rev > -120

            # Check QC6 for each protocol (not just the staircase)
            plot_dir = os.path.join(savedir, 'debug')

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            hergqc = hERGQC(sampling_rate=before_trace.sampling_rate,
                            plot_dir=plot_dir,
                            n_sweeps=before_trace.NofSweeps)

            times = before_trace.get_times()
            voltage = before_trace.get_voltage()
            voltage_protocol = before_trace.get_voltage_protocol()

            voltage_steps = [tstart
                             for tstart, tend, vstart, vend in
                             voltage_protocol.get_all_sections() if vend == vstart]

            current = hergqc.filter_capacitive_spikes(before_corrected - after_corrected,
                                                      times, voltage_steps)

            row_dict['QC6'] = hergqc.qc6(current,
                                         win=hergqc.qc6_win,
                                         label='0')

            #  Assume there is only one sweep for all non-QC protocols
            rseal_before, cm_before, rseries_before = qc_before[well][0]
            rseal_after, cm_after, rseries_after = qc_after[well][0]

            row_dict['QC1'] = all(list(hergqc.qc1(rseal_before, cm_before, rseries_before)) +
                                  list(hergqc.qc1(rseal_after, cm_after, rseries_after)))

            row_dict['QC4'] = all(hergqc.qc4([rseal_before, rseal_after],
                                             [cm_before, cm_after],
                                             [rseries_before, rseries_after]))

            if args.output_traces:
                out_fname = os.path.join(traces_dir,
                                         f"{saveID}-{savename}-{well}-sweep{sweep}-subtracted.csv")

                np.savetxt(out_fname, subtracted_trace.flatten())
            rows.append(row_dict)

            param, leak = fit_linear_leak(current, voltage, times,
                                          *ramp_bounds)

            subtracted_trace = current - leak

            t_step = times[1] - times[0]
            row_dict['total before-drug flux'] = np.sum(current) * (1.0 / t_step)
            res = \
                get_time_constant_of_first_decay(subtracted_trace, times, desc,
                                                 args=args,
                                                 output_path=os.path.join(args.output_dir,
                                                                          'debug',
                                                                          '-120mV time constant',
                                                                          f"{savename}-{well}-sweep"
                                                                          "{sweep}-time-constant-fit.png"))

            row_dict['-120mV decay time constant 1'] = res[0][0]
            row_dict['-120mV decay time constant 2'] = res[0][1]
            row_dict['-120mV decay time constant 3'] = res[1]
            row_dict['-120mV peak current'] = res[2]

        before_leak_current_dict[well] = np.vstack(before_leak_currents)
        after_leak_current_dict[well] = np.vstack(after_leak_currents)

    extract_df = pd.DataFrame.from_dict(rows)
    logging.debug(extract_df)

    times = before_trace.get_times()
    voltages = before_trace.get_voltage()

    before_current_all = before_trace.get_trace_sweeps()
    after_current_all = after_trace.get_trace_sweeps()

    # Convert everything to nA...
    before_current_all = {key: value * 1e-3 for key, value in before_current_all.items()}
    after_current_all = {key: value * 1e-3 for key, value in after_current_all.items()}

    before_leak_current_dict = {key: value * 1e-3 for key, value in before_leak_current_dict.items()}
    after_leak_current_dict = {key: value * 1e-3 for key, value in after_leak_current_dict.items()}

    for well in selected_wells:
        before_current = before_current_all[well]
        after_current = after_current_all[well]

        before_leak_currents = before_leak_current_dict[well]
        after_leak_currents = after_leak_current_dict[well]

        sub_df = extract_df[extract_df.well == well]

        if len(sub_df.index):
            continue

        sweeps = sorted(list(sub_df.sweep.unique()))
        sub_df = sub_df.set_index('sweep')
        logging.debug(sub_df)

        do_subtraction_plot(fig, times, sweeps, before_current, after_current,
                            extract_df, voltages, ramp_bounds, well=well)

        fig.savefig(os.path.join(subtraction_plots_dir,
                                 f"{saveID}-{savename}-{well}-sweep{sweep}-subtraction"))
        fig.clf()

    plt.close(fig)

    protocol_dir = os.path.join(traces_dir, 'protocols')
    if not os.path.exists(protocol_dir):
        try:
            os.makedirs(protocol_dir)
        except FileExistsError:
            pass

    # extract protocol
    protocol = before_trace.get_voltage_protocol()
    protocol.export_txt(os.path.join(protocol_dir,
                                     f"{saveID}-{savename}.txt"))

    json_protocol = before_trace.get_voltage_protocol_json()

    with open(os.path.join(protocol_dir, f"{saveID}-{savename}.json"), 'w') as fout:
        json.dump(json_protocol, fout)

    return extract_df


def run_qc_for_protocol(readname, savename, time_strs, args):
    df_rows = []

    assert len(time_strs) == 2

    filepath_before = os.path.join(args.data_directory,
                                   f"{readname}_{time_strs[0]}")
    json_file_before = f"{readname}_{time_strs[0]}"

    filepath_after = os.path.join(args.data_directory,
                                  f"{readname}_{time_strs[1]}")
    json_file_after = f"{readname}_{time_strs[1]}"

    logging.debug(f"loading {json_file_after} and {json_file_before}")

    before_trace = Trace(filepath_before,
                         json_file_before)

    after_trace = Trace(filepath_after,
                        json_file_after)

    assert before_trace.sampling_rate == after_trace.sampling_rate

    # Convert to s
    sampling_rate = before_trace.sampling_rate

    savedir = args.output_dir
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    before_voltage = before_trace.get_voltage()
    after_voltage = after_trace.get_voltage()

    # Assert that protocols are exactly the same
    assert np.all(before_voltage == after_voltage)

    voltage = before_voltage

    sweeps = [0, 1]
    raw_before_all = before_trace.get_trace_sweeps(sweeps)
    raw_after_all = after_trace.get_trace_sweeps(sweeps)

    selected_wells = []
    for well in args.wells:

        plot_dir = os.path.join(savedir, "debug", f"debug_{well}_{savename}")

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Setup QC instance. We could probably just do this inside the loop
        hergqc = hERGQC(sampling_rate=sampling_rate,
                        plot_dir=plot_dir,
                        voltage=before_voltage)

        qc_before = before_trace.get_onboard_QC_values()
        qc_after = after_trace.get_onboard_QC_values()

        # Check if any cell first!
        if (None in qc_before[well][0]) or (None in qc_after[well][0]):
            # no_cell = True
            continue

        else:
            # no_cell = False
            pass

        nsweeps = before_trace.NofSweeps
        assert after_trace.NofSweeps == nsweeps

        before_currents_corrected = np.empty((nsweeps, before_trace.NofSamples))
        after_currents_corrected = np.empty((nsweeps, after_trace.NofSamples))

        before_currents = np.empty((nsweeps, before_trace.NofSamples))
        after_currents = np.empty((nsweeps, after_trace.NofSamples))

        # Get ramp times from protocol description
        voltage_protocol = VoltageProtocol.from_voltage_trace(voltage,
                                                              before_trace.get_times())

        #  Find start of leak section
        desc = voltage_protocol.get_all_sections()
        ramp_locs = np.argwhere(desc[:, 2] != desc[:, 3]).flatten()
        tstart = desc[ramp_locs[0], 0]
        tend = voltage_protocol.get_ramps()[0][1]

        times = before_trace.get_times()

        ramp_bounds = [np.argmax(times > tstart), np.argmax(times > tend)]

        assert after_trace.NofSamples == before_trace.NofSamples

        for sweep in range(nsweeps):
            before_raw = np.array(raw_before_all[well])[sweep, :]
            after_raw = np.array(raw_after_all[well])[sweep, :]

            before_params1, before_leak = fit_linear_leak(before_raw,
                                                          voltage,
                                                          times,
                                                          *ramp_bounds,
                                                          save_fname=f"{well}-sweep{sweep}-before.png",
                                                          output_dir=savedir)

            after_params1, after_leak = fit_linear_leak(after_raw,
                                                        voltage,
                                                        times,
                                                        *ramp_bounds,
                                                        save_fname=f"{well}-sweep{sweep}-after.png",
                                                        output_dir=savedir)

            before_currents_corrected[sweep, :] = before_raw - before_leak
            after_currents_corrected[sweep, :] = after_raw - after_leak

            before_currents[sweep, :] = before_raw
            after_currents[sweep, :] = after_raw

        logging.info(f"{well} {savename}\n----------")
        logging.info(f"sampling_rate is {sampling_rate}")

        voltage_steps = [tstart
                         for tstart, tend, vstart, vend in
                         voltage_protocol.get_all_sections() if vend == vstart]

        # Run QC with raw currents
        selected, QC = hergqc.run_qc(voltage_steps, times,
                                     before_currents,
                                     after_currents,
                                     np.array(qc_before[well])[0, :],
                                     np.array(qc_after[well])[0, :], nsweeps)

        df_rows.append([well] + list(QC))

        if selected:
            selected_wells.append(well)

        # Save subtracted current in csv file
        header = "\"current\""

        for i in range(nsweeps):

            savepath = os.path.join(savedir,
                                    f"{args.saveID}-{savename}-{well}-sweep{i}.csv")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            subtracted_current = before_currents_corrected[i, :] - after_currents_corrected[i, :]
            np.savetxt(savepath, subtracted_current, delimiter=',',
                       comments='', header=header)

    column_labels = ['well', 'qc1.rseal', 'qc1.cm', 'qc1.rseries', 'qc2.raw',
                     'qc2.subtracted', 'qc3.raw', 'qc3.E4031', 'qc3.subtracted',
                     'qc4.rseal', 'qc4.cm', 'qc4.rseries', 'qc5.staircase',
                     'qc5.1.staircase', 'qc6.subtracted', 'qc6.1.subtracted',
                     'qc6.2.subtracted']

    df = pd.DataFrame(np.array(df_rows), columns=column_labels)

    missing_wells_dfs = []
    # Add onboard qc to dataframe
    for well in args.wells:
        if well not in df['well'].values:
            onboard_qc_df = pd.DataFrame([[well] + [False for col in
                                                    list(df)[1:]]],
                                         columns=list(df))
            missing_wells_dfs.append(onboard_qc_df)
    df = pd.concat([df] + missing_wells_dfs, ignore_index=True)

    df['protocol'] = savename

    return selected_wells, df


def qc3_bookend(readname, savename, time_strs, args):
    plot_dir = os.path.join(args.output_dir, args.savedir,
                            f"{args.saveID}-{savename}-qc3-bookend")

    filepath_first_before = os.path.join(args.data_directory,
                                         f"{readname}_{time_strs[0]}")
    filepath_last_before = os.path.join(args.data_directory,
                                        f"{readname}_{time_strs[1]}")
    json_file_first_before = f"{readname}_{time_strs[0]}"
    json_file_last_before = f"{readname}_{time_strs[1]}"

    #  Each Trace object contains two sweeps
    first_before_trace = Trace(filepath_first_before,
                               json_file_first_before)
    last_before_trace = Trace(filepath_last_before,
                              json_file_last_before)

    times = first_before_trace.get_times()
    voltage = first_before_trace.get_voltage()

    voltage_protocol = first_before_trace.get_voltage_protocol()
    ramp_bounds = detect_ramp_bounds(times,
                                     voltage_protocol.get_all_sections())
    filepath_first_after = os.path.join(args.data_directory,
                                        f"{readname}_{time_strs[2]}")
    filepath_last_after = os.path.join(args.data_directory,
                                       f"{readname}_{time_strs[3]}")
    json_file_first_after = f"{readname}_{time_strs[2]}"
    json_file_last_after = f"{readname}_{time_strs[3]}"

    first_after_trace = Trace(filepath_first_after,
                              json_file_first_after)
    last_after_trace = Trace(filepath_last_after,
                             json_file_last_after)

    # Ensure that all traces use the same voltage protocol
    assert np.all(first_before_trace.get_voltage() == last_before_trace.get_voltage())
    assert np.all(first_after_trace.get_voltage() == last_after_trace.get_voltage())
    assert np.all(first_before_trace.get_voltage() == first_after_trace.get_voltage())
    assert np.all(first_before_trace.get_voltage() == last_before_trace.get_voltage())

    #  Ensure that the same number of sweeps were used
    assert first_before_trace.NofSweeps == last_before_trace.NofSweeps

    first_before_current_dict = first_before_trace.get_trace_sweeps()
    first_after_current_dict = first_after_trace.get_trace_sweeps()
    last_before_current_dict = last_before_trace.get_trace_sweeps()
    last_after_current_dict = last_after_trace.get_trace_sweeps()

    #  Do leak subtraction and store traces for each well
    #  TODO Refactor this code into a single loop. There's no need to store each individual trace.
    before_traces_first = {}
    before_traces_last = {}
    after_traces_first = {}
    after_traces_last = {}
    first_processed = {}
    last_processed = {}

    #  Iterate over all wells
    for well in np.array(all_wells).flatten():
        first_before_current = first_before_current_dict[well][0, :]
        first_after_current = first_after_current_dict[well][0, :]
        last_before_current = last_before_current_dict[well][-1, :]
        last_after_current = last_after_current_dict[well][-1, :]

        before_traces_first[well] = get_leak_corrected(first_before_current,
                                                       voltage, times,
                                                       *ramp_bounds)
        before_traces_last[well] = get_leak_corrected(last_before_current,
                                                      voltage, times,
                                                      *ramp_bounds)

        after_traces_first[well] = get_leak_corrected(first_after_current,
                                                      voltage, times,
                                                      *ramp_bounds)
        after_traces_last[well] = get_leak_corrected(last_after_current,
                                                     voltage, times,
                                                     *ramp_bounds)

        # Store subtracted traces
        first_processed[well] = before_traces_first[well] - after_traces_first[well]
        last_processed[well] = before_traces_last[well] - after_traces_last[well]

    voltage_protocol = VoltageProtocol.from_voltage_trace(voltage, times)

    hergqc = hERGQC(sampling_rate=first_before_trace.sampling_rate,
                    plot_dir=plot_dir,
                    voltage=voltage)

    assert first_before_trace.NofSweeps == last_before_trace.NofSweeps

    voltage_steps = [tstart
                     for tstart, tend, vstart, vend in
                     voltage_protocol.get_all_sections() if vend == vstart]
    res_dict = {}

    fig = plt.figure(figsize=args.figsize)
    ax = fig.subplots()
    for well in args.wells:
        trace1 = hergqc.filter_capacitive_spikes(
            first_processed[well], times, voltage_steps
        ).flatten()

        trace2 = hergqc.filter_capacitive_spikes(
            last_processed[well], times, voltage_steps
        ).flatten()

        passed = hergqc.qc3(trace1, trace2)

        res_dict[well] = passed

        save_fname = os.path.join(args.output_dir,
                                  'debug',
                                  f"debug_{well}_{savename}",
                                  'qc3_bookend')

        ax.plot(times, trace1)
        ax.plot(times, trace2)

        fig.savefig(save_fname)
        ax.cla()

    plt.close(fig)
    return res_dict


def get_time_constant_of_first_decay(trace, times, protocol_desc, args, output_path):

    if output_path:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

    first_120mV_step_index = [i for i, line in enumerate(protocol_desc) if line[2] == 40][0]

    tstart, tend, vstart, vend = protocol_desc[first_120mV_step_index + 1, :]
    assert (vstart == vend)
    assert (vstart == -120.0)

    indices = np.argwhere((times >= tstart) & (times <= tend))

    # find peak current
    peak_current = np.min(trace[indices])
    peak_index = np.argmax(np.abs(trace[indices]))
    peak_time = times[indices[peak_index]][0]

    indices = np.argwhere((times >= peak_time) & (times <= tend - 50))

    def fit_func(x, args=None):
        # Pass 'args=single' when we want to use a single exponential.
        # Otherwise use 2 exponentials
        if args:
            single = args == 'single'
        else:
            single = False

        if not single:
            a, b, c, d = x
            if d < b:
                b, d = d, b
            prediction = c * np.exp((-1.0/d) * (times[indices] - peak_time)) + \
                a * np.exp((-1.0/b) * (times[indices] - peak_time))
        else:
            a, b = x
            prediction = a * np.exp((-1.0/b) * (times[indices] - peak_time))

        return np.sum((prediction - trace[indices])**2)

    bounds = [
        (-np.abs(trace).max()*2, 0),
        (1e-12, 5e3),
        (-np.abs(trace).max()*2, 0),
        (1e-12, 5e3),
    ]

    #  Repeat optimisation with different starting guesses
    x0s = [[np.random.uniform(lower_b, upper_b) for lower_b, upper_b in bounds] for i in range(100)]

    x0s = [[a, b, c, d] if d < b else [a, d, c, b] for (a, b, c, d) in x0s]

    best_res = None
    for x0 in x0s:
        res = scipy.optimize.minimize(fit_func, x0=x0,
                                      bounds=bounds)
        if best_res is None:
            best_res = res
        elif res.fun < best_res.fun and res.success and res.fun != 0:
            best_res = res
    res1 = best_res

    #  Re-run with single exponential
    bounds = [
        (-np.abs(trace).max()*2, 0),
        (1e-12, 5e3),
    ]

    #  Repeat optimisation with different starting guesses
    x0s = [[np.random.uniform(lower_b, upper_b) for lower_b, upper_b in bounds] for i in range(100)]

    best_res = None
    for x0 in x0s:
        res = scipy.optimize.minimize(fit_func, x0=x0,
                                      bounds=bounds, args=('single',))
        if best_res is None:
            best_res = res
        elif res.fun < best_res.fun and res.success and res.fun != 0:
            best_res = res
    res2 = best_res

    if not res2:
        logging.warning('finding 120mv decay timeconstant failed:' + str(res))

    if output_path and res:
        fig = plt.figure(figsize=args.figsize, constrained_layout=True)
        axs = fig.subplots(2)

        for ax in axs:
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylabel(r'$I_\mathrm{obs}$ (pA)')
            ax.set_xlabel(r'$t$ (ms)')

        protocol_ax, fit_ax = axs
        protocol_ax.set_title('a', fontweight='bold')
        fit_ax.set_title('b', fontweight='bold')
        fit_ax.plot(peak_time, peak_current, marker='x', color='red')

        a, b, c, d = res1.x

        if d < b:
            b, d = d, b

        e, f = res2.x

        fit_ax.plot(times[indices], trace[indices], color='grey',
                    alpha=.5)
        fit_ax.plot(times[indices], c * np.exp((-1.0/d) * (times[indices] - peak_time))
                    + a * np.exp(-(1.0/b) * (times[indices] - peak_time)),
                    color='red', linestyle='--')

        res_string = r'$\tau_{1} = ' f"{d:.1f}" r'\mathrm{ms}'\
            r'\; \tau_{2} = ' f"{b:.1f}" r'\mathrm{ms}$'

        fit_ax.annotate(res_string, xy=(0.5, 0.05), xycoords='axes fraction')

        protocol_ax.plot(times, trace)
        protocol_ax.axvspan(peak_time, tend - 50, alpha=.5, color='grey')

        fig.savefig(output_path)
        fit_ax.set_yscale('symlog')

        dirname, filename = os.path.split(output_path)
        filename = 'log10_' + filename
        fig.savefig(os.path.join(dirname, filename))

        fit_ax.cla()

        dirname, filename = os.path.split(output_path)
        filename = 'single_exp_' + filename
        output_path = os.path.join(dirname, filename)

        fit_ax.plot(times[indices], trace[indices], color='grey',
                    alpha=.5)
        fit_ax.plot(times[indices], e * np.exp((-1.0/f) * (times[indices] - peak_time)),
                    color='red', linestyle='--')

        res_string = r'$\tau = ' f"{f:.1f}" r'\mathrm{ms}$'

        fit_ax.annotate(res_string, xy=(0.5, 0.05), xycoords='axes fraction')
        fig.savefig(output_path)

        dirname, filename = os.path.split(output_path)
        filename = 'log10_' + filename
        fit_ax.set_yscale('symlog')
        fig.savefig(os.path.join(dirname, filename))

        plt.close(fig)

    return (d, b), f, peak_current if res else (np.nan, np.nan), np.nan, peak_current


if __name__ == '__main__':
    main()
