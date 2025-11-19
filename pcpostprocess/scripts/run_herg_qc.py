#
# Runs leak subtraction and quality control on syncropatch data
#
#
import argparse
import importlib.util
import json
import logging
import multiprocessing
import os
import re
import string
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from syncropatch_export.trace import Trace
from syncropatch_export.voltage_protocols import VoltageProtocol

from pcpostprocess.detect_ramp_bounds import detect_ramp_bounds
from pcpostprocess.directory_builder import setup_output_directory
from pcpostprocess.hergQC import hERGQC
from pcpostprocess.infer_reversal import infer_reversal_potential
from pcpostprocess.leak_correct import fit_linear_leak, get_leak_corrected
from pcpostprocess.subtraction_plots import do_subtraction_plot


def starmap(n, func, iterable):
    """
    Like ``multiprocessing.Pool.starmap``, but does not use subprocesses when
    n=1.
    """
    if n > 1:  # pragma: no cover
        with multiprocessing.Pool(n, maxtasksperchild=1) as pool:
            return pool.starmap(func, iterable)
    return [func(*args) for args in iterable]


def run_from_command_line():  # pragma: no cover
    """
    Reads arguments from the command line and an ``export_config.py`` and then
    runs herg QC.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='The path to read input from')
    parser.add_argument('-o', '--output_dir', default='output',
                        help='The path to write output to')

    parser.add_argument(
        '-w', '--wells', nargs='+',
        help='A space separated list of wells to include. If not given, all'
             'wells are loaded.')

    parser.add_argument(
        '--output_traces', action='store_true',
        help='Add this flag to store (raw and processed) traces, as .csv files')
    parser.add_argument(
        '--export_failed', action='store_true',
        help='Add this flag to produce full output even for wells failing QC')

    parser.add_argument(
        '--Erev', default=-90.71, type=float,
        help='The calculated or estimated reversal potential.')
    parser.add_argument(
        '--reversal_spread_threshold', type=float, default=10,
        help='The maximum spread in reversal potential (across sweeps) allowed for QC')

    parser.add_argument(
        '-c', '--no_cpus', default=1, type=int,
        help='Number of workers to spawn in the multiprocessing pool (default: 1)')

    parser.add_argument('--figsize', nargs=2, type=int, default=[16, 18])

    parser.add_argument('--log_level', default='WARNING')

    args = parser.parse_args()

    # Import options from configuration file
    spec = importlib.util.spec_from_file_location(
        'export_config', os.path.join(args.data_directory, 'export_config.py'))
    export_config = importlib.util.module_from_spec(spec)
    sys.modules['export_config'] = export_config    # What's happening here!?
    spec.loader.exec_module(export_config)

    # Only from command line: set logging level
    logging.basicConfig(level=args.log_level)

    # Pass in gathered options and run
    run(
        args.data_directory,
        args.output_dir,
        export_config.D2S_QC,
        wells=args.wells,
        write_traces=args.output_traces,
        write_failed_traces=args.export_failed,
        write_map=export_config.D2S,
        reversal_potential=args.Erev,
        reversal_spread_threshold=args.reversal_spread_threshold,
        max_processes=args.no_cpus,
        figure_size=args.figsize,
        save_id=export_config.saveID,
    )


def run(data_path, output_path, qc_map, wells=None,
        write_traces=False, write_failed_traces=False, write_map={},
        reversal_potential=-90, reversal_spread_threshold=10,
        max_processes=1, figure_size=None, save_id=None):
    """
    Imports traces and runs QC+.

    Makes the following assumptions:

    This proceeds with the following steps:

    1. All wells (or those selected with ``wells``) are run through "plain" QC,
       for every protocol in ``qc_map``.






    @param data_path The path to read data from
    @param output_path The path to write output to
    @param qc_map A dictionary mapping input protocol names to output names.
           All protocols in this dictionary will be used for quality control.
    @param wells A list of strings indicating the wells to use, or ``None`` for
           all wells.
    @param write_traces Set to ``True`` to write (raw and processed) traces to
           the output directory in CSV format.
    @param write_failed_traces Set to ``True`` to write traces for wells
           failing quality control.  Ignored if ``write_traces=False`.
    @param write_map A dictionary like ``qc_map``, but specifying protocols to
           write traces for without using them in quality control. Ignored if
           ``write_traces=False`.
    @param reversal_potential The calculated reversal potential, in mV.
    @param reversal_spread_threshold The maximum reversal
    @param max_processes The maximum number of processes to run simultaneously
    @param figure_size An optional tuple specifying the size of figures to
           create

    @param save_id Used in some outputs, e.g. as part of CSV names

    """
    # TODO reversal_spread_threshold should be specified the same way as all
    #      other QC thresholds & parameters.

    # Create output path if necessary, and write info file
    output_path = setup_output_directory(output_path)

    # TODO Remove protocol selection here: this is done via the export file!
    #      Only protocols listed there are accepted

    # Select wells to use
    all_wells = [row + str(i).zfill(2) for row in string.ascii_uppercase[:16]
                 for i in range(1, 25)]
    if wells is None:
        wells = all_wells
    else:
        # Check wells exist
        not_found = set(wells) - set(all_wells)
        if len(not_found) == len(wells):
            raise ValueError(
                'None of the specified well names are in the list of'
                ' available wells.')
        elif len(not_found) > 0:
            raise ValueError(
                f'Specified well names {not_found} are not in the list of'
                ' available wells.')

    #
    # Regex to detect protocol names
    # 1. Starts with at least letter, number, underscore, hyphen, space, or
    #    opening or closing bracket
    # 2. An underscore
    # 3. Ends with a code 00.00.00 (where 0 is any number)
    #
    # TODO: Just want to check that it ends in a 6 digit date code
    protocols_regex = \
        r'^([a-z|A-Z|_|0-9| |\-|\(|\)]+)_([0-9][0-9]\.[0-9][0-9]\.[0-9][0-9])$'
    protocols_regex = re.compile(protocols_regex)

    # Gather protocol directories to use in a dictionary
    # { protocol_name: [time1, time2, time3, ...] }
    # such that protocol_name_time is a directory

    # TODO: Replace this by looping over qc_map and write_map?
    res_dict = {}
    for dirname in os.listdir(data_path):
        print(dirname, os.path.basename(dirname))
        dirname = os.path.basename(dirname)
        match = protocols_regex.match(dirname)

        if match is None:
            continue

        protocol_name = match.group(1)

        if not (protocol_name in qc_map or protocol_name in write_map):
            print(f'Skipping {protocol_name}')
            continue

        time = match.group(2)

        if protocol_name not in res_dict:
            res_dict[protocol_name] = []

        res_dict[protocol_name].append(time)

    def pront(*args):
        print('********')
        for arg in args:
            print(arg)

    pront(res_dict)
    # At this point, despite its name, res_dict is not a dictionary of results,
    # but a map of QC protocol names onto lists of times (see comment above)

    #
    # Prepare arguments to call `run_qc_for_protocol`
    #

    combined_dict = qc_map | write_map

    # Select QC protocols and times
    readnames, savenames, times_list = [], [], []
    for protocol in res_dict:
        if protocol not in qc_map:
            continue

        times = sorted(res_dict[protocol])
        pront(times)

        savename = qc_map[protocol]

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
            assert savename not in write_map.values()
            savenames.append(savename)
            times_list.append([times[1], times[3]])
            readnames.append(protocol)

        else:
            raise ValueError('Expecting 2 or 4 repeats of the QC protocol')

    # For two repeats, we now have
    #   savenames: short user name, one per QC protocol
    #   times_list: the time part of a dirname, [before, after]
    # For four repeats
    #   savenames: short user name, repeat has _2 added on
    #   times_list: [before1, after1], [before2, after2]

    pront(readnames, savenames, times_list)

    if not readnames:
        raise ValueError('No compatible protocols specified.')

    m = len(readnames)
    n = min(max_processes, m)
    args = zip(readnames, savenames, times_list, [output_path] * m,
               [data_path] * m, [wells] * m, [write_traces] * m, [save_id] * m)
    well_selections, qc_dfs = zip(*starmap(n, run_qc_for_protocol, args))

    pront(well_selections)
    pront(qc_dfs)

    #
    # Assuming a single QC protocol. At this point, we have
    #
    #  well_selections: a tuple ``(s1, s2)``, where ``s1`` lists the cells
    #                   passing QC on staircase 1, and ``s2`` lists those
    #                   passing QC on staircase 2.
    #  qc_dfs: a tuple ``(d1, d2)`` where ``d1`` is a dataframe of staircase 1
    #          results. It doesn't need to be a dataframe, really. Just a 2d
    #          matrix of ``n_cells`` rows, and ``n_crit + 2`` QC criteria.
    #          in addition to QC criteria, each row starts with the well code
    #          and ends with the shortened protocol name. The QC crit cells
    #          contain True of False, reasons for failing are not included.
    #

    qc_df = pd.concat(qc_dfs, ignore_index=True)
    pront(qc_df)

    #
    # At this point, ``qc_df`` is a single dataframe containing the information
    # from both qc_dfs. The protocol is indicated with the shortened name,
    # where for the second run it has _2 appended
    #

    # Combine QC protocls into overall_selection
    selection = [set(x) for x in well_selections]
    selection = selection[0].intersection(*selection[1:])

    # Store "plain QC" selections in "selected" files
    fname = os.path.join(output_path, f'selected-{save_id}.txt')
    with open(fname, 'w') as f:
        f.write('\n'.join(selection))
    for partial, protocol in zip(well_selections, list(savenames)):
        fname = os.path.join(output_path, f'selected-{save_id}-{protocol}.txt')
        with open(fname, 'w') as f:
            f.write('\n'.join(partial))

    #
    # Now go over _all_ protocols, including the QC protocols (AGAIN!), and
    # call extract_protocol() on them
    #
    pront(savenames, readnames, times_list)

    # Export all protocols
    savenames, readnames, times_list = [], [], []
    for protocol in res_dict:

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

        # TODO Else raise error?


    pront(savenames, readnames, times_list)

    wells_to_export = wells if write_failed_traces else selection
    logging.info(f'exporting wells {wells_to_export}')
    m = len(readnames)
    n = min(max_processes, m)
    args = zip(readnames, savenames, times_list, [wells_to_export] * m,
               [output_path] * m, [data_path] * m, [figure_size] * m,
               [reversal_potential] * m, [save_id] * m)
    dfs = starmap(n, extract_protocol, args)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    pront(len(dfs))
    for df in dfs:
        pront(df)
    sys.exit(1)

    if dfs:
        extract_df = pd.concat(dfs, ignore_index=True)
        extract_df['selected'] = extract_df['well'].isin(selection)
    else:
        logging.error("Didn't export any data")
        return

    logging.info(f"extract_df: {extract_df}")

    #
    # Do QC3 on first staircase, first sweep VS second staircase, second sweep
    # qc3.bookend check very first and very last staircases are similar
    #

    protocol, savename = list(qc_map.items())[0]
    times = sorted(res_dict[protocol])
    if len(times) == 4:
        qc3_bookend_dict = qc3_bookend(
            protocol, savename, times, wells, output_path, data_path,
            figure_size, save_id)
    else:
        #TODO: Better indicate that it wasn't run?
        qc3_bookend_dict = {well: True for well in qc_df.well.unique()}
    qc_df['qc3.bookend'] = [qc3_bookend_dict[well] for well in qc_df.well]
    pront(qc_df)

    #
    #
    #

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
        passed_QC_Erev_spread = E_rev_spread <= reversal_spread_threshold
        logging.info(f"passed_QC_Erev_spread {passed_QC_Erev_spread}")

        # R_leftover only considered for protocols used for QC (i.e. staircase protocols)
        passed_QC_R_leftover = np.all(sub_df[sub_df.protocol.isin(qc_map.values())]
                                      ['QC.R_leftover'].values)

        logging.info(f"passed_QC_R_leftover {passed_QC_R_leftover}")

        passed_QC_Erev_spread = E_rev_spread <= reversal_spread_threshold

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

    with open(os.path.join(output_path, 'chrono.txt'), 'w') as fout:
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
    qc_styled_df.to_latex(os.path.join(output_path, 'qc_table.tex'))

    # Save in csv format
    qc_df.to_csv(os.path.join(output_path, f'QC-{save_id}.csv'))

    # Write data to JSON file
    qc_df.to_json(os.path.join(output_path, f'QC-{save_id}.json'),
                  orient='records')

    #  Load only QC vals. TODO use a new variabile name to avoid confusion
    qc_vals_df = extract_df[['well', 'sweep', 'protocol', 'Rseal', 'Cm', 'Rseries']].copy()
    qc_vals_df['drug'] = 'before'
    qc_vals_df.to_csv(os.path.join(output_path, 'qc_vals_df.csv'))

    extract_df.to_csv(os.path.join(output_path, 'subtraction_qc.csv'))

    with open(os.path.join(output_path, 'passed_wells.txt'), 'w') as fout:
        for well, passed in passed_qc_dict.items():
            if passed:
                fout.write(well)
                fout.write('\n')


def create_qc_table(qc_df):
    """
    ???
    """

    if len(qc_df.index) == 0:
        return None

    if 'Unnamed: 0' in qc_df:
        qc_df = qc_df.drop('Unnamed: 0', axis='columns')

    qc_criteria = list(qc_df.drop(['protocol', 'well'], axis='columns').columns)

    def agg_func(x):
        x = x.values.flatten().astype(bool)
        return bool(np.all(x))

    qc_df[qc_criteria] = qc_df[qc_criteria].apply(lambda column: [elem == 'True' or elem is True for elem in column])

    qc_df['protocol'] = ['staircaseramp1_2' if p == 'staircaseramp2' else p
                         for p in qc_df.protocol]

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


def extract_protocol(readname, savename, time_strs, selected_wells, savedir,
                     data_path, figure_size, reversal_potential, save_id):
    # TODO: Tidy up argument order
    """
    ???
    """
    print(f'EXTRACT PROTOCOL {savename}')
    #logging.info(f"Exporting {readname} as {savename}")

    savedir = os.path.join(savedir, '2-extract-protocol')
    traces_dir = os.path.join(savedir, 'traces')
    os.makedirs(traces_dir, exist_ok=True)
    subtraction_plots_dir = os.path.join(savedir, 'subtraction_plots')
    os.makedirs(subtraction_plots_dir, exist_ok=True)

    row_dict = {}

    before_trace = Trace(
        os.path.join(data_path, f'{readname}_{time_strs[0]}'),
        f'{readname}_{time_strs[0]}.json')
    after_trace = Trace(
        os.path.join(data_path, f'{readname}_{time_strs[1]}'),
        f'{readname}_{time_strs[1]}')

    # Get times and voltages
    times = before_trace.get_times()
    voltages = before_trace.get_voltage()
    assert len(times) == len(after_trace.get_times())
    assert np.all(np.abs(times - after_trace.get_times()) < 1.8)
    assert np.all(np.abs(voltages - after_trace.get_voltage()) < 1.8)

    #  Find start of leak section
    voltage_protocol = before_trace.get_voltage_protocol()
    desc = voltage_protocol.get_all_sections()
    ramp_bounds = detect_ramp_bounds(times, desc)

    nsweeps = before_trace.NofSweeps
    assert nsweeps == after_trace.NofSweeps

    # Store voltages and times, using 2 different libraries...
    voltage_df = pd.DataFrame(
        np.vstack((times.flatten(), voltages.flatten())).T,
        columns=['time', 'voltage'])
    voltage_df.to_csv(os.path.join(
        traces_dir, f"{save_id}-{savename}-voltages.csv"))
    #np.savetxt(os.path.join(traces_dir, f"{save_id}-{savename}-times.csv"),
    #           times)

    qc_before = before_trace.get_onboard_QC_values()
    qc_after = after_trace.get_onboard_QC_values()

    before_data = before_trace.get_trace_sweeps()
    after_data = after_trace.get_trace_sweeps()

    for i_well, well in enumerate(selected_wells):
        if i_well % 24 == 0:
            logging.info('row ' + well[0])

        if None in qc_before[well] or None in qc_after[well]:
            continue

        # Save before and after drug traces as .csv
        for sweep in range(nsweeps):
            save_fname = os.path.join(
                traces_dir, f'{save_id}-{savename}-{well}-before-sweep{sweep}.csv')
            np.savetxt(save_fname, before_data[well][sweep], delimiter=',',
                       header='"current"')
            save_fname = os.path.join(
                traces_dir, f'{save_id}-{savename}-{well}-after-sweep{sweep}.csv')
            np.savetxt(save_fname, after_data[well][sweep], delimiter=',',
                       header='"current"')

    # plot subtraction
    fig = plt.figure(figsize=figure_size, layout='constrained')

    plot_dir = os.path.join(savedir, 'reversal_plots')
    os.makedirs(plot_dir, exist_ok=True)

    rows = []

    before_leak_current_dict = {}
    after_leak_current_dict = {}

    out1 = os.path.join(savedir, 'leak_correction',
                        f'{save_id}-{savename}-leak_fit-before')
    out2 = os.path.join(savedir, 'leak_correction',
                        f'{save_id}-{savename}-leak_fit-after')
    os.makedirs(out1, exist_ok=True)
    os.makedirs(out2, exist_ok=True)


    hergqc = hERGQC(voltages, before_trace.sampling_rate)
    for well in selected_wells:

        before_leak_currents = []
        after_leak_currents = []

        for sweep in range(nsweeps):
            row_dict = {
                'well': well,
                'sweep': sweep,
                'protocol': savename
            }

            qc_vals = qc_before[well][sweep]
            if qc_vals is None:
                continue
            if len(qc_vals) == 0:
                continue

            row_dict['Rseal'] = qc_vals[0]
            row_dict['Cm'] = qc_vals[1]
            row_dict['Rseries'] = qc_vals[2]

            before_params, before_leak = fit_linear_leak(
                before_data[well][sweep], voltages, times, *ramp_bounds,
                save_fname=os.path.join(out1, f'{well}_sweep{sweep}.png'))
            before_leak_currents.append(before_leak)

            # Convert linear regression parameters into conductance and reversal
            row_dict['gleak_before'] = before_params[1]
            row_dict['E_leak_before'] = -before_params[0] / before_params[1]

            after_params, after_leak = fit_linear_leak(
                after_data[well][sweep], voltages, times, *ramp_bounds,
                save_fname=os.path.join(out2, f'{well}_sweep{sweep}.png'))

            after_leak_currents.append(after_leak)

            # Convert linear regression parameters into conductance and reversal
            row_dict['gleak_after'] = after_params[1]
            row_dict['E_leak_after'] = -after_params[0] / after_params[1]

            subtracted_trace = before_data[well][sweep] - before_leak\
                - (after_data[well][sweep] - after_leak)
            after_corrected = after_data[well][sweep] - after_leak
            before_corrected = before_data[well][sweep] - before_leak

            E_rev_before = infer_reversal_potential(
                before_corrected, times, desc, voltages,
                output_path=os.path.join(
                    plot_dir, f'{well}_{savename}_sweep{sweep}_before'),
                known_Erev=reversal_potential)

            E_rev_after = infer_reversal_potential(
                after_corrected, times, desc, voltages,
                output_path=os.path.join(
                    plot_dir, f'{well}_{savename}_sweep{sweep}_after'),
                known_Erev=reversal_potential)

            E_rev = infer_reversal_potential(
                subtracted_trace, times, desc, voltages,
                output_path=os.path.join(
                    plot_dir, f'{well}_{savename}_sweep{sweep}_subtracted'),
                known_Erev=reversal_potential)

            row_dict['R_leftover'] =\
                np.sqrt(np.sum((after_corrected)**2)/(np.sum(before_corrected**2)))

            QC_R_leftover = np.all(row_dict['R_leftover'] < 0.5)
            row_dict['QC.R_leftover'] = QC_R_leftover

            row_dict['E_rev'] = E_rev
            row_dict['E_rev_before'] = E_rev_before
            row_dict['E_rev_after'] = E_rev_after

            row_dict['QC.Erev'] = E_rev < -50 and E_rev > -120

            # Check QC6 for each protocol (not just the staircase)



            times = before_trace.get_times()
            voltage = before_trace.get_voltage()
            voltage_protocol = before_trace.get_voltage_protocol()

            voltage_steps = [tstart
                             for tstart, tend, vstart, vend in
                             voltage_protocol.get_all_sections() if vend == vstart]

            current = hergqc.filter_capacitive_spikes(before_corrected - after_corrected,
                                                      times, voltage_steps)

            row_dict['QC6'] = hergqc.qc6(current, win=hergqc.qc6_win)[0]

            #  Assume there is only one sweep for all non-QC protocols
            rseal_before, cm_before, rseries_before = qc_before[well][0]
            rseal_after, cm_after, rseries_after = qc_after[well][0]

            qc1_1 = hergqc.qc1(rseal_before, cm_before, rseries_before)
            qc1_2 = hergqc.qc1(rseal_after, cm_after, rseries_after)

            row_dict['QC1'] = all([x for x, _ in qc1_1 + qc1_2])

            qc4 = hergqc.qc4([rseal_before, rseal_after],
                             [cm_before, cm_after],
                             [rseries_before, rseries_after])

            row_dict['QC4'] = all([x for x, _ in qc4])

            out_fname = os.path.join(traces_dir,
                                     f"{save_id}-{savename}-{well}-sweep{sweep}-subtracted.csv")
            np.savetxt(out_fname, subtracted_trace.flatten())

            rows.append(row_dict)

            param, leak = fit_linear_leak(current, voltage, times,
                                          *ramp_bounds)

            subtracted_trace = current - leak

            t_step = times[1] - times[0]
            row_dict['total before-drug flux'] = np.sum(current) * (1.0 / t_step)
            res = get_time_constant_of_first_decay(
                subtracted_trace, times, desc,
                os.path.join(
                    savedir, 'debug', '-120mV time constant',
                    f'{savename}-{well}-sweep{sweep}-time-constant-fit.png'),
                figure_size
            )

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

    for well in selected_wells:
        before_current = before_data[well]
        after_current = after_data[well]

        before_leak_currents = before_leak_current_dict[well]
        after_leak_currents = after_leak_current_dict[well]

        sub_df = extract_df[extract_df.well == well]

        if len(sub_df.index) == 0:
            continue

        sweeps = sorted(list(sub_df.sweep.unique()))

        do_subtraction_plot(fig, times, sweeps, before_current, after_current,
                            voltages, ramp_bounds, well=well,
                            protocol=savename)

        fig.savefig(os.path.join(subtraction_plots_dir,
                                 f'{save_id}-{savename}-{well}-sweep{sweep}-subtraction'))
        fig.clf()

    plt.close(fig)

    return extract_df


def run_qc_for_protocol(readname, savename, time_strs, output_path,
                        data_path, wells, write_traces, save_id):
    """
    Runs a QC procedure for a single protocol, on the selected wells.

    Assumes:
    - time_strs has length 2, corresponding to a before- and after-drug trace
    - each recording has 2 sweeps

    The following procedure is followed:
    - Traces are leak corrected with `fit_linear_leak`
    - Traces are drug subtracted
    - No capacitative spike filtering

    Also:
    - Creates a directory ``output_path/1-qc``
    - Writes leak subtracted traces at ``1-qc/save_id-...csv``
    - Makes leak correction plots at ``output_path/1-qc/leak_correction``.
      These are the plots made by `fit_linear_leak()`.

    @param readname The protocol name, without the time part
    @param savename The shorter name for the protocol
    @param time_strs The time part of the protocol names, must have length 2
    @param output_path The root directory to chuck everything in
    @param data_path The path to read data from
    @param wells The wells to process
    @param write_traces True if traces should be written too
    @param save_id

    Returns a tuple `(selected_wells, detailed_qc)`` where ``selected_wells``
    is a list containing all selected well names, and ``detailed_qc`` is a
    pandas dataframe containing the pass/fail status of all individual QC
    criteria, for all wells.
    """
    print(f'RUN HERG QC FOR {readname}, {time_strs}, {savename}')

    # TODO Tidy up argument order
    assert len(time_strs) == 2

    before_trace = Trace(
        os.path.join(data_path, f'{readname}_{time_strs[0]}'),
        f'{readname}_{time_strs[0]}.json')
    after_trace = Trace(
        os.path.join(data_path, f'{readname}_{time_strs[1]}'),
        f'{readname}_{time_strs[1]}.json')

    # Assert that protocols are exactly the same
    sampling_rate = before_trace.sampling_rate
    assert sampling_rate == after_trace.sampling_rate
    voltage = before_trace.get_voltage()
    assert np.all(voltage == after_trace.get_voltage())
    hergqc = hERGQC(voltage, sampling_rate)

    # Store stuff to "QC/leak_correction"
    save_dir = os.path.join(output_path, '1-qc')
    plot_dir = os.path.join(save_dir, 'leak_correction')
    os.makedirs(plot_dir, exist_ok=True)

    # Assume two sweeps!
    sweeps = [0, 1]
    nsweeps = len(sweeps)
    raw_before = before_trace.get_trace_sweeps(sweeps)
    raw_after = after_trace.get_trace_sweeps(sweeps)

    qc_before = before_trace.get_onboard_QC_values()
    qc_after = after_trace.get_onboard_QC_values()

    # Get ramp bounds
    times = before_trace.get_times()
    nsamples = len(times)
    v_protocol = VoltageProtocol.from_voltage_trace(voltage, times)
    v_sections = v_protocol.get_all_sections()
    ramp_bounds = detect_ramp_bounds(times, v_sections)

    df_rows = []
    selected_wells = []
    for well in wells:
        # Check if any cell first!
        if (None in qc_before[well][0]) or (None in qc_after[well][0]):
            continue

        corrected_before = np.empty((nsweeps, nsamples))
        corrected_after = np.empty((nsweeps, nsamples))
        for isweep in range(nsweeps):
            # Get corrected traces, and make plot
            _, before_leak = fit_linear_leak(
                raw_before[well][isweep], voltage, times, *ramp_bounds,
                save_fname=os.path.join(
                    plot_dir, f'{savename}-{well}-{isweep}-before.png'))
            _, after_leak = fit_linear_leak(
                raw_after[well][isweep], voltage, times, *ramp_bounds,
                save_fname=os.path.join(
                    plot_dir, f'{savename}-{well}-{isweep}-after.png'))

            corrected_before[isweep] = raw_before[well][isweep] - before_leak
            corrected_after[isweep] = raw_after[well][isweep] - after_leak

        # Run QC with raw currents
        QC = hergqc.run_qc(
            [v0 for t0, t1, v0, v1 in v_sections if v1 == v0],
            times, corrected_before, corrected_after,
            qc_before[well][0], qc_after[well][0], nsweeps)

        df_rows.append([well] + QC.passed_list())

        if QC.all_passed():
            selected_wells.append(well)

        # Save subtracted current in csv file
        if write_traces:
            header = '"current"'
            for i in range(nsweeps):
                savepath = os.path.join(
                    save_dir, f'{save_id}-{savename}-{well}-sweep{i}.csv')
                subtracted = corrected_before[i] - corrected_after[i]
                np.savetxt(savepath, subtracted, delimiter=',', header=header)

    # TODO: Depends on QC used
    column_labels = ['well', 'qc1.rseal', 'qc1.cm', 'qc1.rseries', 'qc2.raw',
                     'qc2.subtracted', 'qc3.raw', 'qc3.E4031', 'qc3.subtracted',
                     'qc4.rseal', 'qc4.cm', 'qc4.rseries', 'qc5.staircase',
                     'qc5.1.staircase', 'qc6.subtracted', 'qc6.1.subtracted',
                     'qc6.2.subtracted']

    # Add skipped wells
    ncriteria = len(column_labels) - 1
    for well in set(wells) - set([row[0] for row in df_rows]):
        df_rows.append([well] + [False] * ncriteria)

    # Create data frame and return
    df = pd.DataFrame(np.array(df_rows), columns=column_labels)
    df['protocol'] = savename

    return selected_wells, df


def qc3_bookend(readname, savename, time_strs, wells, output_path,
                data_path, figure_size, save_id):
    """
    Joey's "bookend" test, comparing a staircase before other protocols with a
    staircase after.

    This method assumes 4 time strings are given, for 4 files with 2 sweeps
    each. It loads data, performs leak correction, drug subtraction, and
    capacitance filtering, before comparing the first sweep of the first
    staircase with the second sweep of the second staircase.

    Also:
    - Creates a directory ``output_path/3-qc3-bookend``
    - Creates plots comparing staircase 1, sweep 1, with staircase 2, sweep 2.

    TODO: This method repeats lots of steps, unneccesarily:
    - loading all data
    - leak correction
    - drug subtraction

    @param readname The protocol name, without the time part
    @param savename The shorter name for the protocol
    @param time_strs The time part of the protocol names, must have length 4
    @param wells The wells to process
    @param output_path The root directory to chuck everything in
    @param data_path The path to read data from
    @param figure_size

    Returns a dictionary mapping well names to booleans.
    """
    print('RUN QC3 bookend')
    assert len(time_strs) == 4

    filepath_first_before = os.path.join(data_path, f'{readname}_{time_strs[0]}')
    filepath_last_before = os.path.join(data_path, f'{readname}_{time_strs[1]}')
    json_file_first_before = f"{readname}_{time_strs[0]}"
    json_file_last_before = f"{readname}_{time_strs[1]}"

    #  Each Trace object contains two sweeps
    first_before_trace = Trace(filepath_first_before, json_file_first_before)
    last_before_trace = Trace(filepath_last_before, json_file_last_before)

    times = first_before_trace.get_times()
    voltage = first_before_trace.get_voltage()

    voltage_protocol = first_before_trace.get_voltage_protocol()
    ramp_bounds = detect_ramp_bounds(
        times, voltage_protocol.get_all_sections())

    filepath_first_after = os.path.join(data_path, f'{readname}_{time_strs[2]}')
    filepath_last_after = os.path.join(data_path, f'{readname}_{time_strs[3]}')
    json_file_first_after = f"{readname}_{time_strs[2]}"
    json_file_last_after = f"{readname}_{time_strs[3]}"

    first_after_trace = Trace(filepath_first_after, json_file_first_after)
    last_after_trace = Trace(filepath_last_after, json_file_last_after)

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

    # Get steps for capacitance filtering
    voltage_steps = [tstart
                     for tstart, tend, vstart, vend in
                     voltage_protocol.get_all_sections() if vend == vstart]

    # Create QC object. Only uses qc3 which doesn't plot, so no plot_dir needed
    hergqc = hERGQC(sampling_rate=first_before_trace.sampling_rate,
                    voltage=voltage)

    # Update output path
    output_path = os.path.join(output_path, '3-qc3-bookend')
    os.makedirs(output_path, exist_ok=True)

    # Create figure - will be reused
    fig = plt.figure(figsize=figure_size)
    ax = fig.subplots()

    #  Iterate over all wells, perform qc3-bookend, plot and store
    res_dict = {}
    for well in np.array(wells).flatten():
        # First staircase, before drug and after drug, first sweep
        before_trace_first = get_leak_corrected(
            first_before_current_dict[well][0], voltage, times, *ramp_bounds)
        after_trace_first = get_leak_corrected(
            first_after_current_dict[well][0], voltage, times, *ramp_bounds)
        first = before_trace_first - after_trace_first

        # Second staircase, before drug and after drug, second sweep
        before_trace_last = get_leak_corrected(
            last_before_current_dict[well][-1], voltage, times, *ramp_bounds)
        after_trace_last = get_leak_corrected(
            last_after_current_dict[well][-1], voltage, times, *ramp_bounds)
        last = before_trace_last - after_trace_last

        trace1 = hergqc.filter_capacitive_spikes(first, times, voltage_steps)
        trace2 = hergqc.filter_capacitive_spikes(last, times, voltage_steps)

        res_dict[well] = hergqc.qc3(trace1, trace2)[0]

        save_fname = os.path.join(output_path, f'qc3_bookend-{well}.png')
        ax.cla()
        ax.plot(times, trace1)
        ax.plot(times, trace2)
        fig.savefig(save_fname)

    plt.close(fig)
    return res_dict


def get_time_constant_of_first_decay(
        trace, times, protocol_desc, output_path, figure_size):
    """
    ???
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    first_120mV_step_index = [
        i for i, line in enumerate(protocol_desc) if line[2] == 40][0] + 1

    tstart, tend, vstart, vend = protocol_desc[first_120mV_step_index, :]
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

    # TESTING ONLY
    # np.random.seed(1)

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
        fig = plt.figure(figsize=figure_size, constrained_layout=True)
        axs = fig.subplots(2)

        for ax in axs:
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylabel(r'$I_\mathrm{obs}$ (pA)')

        axs[-1].set_xlabel(r'$t$ (ms)')

        protocol_ax, fit_ax = axs
        protocol_ax.set_title('a', fontweight='bold', loc='left')
        fit_ax.set_title('b', fontweight='bold', loc='left')
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


if __name__ == '__main__':  # pragma: no cover
    run_from_command_line()
