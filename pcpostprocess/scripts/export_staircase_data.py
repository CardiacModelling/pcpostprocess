import argparse
import datetime
import importlib.util
import json
import logging
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
from syncropatch_export.trace import Trace as SyncropatchTrace
from syncropatch_export.voltage_protocols import VoltageProtocol

from pcpostprocess.detect_ramp_bounds import detect_ramp_bounds
from pcpostprocess.hergQC import hERGQC
from pcpostprocess.infer_reversal import infer_reversal_potential
from pcpostprocess.subtraction_plots import do_subtraction_plot
from pcpostprocess.leak_correct import fit_linear_leak

pool_kws = {'maxtasksperchild': 1}

color_cycle = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_cycle)

matplotlib.use('Agg')

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

    qc_dfs = []
    # Do QC which requires both repeats
    passed_qc_dict = {}
    for time_strs, (readname, savename) in zip(times_list, export_config.D2S_QC.items()):
        for well in wells:
            passed, qc_df = run_qc(readname, savename, well, time_strs,
                                   args)
            qc_dfs.append(qc_df)
            passed_qc_dict[well] = True

    qc_df = pd.concat(qc_dfs, ignore_index=True)

    # Write qc_df to file
    qc_df.to_csv(os.path.join(args.savedir, 'QC-%s.csv' % args.saveID))

    # Write data to JSON file
    qc_df.to_json(os.path.join(args.savedir, 'QC-%s.json' % args.saveID),
                  orient='records')

    # Overwrite old files
    for protocol in list(export_config.D2S_QC.values()):
        fname = os.path.join(args.savedir, 'selected-%s-%s.txt' % (args.saveID, protocol))
        with open(fname, 'w') as fout:
            pass

    chrono_dict = {times[0]: prot for prot, times in zip(savenames, times_list)}

    with open(os.path.join(args.output_dir, 'chrono.txt'), 'w') as fout:
        for key in sorted(chrono_dict):
            val = chrono_dict[key]
            #  Output order of protocols
            fout.write(val)
            fout.write('\n')

    for time_strs, (readname, savename) in zip(times_list, export_config.D2S_QC.items()):
        for well in wells:
            qc_df = extract_protocol(readname, savename, well, time_strs,
                                     args)
            qc_dfs.append(qc_df)

    qc_styled_df = create_qc_table(qc_df)
    logging.info(qc_styled_df)

    qc_styled_df.to_excel(os.path.join(args.output_dir, 'qc_table.xlsx'))
    qc_styled_df.to_latex(os.path.join(args.output_dir, 'qc_table.tex'))

    # Save in csv format
    qc_df.to_csv(os.path.join(args.savedir, 'QC-%s.csv' % args.saveID))

    # Write data to JSON file
    qc_df.to_json(os.path.join(args.savedir, 'QC-%s.json' % args.saveID),
                  orient='records')

    #  Load only QC vals. TODO use a new variabile name to avoid confusion
    qc_vals_df = qc_df[['well', 'sweep', 'protocol', 'Rseal', 'Cm', 'Rseries']].copy()
    qc_vals_df['drug'] = 'before'
    qc_vals_df.to_csv(os.path.join(args.output_dir, 'qc_vals_df.csv'))

    qc_df.to_csv(os.path.join(args.output_dir, 'subtraction_qc.csv'))

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

    qc_df[qc_criteria] = qc_df[qc_criteria].apply(lambda column: [elem == 'True' or elem is True for elem in column])

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


def run_qc(readname, savename, well, time_strs, args):

    assert len(time_strs) == 2
    filepath_before = os.path.join(args.data_directory,
                                   f"{readname}_{time_strs[0]}")
    json_file_before = f"{readname}_{time_strs[0]}"

    filepath_after = os.path.join(args.data_directory,
                                  f"{readname}_{time_strs[1]}")
    json_file_after = f"{readname}_{time_strs[1]}"

    logging.debug(f"loading {json_file_after} and {json_file_before}")

    before_trace = SyncropatchTrace(filepath_before,
                                    json_file_before)

    after_trace = SyncropatchTrace(filepath_after,
                                   json_file_after)

    assert before_trace.sampling_rate == after_trace.sampling_rate

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

    hergqc = hERGQC(sampling_rate=sampling_rate,
                    # plot_dir=plot_dir,
                    voltage=before_voltage)

    plot_dir = os.path.join(savedir, "debug", f"debug_{well}_{savename}")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    qc_before = before_trace.get_onboard_QC_values()
    qc_after = after_trace.get_onboard_QC_values()

    # Check if any cell first!
    if (None in qc_before[well][0]) or (None in qc_after[well][0]):
        no_cell = True
        return False, pd.DataFrame()

    else:
        no_cell = False

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

    voltage_steps = [tend
                     for tstart, tend, vstart, vend in
                     voltage_protocol.get_all_sections() if vend == vstart]

    # Run QC with raw currents
    passed, QC = hergqc.run_qc(voltage_steps, times,
                               before_currents_corrected,
                               after_currents_corrected,
                               np.array(qc_before[well])[0, :],
                               np.array(qc_after[well])[0, :], nsweeps)

    QC = list(QC)
    df_rows = [[well] + list(QC)]

    selected = np.all(QC) and not no_cell
    if selected:
        selected_wells.append(well)

    header = "\"current\""
    for i in range(nsweeps):

        savepath = os.path.join(savedir,
                                f"{args.saveID}-{savename}-{well}-sweep{i}.csv")
        subtracted_current = before_currents_corrected[i, :] - after_currents_corrected[i, :]

        if args.output_traces:
            if not os.path.exists(savedir):
                os.makedirs(savedir)

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

    return passed, df


def extract_protocol(readname, savename, well, time_strs, args):
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
    before_trace = SyncropatchTrace(filepath_before,
                                    json_file_before)
    after_trace = SyncropatchTrace(filepath_after,
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

    qc_vals_all = before_trace.get_onboard_QC_values()

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

        QC_R_leftover = np.all(row_dict['R_leftover'] < 0.5)
        row_dict['QC.R_leftover'] = QC_R_leftover

        row_dict['E_rev'] = E_rev
        row_dict['E_rev_before'] = E_rev_before
        row_dict['E_rev_after'] = E_rev_after

        row_dict['QC.Erev'] = E_rev < -50 and E_rev > -120

        if args.output_traces:
            out_fname = os.path.join(traces_dir,
                                     f"{saveID}-{savename}-{well}-sweep{sweep}-subtracted.csv")

            np.savetxt(out_fname, subtracted_trace.flatten())
        rows.append(row_dict)

    extract_df = pd.DataFrame.from_dict(rows)
    logging.debug(extract_df)

    times = before_trace.get_times()
    voltages = before_trace.get_voltage()

    before_current_all = before_trace.get_trace_sweeps()
    after_current_all = after_trace.get_trace_sweeps()

    before_current = before_current_all[well]
    after_current = after_current_all[well]

    sub_df = extract_df[extract_df.well == well]

    if len(sub_df.index) == 0:
        return pd.DataFrame()

    sweeps = sorted(list(sub_df.sweep.unique()))

    do_subtraction_plot(fig, times, sweeps, before_current, after_current,
                        voltages, ramp_bounds, well=well,
                        protocol=savename)

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


if __name__ == '__main__':
    main()
