import argparse
import json
import logging
import os
import string

import cycler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
import scipy
import seaborn as sns
from syncropatch_export.voltage_protocols import VoltageProtocol

from pcpostprocess.scripts.run_herg_qc import create_qc_table


matplotlib.use('Agg')

pool_kws = {'maxtasksperchild': 1}

color_cycle = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_cycle)
sns.set_palette(sns.color_palette(color_cycle))


def get_wells_list(input_dir):
    regex = re.compile(f"{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9])-after")
    wells = []

    for f in filter(regex.match, os.listdir(input_dir)):
        well = re.search(regex, f).groups(2)[1]
        if well not in wells:
            wells.append(well)
    return list(np.unique(wells))


def get_protocol_list(input_dir):
    regex = re.compile(f"{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9])-after")
    protocols = []
    for f in filter(regex.match, os.listdir(input_dir)):
        well = re.search(regex, f).groups(3)[0]
        if protocols not in protocols:
            protocols.append(well)
    return list(np.unique(protocols))


def main():

    description = ""
    parser = argparse.ArgumentParser(description)

    parser.add_argument('data_dir', type=str, help="path to the directory containing the subtract_leak results")
    parser.add_argument('qc_estimates_file')
    parser.add_argument('--cpus', '-c', default=1, type=int)
    parser.add_argument('--wells', '-w', nargs='+', default=None)
    parser.add_argument('--output', '-o', default='output')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('-r', '--reversal', type=float, default=np.nan)
    # parser.add_argument('--selection_file', default=None, type=str)
    parser.add_argument('--experiment_name', default='newtonrun4')
    parser.add_argument('--figsize', type=int, nargs=2, default=[5, 3])
    parser.add_argument('--output_all', action='store_true')
    parser.add_argument('--log_level', default='INFO')

    global args
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=args.log_level)
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level)

    global experiment_name
    experiment_name = args.experiment_name

    global output_dir
    output_dir = os.path.join(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    leak_parameters_df = pd.read_csv(os.path.join(args.data_dir, 'subtraction_qc.csv'))

    qc_df = pd.read_csv(os.path.join(args.data_dir, f"QC-{experiment_name}.csv"))

    qc_styled_df = create_qc_table(qc_df)
    qc_styled_df = qc_styled_df.pivot(columns='protocol', index='crit')
    qc_styled_df.to_excel(os.path.join(output_dir, 'qc_table.xlsx'))
    qc_styled_df.to_latex(os.path.join(output_dir, 'qc_table.tex'))
    qc_vals_df = pd.read_csv(os.path.join(args.qc_estimates_file))

    qc_df.protocol = ['staircaseramp1' if protocol == 'staircaseramp' else protocol
                      for protocol in qc_df.protocol]
    qc_df.protocol = ['staircaseramp1_2' if protocol == 'staircaseramp_2' else protocol
                      for protocol in qc_df.protocol]

    leak_parameters_df.protocol = ['staircaseramp1' if protocol == 'staircaseramp' else protocol
                                   for protocol in leak_parameters_df.protocol]
    leak_parameters_df.protocol = ['staircaseramp1_2' if protocol == 'staircaseramp_2' else protocol
                                   for protocol in leak_parameters_df.protocol]

    print(leak_parameters_df.protocol.unique())

    with open(os.path.join(args.data_dir, 'passed_wells.txt')) as fin:
        global passed_wells
        passed_wells = fin.read().splitlines()

    # Compute new variables
    leak_parameters_df = compute_leak_magnitude(leak_parameters_df)

    global wells
    wells = leak_parameters_df.well.unique()
    global protocols
    protocols = leak_parameters_df.protocol.unique()

    try:
        chrono_fname = os.path.join(args.data_dir, 'chrono.txt')
        with open(chrono_fname, 'r') as fin:
            lines = fin.read().splitlines()
            protocol_order = [line.split(' ')[0] for line in lines]

            protocol_order = ['staircaseramp1' if p == 'staircaseramp' else p
                              for p in protocol_order]

            protocol_order = ['staircaseramp1_2' if p == 'staircaseramp_2' else p
                              for p in protocol_order]

        leak_parameters_df['protocol'] = pd.Categorical(leak_parameters_df['protocol'],
                                                        categories=protocol_order,
                                                        ordered=True)

        qc_vals_df['protocol'] = pd.Categorical(qc_vals_df['protocol'],
                                                categories=protocol_order,
                                                ordered=True)

        leak_parameters_df.sort_values(['protocol', 'sweep'], inplace=True)
    except FileNotFoundError as exc:
        logging.warning(str(exc))
        logger.warning('no chronological information provided. Sorting alphabetically')
        leak_parameters_df.sort_values(['protocol', 'sweep'])

    scatterplot_timescale_E_obs(leak_parameters_df)

    do_chronological_plots(leak_parameters_df)
    do_chronological_plots(leak_parameters_df, normalise=True)

    attrition_df = create_attrition_table(qc_df, leak_parameters_df)
    attrition_df.to_latex(os.path.join(output_dir, 'attrition.tex'))

    if 'passed QC' not in leak_parameters_df.columns and\
       'passed QC6a' in leak_parameters_df.columns:
        leak_parameters_df['passed QC'] = leak_parameters_df['passed QC6a']

    plot_leak_conductance_change_sweep_to_sweep(leak_parameters_df)
    plot_reversal_change_sweep_to_sweep(leak_parameters_df)
    plot_spatial_passed(leak_parameters_df)
    plot_reversal_spread(leak_parameters_df)
    if np.isfinite(args.reversal):
        plot_spatial_Erev(leak_parameters_df)

    leak_parameters_df['passed QC'] = [well in passed_wells for well in leak_parameters_df.well]
    qc_vals_df['passed QC'] = [well in passed_wells for well in qc_vals_df.well]

    # do_scatter_matrices(leak_parameters_df, qc_vals_df)
    plot_histograms(leak_parameters_df, qc_vals_df)

    # Very resource intensive
    # overlay_reversal_plots(leak_parameters_df)
    # do_combined_plots(leak_parameters_df)


def compute_leak_magnitude(df, lims=[-120, 60]):
    def compute_magnitude(g, E, lims=lims):
        # RMSE
        lims = np.array(lims)
        evals = (lims - E)**3 * np.abs(g) / 3
        return np.sqrt(evals[1] - evals[0]) / np.sqrt(lims[1] - lims[0])

    before_lst = []
    after_lst = []
    for i, row in df.iterrows():
        g_before = row['gleak_before']
        E_before = row['E_leak_before']
        leak_magnitude_before = compute_magnitude(g_before, E_before)
        before_lst.append(leak_magnitude_before)

        g_after = row['gleak_after']
        E_after = row['E_leak_after']
        leak_magnitude_after = compute_magnitude(g_after, E_after)
        after_lst.append(leak_magnitude_after)

    df['pre-drug leak magnitude'] = before_lst
    df['post-drug leak magnitude'] = after_lst

    return df


def scatterplot_timescale_E_obs(df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    df = df[(df.well.isin(passed_wells))].sort_values('protocol')

    plot_df = {}

    protocols = list(df.protocol.unique())

    if '-120mV decay time constant 3' in df:
        df['40mV decay time constant'] = df['-120mV decay time constant 3']

    #  Shift values so that reversal ramp is close to -120mV step
    plot_dfs = []
    for well in df.well.unique():
        E_rev_values = df[df.well == well]['E_rev'].values[:-1]
        E_leak_values = df[df.well == well]['E_leak_before'].values[1:]
        decay_values = df[df.well == well]['40mV decay time constant'].values[1:]
        plot_df = pd.DataFrame([(well, p, E_rev, decay, Eleak) for p, E_rev, decay, Eleak
                                in zip(protocols, E_rev_values, decay_values, E_leak_values)],
                               columns=['well', 'protocol', 'E_rev', '40mV decay time constant',
                                        'E_leak'])
        plot_dfs.append(plot_df)

    plot_df = pd.concat(plot_dfs, ignore_index=True)
    print(plot_df)

    plot_df['E_leak'] = (plot_df.set_index('well')['E_leak'] - plot_df.groupby('well')
                         ['E_leak'].mean()).reset_index()['E_leak']

    sns.scatterplot(data=plot_df, y='40mV decay time constant',
                    x='E_rev', ax=ax, hue='well', style='well')

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel(r'$\tau$ (ms)')
    ax.set_xlabel(r'$E_\mathrm{obs}$')

    fig.savefig(os.path.join(output_dir, "decay_timescale_vs_E_rev_scatter.pdf"))
    ax.cla()

    sns.lineplot(data=plot_df, y='40mV decay time constant',
                 x='E_rev', hue='well', style='well',
                 ax=ax)

    ax.set_ylabel(r'$\tau$ (ms)')
    ax.set_xlabel(r'$E_\mathrm{obs}$')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(os.path.join(output_dir, "decay_timescale_vs_E_rev_line.pdf"))
    ax.cla()

    plot_df['E_rev'] = (plot_df.set_index('well')['E_rev'] - plot_df.groupby('well')
                        ['E_rev'].mean()).reset_index()['E_rev']
    sns.scatterplot(data=plot_df, y='E_leak',
                    x='E_rev', ax=ax, hue='well', style='well')

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel(r'$E_\mathrm{leak} - \bar E_\mathrm{leak}$ (ms)')
    ax.set_xlabel(r'$E_\mathrm{obs} - \bar E_\mathrm{obs}$')

    fig.savefig(os.path.join(output_dir, "E_leak_vs_E_rev_scatter.pdf"))
    ax.cla()


def do_chronological_plots(df, normalise=False):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    sub_dir = os.path.join(output_dir, 'chrono_plots')
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    vars = ['gleak_after', 'gleak_before',
            'E_leak_after', 'R_leftover', 'E_leak_before',
            'E_leak_after', 'E_rev', 'pre-drug leak magnitude',
            'post-drug leak magnitude',
            'E_rev_before', 'Cm', 'Rseries',
            '-120mV decay time constant 1',
            '-120mV decay time constant 2',
            '-120mV decay time constant 3',
            '-120mV peak current']

    # df = df[leak_parameters_df['selected']]
    df = df[df['passed QC']].copy()

    relabel_dict = {protocol: r'$d_{' f"{i}" r'}$' for i, protocol in
                    enumerate(df.protocol.unique())}

    df = df.replace({'protocol': relabel_dict})

    units = {
        # 'gleak_after': r'',
        # 'gleak_before':,
        # 'E_leak_after':,
        # 'E_leak_before':,
        'pre-drug leak magnitude': 'pA',
        '-120mV decay time constant 1': 'ms',
        '-120mV decay time constant 2': 'ms',
        '-120mV decay time constant 3': 'ms'
    }

    pretty_vars = {
        'pre-drug leak magnitude': r'$\bar{I}_\mathrm{l}$',
        '-120mV time constant 1': r'$\tau_{1}$',
        '-120mV time constant 2': r'$\tau_{2}$',
        '-120mV time constant 3': r'$\tau$'
    }

    def label_func(p, s):
        p = p[1:-1]
        return r'$' + str(p) + r'^{(' + str(s) + r')}$'

    ax.spines[['top', 'right']].set_visible(False)

    for var in vars:
        if var not in df:
            continue
        df['x'] = [label_func(p, s) for p, s in zip(df.protocol, df.sweep)]
        hist = sns.lineplot(data=df, x='x', y=var, hue='well',
                            legend=True)
        ax = hist.axes

        xlim = list(ax.get_xlim())
        xlim[1] = xlim[1] + 2.5
        ax.set_xlim(xlim)

        ax.legend(frameon=False, fontsize=8)

        if var == 'E_rev' and np.isfinite(args.reversal):
            ax.axhline(args.reversal, linestyle='--', color='grey', label='Calculated Nernst potential')
        ax.set_xlabel('')

        if var in pretty_vars and var in units:
            ax.set_ylabel(f"{pretty_vars[var]} ({units[var]})")

        ax.get_legend().set_title('')
        legend_handles, _ = ax.get_legend_handles_labels()
        ax.legend(legend_handles, ['failed QC', 'passed QC'], bbox_to_anchor=(1.26, 1))

        fig.savefig(os.path.join(sub_dir, f"{var.replace(' ', '_')}.pdf"),
                    format='pdf')
        ax.cla()

    plt.close(fig)


def do_combined_plots(leak_parameters_df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    wells = [well for well in leak_parameters_df.well.unique() if well in passed_wells]

    logger.info(f"passed wells are {passed_wells}")

    protocol_overlaid_dir = os.path.join(output_dir, 'overlaid_by_protocol')
    if not os.path.exists(protocol_overlaid_dir):
        os.makedirs(protocol_overlaid_dir)

    leak_parameters_df = leak_parameters_df[leak_parameters_df.well.isin(passed_wells)]

    palette = sns.color_palette('husl', len(leak_parameters_df.groupby(['well', 'sweep'])))
    for protocol in leak_parameters_df.protocol.unique():
        times_fname = f"{experiment_name}-{protocol}-times.csv"
        try:
            times = np.loadtxt(os.path.join(args.data_dir, 'traces', times_fname)).astype(np.float64).flatten()
        except FileNotFoundError:
            continue

        times = times.flatten().astype(np.float64)

        reference_current = None

        i = 0
        for sweep in leak_parameters_df.sweep.unique():
            for well in wells:
                fname = f"{experiment_name}-{protocol}-{well}-sweep{sweep}.csv"
                try:
                    data = pd.read_csv(os.path.join(args.data_dir, 'traces', fname))

                except FileNotFoundError:
                    continue

                current = data['current'].values.flatten().astype(np.float64)

                if reference_current is None:
                    reference_current = current

                scaled_current = scale_to_reference(current, reference_current)
                col = palette[i]
                i += 1
                ax.plot(times, scaled_current, color=col, alpha=.5, label=well)

        fig_fname = f"{protocol}_overlaid_traces_scaled"
        fig.suptitle(f"{protocol}: all wells")
        ax.set_xlabel(r'time / ms')
        ax.set_ylabel('current scaled to reference trace')
        ax.legend()
        fig.savefig(os.path.join(protocol_overlaid_dir, fig_fname))
        ax.cla()

    plt.close(fig)

    palette = sns.color_palette('husl',
                                len(leak_parameters_df.groupby(['protocol', 'sweep'])))

    fig2 = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs2 = fig2.subplots(1, 2, sharey=True)

    wells_overlaid_dir = os.path.join(output_dir, 'overlaid_by_well')
    if not os.path.exists(wells_overlaid_dir):
        os.makedirs(wells_overlaid_dir)

    logger.info('overlaying traces by well')

    for well in passed_wells:
        i = 0
        for sweep in leak_parameters_df.sweep.unique():
            for protocol in leak_parameters_df.protocol.unique():
                times_fname = f"{experiment_name}-{protocol}-times.csv"
                times = np.loadtxt(os.path.join(args.data_dir, 'traces', times_fname))
                times = times.flatten().astype(np.float64)

                fname = f"{experiment_name}-{protocol}-{well}-sweep{sweep}.csv"
                try:
                    data = pd.read_csv(os.path.join(args.data_dir, 'traces', fname))
                except FileNotFoundError:
                    continue

                current = data['current'].values.flatten().astype(np.float64)

                indices_pre_ramp = times < 3000

                col = palette[i]
                i += 1

                label = f"{protocol}_sweep{sweep}"

                axs2[0].plot(times[indices_pre_ramp], current[indices_pre_ramp], color=col, alpha=.5,
                             label=label)

                indices_post_ramp = times > (times[-1] - 2000)
                post_times = times[indices_post_ramp].copy()
                post_times = post_times - post_times[0] + 5000
                axs2[1].plot(post_times, current[indices_post_ramp], color=col, alpha=.5,
                             label=label)

        axs2[0].legend()
        axs2[0].set_title('before drug')
        axs2[0].set_xlabel(r'time / ms')
        axs2[1].set_title('after drug')
        axs2[1].set_xlabel(r'time / ms')

        axs2[0].set_ylabel('current / pA')
        axs2[1].set_ylabel('current / pA')

        fig2_fname = f"{well}_overlaid_traces"
        fig2.suptitle(f"Leak ramp comparison: {well}")

        fig2.savefig(os.path.join(wells_overlaid_dir, fig2_fname))
        axs2[0].cla()
        axs2[1].cla()

    plt.close(fig2)


def do_scatter_matrices(df, qc_df):
    grid = sns.pairplot(data=df, hue='passed QC', diag_kind='hist',
                        plot_kws={'alpha': 0.4, 'edgecolor': None},
                        hue_order=[True, False])
    grid.savefig(os.path.join(output_dir, 'scatter_matrix_by_QC'))

    if args.reversal:
        true_reversal = args.reversal
    else:
        true_reversal = df['E_rev'].values.mean()

    df['hue'] = df.E_rev.to_numpy() > true_reversal
    grid = sns.pairplot(data=df, hue='hue', diag_kind='hist',
                        plot_kws={'alpha': 0.4, 'edgecolor': None},
                        hue_order=[True, False])
    grid.savefig(os.path.join(output_dir, 'scatter_matrix_by_reversal.pdf'),
                 format='pdf')

    # Now do artefact parameters only
    if 'drug' in qc_df:
        qc_df = qc_df[qc_df.drug == 'before']

    # if args.selection_file and not args.output_all:
    #     qc_df = qc_df[qc_df.well.isin(passed_wells)]

    first_sweep = sorted(list(qc_df.sweep.unique()))[0]
    qc_df = qc_df[(qc_df.protocol == 'staircaseramp1') &
                  (qc_df.sweep == first_sweep)]
    if 'drug' in qc_df:
        qc_df = qc_df[qc_df.drug == 'before']

    qc_df = qc_df.set_index(['protocol', 'well', 'sweep'])
    qc_df = qc_df[['Rseries', 'Cm', 'Rseal', 'passed QC']]
    # qc_df['R_leftover'] = df['R_leftover']
    grid = sns.pairplot(data=qc_df, diag_kind='hist', plot_kws={'alpha': .4,
                                                                'edgecolor': None},
                        hue='passed QC', hue_order=[True, False])

    grid.savefig(os.path.join(output_dir, 'scatter_matrix_QC_params_by_QC'))


def plot_reversal_spread(df):
    df.E_rev = df.E_rev.values.astype(np.float64)

    failed_to_infer = [well for well in df.well.unique() if not
                       np.all(np.isfinite(df[df.well == well]['E_rev'].values))]

    df = df[~df.well.isin(failed_to_infer)]

    def spread_func(x):
        return x.max() - x.min()

    group_df = df[['E_rev', 'well', 'passed QC']].groupby('well').agg(
        {
            'well': 'first',
            'E_rev': spread_func,
            'passed QC': 'min'
        })
    group_df['E_Kr range'] = group_df['E_rev']

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    sns.histplot(data=group_df, x='E_Kr range', hue='passed QC',
                 stat='count', multiple='stack')

    ax.set_xlabel(r'spread in inferred E_Kr / mV')

    fig.savefig(os.path.join(output_dir, 'spread_of_fitted_E_Kr'))
    df.to_csv(os.path.join(output_dir, 'spread_of_fitted_E_Kr.csv'))


def plot_reversal_change_sweep_to_sweep(df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    for protocol in df.protocol.unique():
        sub_df = df[df.protocol == protocol]

        if len(list(sub_df.sweep.unique())) != 2:
            continue

        sub_df = sub_df[['well', 'E_rev', 'sweep']]
        sweep1_vals = sub_df[sub_df.sweep == 0].copy().set_index('well')
        sweep2_vals = sub_df[sub_df.sweep == 1].copy().set_index('well')

        if len(sweep2_vals.index) == 0:
            continue

        rows = []
        for well in sub_df.well.unique():
            delta_rev = sweep2_vals.loc[well]['E_rev'].astype(float)\
                - sweep1_vals.loc[well]['E_rev'].astype(float)
            passed_QC = well in passed_wells
            rows.append([well, delta_rev, passed_QC])

        var_name_ltx = r'$\Delta E_{\mathrm{rev}}$'
        delta_df = pd.DataFrame(rows, columns=['well', var_name_ltx, 'passed QC'])

        sns.histplot(data=delta_df, x=var_name_ltx, hue='passed QC',
                     stat='count', multiple='stack')
        fig.savefig(os.path.join(output_dir, f"E_rev_sweep_to_sweep_{protocol}"))
        ax.cla()

    plt.close(fig)


def plot_leak_conductance_change_sweep_to_sweep(df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    for protocol in df.protocol.unique():
        sub_df = df[df.protocol == protocol]

        if len(list(sub_df.sweep.unique())) != 2:
            continue

        sub_df = sub_df[['well', 'gleak_before', 'sweep']]
        sweep1_vals = sub_df[sub_df.sweep == 0].copy().set_index('well')
        sweep2_vals = sub_df[sub_df.sweep == 1].copy().set_index('well')

        if len(sweep2_vals.index) == 0:
            continue

        rows = []
        for well in sub_df.well.unique():
            delta_rev = float(sweep2_vals.loc[well]['gleak_before']) - \
                float(sweep1_vals.loc[well]['gleak_before'])
            passed_QC = well in passed_wells
            rows.append([well, delta_rev, passed_QC])

        var_name_ltx = r'$\Delta g_{\mathrm{leak}}$'
        delta_df = pd.DataFrame(rows, columns=['well', var_name_ltx, 'passed QC'])

        sns.histplot(data=delta_df, x=var_name_ltx, hue='passed QC',
                     stat='count', multiple='stack', ax=ax)
        fig.savefig(os.path.join(output_dir, f"g_leak_sweep_to_sweep_{protocol}"))

    plt.close(fig)


def plot_spatial_Erev(df):
    def func(protocol, sweep):
        zs = []
        for row in range(16):
            for column in range(24):
                well = f"{string.ascii_uppercase[row]}{column+1:02d}"
                sub_df = df[(df.protocol == protocol) & (df.sweep == sweep)
                            & (df.well == well)]

                if len(sub_df.index) > 1:
                    Exception("Multiple rows values for same (protocol, sweep, well)"
                              "\n ({protocol}, {sweep}, {well})")
                elif len(sub_df.index) == 0:
                    EKr = np.nan
                else:
                    EKr = sub_df['E_rev'].values.astype(np.float64)[0]

                zs.append(EKr)

        zs = np.array(zs)

        if np.all(~np.isfinite(zs)):
            return

        finite_indices = np.isfinite(zs)

        #  This will get casted to float
        zs[finite_indices] = (zs[finite_indices] > zs[finite_indices].mean())
        zs[~np.isfinite(zs)] = 2
        zs = np.array(zs).reshape((16, 24))

        fig = plt.figure(figsize=args.figsize)
        ax = fig.subplots()
        # add black color for NaNs

        cmap = matplotlib.colors.ListedColormap([color_cycle[0], color_cycle[1]], 'indexed')
        ax.pcolormesh(zs, edgecolors='white', cmap=cmap,
                      linewidths=1, antialiased=True)

        ax.plot([], [], ls='None', marker='s', label='high E_rev', color=color_cycle[0])
        ax.plot([], [], ls='None', marker='s', label='low E_rev', color=color_cycle[1])
        ax.legend()

        ax.set_xticks([i + .5 for i in range(24)])
        ax.set_yticks([i + .5 for i in range(16)])

        # Label rows and columns
        ax.set_xticklabels([i + 1 for i in range(24)])
        ax.set_yticklabels(string.ascii_uppercase[:16])

        # Put 'A' row at the top
        ax.invert_yaxis()

        fig.savefig(os.path.join(output_dir, f"{protocol}_sweep{sweep}_E_Kr_map.pdf"),
                    format='pdf')
        plt.close(fig)

    protocol = 'staircaseramp1'
    sweep = 1

    func(protocol, sweep)


def plot_spatial_passed(df):
    fig = plt.figure(figsize=(5, 3))
    ax = fig.subplots()
    zs = []

    for row in range(16):
        for column in range(24):
            well = f"{string.ascii_uppercase[row]}{column+1:02d}"
            passed = well in passed_wells
            zs.append(passed)

    zs = np.array(zs).reshape(16, 24)

    cmap = matplotlib.colors.ListedColormap([color_cycle[0], color_cycle[1]], 'indexed')
    _ = ax.pcolormesh(zs, edgecolors='white',
                      linewidths=1, antialiased=True, cmap=cmap
                      )

    ax.plot([], [], ls='None', marker='s', label='failed QC', color=color_cycle[0])
    ax.plot([], [], ls='None', marker='s', label='passed QC', color=color_cycle[1])
    ax.set_aspect('equal')
    # ax.legend()

    ax.set_xticks([i + .5 for i in list(range(24))[1::2]])
    ax.set_yticks([i + .5 for i in range(16)])

    ax.set_xticklabels([i + 1 for i in list(range(24))[1::2]])
    ax.set_yticklabels(string.ascii_uppercase[:16])

    ax.invert_yaxis()
    fig.savefig(os.path.join(output_dir, "QC_map.pdf"), format='pdf')

    plt.close(fig)


def plot_histograms(df, qc_df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    ax.spines[['top', 'right']].set_visible(False)

    averaged_fitted_EKr = df.groupby(['well'])['E_rev'].mean().copy().to_frame()
    averaged_fitted_EKr['passed QC'] = [np.all(df[df.well == well]['passed QC']) for well in averaged_fitted_EKr.index]

    sns.histplot(averaged_fitted_EKr, x='E_rev', hue='passed QC', ax=ax,
                 multiple='stack', stat='count', legend=False)

    ax.set_xlabel(r'$\mathrm{mean}(E_{\mathrm{obs}})$')
    fig.savefig(os.path.join(output_dir, 'averaged_reversal_potential_histogram'))

    if np.isfinite(args.reversal):
        ax.axvline(args.reversal, linestyle='--', color='grey', label='Calculated Nernst potential')

    fig.savefig(os.path.join(output_dir, 'reversal_potential_histogram'))

    vars = ['pre-drug leak magnitude',
            'post-drug leak magnitude',
            'R_leftover',
            'gleak_before',
            'gleak_after',
            'Rseries',
            'Rseal',
            'Cm'
            ]

    df = df.groupby('well').agg({**{x: 'mean' for x in vars}, **{'passed QC': 'min'}})

    ax.cla()
    sns.histplot(df,
                 x='pre-drug leak magnitude', hue='passed QC', multiple='stack',
                 stat='count', common_norm=False)

    fig.savefig(os.path.join(output_dir, 'pre_drug_leak_magnitude'))
    ax.cla()

    sns.histplot(df,
                 x='post-drug leak magnitude', hue='passed QC',
                 stat='count', common_norm=False, multiple='stack')
    fig.savefig(os.path.join(output_dir, 'post_drug_leak_magnitude'))
    ax.cla()

    ax.cla()
    sns.histplot(df,
                 x='R_leftover', hue='passed QC',
                 multiple='stack',
                 stat='count', common_norm=False)

    ax.get_legend().set_title('')
    legend_handles, _ = ax.get_legend_handles_labels()
    ax.legend(legend_handles, ['failed QC', 'passed QC'], bbox_to_anchor=(1.26, 1))

    fig.savefig(os.path.join(output_dir, 'R_leftover'))
    ax.cla()

    sns.histplot(df,
                 x='gleak_before', hue='passed QC',
                 multiple='stack',
                 stat='count', common_norm=False)
    fig.savefig(os.path.join(output_dir, 'g_leak_before'))
    ax.cla()

    sns.histplot(df,
                 x='gleak_after', hue='passed QC',
                 multiple='stack',
                 stat='count', common_norm=False)
    fig.savefig(os.path.join(output_dir, 'g_leak_after'))
    ax.cla()

    sns.histplot(df,
                 x='Rseries', hue='passed QC',
                 multiple='stack',
                 stat='count', common_norm=False)
    fig.savefig(os.path.join(output_dir, 'Rseries_before'))
    ax.cla()

    sns.histplot(df,
                 x='Rseal', hue='passed QC',
                 multiple='stack',
                 stat='count', common_norm=False)
    fig.savefig(os.path.join(output_dir, 'Rseal_before'))
    ax.cla()

    sns.histplot(df,
                 x='Cm', hue='passed QC', multiple='stack',
                 stat='count', common_norm=False)
    fig.savefig(os.path.join(output_dir, 'Cm_before'))

    plt.close(fig)


def overlay_reversal_plots(leak_parameters_df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    palette = sns.color_palette('husl', len(leak_parameters_df.groupby(['protocol', 'sweep'])))

    sub_dir = os.path.join(output_dir, 'overlaid_reversal_plots')

    # if args.selection_file and not args.output_all:
    #     leak_parameters_df[leak_parameters_df.well.isin(passed_wells)]

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    protocols_to_plot = ['staircaseramp1']
    sweeps_to_plot = [1]

    # leak_parameters_df = leak_parameters_df[leak_parameters_df.well.isin(passed_wells)]

    for well in wells:
        # Setup figure
        if False in leak_parameters_df[leak_parameters_df.well == well]['passed QC'].values:
            continue
        i = 0
        for protocol in protocols_to_plot:
            if protocol == np.nan:
                continue
            for sweep in sweeps_to_plot:
                voltage_fname = os.path.join(args.data_dir, 'traces',
                                             f"{experiment_name}-{protocol}-voltages.csv")
                voltages = pd.read_csv(voltage_fname)['voltage'].values.flatten()

                fname = f"{experiment_name}-{protocol}-{well}-sweep{sweep}.csv"
                try:
                    data = pd.read_csv(os.path.join(args.data_dir, 'traces', fname))
                except FileNotFoundError:
                    continue

                times_fname = f"{experiment_name}-{protocol}-times.csv"
                times = np.loadtxt(os.path.join(args.data_dir, 'traces', times_fname))
                times = times.flatten().astype(np.float64)

                # First, find the reversal ramp
                json_protocol = json.load(os.path.join(args.data_dir, 'traces', 'protocols',
                                          f"{experiment_name}-{protocol}.json"))
                v_protocol = VoltageProtocol.from_json(json_protocol)
                ramps = v_protocol.get_ramps()
                reversal_ramp = ramps[-1]
                ramp_start, ramp_end = reversal_ramp[:2]

                # Next extract steps
                istart = np.argmax(times >= ramp_start)
                iend = np.argmax(times > ramp_end)

                if istart == 0 or iend == 0 or istart == iend:
                    raise Exception("Couldn't identify reversal ramp")

                # Plot voltage vs current
                current = data['current'].values.astype(np.float64)

                col = palette[i]

                ax.scatter(voltages[istart:iend], current[istart:iend], label=protocol,
                           color=col, s=1.2)

                fitted_poly = np.poly1d(np.polyfit(voltages[istart:iend], current[istart:iend], 4))
                ax.plot(voltages[istart:iend], fitted_poly(voltages[istart:iend]), color=col)
                i += 1

        if np.isfinite(args.reversal):
            ax.axvline(args.reversal, linestyle='--', color='grey', label='Calculated Nernst potential')

        ax.legend()
        # Save figure
        fig.savefig(os.path.join(sub_dir, f"overlaid_reversal_ramps_{well}"))

        # Clear figure
        ax.cla()

    plt.close(fig)
    return


def scale_to_reference(trace, reference):
    def error2(p):
        return np.sum((p*trace - reference)**2)

    res = scipy.optimize.minimize_scalar(error2, method='brent')
    return trace * res.x


def create_attrition_table(qc_df, subtraction_df):

    original_qc_criteria = ['qc1.rseal', 'qc1.cm', 'qc1.rseries', 'qc2.raw',
                            'qc2.subtracted', 'qc3.raw', 'qc3.E4031',
                            'qc3.subtracted', 'qc4.rseal', 'qc4.cm',
                            'qc4.rseries', 'qc5.staircase', 'qc5.1.staircase',
                            'qc6.subtracted', 'qc6.1.subtracted',
                            'qc6.2.subtracted']

    subtraction_df_sc = subtraction_df[subtraction_df.protocol.isin(['staircaseramp1',
                                                                     'staircaseramp1_2'])]
    R_leftover_qc = subtraction_df_sc.groupby('well')['R_leftover'].max() < 0.4

    qc_df['QC.R_leftover'] = [R_leftover_qc.loc[well] for well in qc_df.well]

    stage_3_criteria = original_qc_criteria + ['QC1.all_protocols', 'QC4.all_protocols',
                                               'QC6.all_protocols']
    stage_4_criteria = stage_3_criteria + ['qc3.bookend']
    stage_5_criteria = stage_4_criteria + ['QC.Erev.all_protocols', 'QC.Erev.spread']

    stage_6_criteria = stage_5_criteria + ['QC.R_leftover']

    agg_dict = {crit: 'min' for crit in stage_6_criteria}

    qc_df_sc1 = qc_df[qc_df.protocol == 'staircaseramp1']
    print(qc_df_sc1.values.shape)
    n_stage_1_wells = np.sum(np.all(qc_df_sc1.groupby('well')
                                    .agg(agg_dict)[original_qc_criteria].values,
                                    axis=1))

    qc_df_sc_both = qc_df[qc_df.protocol.isin(['staircaseramp1', 'staircaseramp1_2'])]

    n_stage_2_wells = np.sum(np.all(qc_df_sc_both.groupby('well')
                                    .agg(agg_dict)[original_qc_criteria].values,
                                    axis=1))

    n_stage_3_wells = np.sum(np.all(qc_df_sc_both.groupby('well')
                                    .agg(agg_dict)[stage_3_criteria].values,
                                    axis=1))

    n_stage_4_wells = np.sum(np.all(qc_df.groupby('well')
                                    .agg(agg_dict)[stage_4_criteria].values,
                                    axis=1))

    n_stage_5_wells = np.sum(np.all(qc_df.groupby('well')
                                    .agg(agg_dict)[stage_5_criteria].values,
                                    axis=1))

    n_stage_6_wells = np.sum(np.all(qc_df.groupby('well')
                                    .agg(agg_dict)[stage_6_criteria].values,
                                    axis=1))

    passed_qc_df = qc_df.groupby('well').agg(agg_dict)[stage_6_criteria]
    print(passed_qc_df)
    passed_wells = [well for well, row in passed_qc_df.iterrows() if np.all(row.values)]

    print(f"passed wells = {passed_wells}")

    res_dict = {
        'stage1': [n_stage_1_wells],
        'stage2': [n_stage_2_wells],
        'stage3': [n_stage_3_wells],
        'stage4': [n_stage_4_wells],
        'stage5': [n_stage_5_wells],
        'stage6': [n_stage_6_wells],
    }

    res_df = pd.DataFrame.from_records(res_dict)
    return res_df


if __name__ == "__main__":
    main()
