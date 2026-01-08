import argparse
import os
import string

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from pcpostprocess.directory_builder import setup_output_directory
from pcpostprocess.scripts.run_herg_qc import create_qc_table


def run_from_command_line():  # pragma: no cover
    """
    Parses arguments from the command line and then ???
    """

    description = ''    # TODO Describe what this does
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        'data_directory', help='path to the run_herg_qc results')
    parser.add_argument(
        'experiment_name', help='the name of the experiment')
    parser.add_argument('-o', '--output_dir', default='output',
                        help='The path to write output to')
    parser.add_argument(
        '--Erev', default=None, type=float,
        help='The calculated or estimated reversal potential.')
    parser.add_argument(
        '--figsize', type=int, nargs=2, default=(5, 3),
        help='A figure size, to pass to matplotlib')
    args = parser.parse_args()

    run(args.data_directory, args.output_dir, args.experiment_name,
        args.Erev, args.figsize)


def run(data_path, output_path, experiment_name, reversal_potential=None,
        figsize=None):
    """
    Does whatever this does.

    @param data_path The path to read data from
    @param output_path A root path, will be appended with "summarise_herg_export"
    @param experiment_name
    @param reversal_potential The calculated reversal potential, or ``None``
    @param figsize The matplotlib figure size, or ``None``.
    """
    output_path = setup_output_directory(output_path)

    leak_parameters_df = pd.read_csv(os.path.join(data_path, 'subtraction_qc.csv'))

    qc_df = pd.read_csv(os.path.join(data_path, f"QC-{experiment_name}.csv"))

    qc_styled_df = create_qc_table(qc_df)
    qc_styled_df = qc_styled_df.pivot(columns='protocol', index='crit')
    qc_styled_df.to_latex(os.path.join(output_path, 'qc_table.tex'))

    qc_df.protocol = ['staircaseramp1' if protocol == 'staircaseramp' else protocol
                      for protocol in qc_df.protocol]
    qc_df.protocol = ['staircaseramp1_2' if protocol == 'staircaseramp_2' else protocol
                      for protocol in qc_df.protocol]

    leak_parameters_df.protocol = ['staircaseramp1' if protocol == 'staircaseramp' else protocol
                                   for protocol in leak_parameters_df.protocol]
    leak_parameters_df.protocol = ['staircaseramp1_2' if protocol == 'staircaseramp_2' else protocol
                                   for protocol in leak_parameters_df.protocol]

    with open(os.path.join(data_path, 'passed_wells.txt')) as fin:
        passed_wells = fin.read().splitlines()

    # Compute new variables
    leak_parameters_df = compute_leak_magnitude(leak_parameters_df)

    try:
        chrono_fname = os.path.join(data_path, 'chrono.txt')
        with open(chrono_fname, 'r') as fin:
            lines = fin.read().splitlines()
            protocol_order = [line.split(' ')[0] for line in lines]

            protocol_order = ['staircaseramp1' if p == 'staircaseramp' else p
                              for p in protocol_order]

            protocol_order = ['staircaseramp1_2' if p == 'staircaseramp_2' else p
                              for p in protocol_order]

        leak_parameters_df['protocol'] = pd.Categorical(
            leak_parameters_df['protocol'], categories=protocol_order, ordered=True)

        leak_parameters_df.sort_values(['protocol', 'sweep'], inplace=True)
    except FileNotFoundError:
        leak_parameters_df.sort_values(['protocol', 'sweep'])

    scatterplot_timescale_E_obs(output_path, leak_parameters_df, passed_wells, figsize)

    do_chronological_plots(leak_parameters_df, output_path, reversal_potential,
                           figsize=figsize, normalise=False)
    do_chronological_plots(leak_parameters_df, output_path, reversal_potential,
                           figsize=figsize, normalise=True)

    attrition_df = create_attrition_table(qc_df, leak_parameters_df)
    attrition_df.to_latex(os.path.join(output_path, 'attrition.tex'))

    if 'passed QC' not in leak_parameters_df.columns and\
       'passed QC6a' in leak_parameters_df.columns:
        leak_parameters_df['passed QC'] = leak_parameters_df['passed QC6a']

    plot_leak_conductance_change_sweep_to_sweep(
        leak_parameters_df, output_path, passed_wells, figsize)
    plot_reversal_change_sweep_to_sweep(
        leak_parameters_df, output_path, passed_wells, figsize)
    plot_spatial_passed(leak_parameters_df, output_path, passed_wells)
    plot_reversal_spread(leak_parameters_df, output_path, figsize)
    if reversal_potential is not None:
        plot_spatial_Erev(leak_parameters_df, output_path, figsize)

    leak_parameters_df['passed QC'] = [
        well in passed_wells for well in leak_parameters_df.well]

    plot_histograms(leak_parameters_df, output_path, reversal_potential, figsize)


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


def scatterplot_timescale_E_obs(output_path, df, passed_wells, figsize=None):
    """
    ???

    @param output_path
    @param df
    @param passed_wells
    @param figsize

    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
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

    plot_df['E_leak'] = (plot_df.set_index('well')['E_leak'] - plot_df.groupby('well')
                         ['E_leak'].mean()).reset_index()['E_leak']

    sns.scatterplot(data=plot_df, y='40mV decay time constant',
                    x='E_rev', ax=ax, hue='well', style='well')

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel(r'$\tau$ (ms)')
    ax.set_xlabel(r'$E_\mathrm{obs}$')

    fig.savefig(os.path.join(output_path, 'decay_timescale_vs_E_rev_scatter.png'))
    ax.cla()

    sns.lineplot(data=plot_df, y='40mV decay time constant',
                 x='E_rev', hue='well', style='well',
                 ax=ax)

    ax.set_ylabel(r'$\tau$ (ms)')
    ax.set_xlabel(r'$E_\mathrm{obs}$')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(os.path.join(output_path, 'decay_timescale_vs_E_rev_line.png'))
    ax.cla()

    plot_df['E_rev'] = (plot_df.set_index('well')['E_rev'] - plot_df.groupby('well')
                        ['E_rev'].mean()).reset_index()['E_rev']
    sns.scatterplot(data=plot_df, y='E_leak',
                    x='E_rev', ax=ax, hue='well', style='well')

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel(r'$E_\mathrm{leak} - \bar E_\mathrm{leak}$ (ms)')
    ax.set_xlabel(r'$E_\mathrm{obs} - \bar E_\mathrm{obs}$')

    fig.savefig(os.path.join(output_path, 'E_leak_vs_E_rev_scatter.png'))
    ax.cla()


def do_chronological_plots(df, output_path, reversal_potential=None,
                           normalise=False, figsize=None):
    """
    ???

    @param df
    @param output_path
    @param reversal_potential
    @param normalise
    """

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.subplots()

    sub_dir = os.path.join(output_path, 'chrono_plots')
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

        if var == 'E_rev' and reversal_potential is not None:
            ax.axhline(reversal_potential, linestyle='--', color='grey',
                       label='Calculated Nernst potential')
        ax.set_xlabel('')

        if var in pretty_vars and var in units:
            ax.set_ylabel(f"{pretty_vars[var]} ({units[var]})")

        ax.get_legend().set_title('')
        legend_handles, _ = ax.get_legend_handles_labels()
        ax.legend(legend_handles, ['failed QC', 'passed QC'], bbox_to_anchor=(1.26, 1))

        fig.savefig(os.path.join(sub_dir, f'{var.replace(" ", "_")}.png'))
        ax.cla()

    plt.close(fig)


def plot_reversal_spread(df, output_path, figsize=None):
    """
    ???

    @param df
    @param output_path
    @param figsize
    """
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

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.subplots()

    sns.histplot(data=group_df, x='E_Kr range', hue='passed QC',
                 stat='count', multiple='stack')

    ax.set_xlabel(r'spread in inferred E_Kr / mV')

    fig.savefig(os.path.join(output_path, 'spread_of_fitted_E_Kr'))
    df.to_csv(os.path.join(output_path, 'spread_of_fitted_E_Kr.csv'))


def plot_reversal_change_sweep_to_sweep(
        df, output_path, passed_wells, figsize=None):
    """
    ???

    @param df
    @param output_path
    @param passed_wells
    @param figsize
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
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
        fig.savefig(os.path.join(output_path, f"E_rev_sweep_to_sweep_{protocol}"))
        ax.cla()

    plt.close(fig)


def plot_leak_conductance_change_sweep_to_sweep(
        df, output_path, passed_wells, figsize=None):
    """
    ???

    @param df
    @param output_path
    @param passed_wells
    @param figsize
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
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
        fig.savefig(os.path.join(output_path, f"g_leak_sweep_to_sweep_{protocol}"))

    plt.close(fig)


def plot_spatial_Erev(df, output_path, figsize=None):
    """
    ???

    @param df
    @param output_path
    @param figsize
    """
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

        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
        # add black color for NaNs

        color_cycle = ["#5790fc", "#f89c20"]
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

        fig.savefig(os.path.join(
            output_path, f'{protocol}_sweep{sweep}_E_Kr_map.png'))
        plt.close(fig)

    protocol = 'staircaseramp1'
    sweep = 1

    func(protocol, sweep)


def plot_spatial_passed(df, output_path, passed_wells):
    """
    ???

    @param df
    @param output_path
    @param passed_wells
    """
    fig = plt.figure(figsize=(5, 3))
    ax = fig.subplots()
    zs = []

    for row in range(16):
        for column in range(24):
            well = f"{string.ascii_uppercase[row]}{column+1:02d}"
            passed = well in passed_wells
            zs.append(passed)

    zs = np.array(zs).reshape(16, 24)

    color_cycle = ["#5790fc", "#f89c20"]
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
    fig.savefig(os.path.join(output_path, 'QC_map.png'))

    plt.close(fig)


def plot_histograms(df, output_path, reversal_potential=None, figsize=None):
    """
    ???

    @param df
    @param output_path
    @param reversal_potential
    @param figsize
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.subplots()

    ax.spines[['top', 'right']].set_visible(False)

    averaged_fitted_EKr = df.groupby(['well'])['E_rev'].mean().copy().to_frame()
    averaged_fitted_EKr['passed QC'] = [np.all(df[df.well == well]['passed QC']) for well in averaged_fitted_EKr.index]

    sns.histplot(averaged_fitted_EKr, x='E_rev', hue='passed QC', ax=ax,
                 multiple='stack', stat='count', legend=False)

    ax.set_xlabel(r'$\mathrm{mean}(E_{\mathrm{obs}})$')
    fig.savefig(os.path.join(
        output_path, 'averaged_reversal_potential_histogram'))

    if reversal_potential is not None:
        ax.axvline(reversal_potential, linestyle='--', color='grey',
                   label='Calculated Nernst potential')

    fig.savefig(os.path.join(output_path, 'reversal_potential_histogram'))

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

    fig.savefig(os.path.join(output_path, 'pre_drug_leak_magnitude'))
    ax.cla()

    sns.histplot(df,
                 x='post-drug leak magnitude', hue='passed QC',
                 stat='count', common_norm=False, multiple='stack')
    fig.savefig(os.path.join(output_path, 'post_drug_leak_magnitude'))
    ax.cla()

    ax.cla()
    sns.histplot(df,
                 x='R_leftover', hue='passed QC',
                 multiple='stack',
                 stat='count', common_norm=False)

    ax.get_legend().set_title('')
    legend_handles, _ = ax.get_legend_handles_labels()
    ax.legend(legend_handles, ['failed QC', 'passed QC'], bbox_to_anchor=(1.26, 1))

    fig.savefig(os.path.join(output_path, 'R_leftover'))
    ax.cla()

    kwargs = dict(
        hue='passed QC', multiple='stack', stat='count', common_norm=False)
    sns.histplot(df, x='gleak_before', **kwargs)
    fig.savefig(os.path.join(output_path, 'g_leak_before'))
    ax.cla()

    sns.histplot(df, x='gleak_after', **kwargs)
    fig.savefig(os.path.join(output_path, 'g_leak_after'))
    ax.cla()

    sns.histplot(df, x='Rseries', **kwargs)
    fig.savefig(os.path.join(output_path, 'Rseries_before'))
    ax.cla()

    sns.histplot(df, x='Rseal', **kwargs)
    fig.savefig(os.path.join(output_path, 'Rseal_before'))
    ax.cla()

    sns.histplot(df, x='Cm', **kwargs)
    fig.savefig(os.path.join(output_path, 'Cm_before'))

    plt.close(fig)


def scale_to_reference(trace, reference):
    def error2(p):
        return np.sum((p*trace - reference)**2)

    res = scipy.optimize.minimize_scalar(error2, method='brent')
    return trace * res.x


def create_attrition_table(qc_df, subtraction_df):
    """
    ???

    @param qc_df
    @param subtraction_df
    """

    original_qc_criteria = ['qc1.rseal', 'qc1.cm', 'qc1.rseries', 'qc2.raw',
                            'qc2.subtracted', 'qc3.raw', 'qc3.E4031',
                            'qc3.subtracted', 'qc4.rseal', 'qc4.cm',
                            'qc4.rseries', 'qc5.staircase', 'qc5.1.staircase',
                            'qc6.subtracted', 'qc6.1.subtracted',
                            'qc6.2.subtracted']

    # subtraction_df_sc = subtraction_df[
    #    subtraction_df.protocol.isin(['staircaseramp1', 'staircaseramp1_2'])]
    # R_leftover_qc = subtraction_df_sc.groupby('well')['R_leftover'].max() < 0.4
    # qc_df['QC.R_leftover'] = [R_leftover_qc.loc[well] for well in subtraction_df.well.unique()]

    stage_3_criteria = original_qc_criteria + ['QC1.all_protocols', 'QC4.all_protocols',
                                               'QC6.all_protocols']
    stage_4_criteria = stage_3_criteria + ['qc3.bookend']
    stage_5_criteria = stage_4_criteria + ['QC.Erev.all_protocols', 'QC.Erev.spread']

    # stage_6_criteria = stage_5_criteria + ['QC.R_leftover']

    agg_dict = {crit: 'min' for crit in stage_5_criteria}

    qc_df_sc1 = qc_df[qc_df.protocol == 'staircaseramp1']
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

    # n_stage_6_wells = np.sum(
    #    np.all(qc_df.groupby('well').agg(agg_dict)[stage_6_criteria].values,
    #    axis=1))

    res_dict = {
        'stage1': [n_stage_1_wells],
        'stage2': [n_stage_2_wells],
        'stage3': [n_stage_3_wells],
        'stage4': [n_stage_4_wells],
        'stage5': [n_stage_5_wells],
        # 'stage6': [n_stage_6_wells],
    }

    res_df = pd.DataFrame.from_records(res_dict)
    return res_df


if __name__ == "__main__":  # pragma: no cover
    run_from_command_line()
