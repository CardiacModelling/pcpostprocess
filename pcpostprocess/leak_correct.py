import os

import numpy as np
from matplotlib import pyplot as plt


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


def get_QC_dict(QC, bounds={'Rseal': (10e8, 10e12), 'Cm': (1e-12, 1e-10),
                            'Rseries': (1e6, 2.5e7)}):
    '''
    @params:
    QC: QC trace attribute extracted from the JSON file
    bounds: A dictionary of bound tuples, (lower, upper), for each QC variable

    @returns:
    A dictionary where the keys are wells and the values are sweeps that passed QC
    '''
    # TODO decouple this code from syncropatch export

    QC_dict = {}
    for well in QC:
        for sweep in QC[well]:
            if all(sweep):
                if bounds['Rseal'][0] < sweep[0] < bounds['Rseal'][1] and \
                   bounds['Cm'][0] < sweep[1] < bounds['Cm'][1] and \
                   bounds['Rseries'][0] < sweep[2] < bounds['Rseries'][1]:

                    if well in QC_dict:
                        QC_dict[well] = QC_dict[well] + [sweep]
                    else:
                        QC_dict[well] = [sweep]

    max_swp = max(len(QC_dict[well]) for well in QC_dict)
    QC_copy = QC_dict.copy()
    for well in QC_copy:
        if len(QC_dict[well]) != max_swp:
            QC_dict.pop(well)
    return QC_dict


def get_leak_corrected(current, voltages, times, ramp_start_index,
                       ramp_end_index, **kwargs):
    """Leak correct all data in a trace

    @Params:
    current: the observed currents taken from the entire sweep

    voltages: the voltages at each timepoint; has the same shape as current

    ramp_start_index: the index of the observation where the leak ramp begins

    ramp_end_index: the index of the observation where the leak ramp ends

    @Returns: A leak correct trace with the same shape as current

    """

    (b0, b1), I_leak = fit_linear_leak(current, voltages, times, ramp_start_index,
                                       ramp_end_index, **kwargs)


    return current - I_leak


def fit_linear_leak(current, voltage, times, ramp_start_index, ramp_end_index,
                    save_fname='', output_dir='',
                    figsize=(7.5, 6)):
    """


    @params
    current: the observed currents taken from the entire sweep

    voltages: the voltages at each timepoint; has the same shape as current

    ramp_start_index: the index of the observation where the leak ramp begins

    ramp_end_index: the index of the observation where the leak ramp ends

    @Returns: the linear regression parameters obtained from fitting the leak ramp (tuple),
    and the leak current (np.array with the same shape as current)
    """

    if len(current) == 0:
        return (np.nan, np.nan), np.empty(times.shape)

    current = current.flatten()

    # Convert to mV for convinience
    V = voltage

    I_obs = current  # pA
    b_0, b_1 = linear_reg(V[ramp_start_index:ramp_end_index+1],
                          I_obs[ramp_start_index:ramp_end_index+1])
    I_leak = b_1*V + b_0

    if save_fname:
        # fit to leak ramp
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=figsize)

        start_t = times[ramp_start_index]
        end_t = times[ramp_end_index]

        ax1.set_title('current vs time')
        ax1.set_xlabel('time (ms)')
        ax1.set_ylabel('current (pA)')
        ax1.plot(times, I_obs)
        ax1.axvline(start_t, linestyle='--', color='k', alpha=0.5)
        ax1.axvline(end_t, linestyle='--', color='k', alpha=0.5)
        ax1.set_xlim(left=start_t - 1,
                     right=end_t + 1)
        ax1.set_ylim(*np.quantile(I_obs[ramp_start_index:ramp_end_index],
                                  [0, 1]))

        ax2.set_title('voltage vs time')
        ax2.set_xlabel('time (ms)')
        ax2.set_ylabel('voltage (mV)')
        ax2.plot(times, V)
        ax2.axvline(start_t, linestyle='--', color='k', alpha=0.5)
        ax2.axvline(end_t, linestyle='--', color='k', alpha=0.5)
        ax2.set_xlim(left=start_t - 1,
                     right=end_t + 1)

        ax3.set_title('current vs voltage')
        ax3.set_xlabel('voltage (mV)')
        ax3.set_ylabel('current (pA)')
        ax3.plot(V[ramp_start_index:ramp_end_index+1],
                 I_obs[ramp_start_index:ramp_end_index+1], 'x')
        ax3.plot(V[ramp_start_index:ramp_end_index+1],
                 I_leak[ramp_start_index:ramp_end_index+1], '--')

        ax4.set_title(
            f'current vs. time (gleak: {np.round(b_1,1)}, Eleak: {np.round(b_0/b_1,1)})')
        ax4.set_xlabel('time (s)')
        ax4.set_ylabel('current (pA)')
        ax4.plot(times, I_obs, label='I_obs')
        ax4.plot(times, I_leak, linestyle='--', label='I_leak')
        ax4.plot(times, I_obs - I_leak,
                 linestyle='--', alpha=0.5, label='Ikr')
        ax4.legend()

        fig.tight_layout()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if output_dir:
            fig.savefig(os.path.join(output_dir, save_fname))
        plt.close(fig)

    return (b_0, b_1), I_leak
