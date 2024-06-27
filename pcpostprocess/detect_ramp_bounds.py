import numpy as np


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

