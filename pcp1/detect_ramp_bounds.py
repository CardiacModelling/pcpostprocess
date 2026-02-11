import numpy as np


def detect_ramp_bounds(times, voltage_sections, ramp_index=0):
    """
    Extract the timepoint indices at the start and end of the nth ramp in the protocol.

    @param times: np.array containing the time at which each sample was taken
    @param voltage_sections 2d np.array where each row describes a segment of the protocol: (tstart, tend, vstart, end)
    @param ramp_index: the index of the ramp to select. Defaults to 0 - the first ramp

    @returns istart, iend: the start and end timepoint indices for the specified ramp
    """

    ramps = [(tstart, tend, vstart, vend) for tstart, tend, vstart, vend
             in voltage_sections if vstart != vend]
    try:
        ramp = ramps[ramp_index]
    except IndexError:
        print(f"Requested {ramp_index+1}th ramp (ramp_index={ramp_index}),"
              " but there are only {len(ramps)} ramps")

    tstart, tend = ramp[:2]

    ramp_bounds = [np.argmax(times > tstart), np.argmax(times > tend)]
    return ramp_bounds

