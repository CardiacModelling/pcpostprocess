#
# Leak estimation and correction
#
# This file is part of pcpostprocess.
# See https://github.com/CardiacModelling/pcpostprocess for copyright, sharing,
# and licensing details.
#
from . import Trace


def estimate_gE_leak_ramp(
        trace: Trace, ramp: int | None = None, step: int | None = None):
    """
    Estimates ``g * (V - E)`` leak using a "leak ramp" in the voltage protocol.

    An example leak ramp for IKr would go from -120mV to -80mV in 400ms, after
    holding at -80mV.

    Parameters
    ----------
    trace : Trace

    ramp : int, optional
        The index of the ramp in the voltage protocol: 0 for the first segment
        that's a ramp, 1 for the second segment that's a ramp, etc. If neither
        ``ramp_index`` or ``step_index`` is given, a ``ValueError`` will be
        raised.
    step : int, optional
        An alternative way to specify the

    Returns
    -------

    """
    raise NotImplementedError


def estimate_gE_leak_step(
        trace: Trace, step_1: int, step_2: int | None = None):
    """
    Estimates ``g * (V - E)`` leak using two steps in a voltage protocol.

    An example sequence would be holding at -80mV (``step_1``), then performing
    a 20ms step down to -100mV (``step_2``).

    Parameters
    ----------
    trace : Trace
    step_1 : int
        The index (in the trace's voltage protocol) of the first step to use.
    step_2
        The index of the second step to use. If not set, the step immediately
        following ``step_1`` will be used (default).

    Returns
    -------


    """
    #TODO: Some magic number parameter defining how many ms or which percentage
    # of the first and second step to use

    #TODO: Modify Trace in place? Return new Trace (copying meta data?)

    raise NotImplementedError


def correct_gE_leak(trace: Trace, g : float, E : float):
    """
    Corrects...

    """
    raise NotImplementedError


# Alternative: Have a class that contains the above method (so basically two
# constructors
class gELeak:

    def __init__(self, g, E):
        pass

    @classmethod
    def from_ramp(trace, ramp_index=None, step_index=None):
        pass

    def from_steps(trace, step_1, step_2):
        pass

