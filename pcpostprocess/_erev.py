#
# Reversal potential estimation
#
# This file is part of pcpostprocess.
# See https://github.com/CardiacModelling/pcpostprocess for copyright, sharing,
# and licensing details.
#
from . import Trace


def estimate_reversal_potential_ramp_poly(
        trace: Trace, ramp: int) -> float:
    """
    Estimates ``g * (V - E)`` leak using a "reversal ramp" in the voltage
    protocol.

    An example reversal ramp for IKr would go from




    Parameters
    ----------
    trace : Trace
    ramp : int
        The index of the ramp in the voltage protocol: 0 for the first segment
        that's a ramp, 1 for the second segment that's a ramp, etc. If neither
        ``ramp_index`` or ``step_index`` is given, a ``ValueError`` will be
        raised.
    step : int, optional
        An alternative way to specify the
    order: int
        The order of polynomial to fit (1, 3, or 4)




    Returns
    -------

    """
    raise NotImplementedError



def estimate_reversal_potential_ramp_smoothing(
        trace: Trace, ramp: int) -> float:
    """
    Similar but by smoothing first and then just finding the crossing.

    """
