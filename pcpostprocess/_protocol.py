#
# Voltage protocol
#
# This file is part of pcpostprocess.
# See https://github.com/CardiacModelling/pcpostprocess for copyright, sharing,
# and licensing details.
#
#import numpy as np


class VoltageProtocol:
    """
    Represents a voltage protocol consisting of a sequence of steps and/or
    ramps.
    """
    # Internal format: Sequence of (duration, v_start, v_end) objects.

