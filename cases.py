# cases.py
import numpy as np
from powerflow_core import PQ, PV, SLACK

def case_3bus_pq():
    """ Slack + PQ + PQ, good for Gauss–Seidel demo """
    n = 3
    bus = {
        'type': np.array([SLACK, PQ, PQ]),
        'Pd':   np.array([0.00, 0.10, 0.08]),
        'Qd':   np.array([0.00, 0.05, 0.04]),
        'Gs':   np.zeros(n),
        'Bs':   np.zeros(n),
        'Pg':   np.zeros(n),
        'Qg':   np.zeros(n),
        'Vm':   np.array([1.04, 1.00, 1.00]),
        'Va':   np.array([0.00, 0.00, 0.00]),
        'Vset': np.array([1.04, 1.00, 1.00]),
        'Qmin': np.array([-0.3, -0.3, -0.3]),
        'Qmax': np.array([ 0.3,  0.3,  0.3]),
    }
    branches = [
        {'f':0, 't':1, 'r':0.02, 'x':0.06, 'b':0.02, 'tap':1.0, 'shift':0.0},
        {'f':0, 't':2, 'r':0.08, 'x':0.24, 'b':0.02, 'tap':1.0, 'shift':0.0},
        {'f':1, 't':2, 'r':0.06, 'x':0.18, 'b':0.02, 'tap':1.0, 'shift':0.0},
    ]
    return bus, branches

def case_3bus_pv():
    """ Slack + PV + PQ, good for Newton–Raphson demo (with Q-limits) """
    n = 3
    bus = {
        'type': np.array([SLACK, PV, PQ]),
        'Pd':   np.array([0.00, 0.05, 0.06]),
        'Qd':   np.array([0.00, 0.02, 0.03]),
        'Gs':   np.zeros(n),
        'Bs':   np.zeros(n),
        'Pg':   np.array([0.00, 0.10, 0.00]),  # PV bus generates active power
        'Qg':   np.zeros(n),
        'Vm':   np.array([1.04, 1.02, 1.00]),  # initial guesses
        'Va':   np.array([0.00, 0.00, 0.00]),
        'Vset': np.array([1.04, 1.02, 1.00]),  # setpoints for Slack/PV
        'Qmin': np.array([-0.3, -0.2, -0.3]),  # PV Q-limits matter for bus 2
        'Qmax': np.array([ 0.3,  0.2,  0.3]),
    }
    branches = [
        {'f':0, 't':1, 'r':0.02, 'x':0.06, 'b':0.02, 'tap':1.0, 'shift':0.0},
        {'f':0, 't':2, 'r':0.04, 'x':0.20, 'b':0.02, 'tap':1.0, 'shift':0.0},
        {'f':1, 't':2, 'r':0.03, 'x':0.15, 'b':0.02, 'tap':1.0, 'shift':0.0},
    ]
    return bus, branches
