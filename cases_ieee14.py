# cases_ieee14.py
import numpy as np
from powerflow_core import PQ, PV, SLACK

def case_ieee14():
    """
    IEEE 14-bus test case (per-unit on 100 MVA base).
    Data source: PYPOWER/MATPOWER case14 (same numbers).
    """
    baseMVA = 100.0   # MATPOWER base

    # --- BUS TABLE (MATPOWER columns) ---
    # bus_i type Pd  Qd  Gs  Bs  area Vm    Va(deg) baseKV zone Vmax Vmin
    bus_raw = np.array([
        [ 1, 3,   0.0,   0.0,  0.0,  0.0, 1, 1.06,   0.00,   0, 1, 1.06, 0.94],
        [ 2, 2,  21.7,  12.7,  0.0,  0.0, 1, 1.045, -4.98,   0, 1, 1.06, 0.94],
        [ 3, 2,  94.2,  19.0,  0.0,  0.0, 1, 1.01, -12.72,   0, 1, 1.06, 0.94],
        [ 4, 1,  47.8,  -3.9,  0.0,  0.0, 1, 1.019,-10.33,   0, 1, 1.06, 0.94],
        [ 5, 1,   7.6,   1.6,  0.0,  0.0, 1, 1.02,  -8.78,   0, 1, 1.06, 0.94],
        [ 6, 2,  11.2,   7.5,  0.0,  0.0, 1, 1.07, -14.22,   0, 1, 1.06, 0.94],
        [ 7, 1,   0.0,   0.0,  0.0,  0.0, 1, 1.062,-13.37,   0, 1, 1.06, 0.94],
        [ 8, 2,   0.0,   0.0,  0.0,  0.0, 1, 1.09, -13.36,   0, 1, 1.06, 0.94],
        [ 9, 1,  29.5,  16.6,  0.0, 19.0, 1, 1.056,-14.94,   0, 1, 1.06, 0.94],
        [10, 1,   9.0,   5.8,  0.0,  0.0, 1, 1.051,-15.10,   0, 1, 1.06, 0.94],
        [11, 1,   3.5,   1.8,  0.0,  0.0, 1, 1.057,-14.79,   0, 1, 1.06, 0.94],
        [12, 1,   6.1,   1.6,  0.0,  0.0, 1, 1.055,-15.07,   0, 1, 1.06, 0.94],
        [13, 1,  13.5,   5.8,  0.0,  0.0, 1, 1.050,-15.16,   0, 1, 1.06, 0.94],
        [14, 1,  14.9,   5.0,  0.0,  0.0, 1, 1.036,-16.04,   0, 1, 1.06, 0.94]
    ], dtype=float)

    n = bus_raw.shape[0]
    # Per-unit conversions for loads/shunts: MW/Mvar -> pu on 100 MVA
    Pd_pu = bus_raw[:,2] / baseMVA
    Qd_pu = bus_raw[:,3] / baseMVA
    Gs_pu = bus_raw[:,4] / baseMVA
    Bs_pu = bus_raw[:,5] / baseMVA

    bus = {
        'type': bus_raw[:,1].astype(int),         # 1 PQ, 2 PV, 3 Slack (same coding)
        'Pd':   Pd_pu,
        'Qd':   Qd_pu,
        'Gs':   Gs_pu,
        'Bs':   Bs_pu,
        'Pg':   np.zeros(n),                      # filled from gen table below
        'Qg':   np.zeros(n),                      # filled from gen table below (initial guess)
        'Vm':   bus_raw[:,7].copy(),              # initial Vm guesses (MATPOWER's Vm column)
        'Va':   np.radians(bus_raw[:,8].copy()),  # convert degrees -> radians for init guess
        'Vset': bus_raw[:,7].copy(),              # setpoints used for PV/Slack
        'Qmin': np.full(n, -1e9),
        'Qmax': np.full(n,  1e9),
    }

    # --- GENERATOR TABLE (MATPOWER columns) ---
    # bus, Pg,   Qg,   Qmax, Qmin, Vg, ...
    gen_raw = np.array([
        [ 1, 232.4, -16.9,  10,   0,   1.06],
        [ 2,  40.0,  42.4,  50, -40,   1.045],
        [ 3,   0.0,  23.4,  40,   0,   1.01],
        [ 6,   0.0,  12.2,  24,  -6,   1.07],
        [ 8,   0.0,  17.4,  24,  -6,   1.09]
    ], dtype=float)

    # Map generator data into bus-sized arrays (convert MW/Mvar -> pu)
    for row in gen_raw:
        bus_i = int(row[0]) - 1   # to 0-index
        Pg, Qg, Qmax, Qmin, Vg = row[1]/baseMVA, row[2]/baseMVA, row[3]/baseMVA, row[4]/baseMVA, row[5]
        bus['Pg'][bus_i] = Pg
        bus['Qg'][bus_i] = Qg
        bus['Qmax'][bus_i] = Qmax
        bus['Qmin'][bus_i] = Qmin
        # Vset should match specified generator voltage at PV/Slack buses
        bus['Vset'][bus_i] = Vg

    # --- BRANCH TABLE (MATPOWER columns) ---
    # fbus tbus   r         x         b        rateA rateB rateC ratio angle status angmin angmax
    br_raw = np.array([
        [ 1,  2, 0.01938, 0.05917, 0.0528,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 1,  5, 0.05403, 0.22304, 0.0492,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 2,  3, 0.04699, 0.19797, 0.0438,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 2,  4, 0.05811, 0.17632, 0.0340,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 2,  5, 0.05695, 0.17388, 0.0346,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 3,  4, 0.06701, 0.17103, 0.0128,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 4,  5, 0.01335, 0.04211, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 4,  7, 0.00000, 0.20912, 0.0000,  0, 0, 0, 0.978,  0, 1, -360, 360],  # xfmrs with tap
        [ 4,  9, 0.00000, 0.55618, 0.0000,  0, 0, 0, 0.969,  0, 1, -360, 360],
        [ 5,  6, 0.00000, 0.25202, 0.0000,  0, 0, 0, 0.932,  0, 1, -360, 360],
        [ 6, 11, 0.09498, 0.19890, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 6, 12, 0.12291, 0.25581, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 6, 13, 0.06615, 0.13027, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 7,  8, 0.00000, 0.17615, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 7,  9, 0.00000, 0.11001, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 9, 10, 0.03181, 0.08450, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [ 9, 14, 0.12711, 0.27038, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [10, 11, 0.08205, 0.19207, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [12, 13, 0.22092, 0.19988, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360],
        [13, 14, 0.17093, 0.34802, 0.0000,  0, 0, 0, 0.000,  0, 1, -360, 360]
    ], dtype=float)

    # Convert to your branch schema (0-indexed, radians shift)
    branches = []
    for row in br_raw:
        f = int(row[0]) - 1
        t = int(row[1]) - 1
        r, x, b = float(row[2]), float(row[3]), float(row[4])
        ratio = float(row[8])   # MATPOWER 'ratio' (tap); 0 means a=1
        angle_deg = float(row[9])
        tap = ratio if ratio != 0.0 else 1.0
        shift = np.radians(angle_deg) if angle_deg != 0.0 else 0.0
        branches.append({'f': f, 't': t, 'r': r, 'x': x, 'b': b, 'tap': tap, 'shift': shift})

    return bus, branches
