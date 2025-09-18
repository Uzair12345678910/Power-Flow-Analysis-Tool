# powerflow_core.py
import numpy as np
from dataclasses import dataclass

PQ, PV, SLACK = 1, 2, 3

@dataclass
class PowerFlowResult:
    Vm: np.ndarray
    Va: np.ndarray
    P: np.ndarray     # computed net active injection at buses (pu)
    Q: np.ndarray     # computed net reactive injection at buses (pu)
    iterations: int
    converged: bool
    metric: float     # final mismatch (NR) or max |ΔV| (GS)

def build_ybus(nbus, branches):
    """
    branches: list of dicts with keys
      f, t: 0-indexed bus ids
      r, x: series impedance (pu)
      b: total line charging susceptance (pu)
      tap: off-nominal tap (default 1.0)
      shift: phase shift (radians, default 0.0)
    """
    Y = np.zeros((nbus, nbus), dtype=complex)
    for br in branches:
        f, t = br['f'], br['t']
        r, x = br['r'], br['x']
        b = br.get('b', 0.0)
        tap = br.get('tap', 1.0)
        shift = br.get('shift', 0.0)

        z = complex(r, x)
        y = 0j if z == 0 else 1/z
        a = tap * np.exp(1j*shift) if tap != 0 else 1.0

        # Shunt b is total -> split half to each end
        Y[f, f] += (y / (a*np.conj(a))) + 1j*(b/2)
        Y[t, t] += y + 1j*(b/2)
        Y[f, t] += -y / np.conj(a)
        Y[t, f] += -y / a
    return Y

def calc_injections(Vm, Va, Ybus, Gs=None, Bs=None):
    """
    Returns P, Q as *net injections* (+ means injected into the network).
    Shunts (Gs - jBs) are treated as loads subtracted from S_inj.
    """
    n = len(Vm)
    if Gs is None: Gs = np.zeros(n)
    if Bs is None: Bs = np.zeros(n)

    V = Vm * np.exp(1j*Va)
    I = Ybus @ V
    S_inj = V * np.conj(I) - (Gs - 1j*Bs)*(Vm**2)  # subtract shunt load
    P = S_inj.real
    Q = S_inj.imag
    return P, Q

def gauss_seidel_pq(bus, branches, tol=1e-6, max_it=300, verbose=False):
    """
    Slack + PQ buses only (no PV support).
    Uses classic GS update. Metric = max |ΔV| per iteration.
    """
    Vm = bus['Vm'].astype(float).copy()
    Va = bus['Va'].astype(float).copy()
    Pd = bus['Pd'].astype(float)
    Qd = bus['Qd'].astype(float)
    Gs = bus.get('Gs', np.zeros_like(Vm)).astype(float)
    Bs = bus.get('Bs', np.zeros_like(Vm)).astype(float)

    n = len(Vm)
    typ = bus['type'].astype(int)
    s_idx = np.where(typ == SLACK)[0]
    if len(s_idx) != 1:
        raise ValueError("Exactly one slack bus required.")
    s = s_idx[0]

    Y = build_ybus(n, branches)
    V = Vm * np.exp(1j*Va)
    V[s] = Vm[s] * np.exp(1j*Va[s])
    PQ = [i for i in range(n) if i != s]

    metric, iters, converged = 0.0, 0, False
    for it in range(1, max_it+1):
        iters = it
        max_dev = 0.0
        for i in PQ:
            Sspec = -(Pd[i] + 1j*Qd[i])  # load as negative injection
            Yii = Y[i, i]
            sumYV = (Y[i, :] @ V) - Yii*V[i]
            # include shunt on bus i in the power balance:
            # Sbus = Sspec + (Gs - jBs)*|V|^2; in GS formula it appears via conj(S)/conj(V)
            S_eff = Sspec + (Gs[i] - 1j*Bs[i]) * (abs(V[i])**2)
            Vnew = (1/Yii) * ((np.conj(S_eff)/np.conj(V[i])) - sumYV)
            dev = abs(Vnew - V[i])
            if dev > max_dev: max_dev = dev
            V[i] = Vnew

        metric = max_dev
        if verbose:
            print(f"GS iter {it}: max|ΔV|={max_dev:.3e}")
        if max_dev < tol:
            converged = True
            break

    Vm = np.abs(V)
    Va = np.angle(V)
    P, Q = calc_injections(Vm, Va, Y, Gs, Bs)
    return PowerFlowResult(Vm=Vm, Va=Va, P=P, Q=Q,
                           iterations=iters, converged=converged, metric=metric)

def newton_raphson(bus, branches, tol=1e-8, max_it=30, enforce_q_limits=True, verbose=False):
    """
    Full AC power flow with Slack, PV, PQ support.
    PV Q-limits enforced by converting PV->PQ when violated (if enabled).
    """
    Vm = bus['Vm'].astype(float).copy()
    Va = bus['Va'].astype(float).copy()
    typ = bus['type'].astype(int)
    Pd = bus['Pd'].astype(float)
    Qd = bus['Qd'].astype(float)
    Pg = bus.get('Pg', np.zeros_like(Vm)).astype(float)
    Qg = bus.get('Qg', np.zeros_like(Vm)).astype(float)
    Gs = bus.get('Gs', np.zeros_like(Vm)).astype(float)
    Bs = bus.get('Bs', np.zeros_like(Vm)).astype(float)
    Vset = bus.get('Vset', Vm).astype(float)
    Qmin = bus.get('Qmin', np.full_like(Vm, -1e9)).astype(float)
    Qmax = bus.get('Qmax', np.full_like(Vm,  1e9)).astype(float)

    n = len(Vm)
    s_idx = np.where(typ == SLACK)[0]
    if len(s_idx) != 1:
        raise ValueError("Exactly one slack bus required.")
    s = s_idx[0]

    Y = build_ybus(n, branches)
    G, B = Y.real, Y.imag

    # keep an original PV mask to know which are candidate PVs
    pv_set = np.where(typ == PV)[0]
    pq_set = np.where(typ == PQ)[0]

    # enforce target magnitudes on PV + Slack initially
    Vm[pv_set] = Vset[pv_set]
    Vm[s] = Vset[s]

    iterations, converged, metric = 0, False, 0.0

    def state_indices():
        theta_idx = [i for i in range(n) if i != s]
        v_idx = list(np.where(typ == PQ)[0])  # V updates only on PQ buses
        return theta_idx, v_idx

    for it in range(1, max_it+1):
        iterations = it

        # compute injections
        P, Q = calc_injections(Vm, Va, Y, Gs, Bs)
        Pspec = Pg - Pd
        Qspec = Qg - Qd

        # optionally enforce PV Q-limits *this iteration*
        if enforce_q_limits and len(pv_set) > 0:
            # Estimate generator Q at PV buses from mismatch:
            # Qgen_i = Qcalc_i + Qd_i  (since Qspec = Qg - Qd, and Qcalc is network)
            # more directly: required Qg = Qcalc + Qd to meet ΔQ=0 at PV bus if it were PQ
            Qgen_req = Q + Qd
            for i in pv_set:
                if Qgen_req[i] < Qmin[i] - 1e-12:
                    # hit lower limit: fix Qg at Qmin and convert to PQ (for the rest of the solve)
                    Qg[i] = Qmin[i]
                    typ[i] = PQ
                elif Qgen_req[i] > Qmax[i] + 1e-12:
                    Qg[i] = Qmax[i]
                    typ[i] = PQ
            # update sets after any conversions
            pv_set = np.where(typ == PV)[0]
            pq_set = np.where(typ == PQ)[0]

        # recompute with possibly changed types
        theta_idx, v_idx = state_indices()

        # mismatch vector: ΔP for all non-slack; ΔQ for PQ only
        dP = Pspec - P
        dQ = Qspec - Q

        mism = []
        for i in range(n):
            if i == s: continue
            mism.append(dP[i])
        for i in v_idx:
            mism.append(dQ[i])
        mism = np.array(mism, dtype=float)
        metric = 0.0 if mism.size == 0 else np.max(np.abs(mism))
        if verbose:
            print(f"NR iter {it}: max mism = {metric:.3e}")
        if metric < tol:
            converged = True
            break

        # Build Jacobian blocks
        n_theta = n - 1
        n_v = len(v_idx)
        J11 = np.zeros((n_theta, n_theta))
        J12 = np.zeros((n_theta, n_v))
        J21 = np.zeros((n_v, n_theta))
        J22 = np.zeros((n_v, n_v))

        theta_pos = {bus_k: i for i, bus_k in enumerate(theta_idx)}
        v_pos = {bus_k: i for i, bus_k in enumerate(v_idx)}

        for i in range(n):
            Vi = Vm[i]
            for k in range(n):
                Vk = Vm[k]
                Gik, Bik = G[i, k], B[i, k]
                angle = Va[i] - Va[k]

                # dP/dθ
                if i != s and k != s:
                    if i == k:
                        val = 0.0
                        for m in range(n):
                            if m == i: continue
                            Gim, Bim = G[i, m], B[i, m]
                            ang = Va[i] - Va[m]
                            val += Vi*Vm[m]*(-Gim*np.sin(ang) + Bim*np.cos(ang))
                        J11[theta_pos[i], theta_pos[i]] = -val
                    else:
                        J11[theta_pos[i], theta_pos[k]] = Vi*Vk*(-Gik*np.sin(angle) + Bik*np.cos(angle))

                # dQ/dθ rows exist only for PQ buses
                if (i in v_pos) and (k != s):
                    if i == k:
                        valQ = 0.0
                        for m in range(n):
                            if m == i: continue
                            Gim, Bim = G[i, m], B[i, m]
                            ang = Va[i] - Va[m]
                            valQ += Vi*Vm[m]*(-Gim*np.cos(ang) - Bim*np.sin(ang))
                        J21[v_pos[i], theta_pos[i]] = -valQ
                    else:
                        J21[v_pos[i], theta_pos[k]] = -Vi*Vk*(Gik*np.cos(angle) + Bik*np.sin(angle))

                # dP/dV columns: only for PQ buses
                if (k in v_pos) and (i != s):
                    J12[theta_pos[i], v_pos[k]] = Vi*(Gik*np.cos(angle) + Bik*np.sin(angle))

                # dQ/dV rows+cols: only for PQ buses
                if (i in v_pos) and (k in v_pos):
                    J22[v_pos[i], v_pos[k]] = Vi*(Gik*np.sin(angle) - Bik*np.cos(angle))

            # Diagonal dP/dV_i and dQ/dV_i (only for PQ V columns)
            if i in v_pos:
                dPdVi = 2*Vi*G[i, i]
                dQdVi = -2*Vi*B[i, i]
                for m in range(n):
                    if m == i: continue
                    Gim, Bim = G[i, m], B[i, m]
                    ang = Va[i] - Va[m]
                    dPdVi += Vm[m]*(Gim*np.cos(ang) + Bim*np.sin(ang))
                    dQdVi += Vm[m]*(Gim*np.sin(ang) - Bim*np.cos(ang))
                J12[theta_pos[i], v_pos[i]] = dPdVi
                J22[v_pos[i], v_pos[i]] = dQdVi

        # Assemble and solve
        if n_v == 0:
            J = J11
        else:
            J = np.block([[J11, J12],
                          [J21, J22]])
        dx = np.linalg.solve(J, mism)

        # Update states
        dtheta = dx[:len(theta_idx)]
        Va[theta_idx] += dtheta
        if n_v > 0:
            dV = dx[len(theta_idx):]
            Vm[v_idx] += dV

        # Enforce PV and Slack magnitudes
        Vm[pv_set] = Vset[pv_set]
        Vm[s] = Vset[s]

    # Final injections
    P, Q = calc_injections(Vm, Va, Y, Gs, Bs)
    return PowerFlowResult(Vm=Vm, Va=Va, P=P, Q=Q,
                           iterations=iterations, converged=converged, metric=metric)
