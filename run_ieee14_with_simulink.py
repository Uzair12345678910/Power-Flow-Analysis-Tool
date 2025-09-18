import numpy as np
import matplotlib.pyplot as plt
import matlab.engine

from powerflow_core import newton_raphson
from cases_ieee14 import case_ieee14

def solve_powerflow():
    bus, branches = case_ieee14()
    res = newton_raphson(bus, branches, tol=1e-8, max_it=40, verbose=True)
    if not res.converged:
        raise RuntimeError(f"NR did not converge (metric={res.metric})")
    print("\nPower flow solved. Sample:")
    for i, (vm, va) in enumerate(zip(res.Vm[:5], np.degrees(res.Va[:5])), start=1):
        print(f"Bus {i:02d}: Vm={vm:.4f} pu, Va={va:.2f} deg")
    return res, bus

def push_init_to_matlab(eng, res, bus):
    Vm0 = matlab.double(res.Vm.tolist())
    Va0 = matlab.double(res.Va.tolist())  # radians
    Pg0 = matlab.double(bus['Pg'].tolist()) if 'Pg' in bus else matlab.double([0.0]*len(res.Vm))
    Qg0 = matlab.double(bus['Qg'].tolist()) if 'Qg' in bus else matlab.double([0.0]*len(res.Vm))

    eng.workspace['Vm0'] = Vm0
    eng.workspace['Va0'] = Va0
    eng.workspace['Pg0'] = Pg0
    eng.workspace['Qg0'] = Qg0
    print("Pushed Vm0, Va0, Pg0, Qg0 into MATLAB workspace.")

def run_simulink(eng, model='ieee14_dynamic', stop_time='5'):
    print(f"Loading Simulink model: {model}.slx")
    eng.load_system(model, nargout=0)
    eng.set_param(model, 'StopTime', stop_time, nargout=0)
    print("Starting simulation…")
    simout = eng.sim(model, nargout=1)
    print("Simulation complete.")
    try:
        logsout = simout.get('logsout')
    except Exception:
        logsout = None
    return logsout

def pull_and_plot_logs(logsout):
    if logsout is None:
        print("No logsout found. Enable 'Single simulation output' or log signals.")
        return
    try:
        n = int(logsout.numElements())
    except Exception:
        try:
            n = int(logsout.getLength())
        except Exception:
            n = 0
    if n == 0:
        print("No logged elements in logsout. Add signal logging in the model.")
        return

    for i in range(n):
        try:
            el = logsout.get(i)
            name = el.Name
            ts = el.Values
            t = np.array(ts.Time).astype(float)
            y = np.array(ts.Data).astype(float).squeeze()
            plt.figure()
            plt.plot(t, y)
            plt.xlabel('Time (s)')
            plt.ylabel(name)
            plt.title(f"Simulink signal: {name}")
            plt.grid(True)
        except Exception as e:
            print(f"Could not plot element {i}: {e}")
    plt.show()

def main():
    res, bus = solve_powerflow()

    print("\nStarting MATLAB Engine…")
    eng = matlab.engine.start_matlab()
    # Example: add your Simulink models folder if needed
    # eng.addpath(r'C:\path\to\simulink_models', nargout=0)

    push_init_to_matlab(eng, res, bus)
    logsout = run_simulink(eng, model='ieee14_dynamic', stop_time='5')
    pull_and_plot_logs(logsout)

    print("\nDone. Closing MATLAB.")
    eng.quit()

if __name__ == "__main__":
    main()
