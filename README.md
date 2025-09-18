# PowerFlow Analysis Tool


A Python-based AC power flow solver with Gauss-Seidel (GS) and Newton–Raphson (NR) methods, plus an optional Python ↔ Simulink integration for dynamic validation. Tested on 3-bus examples and the IEEE-14 bus standard case. Designed to demonstrate numerical methods, convergence handling, and cross-tool workflows valued in power systems engineering.

**Key Features**

Solvers

-Gauss–Seidel (Slack + PQ)

-Newton–Raphson (Slack + PV + PQ) with PV Q-limit enforcement

Network Modeling

-Y-bus assembly with line charging, off-nominal taps, and phase shift

-Shunt elements per bus (Gs − jBs)

Built-in Cases

-3-bus PQ and PV test systems

-IEEE-14 bus system (per-unit on 100 MVA)

Outputs

-Terminal summary of Vm (pu) and Va (deg)

-CSV export of per-bus Vm, Va, P, Q

-Optional voltage magnitude plot

Simulink Integration 

-Solve steady-state in Python, push Vm/Va/Pg/Qg to MATLAB

-Run a Simulink model and pull logged signals back to Python for plotting

**Repository Structure**

powerflow_core.py Core algorithms (Y-bus, GS, NR, PV Q-limits)
cases.py 3-bus example cases
cases_ieee14.py IEEE-14 case data (bus, gen, branch)
run.py CLI runner for GS/NR and CSV/plots
run_ieee14_with_simulink.py Python ↔ Simulink demo script
requirements.txt Python dependencies
.gitignore Ignore list (adjust to your preference)
README.md This file
Terminal Reports/ Figures and screenshots (if you choose to track them)

**Installation (Python)**

Clone the repository
git clone https://github.com/
<your-username>/powerflow-solver.git
cd powerflow-solver

Install dependencies
pip install -r requirements.txt

Dependencies: numpy, matplotlib

**Quick Start (Power Flow Only)**

Gauss–Seidel on 3-bus PQ
python run.py --solver gs --case 3bus_pq --tol 1e-6 --max_it 300 --verbose

Newton–Raphson on 3-bus PV
python run.py --solver nr --case 3bus_pv --tol 1e-8 --max_it 40 --verbose

Newton–Raphson on IEEE-14
python run.py --solver nr --case ieee14 --tol 1e-8 --max_it 40 --verbose

Disable PV Q-limit enforcement (for experimentation)
python run.py --solver nr --case ieee14 --no_q_limits --verbose

Notes:

The script saves a CSV: <case>_<solver>_results.csv

A voltage magnitude plot is shown after each run

If your matplotlib version errors on use_line_collection, this repo already uses a compatible plt.stem call

**Simulink Integration**

What this does

Solve IEEE-14 with Python NR

Start MATLAB from Python via MATLAB Engine for Python

Push Vm/Va/Pg/Qg into MATLAB base workspace

Load and run a Simulink model (ieee14_dynamic.slx)

Pull logged signals (logsout) back to Python and plot

One-time MATLAB Engine setup (do once)

Open an elevated terminal and run:
cd "C:\Program Files\MATLAB\R2024a\extern\engines\python"
python -m pip install .

Prepare a Simulink model

Create or copy a model named: ieee14_dynamic.slx

Put it somewhere on the MATLAB path (or addpath in the script)

In Model Settings → Data Import/Export, enable: Single simulation output

Log at least one signal (right-click a line → Log Selected Signals)

Optionally use From Workspace blocks to consume Vm0, Va0, Pg0, Qg0

Run the integration demo
python run_ieee14_with_simulink.py

What to expect

Python prints convergence info for IEEE-14

MATLAB opens and runs the model for the specified StopTime

The script plots any logged Simulink signals found in logsout

Troubleshooting

If MATLAB Engine is not found, re-run the pip install step above

If the model is not found, either save it as ieee14_dynamic.slx or edit the script’s model name

If logsout is empty, enable Single simulation output and log at least one signal

**Sample Terminal Output (abbreviated)**

Converged: True Iterations: 7 Metric: 5.691e-07
Bus | Vm (pu) | Va (deg)
1 | 1.0400 | 0.000
2 | 1.0355 | -0.370
3 | 1.0329 | -0.617
...

CSV saved to: ieee14_nr_results.csv
Voltage magnitude plot displayed.

**Design Notes**

Conventions: Net injections P, Q are positive when injected into the network. Loads enter the specification via Pd, Qd.

GS: Simple and robust for small Slack+PQ systems

NR: Quadratic convergence near solution; supports PV magnitude setpoints and Q-limit enforcement by PV→PQ conversion when violated

Y-bus: Includes line charging and transformer taps/phase shift at assembly time

Extensibility: Add cases by creating bus/branch dicts matching the provided schema

**Roadmap**

IEEE-30 and larger cases

Line flow reporting and losses per branch

Basic UI or notebook widgets for case creation

Optional damping/step control in NR for stressed cases

**Credits**

Author: Uzair Ur Rahman, Third Year Electrical Engineering and CS Minor Student At Western University 
LinkedIn: https://www.linkedin.com/in/uzairurrahman

