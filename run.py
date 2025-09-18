# run.py
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from powerflow_core import gauss_seidel_pq, newton_raphson
from cases import case_3bus_pq, case_3bus_pv
from cases_ieee14 import case_ieee14


def run_and_report(result, title, out_csv):
    print(f"\nConverged: {result.converged}  "
          f"Iterations: {result.iterations}  "
          f"Metric: {result.metric:.3e}")

    print("\nBus |   Vm (pu) |  Va (deg)")
    for i, (vm, va) in enumerate(zip(result.Vm, result.Va), start=1):
        print(f"{i:3d} | {vm:9.4f} | {np.degrees(va):8.3f}")

    # Save per-bus results
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Bus", "Vm(pu)", "Va(deg)", "P_inj(pu)", "Q_inj(pu)"])
        for i, (vm, va_deg, p, q) in enumerate(
            zip(result.Vm, np.degrees(result.Va), result.P, result.Q), start=1
        ):
            w.writerow([i, f"{vm:.6f}", f"{va_deg:.6f}", f"{p:.6f}", f"{q:.6f}"])
    print(f"\nSaved CSV -> {out_csv}")

    # Plot voltage profile
    plt.figure()
    plt.stem(range(1, len(result.Vm) + 1), result.Vm)
    plt.title(f"Voltage Magnitudes (pu) — {title}")
    plt.xlabel("Bus")
    plt.ylabel("Vm (pu)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="AC Power Flow Runner")
    parser.add_argument(
        "--solver", choices=["gs", "nr"], default="gs",
        help="gs = Gauss–Seidel (Slack+PQ), nr = Newton–Raphson (Slack+PV+PQ)"
    )
    parser.add_argument(
        "--case", choices=["3bus_pq", "3bus_pv", "ieee14"], default="3bus_pq",
        help="Select a test system to run"
    )
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--max_it", type=int, default=300, help="Max iterations")
    parser.add_argument("--verbose", action="store_true", help="Print iteration details")
    parser.add_argument(
        "--no_q_limits", action="store_true",
        help="Disable PV reactive power limit enforcement (NR only)"
    )
    args = parser.parse_args()

    # Load case
    if args.case == "3bus_pq":
        bus, branches = case_3bus_pq()
    elif args.case == "3bus_pv":
        bus, branches = case_3bus_pv()
    else:  # ieee14
        bus, branches = case_ieee14()

    # Run solver
    if args.solver == "gs":
        if args.case != "3bus_pq":
            print("Note: Gauss–Seidel only supports Slack+PQ. "
                  "For IEEE-14 or PV buses, use --solver nr.")
        title = f"GS / {args.case}"
        res = gauss_seidel_pq(
            bus, branches, tol=args.tol, max_it=args.max_it, verbose=args.verbose
        )
        out_csv = f"{args.case}_gs_results.csv"
    else:
        title = f"NR / {args.case}"
        res = newton_raphson(
            bus, branches, tol=args.tol,
            max_it=args.max_it,
            enforce_q_limits=(not args.no_q_limits),
            verbose=args.verbose
        )
        out_csv = f"{args.case}_nr_results.csv"

    # Report & save
    run_and_report(res, title, out_csv)


if __name__ == "__main__":
    main()
