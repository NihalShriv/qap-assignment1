import argparse
import json
import sys
from pathlib import Path

import l1_states
import l2_s2_circuit
import l2_s3_entanglement
import l2_s4_noise_sweep
import l2_s5_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()

    results = {
        "team": ["<id 1>", "<id 2>"],
        "level_attempted": args.level,
        "versions": {},
        "L1": {},
        "L2": {},
        "L3": {},
    }

    try:
        import qiskit, qiskit_aer
        results["versions"]["qiskit"]     = qiskit.__version__
        results["versions"]["qiskit-aer"] = qiskit_aer.__version__
    except ImportError as e:
        print(f"[warn] {e}")

    print("=== S1: single-qubit state checks ===")
    l1_out = l1_states.run()
    results["L1"].update(l1_out)
    print(f"  purity_third_state = {l1_out.get('purity_third_state')}")

    if args.level >= 2:
        print("=== S2: circuit consistency ===")
        results["L2"].update(l2_s2_circuit.run())

        print("=== S3: entanglement (partial trace) ===")
        l2s3 = l2_s3_entanglement.run()
        results["L2"].update(l2s3)
        print(f"  bell_reduced_purity = {l2s3.get('bell_reduced_purity')}")

        print("=== S4: noise sweep ===")
        results["L2"].update(l2_s4_noise_sweep.run())

        print("=== S5: dephasing trajectories ===")
        results["L2"].update(l2_s5_trajectories.run())

    if args.level >= 3:
        instance_path = Path("instance.json")
        if not instance_path.exists():
            print("[error] instance.json not found.")
            sys.exit(1)
        import l3_s6_instance, l3_s7_traj_instance, l3_s8_scaling

        print("=== S6: instance simulation ===")
        l3s6 = l3_s6_instance.run(instance_path)
        results["L3"].update(l3s6)
        print(f"  purity={l3s6.get('purity')}  S(rho)={l3s6.get('entropy')}  S(rho_A)={l3s6.get('entropy_A')}")

        print("=== S7: instance trajectories ===")
        results["L3"].update(l3_s7_traj_instance.run(instance_path))

        print("=== S8: scaling benchmark ===")
        results["L3"].update(l3_s8_scaling.run())

    out_path = Path("results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
