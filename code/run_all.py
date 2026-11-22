"""
Run every simulation and write results.json.

    python run_all.py                  # highest level possible (3 if instance.json exists, else 2)
    python run_all.py --level 2        # stop after Level 2
    python run_all.py --s8 full        # report-grade S8 ladder (slow); --s8 skip to leave S8 out

A failing step does not stop the run: its traceback is printed, the other steps still
run, results.json is still written, and the exit code is 1.
"""
import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)                      # every module writes its figures / json next to the code

TEAM = ["2023A7PS0397G", "2023B5A70986G"]        # TODO: the two student IDs that were used for make_instance.py

REQUIRED = {                        # the keys the results.json schema asks for
    "L1": ["purity_third_state"],
    "L2": ["bell_reduced_purity"],
    "L3": ["purity", "entropy", "entropy_A", "traj_estimate_purity", "n_traj"],
}


def find_instance():
    for p in (HERE / "instance.json", HERE.parent / "instance.json"):
        if p.exists():
            return p
    return None


def _json_default(o):
    """Safety net so numpy scalars / arrays / Paths never break json.dump."""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--s8", choices=["quick", "full", "skip"], default="quick")
    args = ap.parse_args()

    instance = find_instance()
    level = args.level or (3 if instance else 2)
    if level == 3 and instance is None:
        print("[error] Level 3 needs instance.json (next to run_all.py); running up to Level 2 instead.")
        level = 2

    results = {"team": TEAM, "level_attempted": level, "versions": {}, "L1": {}, "L2": {}, "L3": {}}
    failures = []

    if any("<" in str(t) for t in TEAM):
        print("[warn] TEAM still has placeholder IDs: results.json will not match your instance.")

    for pkg, key in [("qiskit", "qiskit"), ("qiskit_aer", "qiskit-aer")]:
        try:
            results["versions"][key] = importlib.import_module(pkg).__version__
        except ImportError as e:
            print(f"[warn] {e}")

    def step(title, block, module, *a, **kw):
        print(f"=== {title} ===")
        t0 = time.perf_counter()
        try:
            out = importlib.import_module(module).run(*a, **kw) or {}
            results[block].update(out)
            print(f"  ok ({time.perf_counter() - t0:.1f}s)  {out if out else ''}")
        except Exception:
            traceback.print_exc()
            failures.append(title)
            print(f"  FAILED ({title})")

    step("S1: single-qubit state checks", "L1", "l1_states")

    if level >= 2:
        step("S2: circuit consistency", "L2", "l2_s2_circuit")
        step("S3: entanglement (partial trace)", "L2", "l2_s3_entanglement")
        step("S4: noise sweep", "L2", "l2_s4_noise_sweep")
        step("S5: dephasing trajectories", "L2", "l2_s5_trajectories")

    if level >= 3:
        step("S6: instance simulation", "L3", "l3_s6_instance", instance)
        step("S7: instance trajectories", "L3", "l3_s7_traj_instance", instance)
        if args.s8 != "skip":
            step(f"S8: scaling benchmark ({args.s8})", "L3", "l3_s8_scaling", full=(args.s8 == "full"))

    for block in ("L1", "L2", "L3"):
        if int(block[1]) <= level:
            missing = [k for k in REQUIRED[block] if k not in results[block]]
            if missing:
                print(f"[warn] {block} is missing required keys: {missing}")
                failures.append(f"{block} keys {missing}")

    targets = [HERE / "results.json"]
    if HERE.name == "code":                  # the spec lists results.json at package level: write both
        targets.append(HERE.parent / "results.json")
    for out_path in targets:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        print(f"\nWrote {out_path}")
    if failures:
        print("Problems:", "; ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()