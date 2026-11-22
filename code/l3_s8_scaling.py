"""
S8 (Level 3): scaling benchmark, Aer statevector vs density-matrix.

Circuit family: DEPTH layers of Ry/Rz on every qubit plus a brick-wall of CX, so the
depth is fixed and the gate count grows only linearly with n. Four configurations:

  statevector            noiseless, 1 shot
  density_matrix         noiseless, 1 shot
  statevector_noisy      noisy,     1 shot   = the cost of ONE trajectory  (for A3.3)
  density_matrix_noisy   noisy,     1 shot   = exact noisy evolution

Method:
  * every (configuration, n) runs in its OWN subprocess, so peak memory is not
    polluted by earlier, larger runs;
  * in the subprocess: one warm-up run on a 2-qubit circuit (excluded), then
    `repeats` timed runs; the median is reported;
  * wall time = Python-side run()+result(); aer time = Aer's own result.time_taken.
    The gap between them is simulator/interface overhead (A3.2, Q3.3);
  * memory = peak RSS of the worker, sampled by the PARENT every 5 ms while the worker
    runs, minus the worker's own baseline after imports and warm-up; shown next to the
    theoretical state size 16*2^n (statevector) or 16*4^n (density matrix) bytes;
  * each ladder of n stops by itself when the next size will not fit in memory or
    the previous size was slower than the time cap ("largest size you can run").

python l3_s8_scaling.py                       quick ladder (about a minute)
python l3_s8_scaling.py --full                report-grade ladder (larger n, more repeats)
python l3_s8_scaling.py --full --threads 1    same, single-threaded: separates the
                                              algorithm's cost from Aer's multi-threading

Writes s8_scaling.json / .png for --full (s8_scaling_t1.* with --threads 1). A quick run writes
s8_scaling_quick*.json / .png instead, so it can never overwrite the report-grade full data. run() returns {} (S8 has no key in the
results.json schema).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

DEPTH = 10                  # fixed number of layers for every n
CIRCUIT_SEED = 7
NOISE_P = 0.01              # depolarising probability on every gate (benchmark noise)
CONFIGS = ["statevector", "density_matrix", "statevector_noisy", "density_matrix_noisy"]

QUICK = dict(n_cap={"statevector": 18, "statevector_noisy": 18,
                    "density_matrix": 10, "density_matrix_noisy": 10},
             repeats=3, time_cap=8.0)
FULL = dict(n_cap={"statevector": 28, "statevector_noisy": 28,
                   "density_matrix": 14, "density_matrix_noisy": 14},
            repeats=5, time_cap=60.0)
MEM_FRACTION = 0.5          # never plan a run needing more than this share of free RAM
MEM_SAFETY = 1.4            # measured: peak = 1.0 x state size + ~2 MiB, so 1.4 is a safe margin
FIT_MIN_TIME = 0.1          # s: time fits use only sizes slower than this (above the thread ramp-up)


# ----------------------------------------------------------------- circuit / noise
def build_circuit(n, depth=DEPTH, seed=CIRCUIT_SEED):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for layer in range(depth):
        for q in range(n):
            qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
            qc.rz(float(rng.uniform(0, 2 * np.pi)), q)
        for q in range(layer % 2, n - 1, 2):
            qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def make_noise(p=NOISE_P):
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["ry", "rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    return nm


def predicted_bytes(config, n):
    return 16 * (4 ** n if config.startswith("density_matrix") else 2 ** n)


def rss_bytes():
    import psutil
    return psutil.Process().memory_info().rss



# --------------------------------------------------------------------- worker
def worker(config, n, repeats, threads=0):
    """Runs in a fresh subprocess; prints one 'RESULT {json}' line.
    threads = 0 lets Aer use all cores (its default); threads = 1 gives single-thread timings."""
    from qiskit_aer import AerSimulator
    method = "density_matrix" if config.startswith("density_matrix") else "statevector"
    noisy = config.endswith("_noisy")
    extra = {"max_parallel_threads": threads} if threads else {}
    sim = AerSimulator(method=method, noise_model=make_noise() if noisy else None, **extra)

    sim.run(build_circuit(2), shots=1).result()            # warm-up (not timed)
    qc = build_circuit(n)
    base = rss_bytes()                                     # RSS after imports + warm-up

    walls, aers = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = sim.run(qc, shots=1, seed_simulator=1).result()
        walls.append(time.perf_counter() - t0)
        assert res.success, "Aer run failed"
        aers.append(float(getattr(res, "time_taken", float("nan"))))

    print("RESULT " + json.dumps({
        "config": config, "n": n, "threads": threads,
        "gates": sum(1 for i in qc.data if i.operation.name in ("ry", "rz", "cx")),
        "wall_s": walls, "aer_s": aers, "base_mb": base / 2**20,
        "theory_mb": predicted_bytes(config, n) / 2**20}))


def run_one(config, n, repeats, timeout, threads=0, poll=0.005):
    """Start the worker subprocess and sample ITS resident memory from out here.

    Sampling from inside the worker does not work: Aer's C++ simulation holds the GIL,
    so a sampler thread in the same process cannot run during the simulation. And
    ru_maxrss is useless because a child inherits its parent's peak across exec.
    """
    import psutil
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", config, str(n), str(repeats), str(threads)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=HERE)
    ps, peak, t0 = psutil.Process(proc.pid), 0, time.time()
    while proc.poll() is None:
        try:
            peak = max(peak, ps.memory_info().rss)
        except psutil.Error:
            break
        if time.time() - t0 > timeout:
            proc.kill()
            proc.communicate()
            return None, "timeout"
        time.sleep(poll)
    out, err = proc.communicate()
    for line in reversed(out.splitlines()):
        if line.startswith("RESULT "):
            row = json.loads(line[7:])
            row["peak_mb"] = peak / 2**20
            row["delta_mb"] = max(0.0, peak / 2**20 - row["base_mb"])
            return row, None
    return None, (err.strip().splitlines() or ["no output"])[-1]


# --------------------------------------------------------------------- ladder
def ladder(config, n_cap, repeats, time_cap, threads=0, verbose=True):
    import psutil
    rows = []
    for n in range(2, n_cap + 1):
        avail = psutil.virtual_memory().available
        if predicted_bytes(config, n) * MEM_SAFETY > MEM_FRACTION * avail:
            if verbose:
                print(f"  {config}: stop before n={n} (needs ~{predicted_bytes(config, n) * MEM_SAFETY / 2**30:.1f} GiB, "
                      f"free {avail / 2**30:.1f} GiB)")
            break
        row, err = run_one(config, n, repeats, timeout=max(120.0, time_cap * (repeats + 2) * 4), threads=threads)
        if row is None:
            if verbose:
                print(f"  {config}: n={n} failed ({err}); stopping this ladder")
            break
        row["wall_median_s"] = float(np.median(row["wall_s"]))
        row["aer_median_s"] = float(np.nanmedian(row["aer_s"]))
        rows.append(row)
        if verbose:
            print(f"  {config:<21} n={n:>2}  wall {row['wall_median_s'] * 1e3:>10.2f} ms  "
                  f"aer {row['aer_median_s'] * 1e3:>10.2f} ms  "
                  f"mem +{row['delta_mb']:>8.1f} MiB (theory {row['theory_mb']:>8.1f})")
        if row["wall_median_s"] > time_cap:
            if verbose:
                print(f"  {config}: n={n} exceeded the {time_cap:g}s cap; stopping this ladder")
            break
    return rows


def fit_slope(ns, ys):
    """(slope, intercept) of log2(y) against n, or None if fewer than 2 usable points."""
    ns, ys = np.asarray(ns, float), np.asarray(ys, float)
    ok = ys > 0
    if ok.sum() < 2:
        return None
    slope, intercept = np.polyfit(ns[ok], np.log2(ys[ok]), 1)
    return float(slope), float(intercept)


def fit_window(rows):
    """Sizes used for the time fit: those slower than FIT_MIN_TIME, so that fixed
    overhead and the multi-thread ramp-up (where time barely grows with n) are left out.
    Falls back to the last three sizes when fewer than three qualify."""
    sel = [r for r in rows if r["wall_median_s"] >= FIT_MIN_TIME]
    return sel if len(sel) >= 3 else rows[-3:]


def run(full=False, outdir=None, verbose=True, threads=0, repeats=None):
    import psutil
    outdir = Path(outdir) if outdir else Path.cwd()
    cfg = dict(FULL if full else QUICK)
    if repeats:
        cfg["repeats"] = repeats
    suffix = ("" if full else "_quick") + (f"_t{threads}" if threads else "")   # quick never overwrites full data
    if verbose:
        print(f"S8 ({'full' if full else 'quick'}): depth={DEPTH}, noise p={NOISE_P}, repeats={cfg['repeats']}, "
              f"threads={threads or 'Aer default'}, cpus={os.cpu_count()}, "
              f"free RAM={psutil.virtual_memory().available / 2**30:.1f} GiB")

    data = {c: ladder(c, cfg["n_cap"][c], cfg["repeats"], cfg["time_cap"], threads, verbose) for c in CONFIGS}

    fits, info = {}, {}
    for c, rows in data.items():
        if not rows:
            continue
        win = fit_window(rows)
        big = [r for r in rows if r["delta_mb"] > 5]                 # memory below ~5 MiB is noise
        fits[c] = {"time_wall": fit_slope([r["n"] for r in win], [r["wall_median_s"] for r in win]),
                   "time_aer": fit_slope([r["n"] for r in win], [r["aer_median_s"] for r in win]),
                   "memory": fit_slope([r["n"] for r in big], [r["delta_mb"] for r in big]) if len(big) >= 2 else None}
        last = rows[-1]
        info[c] = {"time_fit_n_range": [win[0]["n"], win[-1]["n"]], "n_max": last["n"],
                   "mem_over_theory_at_n_max": last["delta_mb"] / last["theory_mb"] if last["theory_mb"] > 0.5 else None,
                   "overhead_wall_ms_at_n2": rows[0]["wall_median_s"] * 1e3}
    if verbose:
        print("\nslopes of log2(y) vs n (state-size theory: statevector 1, density matrix 2):")
        for c, f in fits.items():
            fmt = lambda v: "n/a" if v is None else f"{v[0]:+.2f}"
            i = info[c]
            ratio = "n/a" if i["mem_over_theory_at_n_max"] is None else f"{i['mem_over_theory_at_n_max']:.3f}"
            print(f"  {c:<21} time(wall) {fmt(f['time_wall'])} over n={i['time_fit_n_range'][0]}..{i['time_fit_n_range'][1]}   "
                  f"memory {fmt(f['memory'])}   mem/theory at n={i['n_max']}: {ratio}   n=2 overhead {i['overhead_wall_ms_at_n2']:.2f} ms")

    meta = {"depth": DEPTH, "noise_p": NOISE_P, "repeats": cfg["repeats"], "mode": "full" if full else "quick",
            "threads": threads or "default", "cpu_count": os.cpu_count(),
            "ram_total_gib": psutil.virtual_memory().total / 2**30,
            "platform": platform.platform(), "python": sys.version.split()[0]}
    with open(outdir / f"s8_scaling{suffix}.json", "w") as f:
        json.dump({"meta": meta, "fits_log2_slope_intercept": fits, "fit_info": info, "data": data}, f, indent=2)
    _plot(data, outdir / f"s8_scaling{suffix}.png")
    return {}


def _plot(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    style = {"statevector": ("C0", "o-"), "statevector_noisy": ("C0", "s--"),
             "density_matrix": ("C3", "o-"), "density_matrix_noisy": ("C3", "s--")}
    for c, rows in data.items():
        if not rows:
            continue
        col, st = style[c]
        ns = [r["n"] for r in rows]
        a1.semilogy(ns, [r["wall_median_s"] for r in rows], st, color=col, label=c)
        big = [r for r in rows if r["delta_mb"] > 5]
        if big:
            a2.semilogy([r["n"] for r in big], [r["delta_mb"] for r in big], st, color=col, label=c)
    nn = np.arange(2, 29)
    a2.semilogy(nn, 16 * 2.0 ** nn / 2**20, "k:", label="16*2^n bytes")
    a2.semilogy(nn[:15], 16 * 4.0 ** nn[:15] / 2**20, "k-.", label="16*4^n bytes")
    a2.set_ylim(1, None)
    a1.set_xlabel("qubits n"); a1.set_ylabel("median wall time per run (s)"); a1.set_title("run time")
    a2.set_xlabel("qubits n"); a2.set_ylabel("peak-memory growth (MiB)"); a2.set_title("memory")
    a1.legend(fontsize=7); a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="Aer threads (0 = Aer default, 1 = single thread)")
    ap.add_argument("--repeats", type=int, default=None, help="timed repeats per size")
    ap.add_argument("--worker", nargs=4, metavar=("CONFIG", "N", "REPEATS", "THREADS"))
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]), int(args.worker[3]))
    else:
        run(full=args.full, threads=args.threads, repeats=args.repeats)