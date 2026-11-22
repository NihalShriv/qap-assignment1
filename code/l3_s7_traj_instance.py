import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator

from l3_s6_instance import instance_circuit, kraus_dm, load_instance, noise_model
from utils import purity

N_MAX = 8192
NS = [2 ** k for k in range(4, 14)]


def trajectories(inst, shots=N_MAX):
    qc = instance_circuit(inst)
    qc.save_statevector(pershot=True)
    sim = AerSimulator(
        method="statevector",
        noise_model=noise_model(inst),
        seed_simulator=inst["traj_seed"],
    )
    result = sim.run(qc, shots=shots).result()
    states = result.data(0)["statevector"]
    return np.array([np.asarray(s) for s in states])


def _avg_dm(block):
    return block.T @ block.conj() / len(block)


def run(instance_path="instance.json", verbose=True):
    inst = load_instance(instance_path)
    n = inst["n"]
    rho_exact = kraus_dm(inst)
    gamma_exact = purity(rho_exact)

    t0 = time.perf_counter()
    psi = trajectories(inst)
    traj_time = time.perf_counter() - t0
    assert psi.shape == (N_MAX, 2 ** n), psi.shape
    assert np.allclose(np.linalg.norm(psi, axis=1), 1.0, atol=1e-10), "trajectory not normalised"

    a = trajectories(inst, shots=64)
    b = trajectories(inst, shots=64)
    assert np.allclose(a, b), "same seed gave different trajectories"

    errs = []
    for N in NS:
        blocks = N_MAX // N
        e = [
            np.linalg.norm(_avg_dm(psi[i * N:(i + 1) * N]) - rho_exact)
            for i in range(blocks)
        ]
        errs.append(float(np.mean(e)))

    slope, intercept = np.polyfit(np.log10(NS), np.log10(errs), 1)

    rho_N = _avg_dm(psi)
    traj_purity = purity(rho_N)
    unbiased = (N_MAX * traj_purity - 1) / (N_MAX - 1)
    assert abs(traj_purity - gamma_exact) < 0.01, f"{traj_purity} vs {gamma_exact}"

    ns = np.array(NS, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(ns, errs, "o", label="mean error over blocks")
    ax.loglog(ns, errs[0] * (ns[0] / ns) ** 0.5, "--", label="1/sqrt(N) reference")
    ax.loglog(ns, 10 ** intercept * ns ** slope, "-", label=f"fit slope = {slope:.3f}")
    ax.loglog(ns, np.sqrt((1 - gamma_exact) / ns), ":", label="theory sqrt((1-purity)/N)")
    ax.set_xlabel("N (trajectories)")
    ax.set_ylabel("||rho_N - rho_exact||_F")
    ax.set_title(f"S7 instance {inst['code']}: trajectory error vs N")
    ax.legend()
    fig.tight_layout()
    out = Path(__file__).parent / "s7_traj_error.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)

    if verbose:
        print(f"seed = {inst['traj_seed']}, N_MAX = {N_MAX}")
        print(f"{'N':>6} {'err_N':>12}")
        for N, e in zip(NS, errs):
            print(f"{N:>6} {e:>12.6f}")
        print(f"fitted slope = {slope:.4f} (expected -0.5)")
        print(f"exact purity = {gamma_exact:.12g}")
        print(f"trajectory purity at N_MAX = {traj_purity:.12g}")
        print(f"bias-corrected purity = {unbiased:.12g}")
        print(f"difference from exact: raw = {traj_purity - gamma_exact:.3e}, corrected = {unbiased - gamma_exact:.3e}")
        print(f"expected bias (1 - gamma_exact) / N_MAX = {(1 - gamma_exact) / N_MAX:.3e}")
        print(f"trajectory run time = {traj_time:.3f} s for {N_MAX} shots ({traj_time / N_MAX * 1e6:.1f} us per trajectory)")
        print(f"plot saved: {out.name}")

    return {"traj_estimate_purity": float(traj_purity), "n_traj": int(N_MAX)}


if __name__ == "__main__":
    print(run())
