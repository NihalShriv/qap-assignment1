import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

from utils import dm_pure, kraus_dephasing, apply_kraus

TOL = 1e-10
P = 0.3
NS = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
N_MAX = NS[-1]
REPEATS = 20
SEED = 2026

PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)


def exact_rho(p):
    rho = apply_kraus(dm_pure(PLUS), kraus_dephasing(p))
    closed = 0.5 * np.array([[1, 1 - 2 * p], [1 - 2 * p, 1]], dtype=complex)
    qiskit = DensityMatrix(Statevector(PLUS)).evolve(Kraus(kraus_dephasing(p))).data
    assert np.allclose(rho, closed, atol=TOL)
    assert np.allclose(rho, qiskit, atol=TOL)
    return rho


def make_aer(p):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(pauli_error([("Z", p), ("I", 1 - p)]), ["id"])
    return AerSimulator(method="statevector", noise_model=nm)


def aer_trajectories(sim, shots, seed):
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.id(0)
    qc.save_statevector(pershot=True)
    res = sim.run(qc, shots=shots, seed_simulator=seed).result()
    svs = np.array([np.asarray(s) for s in res.data(0)["statevector"]])
    if len(svs) == 1 and shots > 1:
        svs = np.tile(svs, (shots, 1))
    assert len(svs) == shots
    return svs


def numpy_trajectories(p, shots, rng):
    ops = kraus_dephasing(p)
    probs = np.array([np.real(np.vdot(PLUS, K.conj().T @ K @ PLUS)) for K in ops])
    assert abs(probs.sum() - 1) < TOL
    ks = rng.choice(len(ops), size=shots, p=probs)
    out = np.empty((shots, 2), dtype=complex)
    for k, K in enumerate(ops):
        psi = K @ PLUS
        out[ks == k] = psi / np.linalg.norm(psi)
    return out


def running_errors(svs, rho_exact, ns):
    rhos = np.einsum("si,sj->sij", svs, svs.conj())
    cums = np.cumsum(rhos, axis=0)
    return np.array([np.linalg.norm(cums[n - 1] / n - rho_exact) for n in ns])


def sanity_checks():
    for p, expect in [(0.0, "+"), (1.0, "-")]:
        svs = aer_trajectories(make_aer(p), 200, SEED)
        target = dm_pure(Statevector.from_label(expect).data)
        err = running_errors(svs, target, [200])[0]
        assert err < TOL, f"p={p}: Aer is not giving |{expect}> every shot"
    svs = aer_trajectories(make_aer(P), 500, SEED)
    assert np.allclose(np.linalg.norm(svs, axis=1), 1, atol=TOL)


def run(outdir="."):
    sanity_checks()
    rho_ex = exact_rho(P)
    sim = make_aer(P)

    seeds = np.random.default_rng(SEED).integers(0, 2**31 - 1, size=REPEATS)
    aer_err = np.array([running_errors(aer_trajectories(sim, N_MAX, int(sd)), rho_ex, NS)
                        for sd in seeds])
    rng = np.random.default_rng(SEED)
    np_err = np.array([running_errors(numpy_trajectories(P, N_MAX, rng), rho_ex, NS)
                       for _ in range(REPEATS)])

    ns = np.array(NS, dtype=float)
    mean_aer, mean_np = aer_err.mean(axis=0), np_err.mean(axis=0)
    theory = 2 * np.sqrt(P * (1 - P) / (np.pi * ns))
    slope_aer = np.polyfit(np.log(ns), np.log(mean_aer), 1)[0]
    slope_np = np.polyfit(np.log(ns), np.log(mean_np), 1)[0]

    print(f"S5: p={P}, repeats={REPEATS}, exact rho off-diagonal = {rho_ex[0, 1].real:.3f}")
    print(f"{'N':>7}{'Aer err':>12}{'NumPy err':>12}{'theory':>12}")
    for n, a, b, t in zip(NS, mean_aer, mean_np, theory):
        print(f"{n:>7}{a:>12.5f}{b:>12.5f}{t:>12.5f}")
    print(f"log-log slope: Aer {slope_aer:+.3f}, NumPy {slope_np:+.3f}  (expected -0.5)")
    assert abs(slope_aer + 0.5) < 0.15 and abs(slope_np + 0.5) < 0.15

    fig, ax = plt.subplots(figsize=(6, 4))
    lo, hi = np.percentile(aer_err, [16, 84], axis=0)
    ax.fill_between(ns, lo, hi, alpha=0.2, label="Aer 16-84% band")
    ax.loglog(ns, mean_aer, "o-", label=f"Aer mean (slope {slope_aer:+.2f})")
    ax.loglog(ns, mean_np, "s--", label=f"NumPy O8 mean (slope {slope_np:+.2f})")
    ax.loglog(ns, theory, "k:", label=r"theory $2\sqrt{p(1-p)/(\pi N)}$")
    ax.set_xlabel("number of trajectories N")
    ax.set_ylabel(r"$\|\bar\rho_N-\rho_{exact}\|_F$")
    ax.set_title(f"S5: dephasing trajectories of |+>, p={P}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/s5_error_vs_N.png", dpi=150)

    with open(f"{outdir}/s5_data.json", "w") as f:
        json.dump({"p": P, "N": NS, "repeats": REPEATS, "seed": SEED,
                   "aer_mean_error": mean_aer.tolist(), "numpy_mean_error": mean_np.tolist(),
                   "theory": theory.tolist(), "slope_aer": float(slope_aer),
                   "slope_numpy": float(slope_np)}, f, indent=2)
    return {}


if __name__ == "__main__":
    run()
