import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error
from qiskit.quantum_info import DensityMatrix, Statevector

from utils import purity, kraus_bit_flip, kraus_dephasing, kraus_depolarising, apply_kraus

TOL = 1e-10
PS  = np.linspace(0, 1, 21)

RHO = {
    "|0>": DensityMatrix.from_label("0").data,
    "|+>": DensityMatrix(Statevector.from_label("+")).data,
}


def _aer_dm(is_plus, error):
    qc = QuantumCircuit(1)
    if is_plus:
        qc.h(0)
    qc.id(0)
    qc.save_density_matrix()
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(error, ["id"])
    result = AerSimulator(method="density_matrix", noise_model=nm).run(qc).result()
    return np.array(result.data(0)["density_matrix"])


def _analytic(channel, state, p):
    if channel == "bit_flip":
        return 1.0 if state == "|+>" else (1 + (1 - 2 * p) ** 2) / 2
    if channel == "dephasing":
        return 1.0 if state == "|0>" else (1 + (1 - 2 * p) ** 2) / 2
    return (1 + (1 - p) ** 2) / 2


CHANNELS = {
    "bit_flip":     (kraus_bit_flip,     lambda p: pauli_error([("X", p), ("I", 1 - p)])),
    "dephasing":    (kraus_dephasing,    lambda p: pauli_error([("Z", p), ("I", 1 - p)])),
    "depolarising": (kraus_depolarising, lambda p: depolarizing_error(p, 1)),
}


def run(verbose=True):
    max_diff = 0.0

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    channel_list = list(CHANNELS)
    state_list   = ["|0>", "|+>"]

    for ci, ch in enumerate(channel_list):
        kraus_fn, aer_fn = CHANNELS[ch]
        for si, st in enumerate(state_list):
            rho_init = RHO[st]
            is_plus  = (st == "|+>")

            purities_kraus = []
            purities_aer   = []

            for p in PS:
                rho_k = apply_kraus(rho_init, kraus_fn(p))
                rho_a = _aer_dm(is_plus, aer_fn(p))

                diff = float(np.max(np.abs(rho_k - rho_a)))
                max_diff = max(max_diff, diff)
                assert diff < TOL, f"{ch}, {st}, p={p:.2f}: diff={diff:.2e}"

                purities_kraus.append(purity(rho_k))
                purities_aer.append(purity(rho_a))

            analytic = [_analytic(ch, st, p) for p in PS]
            assert np.allclose(purities_kraus, analytic, atol=1e-6), \
                f"Kraus purity curve wrong for {ch}, {st}"

            ax = axes[si, ci]
            ax.plot(PS, purities_kraus, "b.",  label="Kraus",    markersize=8)
            ax.plot(PS, purities_aer,   "r+",  label="Aer",      markersize=8)
            ax.plot(PS, analytic,        "k-",  label="Analytic", linewidth=1)
            ax.set_title(f"{ch}  {st}", fontsize=9)
            ax.set_xlabel("p")
            ax.set_ylabel("purity")
            ax.set_ylim(0.45, 1.05)
            ax.legend(fontsize=7)

    plt.tight_layout()
    out = Path(__file__).parent / "s4_noise_sweep.png"
    plt.savefig(out, dpi=100)
    plt.close()

    if verbose:
        print(f"Max Kraus vs Aer diff (all 21 p): {max_diff:.2e}  TOL={TOL}")
        print(f"Plot saved: {out.name}")

        from qiskit_aer.noise import depolarizing_error as _dep
        from qiskit.quantum_info import Kraus, SuperOp
        from utils import kraus_depolarising
        p_demo = 0.3
        qiskit_sop = SuperOp(Kraus(_dep(p_demo, 1)))
        our_sop    = SuperOp(Kraus(kraus_depolarising(p_demo)))
        sop_diff   = float(np.max(np.abs((qiskit_sop - our_sop).data)))
        print(f"\nQ2.4 numerical example (p={p_demo}):")
        print(f"  depolarizing_error({p_demo},1) Kraus operators weight:")
        for k in Kraus(_dep(p_demo, 1)).data:
            print(f"    norm={float(np.linalg.norm(k)):.6f}")
        print(f"  Our kraus_depolarising({p_demo}) weights:")
        for k in kraus_depolarising(p_demo):
            print(f"    norm={float(np.linalg.norm(k)):.6f}")
        print(f"  SuperOp difference: {sop_diff:.2e}  (channels are identical)")
        print(f"  Interpretation: depolarizing_error(p,1) == (1-p)*rho + p*I/2")
        print(f"  i.e. Qiskit's parameter = O7's p directly (no 3/4 factor needed).")
        print(f"  Gotcha: a naive pauli_error([X,Y,Z each p/3]) would be a")
        print(f"  different channel; use depolarizing_error to match O7.")

    return {}


if __name__ == "__main__":
    result = run()
    print(result)
