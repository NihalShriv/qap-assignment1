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
        print("\nQ2.4 convention note:")
        print("  pauli_error([('X', p), ('I', 1-p)]) applies X with prob p,")
        print("  matching O7's bit-flip directly.")
        print("  depolarizing_error(p, 1) implements (1-p)rho + p*I/2,")
        print("  matching O7's depolarising with the same parameter p.")
        print("  Qiskit's little-endian qubit ordering is a second gotcha")
        print("  that affects multi-qubit Kraus placement (see S3 ordering check).")

    return {"s4_max_route_diff": float(max_diff)}


if __name__ == "__main__":
    result = run()
    print(result)
