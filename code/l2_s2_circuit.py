import json
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, Statevector
from qiskit_aer import AerSimulator

from utils import check_valid, purity

TOL = 1e-10


def build_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.7, 2)
    qc.cx(1, 2)
    qc.t(0)
    qc.rz(1.1, 1)
    qc.cz(0, 2)
    qc.h(1)
    qc.s(2)
    qc.cx(2, 0)
    assert len(qc.data) == 10, "S2 asks for exactly 10 gates"
    return qc


def ordering_test():
    for qubit, expected in [(0, "001"), (1, "010"), (2, "100")]:
        qc = QuantumCircuit(3)
        qc.x(qubit)
        d = Statevector(qc).probabilities_dict()
        assert set(d) == {expected} and abs(d[expected] - 1) < TOL, (qubit, d)
        c = qc.copy()
        c.measure_all()
        counts = AerSimulator().run(c, shots=50).result().get_counts()
        assert counts == {expected: 50}, (qubit, counts)


def aer_statevector(qc):
    c = qc.copy()
    c.save_statevector()
    res = AerSimulator(method="statevector").run(c, shots=1).result()
    return np.asarray(res.data(0)["statevector"])


def aer_density_matrix(qc):
    c = qc.copy()
    c.save_density_matrix()
    res = AerSimulator(method="density_matrix").run(c, shots=1).result()
    return np.asarray(res.data(0)["density_matrix"])


def rho_of(psi):
    psi = np.asarray(psi, dtype=complex).reshape(-1, 1)
    return psi @ psi.conj().T


def global_phase_demo(qc):
    qc2 = qc.copy()
    qc2.global_phase += np.pi / 3
    v1, v2 = Statevector(qc).data, Statevector(qc2).data
    assert not np.allclose(v1, v2, atol=TOL)
    assert np.allclose(rho_of(v1), rho_of(v2), atol=TOL)
    return float(np.max(np.abs(v1 - v2)))


def run(outdir="."):
    ordering_test()
    qc = build_circuit()
    with open(f"{outdir}/s2_circuit.txt", "w",encoding="utf-8") as f:
        f.write(str(qc.draw("text")))

    psi_A = Statevector(qc).data
    rho_B = DensityMatrix(qc).data
    psi_C = aer_statevector(qc)
    rho_D = aer_density_matrix(qc)
    U = Operator(qc).data
    psi_E = U[:, 0]
    e0 = np.zeros((8, 8), dtype=complex); e0[0, 0] = 1
    rho_E = U @ e0 @ U.conj().T

    rhos = {"A statevector (quantum_info)": rho_of(psi_A),
            "B density matrix (quantum_info)": rho_B,
            "C Aer statevector": rho_of(psi_C),
            "D Aer density_matrix": rho_D,
            "E NumPy U rho U^dag": rho_E}
    ref = rhos["B density matrix (quantum_info)"]

    print(f"{'method':<34}{'valid':>6}{'purity':>10}{'max|rho - rho_B|':>20}")
    diffs = {}
    for name, rho in rhos.items():
        v = check_valid(rho, TOL)
        diffs[name] = float(np.max(np.abs(rho - ref)))
        print(f"{name:<34}{str(bool(v['valid'])):>6}{purity(rho):>10.6f}{diffs[name]:>20.2e}")
        assert v["valid"] and abs(purity(rho) - 1) < TOL
        assert diffs[name] < TOL, f"{name} disagrees with the density-matrix method"

    overlap = float(abs(np.vdot(psi_A, psi_C)))
    raw_diff = float(np.max(np.abs(psi_A - psi_C)))
    assert abs(overlap - 1) < TOL and abs(abs(np.vdot(psi_A, psi_E)) - 1) < TOL
    print(f"\n|<psi_A|psi_C>| = {overlap:.12f}, raw max|psi_A - psi_C| = {raw_diff:.2e}")

    wrong = DensityMatrix(qc.reverse_bits()).data
    sens = float(np.max(np.abs(wrong - ref)))
    assert sens > 1e-3, "circuit is symmetric under qubit reversal; pick another circuit"
    print(f"reversed-qubit-order circuit differs by {sens:.3f}  (so the test would catch an ordering bug)")

    phase_gap = global_phase_demo(qc)
    print(f"global-phase demo: raw vectors differ by {phase_gap:.3f}, density matrices agree to {TOL:g}")

    with open(f"{outdir}/s2_data.json", "w") as f:
        json.dump({"tolerance": TOL, "max_rho_diff_vs_B": diffs,
                   "statevector_overlap_A_C": overlap, "raw_statevector_diff_A_C": raw_diff,
                   "ordering_sensitivity": sens, "global_phase_demo_vector_gap": phase_gap},
                  f, indent=2)
    return {}


if __name__ == "__main__":
    run()
