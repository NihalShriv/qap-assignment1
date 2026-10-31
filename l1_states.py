"""
S1 (Level 1): five single-qubit states, checked with Qiskit and NumPy.

Covers O1-O4 (density matrix, validity, purity / Bloch vector, measurement)
and gives the numbers needed for Q1.1, Q1.2 and Q1.3.

The state vectors / density matrices are built with Qiskit's DensityMatrix,
and every quantity is computed twice (our NumPy code in utils.py and
Qiskit's own methods) and the two are required to agree to 1e-10.
"""
import numpy as np
from qiskit.circuit.library import HGate
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector

from utils import check_valid, purity, bloch_vector

TOL = 1e-10  # float64 round-off here is ~1e-16, so 1e-10 leaves a wide margin

# ---------------------------------------------------------------- states (O1)
ZERO = DensityMatrix.from_label("0")                      # |0><0|
ONE = DensityMatrix.from_label("1")                       # |1><1|
PLUS = DensityMatrix(Statevector.from_label("+"))         # |+><+|
MINUS = DensityMatrix(Statevector.from_label("-"))        # |-><-|

# Order matters: "third state" in results.json = equal mixture of |0>,|1>.
STATES = {
    "|0>":          ZERO,
    "|+>":          PLUS,
    "mix(|0>,|1>)": 0.5 * ZERO + 0.5 * ONE,
    "mix(|+>,|->)": 0.5 * PLUS + 0.5 * MINUS,
    "partly_mixed": 0.8 * ZERO + 0.2 * ONE,   # purity 0.68, strictly in (1/2, 1)
}

# Predictions written BEFORE running (also in predictions.md): purity, Bloch.
EXPECTED = {
    "|0>":          (1.00, (0, 0, 1.0)),
    "|+>":          (1.00, (1.0, 0, 0)),
    "mix(|0>,|1>)": (0.50, (0, 0, 0)),
    "mix(|+>,|->)": (0.50, (0, 0, 0)),
    "partly_mixed": (0.68, (0, 0, 0.6)),
}

# ------------------------------------------------------------ measurement (O4)
Z_BASIS = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
X_BASIS = [np.array([1, 1], dtype=complex) / np.sqrt(2),
           np.array([1, -1], dtype=complex) / np.sqrt(2)]


def measure_probs(rho, basis):
    """p_m = <m| rho |m> for each basis vector m."""
    return np.array([float(np.real(np.vdot(m, rho @ m))) for m in basis])


def qiskit_probs(dm, basis):
    """Same probabilities from Qiskit: rotate with H for the X basis."""
    if basis is X_BASIS:
        dm = dm.evolve(HGate())          # H maps |+> -> |0>, |-> -> |1>
    return np.asarray(dm.probabilities())


def run(verbose=True):
    rows = {}
    if verbose:
        print(f"{'State':<14}{'Valid':>6}{'Purity':>9}{'Bloch (x,y,z)':>26}"
              f"{'P(Z=0,1)':>16}{'P(X=+,-)':>16}")
        print("-" * 87)

    for name, dm in STATES.items():
        rho = dm.data

        # --- O2 validity: NumPy check and Qiskit check must agree
        v = check_valid(rho, TOL)
        assert v["valid"], f"{name}: not a valid density matrix {v}"
        assert dm.is_valid(atol=TOL), f"{name}: Qiskit says invalid"

        # --- O3 purity: Tr(rho^2) via utils, via eigenvalues, via Qiskit
        gamma = purity(rho)
        eig_gamma = float(np.sum(np.linalg.eigvalsh(rho) ** 2))
        assert abs(gamma - eig_gamma) < TOL, name
        assert abs(gamma - dm.purity()) < TOL, f"{name}: Qiskit purity differs"

        # --- O3 Bloch vector: Tr(rho sigma_k) via utils and via Qiskit Paulis
        r = bloch_vector(rho)
        r_qiskit = np.array([np.real(dm.expectation_value(Pauli(k))) for k in "XYZ"])
        assert np.allclose(r, r_qiskit, atol=TOL), f"{name}: Bloch vectors differ"

        # --- purity <-> Bloch length link: gamma = (1 + |r|^2) / 2
        assert abs(gamma - (1 + r @ r) / 2) < TOL, f"{name}: purity/Bloch mismatch"

        # --- O4 measurement in Z and X bases, NumPy vs Qiskit
        pz, px = measure_probs(rho, Z_BASIS), measure_probs(rho, X_BASIS)
        assert np.allclose(pz, qiskit_probs(dm, Z_BASIS), atol=TOL), name
        assert np.allclose(px, qiskit_probs(dm, X_BASIS), atol=TOL), name

        # --- compare with the predictions made before running
        exp_gamma, exp_r = EXPECTED[name]
        assert abs(gamma - exp_gamma) < TOL, f"{name}: purity != prediction"
        assert np.allclose(r, exp_r, atol=TOL), f"{name}: Bloch != prediction"

        rows[name] = dict(purity=gamma, bloch=r, pz=pz, px=px)
        if verbose:
            print(f"{name:<14}{str(v['valid']):>6}{gamma:>9.6f}"
                  f"  ({r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f})"
                  f"   ({pz[0]:.3f}, {pz[1]:.3f})   ({px[0]:.3f}, {px[1]:.3f})")

    # --- partly mixed state must be strictly between 1/2 and 1
    g5 = rows["partly_mixed"]["purity"]
    assert 0.5 + TOL < g5 < 1.0 - TOL, "partly_mixed purity must be in (1/2, 1)"

    # --- Q1.1: |+> and mix(|0>,|1>) agree in Z, differ in X
    plus, mixz = rows["|+>"], rows["mix(|0>,|1>)"]
    assert np.allclose(plus["pz"], mixz["pz"], atol=TOL)        # same Z stats
    assert not np.allclose(plus["px"], mixz["px"], atol=1e-3)   # different X stats

    # --- Q1.3: mix(|0>,|1>) and mix(|+>,|->) are the SAME matrix (I/2)
    rho3, rho4 = STATES["mix(|0>,|1>)"].data, STATES["mix(|+>,|->)"].data
    assert np.allclose(rho3, rho4, atol=TOL)
    assert np.allclose(rho3, np.eye(2) / 2, atol=TOL)

    if verbose:
        print("\nQ1.1  Z probs equal for |+> and mix(|0>,|1>); X probs differ:")
        print(f"      |+>          : Z={plus['pz']}  X={plus['px']}")
        print(f"      mix(|0>,|1>) : Z={mixz['pz']}  X={mixz['px']}")
        print("Q1.3  rho(mix(|0>,|1>)) == rho(mix(|+>,|->)) == I/2:")
        print(np.round(rho3, 12))
        print(f"\npurity_third_state = {rows['mix(|0>,|1>)']['purity']}")

    # results.json entry: third state = equal mixture of |0> and |1>
    return {"purity_third_state": float(rows["mix(|0>,|1>)"]["purity"])}


if __name__ == "__main__":
    run()
