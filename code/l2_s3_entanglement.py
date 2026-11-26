import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, entropy

from utils import purity, partial_trace_np, von_neumann

TOL = 1e-10


def _bell():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return DensityMatrix(Statevector(qc))


def _ghz():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return DensityMatrix(Statevector(qc))


def _ordering_check():
    qc = QuantumCircuit(2)
    qc.x(0)
    dm = DensityMatrix(Statevector(qc))
    q0_qiskit = partial_trace(dm, [1]).data
    q0_np     = partial_trace_np(dm.data, keep=[1], dims=[2, 2])
    assert np.allclose(q0_qiskit, q0_np, atol=TOL), "Ordering check failed"
    assert np.allclose(q0_qiskit, np.array([[0, 0], [0, 1]], dtype=complex), atol=TOL)


def run(verbose=True):
    _ordering_check()

    bell = _bell()
    ghz  = _ghz()

    assert abs(purity(bell.data) - 1.0) < TOL
    assert abs(purity(ghz.data)  - 1.0) < TOL

    bell_q1 = partial_trace(bell, [1])
    bell_q0 = partial_trace(bell, [0])

    assert np.allclose(bell_q1.data, np.eye(2) / 2, atol=TOL)
    assert np.allclose(bell_q0.data, np.eye(2) / 2, atol=TOL)

    bell_purity = purity(bell_q1.data)
    assert abs(bell_purity - 0.5) < TOL

    bell_entropy = von_neumann(bell_q1.data)
    assert abs(bell_entropy - 1.0) < TOL
    assert abs(entropy(bell_q1, base=2) - 1.0) < TOL

    np_bell_q1 = partial_trace_np(bell.data, keep=[1], dims=[2, 2])
    assert np.allclose(np_bell_q1, bell_q1.data, atol=TOL), "NumPy vs Qiskit mismatch (Bell)"

    ghz_2q = partial_trace(ghz, [2])
    ghz_1q = partial_trace(ghz, [1, 2])

    expected_2q = np.zeros((4, 4), dtype=complex)
    expected_2q[0, 0] = expected_2q[3, 3] = 0.5
    assert np.allclose(ghz_2q.data, expected_2q, atol=TOL)
    assert abs(purity(ghz_2q.data) - 0.5) < TOL
    assert abs(entropy(ghz_2q, base=2) - 1.0) < TOL

    assert np.allclose(ghz_1q.data, np.eye(2) / 2, atol=TOL)
    assert abs(purity(ghz_1q.data) - 0.5) < TOL

    np_ghz_2q = partial_trace_np(ghz.data, keep=[1, 2], dims=[2, 2, 2])
    np_ghz_1q = partial_trace_np(ghz.data, keep=[2],    dims=[2, 2, 2])
    assert np.allclose(np_ghz_2q, ghz_2q.data, atol=TOL), "NumPy vs Qiskit mismatch (GHZ 2q)"
    assert np.allclose(np_ghz_1q, ghz_1q.data, atol=TOL), "NumPy vs Qiskit mismatch (GHZ 1q)"

    if verbose:
        print("Bell state (pure, purity=1):")
        print(f"  Reduced qubit 0: purity={bell_purity:.6f}  entropy={bell_entropy:.6f}")
        print(f"  rho_A = I/2 =\n{np.round(bell_q1.data, 10)}")
        print("\nGHZ state (pure, purity=1):")
        print(f"  Reduced 2-qubit: purity={purity(ghz_2q.data):.6f}  entropy={entropy(ghz_2q, base=2):.6f}")
        print(f"  Reduced 1-qubit: purity={purity(ghz_1q.data):.6f}")
        print("\nQ2.1: The global Bell state is pure, but it is entangled and cannot be written")
        print("as a product state. Tracing out one qubit discards all inter-qubit correlations,")
        print("leaving rho_A = I/2, the maximally mixed state with entropy 1 bit. The mixedness")
        print("of the subsystem is entirely entanglement-induced, not classical uncertainty.")
        print("\nNumPy partial_trace_np agrees with Qiskit to 1e-10: PASS")

    return {"bell_reduced_purity": float(bell_purity)}


if __name__ == "__main__":
    result = run()
    print(f"\nbell_reduced_purity = {result['bell_reduced_purity']}")
