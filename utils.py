import numpy as np
from qiskit.quantum_info import partial_trace as qiskit_partial_trace


def dm_pure(statevector):
    psi = np.asarray(statevector, dtype=complex).reshape(-1, 1)
    return psi @ psi.conj().T


def dm_mixture(states, probs):
    probs = np.asarray(probs, dtype=float)
    assert abs(probs.sum() - 1.0) < 1e-10
    return sum(p * dm_pure(s) for p, s in zip(probs, states))


def check_valid(rho, tol=1e-10):
    hermitian = np.allclose(rho, rho.conj().T, atol=tol)
    trace_one  = abs(np.trace(rho) - 1.0) < tol
    eigvals    = np.linalg.eigvalsh(rho)
    psd        = bool(np.all(eigvals >= -tol))
    return {"hermitian": hermitian, "trace_one": trace_one, "psd": psd, "valid": hermitian and trace_one and psd}


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def bloch_vector(rho):
    assert rho.shape == (2, 2)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.array([np.real(np.trace(rho @ s)) for s in (sx, sy, sz)])


def kraus_bit_flip(p):
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    return [np.sqrt(1 - p) * I, np.sqrt(p) * X]


def kraus_dephasing(p):
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [np.sqrt(1 - p) * I, np.sqrt(p) * Z]


def kraus_depolarising(p):
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [
        np.sqrt(1 - 3 * p / 4) * I,
        np.sqrt(p / 4) * X,
        np.sqrt(p / 4) * Y,
        np.sqrt(p / 4) * Z,
    ]


def apply_kraus(rho, kraus_ops):
    return sum(K @ rho @ K.conj().T for K in kraus_ops)


def von_neumann(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def partial_trace_np(rho, keep, dims):
    d_keep = int(np.prod([dims[i] for i in keep]))
    trace_out = sorted([i for i in range(len(dims)) if i not in keep], reverse=True)
    rho = rho.reshape(list(dims) * 2)
    for ax in trace_out:
        rho = np.trace(rho, axis1=ax, axis2=ax + rho.ndim // 2)
    return rho.reshape(d_keep, d_keep)


if __name__ == "__main__":
    rho0 = dm_pure([1, 0])
    assert abs(purity(rho0) - 1.0) < 1e-10
    assert np.allclose(bloch_vector(rho0), [0, 0, 1], atol=1e-10)

    rho_mix = dm_mixture([[1, 0], [0, 1]], [0.5, 0.5])
    assert abs(purity(rho_mix) - 0.5) < 1e-10
    assert np.allclose(bloch_vector(rho_mix), [0, 0, 0], atol=1e-10)

    for ops in [kraus_bit_flip(0.3), kraus_dephasing(0.3), kraus_depolarising(0.3)]:
        total = sum(K.conj().T @ K for K in ops)
        assert np.allclose(total, np.eye(2), atol=1e-10)

    assert abs(von_neumann(np.eye(2) / 2) - 1.0) < 1e-10
    assert abs(von_neumann(rho0)) < 1e-10

    rho_plus = dm_pure([1, 1] / np.sqrt(2))
    assert np.allclose(apply_kraus(rho_plus, kraus_bit_flip(0.3)), rho_plus, atol=1e-10)

    rho_dep = apply_kraus(rho0, kraus_depolarising(0.5))
    assert np.allclose(rho_dep, 0.5 * rho0 + 0.5 * np.eye(2) / 2, atol=1e-10)

    ghz = np.zeros((8, 8), dtype=complex)
    ghz[0, 0] = ghz[0, 7] = ghz[7, 0] = ghz[7, 7] = 0.5
    reduced_1q = partial_trace_np(ghz, [0], [2, 2, 2])
    assert np.allclose(reduced_1q, np.eye(2) / 2, atol=1e-10), "GHZ 1-qubit trace failed"
    reduced_2q = partial_trace_np(ghz, [0, 1], [2, 2, 2])
    assert abs(purity(reduced_2q) - 0.5) < 1e-10, "GHZ 2-qubit trace purity failed"

    print("utils.py self-check passed.")
