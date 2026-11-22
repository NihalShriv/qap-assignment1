import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, HGate, SGate, TGate, XGate
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, Statevector, entropy, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

import utils
from instance_circuit import build_circuit

TOL = 1e-10

GATES = {"H": HGate, "S": SGate, "T": TGate, "X": XGate, "CNOT": CXGate}
KRAUS = {
    "bitflip": utils.kraus_bit_flip,
    "dephasing": utils.kraus_dephasing,
    "depolarising": utils.kraus_depolarising,
}


def load_instance(instance_path="instance.json"):
    path = Path(instance_path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).parent / path
    with open(path) as f:
        return json.load(f)


def instance_circuit(inst):
    qc = QuantumCircuit(inst["n"])
    for gate in inst["gates"]:
        g, q = gate["g"], gate["q"]
        if g == "H":
            qc.h(q[0])
        elif g == "S":
            qc.s(q[0])
        elif g == "T":
            qc.t(q[0])
        elif g == "X":
            qc.x(q[0])
        elif g == "CNOT":
            qc.cx(q[0], q[1])
        else:
            raise ValueError(g)
    return qc


def noise_model(inst):
    channel = inst["noise"]["channel"]
    p = inst["noise"]["p"]
    if channel == "bitflip":
        err = pauli_error([("X", p), ("I", 1 - p)])
    elif channel == "dephasing":
        err = pauli_error([("Z", p), ("I", 1 - p)])
    elif channel == "depolarising":
        err = depolarizing_error(p, 1)
    else:
        raise ValueError(channel)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(err, ["h", "s", "t", "x"])
    nm.add_all_qubit_quantum_error(err.tensor(err), ["cx"])
    return nm


def aer_dm(inst):
    qc = instance_circuit(inst)
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model(inst))
    result = sim.run(qc).result()
    return np.array(result.data(0)["density_matrix"])


def kraus_dm(inst):
    rho = DensityMatrix.from_label("0" * inst["n"])
    kraus_fn = KRAUS[inst["noise"]["channel"]]
    p = inst["noise"]["p"]
    for gate in inst["gates"]:
        rho = rho.evolve(Operator(GATES[gate["g"]]()), qargs=gate["q"])
        for q in gate["q"]:
            rho = rho.evolve(Kraus(kraus_fn(p)), qargs=[q])
    return rho.data


def _ordering_check():
    qc = QuantumCircuit(6)
    qc.x(0)
    dm = DensityMatrix(Statevector(qc))
    red = partial_trace(dm, [5]).data
    expected = np.zeros((32, 32), dtype=complex)
    expected[1, 1] = 1
    assert red.shape == (32, 32)
    assert np.allclose(red, expected, atol=TOL), "Ordering check failed"


def run(instance_path="instance.json", verbose=True):
    _ordering_check()

    inst = load_instance(instance_path)
    n = inst["n"]
    A = inst["subsystem_A"]
    B = [q for q in range(n) if q not in A]

    assert np.allclose(
        Operator(instance_circuit(inst)).data,
        Operator(build_circuit()).data,
        atol=1e-12,
    ), "Circuit mismatch"

    psi = Statevector(instance_circuit(inst))
    rho0 = DensityMatrix(psi)
    rho0_A = partial_trace(rho0, B).data
    pur0 = utils.purity(rho0.data)
    pur0_A = utils.purity(rho0_A)
    S0 = utils.von_neumann(rho0.data)
    S0_A = utils.von_neumann(rho0_A)

    rho_aer = aer_dm(inst)
    rho_k = kraus_dm(inst)
    max_diff = float(np.max(np.abs(rho_aer - rho_k)))
    assert max_diff < TOL, f"Aer vs Kraus mismatch {max_diff}"
    assert utils.check_valid(rho_aer)["valid"]
    assert utils.check_valid(rho_k)["valid"]

    pur = utils.purity(rho_k)
    ent = utils.von_neumann(rho_k)
    assert abs(ent - entropy(DensityMatrix(rho_k), base=2)) < 1e-8

    rho_A = partial_trace(DensityMatrix(rho_k), B).data
    rho_B = partial_trace(DensityMatrix(rho_k), A).data
    ent_A = utils.von_neumann(rho_A)
    ent_B = utils.von_neumann(rho_B)
    pur_A = utils.purity(rho_A)
    rho_A_np = utils.partial_trace_np(rho_k, keep=sorted(n - 1 - q for q in A), dims=[2] * n)
    assert np.allclose(rho_A, rho_A_np, atol=TOL), "NumPy vs Qiskit partial trace mismatch"

    if verbose:
        mi = ent_A + ent_B - ent
        rel0 = ">" if S0_A > S0 else "<"
        rel = ">" if ent_A > ent else "<"
        print(f"Instance {inst['code']}, fingerprint {inst['fingerprint']}")
        print(f"n = {n}, A = {A}, B = {B}, channel = {inst['noise']['channel']}, p = {inst['noise']['p']}")
        print(f"Noiseless full state: purity = {pur0:.12g}, S = {S0:.12g}")
        print(f"Noiseless rho_A: purity = {pur0_A:.12g}, S = {S0_A:.12g}")
        print(f"Noisy: purity = {pur:.12g}, S(rho) = {ent:.12g}")
        print(f"Noisy: S(rho_A) = {ent_A:.12g}, S(rho_B) = {ent_B:.12g}, purity(rho_A) = {pur_A:.12g}")
        print(f"Max Aer vs Kraus elementwise diff = {max_diff:.3e}")
        print(f"Mutual information S_A + S_B - S = {mi:.12g}")
        print(f"Q3.1 noiseless: S(rho_A) {rel0} S(rho): {S0_A:.12g} {rel0} {S0:.12g}")
        print(f"Q3.1 noisy: S(rho_A) {rel} S(rho): {ent_A:.12g} {rel} {ent:.12g}")

    return {"purity": float(pur), "entropy": float(ent), "entropy_A": float(ent_A)}


if __name__ == "__main__":
    print(run())
