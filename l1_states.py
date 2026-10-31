import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector

from utils import dm_pure, dm_mixture, check_valid, purity, bloch_vector


STATES = {
    "|0>": DensityMatrix.from_label("0").data,
    "|+>": DensityMatrix(Statevector.from_label("+")).data,
    "mix(|0>,|1>)": dm_mixture([[1, 0], [0, 1]], [0.5, 0.5]),
    "mix(|+>,|->)": dm_mixture(
        [np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)],
        [0.5, 0.5],
    ),
    "partly_mixed": dm_mixture([[1, 0], [0, 1]], [0.8, 0.2]),
}


def run():
    print(f"\n{'State':<20} {'Valid':>6} {'Purity':>8} {'Bloch (x,y,z)':>30}")
    print("-" * 68)

    for name, rho in STATES.items():
        v = check_valid(rho)
        p = purity(rho)
        bv = bloch_vector(rho)
        print(f"{name:<20} {str(v['valid']):>6} {p:>8.6f}  ({bv[0]:+.4f}, {bv[1]:+.4f}, {bv[2]:+.4f})")
        assert v["valid"], f"Invalid density matrix: {name}"

    partly_mixed_purity = purity(STATES["partly_mixed"])
    assert 0.5 < partly_mixed_purity < 1.0, "partly_mixed purity must be strictly between 0.5 and 1"

    return {"purity_third_state": partly_mixed_purity}


if __name__ == "__main__":
    result = run()
    print(f"\npurity_third_state = {result['purity_third_state']}")
