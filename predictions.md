# Predictions

Write 1–2 sentences per item **before** running the simulation.
Then add one line after: what you actually saw and whether it matched.

---

## S1 — Single-qubit states (Person A)

### |0⟩
**Before:** Pure state, purity = 1, Bloch vector (0, 0, 1).
**After:** _fill in_

### |+⟩ = (|0⟩+|1⟩)/√2
**Before:** Pure state, purity = 1, Bloch vector (1, 0, 0).
**After:** _fill in_

### Equal mixture of |0⟩ and |1⟩
**Before:** Mixed state, purity = 0.5, Bloch vector (0, 0, 0). Both ensembles give the same ρ = I/2.
**After:** _fill in_

### Equal mixture of |+⟩ and |−⟩
**Before:** Also ρ = I/2, purity = 0.5, same as the |0⟩/|1⟩ mixture — illustrates that different ensembles share one density matrix.
**After:** _fill in_

### Partly mixed state (our choice)
**Before:** Purity strictly between 0.5 and 1, Bloch vector length between 0 and 1.
**After:** _fill in_

---

## S2 — Circuit consistency (Person B)

**Before:** Statevector and density-matrix simulation should give the same reduced state to floating-point precision (~1e-10). Any difference will be from global phase, which drops out in the density matrix.
**After:** _fill in_

---

## S3 — Entanglement (Person B)

### Bell state, trace out qubit 1
**Before:** The reduced state should be I/2, purity = 0.5. Maximum mixedness because the full state is pure and maximally entangled.
**After:** _fill in_

### GHZ state, trace out one qubit
**Before:** Reduced state on the remaining two qubits should be a mixed state, purity < 1.
**After:** _fill in_

---

## S4 — Noise sweep (Person A)

### Bit-flip on |+⟩
**Before:** |+⟩ is an eigenstate of X, so bit-flip should leave it unchanged. Purity stays at 1 regardless of p.
**After:** _fill in_

### Bit-flip on |0⟩
**Before:** Purity decreases from 1 (at p=0) to 0.5 (at p=0.5) then increases back to 1 (at p=1, which is a clean bit flip).
**After:** _fill in_

### Dephasing on |+⟩
**Before:** Dephasing kills off-diagonal elements in the Z basis. |+⟩ has equal superposition, so purity drops to 0.5 as p→1.
**After:** _fill in_

### Dephasing on |0⟩
**Before:** |0⟩ is a Z eigenstate, so dephasing (Z ρ Z) leaves it unchanged. Purity stays at 1.
**After:** _fill in_

### Depolarising on |+⟩ and |0⟩
**Before:** Depolarising shrinks the Bloch vector uniformly. Both states should reach purity 0.5 around p=0.75 (since the channel maps to I/2 at p=1 in our convention).
**After:** _fill in_

---

## S5 — Dephasing trajectories (Person A)

**Before:** Each single trajectory is a pure state (one Kraus operator applied, then renormalised). The average over N trajectories should converge to the exact density-matrix result. Error should shrink as ~1/√N.
**After:** _fill in_

---

## L3 (fill in once instance.json is generated)

### S6 — Instance simulation
**Before:** _fill in after seeing the circuit_
**After:** _fill in_

### S7 — Instance trajectories
**Before:** Error vs N should follow ~1/√N on a log-log plot (slope ≈ −0.5).
**After:** _fill in_

### S8 — Scaling benchmark
**Before:** Statevector simulation memory scales as 2ⁿ, density-matrix as 4ⁿ. Density matrix should become significantly slower around n=4 or 5.
**After:** _fill in_
