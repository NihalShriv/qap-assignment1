# Predictions

Write 1–2 sentences per item **before** running the simulation.
Then add one line after: what you actually saw and whether it matched.

---

## S1 — Single-qubit states (Person A)

### |0⟩
**Before:** Pure state, purity = 1, Bloch vector (0, 0, 1).
**After:** Matched exactly. Purity = 1.0, Bloch = (0, 0, +1). Valid.

### |+⟩ = (|0⟩+|1⟩)/√2
**Before:** Pure state, purity = 1, Bloch vector (1, 0, 0).
**After:** Matched exactly. Purity = 1.0, Bloch = (+1, 0, 0). Valid.

### Equal mixture of |0⟩ and |1⟩
**Before:** Mixed state, purity = 0.5, Bloch vector (0, 0, 0). Both ensembles give the same ρ = I/2.
**After:** Matched. Purity = 0.5, Bloch = (0, 0, 0). ρ = I/2 as expected.

### Equal mixture of |+⟩ and |−⟩
**Before:** Also ρ = I/2, purity = 0.5, same as the |0⟩/|1⟩ mixture — illustrates that different ensembles share one density matrix.
**After:** Matched. Purity = 0.5, Bloch = (0, 0, 0). Identical ρ to the previous state, confirming ensemble non-uniqueness.

### Partly mixed state (our choice)
**Before:** Purity strictly between 0.5 and 1, Bloch vector length between 0 and 1.
**After:** Matched. 0.8|0⟩ + 0.2|1⟩ mixture gives purity = 0.68, Bloch = (0, 0, +0.6). Both strictly between bounds.

---

## S2 — Circuit consistency (Person B)

**Before:** Statevector and density-matrix simulation should give the same density matrix to floating-point precision (~1e-10). Any difference from global phase drops out in the density matrix.
**After:** _fill in_

---

## S3 — Entanglement (Person A)

### Bell state, trace out qubit 1
**Before:** The reduced state should be I/2, purity = 0.5, entropy = 1 bit. Maximum mixedness because the full state is pure and maximally entangled. The information sits entirely in the correlations.
**After:** Matched. Reduced state = I/2, purity = 0.5, entropy = 1.0 bit. NumPy and Qiskit partial_trace agree to 1e-10.

### GHZ state, trace out one qubit
**Before:** Reduced two-qubit state is (|00⟩⟨00| + |11⟩⟨11|)/2, purity exactly 0.5, entropy 1 bit. It is separable and only classically correlated. Tracing out two qubits gives I/2.
**After:** Matched. Two-qubit reduced state = diag(0.5, 0, 0, 0.5), purity = 0.5, entropy = 1.0 bit. One-qubit reduced state = I/2, purity = 0.5.

---

## S4 — Noise sweep (Person A)

### Bit-flip on |+⟩
**Before:** |+⟩ is an eigenstate of X, so bit-flip leaves it unchanged. Purity stays at 1 for all p.
**After:** Matched. Purity = 1.0 for all 21 p values. Kraus and Aer agree to 4.4e-16.

### Bit-flip on |0⟩
**Before:** Purity = (1+(1−2p)²)/2. Decreases from 1 at p=0 to 0.5 at p=0.5, then increases back to 1 at p=1 (clean bit flip).
**After:** Matched. Symmetric U-shape, minimum 0.5 at p=0.5, back to 1 at p=1.

### Dephasing on |+⟩
**Before:** Dephasing shrinks the x-component of the Bloch vector: r_x → (1−2p)r_x. Purity = (1+(1−2p)²)/2, reaching 0.5 at p=0.5 and returning to 1 at p=1 (state becomes |−⟩).
**After:** Matched. Same U-shape as bit-flip on |0⟩, by symmetry of the Bloch sphere picture.

### Dephasing on |0⟩
**Before:** |0⟩ is a Z eigenstate, so Z ρ Z = ρ. Dephasing leaves it completely unchanged. Purity stays at 1 for all p.
**After:** Matched. Purity = 1.0 for all 21 p values. Dephasing is the channel that leaves |0⟩ unchanged.

### Depolarising on |+⟩ and |0⟩
**Before:** Depolarising shrinks the Bloch vector uniformly: r → (1−p)r. Purity = (1+(1−p)²)/2 for both states. Reaches 0.5 only at p=1 in the O7 convention.
**After:** Matched. Both states give identical purity curves, monotonically decreasing from 1 to 0.5. Max Kraus vs Aer diff across all 126 cases: 4.4e-16.

---

## S5 — Dephasing trajectories (Person B)

**Before:** Each single trajectory is a pure state (one Kraus operator applied, then renormalised). The average over N trajectories converges to the exact density-matrix result. Error shrinks as ~1/√N.
**After:** _fill in_

---

## L3 (fill in once instance.json is generated)

### S6 — Instance simulation (Person A)
**Before:** _fill in after seeing the circuit_
**After:** _fill in_

### S7 — Instance trajectories (Person A)
**Before:** Error vs N should follow ~1/√N on a log-log plot (slope ≈ −0.5).
**After:** _fill in_

### S8 — Scaling benchmark (Person B)
**Before:** Statevector memory scales as 2ⁿ, density-matrix as 4ⁿ. Density matrix becomes significantly slower around n=4 or 5.
**After:** _fill in_
