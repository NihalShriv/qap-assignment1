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
**Before:** Instance 0986: 6 qubits, A = qubits 0-4, B = qubit 5, dephasing p = 0.121 after every gate. Without noise the global state is pure, so S(ρ) = 0, while S(ρ_A) = S(ρ_B) > 0 and at most 1 bit because B is a single qubit (Schmidt rank at most 2). With noise ρ becomes clearly mixed (S(ρ) of a few bits), and by Araki-Lieb S(ρ_A) stays within S(ρ_B) ≤ 1 bit of S(ρ). With noise, Araki-Lieb allows either sign: S(ρ_A) ≥ S(ρ) only if S(B|A) ≤ 0, i.e. if A-B entanglement still outweighs the noise on B, so we expect S(ρ_A) and S(ρ) within 1 bit of each other and are unsure of the sign. The Aer and Kraus routes should agree to about 1e-10.
**After:** Noiseless: S(ρ) = 0, S(ρ_A) = 1 bit, purity(ρ_A) = 0.5. Noisy: purity = 0.111549021553, S(ρ) = 3.526117392457, S(ρ_A) = 3.553370440592, S(ρ_B) = 1. So S(ρ_A) > S(ρ) in both cases, but with noise only by 0.027 bit; the sign is settled by entanglement only just outweighing the noise. Aer vs Kraus max difference 9.7e-17.

### S7 — Instance trajectories (Person A)
**Before:** Error vs N should follow ~1/√N on a log-log plot (slope ≈ −0.5). Every single trajectory is pure, so the purity of the N-trajectory average should start near 1 and fall towards the exact value, overshooting it by about (1 − γ)/N.
**After:** Matched. Fitted slope −0.520 over N = 16 to 8192, close to −0.5, and the errors follow √((1−γ)/N). At N = 8192 the trajectory purity is 0.111427 vs exact 0.111549, a difference of −1.2e-4, well inside the standard error of 1.6e-3; the (1−γ)/N bias of 1.1e-4 is too small to see at this N. Cost: 234 µs per trajectory.

### S8 — Scaling benchmark (Person B)
**Before:** Statevector memory scales as 2ⁿ, density-matrix as 4ⁿ. Density matrix becomes significantly slower around n=4 or 5.
**After:** _fill in_
