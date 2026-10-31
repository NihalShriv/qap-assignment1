# QAP Assignment I — Pure vs Mixed States

## Run

```
pip install -r environment.txt
python run_all.py           # L1 + L2
python run_all.py --level 3 # L1 + L2 + L3 (needs instance.json)
```

## Structure

| File | Owner | What it does |
|------|-------|--------------|
| `utils.py` | A | Shared helpers: purity, Bloch vector, Kraus operators, von Neumann entropy |
| `l1_states.py` | A | S1: five single-qubit states |
| `l2_s2_circuit.py` | B | S2: 10-gate 3-qubit circuit, statevector vs density matrix |
| `l2_s3_entanglement.py` | A | S3: Bell and GHZ partial trace |
| `l2_s4_noise_sweep.py` | A | S4: bit-flip, dephasing, depolarising sweep |
| `l2_s5_trajectories.py` | B | S5: dephasing trajectories |
| `l3_s6_instance.py` | A | S6: instance simulation, entropies |
| `l3_s7_traj_instance.py` | A | S7: trajectories on instance |
| `l3_s8_scaling.py` | B | S8: scaling benchmark |
| `run_all.py` | B | Single entry point, writes results.json |
| `predictions.md` | Both | Predictions written before each simulation |

## Level 3

```
python make_instance.py <id1> <id2> <salt>
python run_all.py --level 3
```
