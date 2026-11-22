# QAP Assignment I: Pure vs Mixed States

Team: A = Nihal Krishna Shrivastava (2023B5A70986G), B = Kartik Garg (2023A7PS0397G). Level attempted: 3.

## Environment

Python 3.11 (tested with 3.11.3 on Windows 10). Install the libraries:

```
pip install -r environment.txt
```

`environment.txt` pins qiskit, qiskit-aer, numpy, scipy and matplotlib. S8 also uses psutil to measure
memory; it is installed as a dependency of qiskit-aer, so it needs no line of its own.

## Run (single command)

```
python run_all.py
```

Run it from this folder (`python code/run_all.py` from the package root also works). It needs
`instance.json` for Level 3. It takes about 4 to 7 minutes, mostly S8; add `--s8 skip` to finish in under a minute. S8 time depends mostly on the machine.

Options (all optional):

| Option                   | Meaning                                                                        |
| ------------------------ | ------------------------------------------------------------------------------ |
| `--level 1\|2\|3`        | stop after this level (default: 3 if `instance.json` exists, otherwise 2)      |
| `--s8 quick\|full\|skip` | S8 ladder size (default `quick`; `full` is slow and is not needed for grading) |

A failing step does not stop the others: its traceback is printed, the remaining steps still run,
`results.json` is still written, and the exit code is 1.

## What it writes

- `results.json`: the schema from the assignment (written here and one folder up, because the
  assignment lists it at package level).
- `s2_data.json`, `s2_circuit.txt` (S2), `s4_noise_sweep.png` (S4), `s5_data.json`, `s5_error_vs_N.png` (S5),
  `s7_traj_error.png` (S7), `s8_scaling_quick.json` / `.png` (S8, from the default run; git-ignored).

## Files

| File                     | Owner | What it does                                                               |
| ------------------------ | ----- | -------------------------------------------------------------------------- |
| `run_all.py`             | B     | Single entry point; runs every step and writes `results.json`              |
| `utils.py`               | A     | Shared helpers: purity, Bloch vector, Kraus operators, von Neumann entropy |
| `l1_states.py`           | A     | S1: five single-qubit states, checked with NumPy and Qiskit                |
| `l2_s2_circuit.py`       | B     | S2: 10-gate 3-qubit circuit, statevector vs density matrix (five routes)   |
| `l2_s3_entanglement.py`  | A     | S3: Bell and GHZ partial trace                                             |
| `l2_s4_noise_sweep.py`   | A     | S4: bit-flip, dephasing, depolarising sweep (Aer vs exact Kraus)           |
| `l2_s5_trajectories.py`  | B     | S5: dephasing trajectories, Aer vs a NumPy sampler from O8                 |
| `instance_circuit.py`    | A     | Builds the instance circuit from `instance.json`                           |
| `instance.json`          | A     | Our Level 3 instance (n, circuit, noise, bipartition, seed)                |
| `l3_s6_instance.py`      | A     | S6: noisy instance, Aer density matrix vs exact Kraus, entropies           |
| `l3_s7_traj_instance.py` | A     | S7: trajectories on the instance, error against N                          |
| `l3_s8_scaling.py`       | B     | S8: scaling benchmark, statevector vs density matrix                       |

A and B are the team members named at the top. See `contribution.txt` in the package root for who
did what.

## Level 3 instance

`instance.json` was generated with the instructor script `make_instance.py` (the script is not part of this repository), using the last 4 digits of ID 2023B5A70986G. To regenerate it:

```
python make_instance.py 0986
```

and save the output as `instance.json` in this folder. `results.json` is then specific to that instance.

## Report-grade S8 data (slow, not run by `run_all.py` by default)

```
python l3_s8_scaling.py --full
python l3_s8_scaling.py --full --threads 1 --repeats 3
```

The first command writes `s8_scaling.json` / `.png`, the second `s8_scaling_t1.json` / `.png`. The
report uses the single-thread `s8_scaling_t1.*` data, which separates the algorithm's cost from Aer's
multi-threading. Timings and memory limits depend on
the machine (number of cores, free RAM); the ladders stop by themselves when the next size would not
fit in free memory.

## Reproducibility notes

- Tolerance 1e-10 for exactly computable quantities (purity, entropy, density-matrix agreement).
- S7 uses the seed in `instance.json`; error against N is computed from disjoint blocks of one run.
- S5 uses seed 2026; the repeats use well-separated seeds, because Aer shot streams for consecutive
  seeds are shifted copies of each other.
- Qubit order is Qiskit's little-endian (qubit 0 is the rightmost bit); S2 tests this on known states.

## Predictions

`../predictions.md` holds the predictions and their comparison with the results.
