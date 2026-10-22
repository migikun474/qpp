# QHack 2026 – "Personal Best" unitary synthesis

`solution_part.py` is the code to paste into the challenge editor: it replaces the empty
`submission_circ` at the end of the template (the helpers go just above it).

Local check (needs `pip install pennylane`): `python3 run_local.py`

| n | rotations | cost | baseline | saving |
|---|-----------|------|----------|--------|
| 2 | 15   | 660   | 660   | 0.00%  |
| 3 | 63   | 2772  | 3168  | 12.50% |
| 4 | 255  | 11220 | 13728 | 18.27% |
| 5 | 1023 | 45012 | 57024 | 21.06% |

Max error is about 1e-14, with no auxiliary wires and under 3 s for every case.

The circuit uses exactly 4^n - 1 rotations. That is the dimension of U(2^n) once the
global phase is removed, so no exact construction can use fewer, and 21.06% is the most
any exact rotation circuit can save at 5 qubits. The 60% Gold mark (518 rotations or
fewer) is out of reach for exact synthesis under this cost model. See the solution header
for how the construction works.
