"""Measures what Clifford+T (gridsynth) replacements of rotations cost under the grader.
Needs: pip install pennylane pygridsynth. Run: python3 t_gate_experiment.py"""
import pathlib
exec(compile((pathlib.Path(__file__).parent / "personal_best_full.py").read_text(), "personal_best_full.py", "exec"))
import mpmath, random as pyrandom
from pygridsynth import gridsynth_gates

u = haar_unitary(5, 2026)
tape = compile_circuit(submission_circ, u)
rot_idx = [i for i, op in enumerate(tape.operations) if type(op) in (qp.RX, qp.RY, qp.RZ)]
print("rotations:", len(rot_idx), "cost:", tape_cost(tape, 5))

# 1) How accurate must each rotation be? Give every rotation an error of size eps.
rng = np.random.default_rng(0)
for eps in (1e-8, 3e-9, 1e-9, 5e-10, 2e-10):
    ops = list(tape.operations)
    for i in rot_idx:
        op = ops[i]
        ops[i] = type(op)(float(op.data[0]) + 2 * eps * rng.choice([-1, 1]), wires=op.wires)
    err = float(np.abs(tape_matrix(qp.tape.QuantumScript(ops), 5) - u).max())
    print(f"every rotation off by {eps:.0e}: entrywise error {err:.1e}  {'PASS' if err <= ATOL else 'FAIL'}")

# 2) Replace K rotations by gridsynth Clifford+T sequences and grade the real circuit.
GATES = {"H": qp.Hadamard, "S": qp.S, "T": qp.T, "X": qp.PauliX, "Y": qp.PauliY, "Z": qp.PauliZ}
M = {"H": _FIXED_GATES[qp.Hadamard], "S": _FIXED_GATES[qp.S], "T": _FIXED_GATES[qp.T],
     "X": _FIXED_GATES[qp.PauliX], "Y": _FIXED_GATES[qp.PauliY], "Z": _FIXED_GATES[qp.PauliZ]}
def clifford_t(op, eps):
    theta = float(op.data[0]); w = op.wires[0]
    pre, post = [], []
    if type(op) is qp.RX:   # RX = H RZ H
        pre, post = [qp.Hadamard(w)], [qp.Hadamard(w)]
    elif type(op) is qp.RY:  # RY = S H RZ H S^dag   (S^dag = S S S)
        pre, post = [qp.S(w), qp.S(w), qp.S(w), qp.Hadamard(w)], [qp.Hadamard(w), qp.S(w)]
    seq = [c for c in str(gridsynth_gates(theta=mpmath.mpf(theta), epsilon=mpmath.mpf(eps))) if c in GATES]
    mat = np.eye(2, dtype=complex)
    for c in seq:            # string is a matrix product, leftmost applied last
        mat = mat @ M[c]
    rz = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
    phase = np.angle(np.trace(rz @ mat.conj().T))
    body = [GATES[c](w) for c in reversed(seq)] + [qp.GlobalPhase(-phase)]
    return pre + body + post
for K in (10, 100, 300):
    ops = []
    chosen = set(rot_idx[:K])
    for i, op in enumerate(tape.operations):
        ops.extend(clifford_t(op, 5e-10) if i in chosen else [op])
    new = qp.tape.QuantumScript(ops)
    err = float(np.abs(tape_matrix(new, 5) - u).max())
    nrot = sum(type(o) in (qp.RX, qp.RY, qp.RZ) for o in ops); nt = sum(type(o) is qp.T for o in ops)
    c = tape_cost(new, 5)
    print(f"replace {K:3d} rotations: {nrot} rotations + {nt} T  ->  cost {c} (was 45012, saving {100*(1-c/57024):.1f}%), error {err:.1e}")
