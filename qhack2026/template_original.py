import json
import pennylane as qp
import pennylane.numpy as np

# The unitary sizes your circuit is graded on, and the wire budget.
TEST_SIZES = (2, 3, 4, 5)
MAX_TOTAL_WIRES = 8
# The grader has to finish inside the challenge platform's time limit, so a decomposed
# circuit may hold at most this many gates, which is more than a good solution should need.
MAX_GATES = 100_000

ATOL = 1e-8
GRADES = {"Bronze": 5.0, "Silver": 20.0, "Gold": 60.0}  # minimum saving at TEST_SIZES[-1]

# ---------------------------------------------------------------------------------------
# The cost model. You do not need to change any of this, but read it: it is the scoring.
#
# 1. Your circuit is decomposed with qp.decompose into the gate set below.
#
# 2. Every surviving gate must be one of these exact PennyLane classes. Anything else, a
#    subclass, a look-alike with a borrowed name, or an operator with no decomposition,
#    is rejected rather than charged.
#
# 3. Each gate is charged in T-gate equivalents: a rotation 44, a T gate 1, the rest 0.
#
# 4. Qubits are charged too. Every wire beyond the n data wires multiplies the whole cost
#    by AUX_WIRE_FACTOR. A circuit on more than MAX_TOTAL_WIRES wires is rejected without
#    being simulated.
# ---------------------------------------------------------------------------------------
GATE_COSTS = {
    qp.RX: 44, qp.RY: 44, qp.RZ: 44,
    qp.T: 1,
    qp.S: 0, qp.Hadamard: 0, qp.PauliX: 0, qp.PauliY: 0, qp.PauliZ: 0,
    qp.CNOT: 0, qp.GlobalPhase: 0,
}
TARGET_GATE_SET = {"RX", "RY", "RZ", "T", "S", "H", "X", "Y", "Z", "CNOT", "GlobalPhase"}
ROTATION_COST = 44
AUX_WIRE_FACTOR = 2


class IllegalGateError(ValueError):
    """Raised when a decomposed circuit contains a gate outside the cost model."""


def compile_circuit(circuit, u_mat):
    """Decomposes `circuit(u_mat)` into the target gate set and checks every gate.

    Returns:
        qp.tape.QuantumScript: the decomposed circuit, which is what is charged and
            what is simulated.
    """
    tape = qp.tape.make_qscript(circuit)(u_mat)
    (decomposed,), _ = qp.decompose(tape, gate_set=TARGET_GATE_SET)
    for op in decomposed.operations:
        if type(op) not in GATE_COSTS:
            raise IllegalGateError(
                f"{type(op).__name__} on wires {list(op.wires)} is not in the gate set. Only "
                "the exact PennyLane classes RX, RY, RZ, T, S, H, X, Y, Z, CNOT and "
                "GlobalPhase are accepted after decomposition; subclasses and custom "
                "operators are rejected, whatever they are called."
            )
    return decomposed


def wire_usage(tape, num_qubits):
    """(total wires, auxiliary wires) used by a tape whose data wires are 0 to num_qubits-1."""
    data = set(range(num_qubits))
    used = set(tape.wires.tolist())
    return len(used | data), len(used - data)


def tape_cost(tape, num_qubits):
    """The T-gate-equivalent cost of an already compiled tape, (44 N_rot + N_T) * 2**N_aux.
    A tape on more than MAX_TOTAL_WIRES wires is not charged: the grader rejects it."""
    gates = sum(GATE_COSTS[type(op)] for op in tape.operations)
    _, aux_wires = wire_usage(tape, num_qubits)
    return int(gates * AUX_WIRE_FACTOR ** aux_wires)


def cost_of(circuit, u_mat):
    """The T-gate-equivalent cost of `circuit(u_mat)`. Call this on your own circuit."""
    num_qubits = int(round(np.log2(len(u_mat))))
    return tape_cost(compile_circuit(circuit, u_mat), num_qubits)


def _plain(array):
    """A plain NumPy array, without pennylane.numpy's autodiff bookkeeping (faster)."""
    array = np.array(array, dtype=complex, requires_grad=False)
    return array.unwrap() if hasattr(array, "unwrap") else array


_FIXED_GATES = {
    qp.T: _plain([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
    qp.S: _plain([[1, 0], [0, 1j]]),
    qp.Hadamard: _plain([[2 ** -0.5, 2 ** -0.5], [2 ** -0.5, -(2 ** -0.5)]]),
    qp.PauliX: _plain([[0, 1], [1, 0]]),
    qp.PauliY: _plain([[0, -1j], [1j, 0]]),
    qp.PauliZ: _plain([[1, 0], [0, -1]]),
    qp.CNOT: _plain([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
}


def _gate_matrix(op):
    """The grader's own matrix for each gate. Gates are never asked for their matrices."""
    kind = type(op)
    if kind in (qp.RX, qp.RY, qp.RZ):
        half = float(op.data[0]) / 2
        c, s = float(np.cos(half)), float(np.sin(half))
        if kind is qp.RX:
            return _plain([[c, -1j * s], [-1j * s, c]])
        if kind is qp.RY:
            return _plain([[c, -s], [s, c]])
        return _plain([[complex(c, -s), 0], [0, complex(c, s)]])
    return _FIXED_GATES[kind]


def tape_matrix(tape, num_qubits):
    """The matrix a compiled tape implements on wires 0 to num_qubits-1, every other
    wire starting in |0>. Simulated with the grader's own gate matrices.

    Runs of single-qubit gates on a wire are multiplied together before they are applied,
    and a CNOT is applied as a swap of two slices, which keeps grading fast."""
    order = list(range(num_qubits)) + sorted(
        (w for w in tape.wires.tolist() if w not in range(num_qubits)), key=str
    )
    position = {wire: k for k, wire in enumerate(order)}
    dim = 2 ** num_qubits
    states = _plain(np.zeros((dim,) + (2,) * len(order)))
    for column in range(dim):
        bits = [(column >> (num_qubits - 1 - k)) & 1 for k in range(num_qubits)]
        states[(column, *bits) + (0,) * (len(order) - num_qubits)] = 1.0
    every = [slice(None)] * states.ndim

    def axis_slice(fixed):
        index = list(every)
        for axis, value in fixed.items():
            index[axis] = value
        return tuple(index)

    def apply_single(gate, axis):
        zero, one = states[axis_slice({axis: 0})], states[axis_slice({axis: 1})]
        new_zero = gate[0, 0] * zero + gate[0, 1] * one
        one *= gate[1, 1]
        one += gate[1, 0] * zero
        zero[...] = new_zero

    pending = {}  # wire -> product of the single-qubit gates not applied yet

    def flush(wire):
        gate = pending.pop(wire, None)
        if gate is not None:
            apply_single(gate, 1 + position[wire])

    phase = 1.0 + 0j
    for op in tape.operations:
        kind = type(op)
        if kind is qp.GlobalPhase:
            phase *= complex(np.exp(-1j * float(op.data[0])))
            continue
        wires = op.wires.tolist()
        if len(wires) == 1:
            gate = _gate_matrix(op)
            pending[wires[0]] = gate @ pending[wires[0]] if wires[0] in pending else gate
            continue
        for wire in wires:
            flush(wire)
        axes = [1 + position[w] for w in wires]
        if kind is qp.CNOT:
            low = axis_slice({axes[0]: 1, axes[1]: 0})
            high = axis_slice({axes[0]: 1, axes[1]: 1})
            swapped = states[low].copy()
            states[low] = states[high]
            states[high] = swapped
        else:
            gate = _gate_matrix(op).reshape((2,) * (2 * len(axes)))
            states = np.tensordot(gate, states, axes=(list(range(len(axes), 2 * len(axes))), axes))
            states = np.moveaxis(states, list(range(len(axes))), axes)
    for wire in list(pending):
        flush(wire)
    kept = states.reshape(dim, dim, -1)[:, :, 0]
    return np.array(phase * kept.T, requires_grad=False)


def circuit_matrix(circuit, u_mat):
    """The matrix of `circuit(u_mat)` on wires 0 to n-1, every other wire starting in |0>."""
    num_qubits = int(round(np.log2(len(u_mat))))
    return tape_matrix(compile_circuit(circuit, u_mat), num_qubits)


def baseline_cost(num_qubits):
    """The cost of the 2006 Shannon-decomposition baseline, 44 * (21/16 * 4**n - 3 * 2**(n-1))."""
    return ROTATION_COST * ((21 * 4 ** num_qubits) // 16 - 3 * 2 ** (num_qubits - 1))


def haar_unitary(num_qubits, seed):
    """A Haar-random unitary of dimension 2**num_qubits. You do not need to change this."""
    rng = np.random.default_rng(seed)
    dim = 2 ** num_qubits
    ginibre = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q_mat, r_mat = np.linalg.qr(ginibre)
    diagonal = np.diag(r_mat)
    return np.array(q_mat * (diagonal / np.abs(diagonal)), requires_grad=False)


# Two pieces of linear algebra you are given. You do not need to change these.
_TOL = 1e-8
_MIXERS = (0.6180339887498949, 1.3027756377319946, 0.2360679774997896, 2.7320508075688772)


def diagonalize_unitary(d_mat):
    """Eigendecomposes a unitary in an orthonormal basis. Returns (v_mat, phases) with

        d_mat == v_mat @ diag(exp(1j * phases)) @ v_mat.conj().T
    """
    herm = d_mat + d_mat.conj().T
    skew = -1j * (d_mat - d_mat.conj().T)
    # herm and skew are Hermitian and commute with d_mat, so an eigenbasis of a generic
    # real combination of the two is an orthonormal eigenbasis of d_mat itself.
    best = None
    for mixer in _MIXERS:
        _, v_mat = np.linalg.eigh(herm + mixer * skew)
        rotated = v_mat.conj().T @ d_mat @ v_mat
        residual = np.abs(rotated - np.diag(np.diag(rotated))).max()
        if best is None or residual < best[0]:
            best = (residual, v_mat, np.angle(np.diag(rotated)))
        if residual < 1e-13:
            break
    return best[1], best[2]


def cs_decomposition(u_mat):
    """Cosine-sine decomposition of a unitary of even dimension 2m. Returns
    (l_0, l_1, theta, r_0, r_1) such that, in m x m blocks,

        u_mat == [[l_0, 0], [0, l_1]] @ [[C, -S], [S, C]] @ [[r_0, 0], [0, r_1]]

    with C = diag(cos(theta)) and S = diag(sin(theta)).
    """
    half = len(u_mat) // 2
    u_00, u_01 = u_mat[:half, :half], u_mat[:half, half:]
    u_10, u_11 = u_mat[half:, :half], u_mat[half:, half:]
    l_0, cosines, r_0 = np.linalg.svd(u_00)
    lower = u_10 @ r_0.conj().T
    sines = np.linalg.norm(lower, axis=0)
    theta = np.arctan2(sines, cosines)
    big_s, big_c = sines > _TOL, cosines > _TOL
    l_1 = _complete_basis(lower / np.where(big_s, sines, 1.0), big_s)
    r_1 = np.where(
        big_s[:, None],
        -(l_0.conj().T @ u_01) / np.where(big_s, sines, 1.0)[:, None],
        (l_1.conj().T @ u_11) / np.where(big_c, cosines, 1.0)[:, None],
    )
    # l_1 and r_1 come from dividing by small sines, so they drift from unitarity; snap
    # each to the nearest unitary so the drift is not amplified by a deep recursion.
    return l_0, _nearest_unitary(l_1), theta, r_0, _nearest_unitary(r_1)


def _nearest_unitary(mat):
    """The unitary closest to `mat` in Frobenius norm (its polar factor)."""
    left, _, right = np.linalg.svd(mat)
    return left @ right


def _complete_basis(partial, mask):
    """Fills the columns of `partial` where `mask` is False with an orthonormal completion."""
    missing = int((~mask).sum())
    if missing == 0:
        return partial
    good = partial[:, mask]
    extra, _, _ = np.linalg.svd(np.eye(len(partial)) - good @ good.conj().T)
    out = np.array(partial, dtype=complex)
    out[:, ~mask] = extra[:, :missing]
    return out


