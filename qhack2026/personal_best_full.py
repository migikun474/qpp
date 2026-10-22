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


# ---------------------------------------------------------------------------------------
# Solution: rotation-optimal synthesis, exactly 4**n - 1 rotations and no auxiliary wire.
#
# 1. Recursive cosine-sine decomposition. Splitting on wire 0, then wire 1, ... turns
#    u_mat into an alternating time-ordered sequence
#        Leaf, Y, Leaf, Y, ..., Y, Leaf
#    of 2**(n-1) "leaves" (uniformly controlled 2x2 unitaries on wire n-1, controlled
#    by wires 0..n-2) and 2**(n-1) - 1 uniformly controlled RY gates, the Y on wire t
#    being controlled by every other wire.
#
# 2. Each leaf block is e^{i phi} RZ(alpha) RY(beta) RZ(gamma). A Y on wire t commutes
#    with any diagonal that does not involve wire t, so the part of the leaf's final
#    RZ(alpha) and phase phi that does not depend on wire t (the average over that bit)
#    is pushed through the Y into the next leaf for free. What stays behind is
#    antisymmetric in wire t, and costs half as many rotations. Every leaf except the
#    last one then costs 3 * 2**(n-1) rotations instead of 4 * 2**(n-1).
#
# 3. Multiplexed rotations use the Gray-code construction (2**k rotations and CNOTs for
#    k controls), the global phase is a free GlobalPhase.
#
# Count: (2**(n-1) - 1) * (3 + 1) * 2**(n-1) + 4 * 2**(n-1) - 1 = 4**n - 1 rotations,
# which is the dimension of U(2**n) modulo global phase, so it is the least possible
# number of rotations for exact synthesis: 15, 63, 255, 1023 for n = 2, 3, 4, 5.
# ---------------------------------------------------------------------------------------
_ZERO_ANGLE = 1e-14


def _multiplexed_rotation(gate, angles, controls, target):
    """Uniformly controlled gate(angles[x]) on target, x = control values (controls[0]
    is the most significant bit). Uses 2**k rotations and 2**k CNOTs."""
    angles = np.asarray(angles, dtype=float)
    k = len(controls)
    if k == 0:
        if abs(angles[0]) > _ZERO_ANGLE:
            gate(float(angles[0]), wires=target)
        return
    # Walsh-Hadamard transform: angles[x] = sum_S w[S] (-1)^{popcount(x & S)}.
    walsh = angles.copy()
    step = 1
    while step < len(walsh):
        for start in range(0, len(walsh), 2 * step):
            low = walsh[start:start + step].copy()
            high = walsh[start + step:start + 2 * step].copy()
            walsh[start:start + step] = low + high
            walsh[start + step:start + 2 * step] = low - high
        step *= 2
    walsh /= len(walsh)
    size = 2 ** k
    for i in range(size):
        gray = i ^ (i >> 1)
        if abs(walsh[gray]) > _ZERO_ANGLE:
            gate(float(walsh[gray]), wires=target)
        nxt = (i + 1) ^ ((i + 1) >> 1) if i + 1 < size else 0
        changed = (gray ^ nxt).bit_length() - 1
        qp.CNOT(wires=[controls[k - 1 - changed], target])


def _diagonal(phases, wires):
    """diag(exp(1j * phases)) on wires (wires[0] most significant), 2**m - 1 rotations."""
    phases = np.asarray(phases, dtype=float)
    if len(wires) == 0:
        qp.GlobalPhase(-float(phases[0]))
        return
    even, odd = phases[0::2], phases[1::2]
    # exp(1j * (-1)^b * q) = RZ(-2 q) on the last wire, multiplexed by the others.
    _multiplexed_rotation(qp.RZ, -(even - odd), wires[:-1], wires[-1])
    _diagonal((even + odd) / 2, wires[:-1])


def _zyz(u):
    """u == exp(1j phi) RZ(alpha) RY(beta) RZ(gamma). Returns (phi, alpha, beta, gamma)."""
    phi = np.angle(np.linalg.det(u)) / 2
    v = u * np.exp(-1j * phi)
    beta = 2 * np.arctan2(abs(v[1, 0]), abs(v[0, 0]))
    plus = np.angle(v[1, 1]) if abs(v[1, 1]) > 1e-12 else 0.0
    minus = np.angle(v[1, 0]) if abs(v[1, 0]) > 1e-12 else 0.0
    return phi, plus + minus, beta, plus - minus


def _csd_sequence(blocks, level, num_qubits, out):
    """Appends the time-ordered leaves and multiplexed RY gates of the multiplexor
    diag(blocks), which is controlled by wires 0..level-1, to out."""
    if len(blocks[0]) == 2:
        out.append(("leaf", [np.array(b, dtype=complex) for b in blocks]))
        return
    lefts, rights, angles = [], [], []
    for block in blocks:
        l_0, l_1, theta, r_0, r_1 = cs_decomposition(block)
        lefts += [l_0, l_1]
        rights += [r_0, r_1]
        angles.append(2 * np.asarray(theta, dtype=float))
    _csd_sequence(rights, level + 1, num_qubits, out)
    out.append(("ry", level, np.concatenate(angles)))
    _csd_sequence(lefts, level + 1, num_qubits, out)


def submission_circ(u_mat):
    """Queues a circuit equal to u_mat to within ATOL, global phase included, at low cost.

    Args:
        u_mat (np.ndarray): A 2**n x 2**n unitary matrix, with n in TEST_SIZES. It is
            Haar-random, so you may assume no special structure whatsoever.

    Returns:
        None. The gates you queue on wires 0 to n-1 are the answer. Auxiliary wires are
        allowed, up to MAX_TOTAL_WIRES in total, but each one doubles your cost.
    """
    u_mat = _plain(u_mat)
    n = int(round(np.log2(len(u_mat))))
    if n == 1:
        phi, alpha, beta, gamma = _zyz(u_mat)
        qp.RZ(float(gamma), wires=0)
        qp.RY(float(beta), wires=0)
        qp.RZ(float(alpha), wires=0)
        qp.GlobalPhase(-float(phi))
        return

    sequence = []
    _csd_sequence([u_mat], 0, n, sequence)
    target, controls = n - 1, list(range(n - 1))
    carry = None  # diagonal pushed out of the previous leaf, as (phases, rz angles)

    for index in range(0, len(sequence), 2):
        blocks = sequence[index][1]
        if carry is not None:
            blocks = [
                b @ np.diag([np.exp(1j * (f - g / 2)), np.exp(1j * (f + g / 2))])
                for b, f, g in zip(blocks, *carry)
            ]
        phi, alpha, beta, gamma = (np.array(v) for v in zip(*(_zyz(b) for b in blocks)))
        _multiplexed_rotation(qp.RZ, gamma, controls, target)
        _multiplexed_rotation(qp.RY, beta, controls, target)

        if index + 1 == len(sequence):  # the last leaf is implemented in full
            _multiplexed_rotation(qp.RZ, alpha, controls, target)
            _diagonal(phi, controls)
            break

        _, wire, ry_angles = sequence[index + 1]
        # The diagonal exp(1j f) RZ(g) with f, g independent of `wire` commutes with the
        # RY on `wire`: push it into the next leaf, keep the part odd in `wire`.
        bit = 1 << (n - 2 - wire)
        partner = np.arange(len(blocks)) ^ bit
        carry = ((phi + phi[partner]) / 2, (alpha + alpha[partner]) / 2)
        others = [w for w in controls if w != wire]
        zero = np.array([x for x in range(len(blocks)) if not x & bit])
        # RZ((-1)^{x_wire} a(y)) on the target, and exp(1j (-1)^{x_wire} h(y)) on `wire`.
        qp.CNOT(wires=[wire, target])
        _multiplexed_rotation(qp.RZ, (alpha - carry[1])[zero], others, target)
        qp.CNOT(wires=[wire, target])
        _multiplexed_rotation(qp.RZ, -2 * (phi - carry[0])[zero], others, wire)

        _multiplexed_rotation(qp.RY, ry_angles, [w for w in range(n) if w != wire], wire)
