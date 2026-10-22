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
