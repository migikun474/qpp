# Local re-implementation of the hidden grader rules (the platform's own harness is locked).
import sys, time


def run(seed, num_samples):
    costs, errors, aux = [], 0.0, 0
    for size in TEST_SIZES:
        total = 0
        for k in range(num_samples):
            u = haar_unitary(size, seed * 1000 + 17 * size + k)
            tape = compile_circuit(submission_circ, u)
            assert len(tape.operations) <= MAX_GATES
            wires, a = wire_usage(tape, size)
            assert wires <= MAX_TOTAL_WIRES
            aux = max(aux, a)
            errors = max(errors, float(np.abs(tape_matrix(tape, size) - u).max()))
            total += tape_cost(tape, size)
        costs.append(total / num_samples)
    base = [baseline_cost(s) for s in TEST_SIZES]
    saving = [100 * (1 - c / b) for c, b in zip(costs, base)]
    ok = errors <= ATOL and all(c <= b for c, b in zip(costs, base)) and all(
        saving[i + 1] >= saving[i] - 1e-9 for i in range(len(saving) - 1))
    grade = "None"
    for name, need in GRADES.items():
        if ok and saving[-1] >= need:
            grade = name
    print(json.dumps({"grade": grade, "saving_at_5_qubits": f"{saving[-1]:.1f}%",
                      "saving_by_size": [f"{s:.2f}%" for s in saving], "cost": costs,
                      "baseline": base, "max_error": f"{errors:.1e}", "aux_wires": aux}))


if __name__ == "__main__":
    for case in ([2026, 2], [7, 2], [1, 5]):
        t = time.time(); run(*case); print(f"  ({time.time() - t:.1f}s)")
