import sys
import math
import numpy as np

# ---------------------------
# Pauli matrices and helpers
# ---------------------------
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
PAULI_LABELS = ['X', 'Y', 'Z']

def kron2(A, B):
    return np.kron(A, B)

def state_vector_from_amplitudes(a, b, c, d):
    v = np.array([a, b, c, d], dtype=complex)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        raise ValueError("Zero state provided")
    return v / nrm

def pure_density_from_state(vec):
    v = vec.reshape((4,1))
    return v @ v.conj().T

# ---------------------------
# Measurement settings
# ---------------------------
def all_measurement_settings():
    """Return a list of measurement settings (pauli1, pauli2). Include local (p,I) and (I,p)."""
    settings = []
    # local single-qubit settings for qubit 1: X⊗I, Y⊗I, Z⊗I
    for p in PAULI_LABELS:
        settings.append((p, 'I'))
    # local single-qubit settings for qubit 2: I⊗X, I⊗Y, I⊗Z
    for p in PAULI_LABELS:
        settings.append(('I', p))
    # two-qubit correlations: X⊗X ... Z⊗Z
    for p1 in PAULI_LABELS:
        for p2 in PAULI_LABELS:
            settings.append((p1, p2))
    # total should be 3 + 3 + 9 = 15
    return settings

def eigenstates_of_single_pauli(p):
    """Return eigenvalues [1,-1] and the corresponding projectors for single-qubit Pauli p."""
    if p == 'X':
        plus = np.array([1, 1], dtype=complex) / math.sqrt(2)
        minus = np.array([1, -1], dtype=complex) / math.sqrt(2)
        return [1, -1], [np.outer(plus, plus.conj()), np.outer(minus, minus.conj())]
    if p == 'Y':
        plus = np.array([1, 1j], dtype=complex) / math.sqrt(2)
        minus = np.array([1, -1j], dtype=complex) / math.sqrt(2)
        return [1, -1], [np.outer(plus, plus.conj()), np.outer(minus, minus.conj())]
    if p == 'Z':
        zero = np.array([1, 0], dtype=complex)
        one = np.array([0, 1], dtype=complex)
        return [1, -1], [np.outer(zero, zero.conj()), np.outer(one, one.conj())]
    if p == 'I':
        # For identity, treat as single "outcome" projector equal to I (but we'll never measure (I,I) alone)
        return [1], [I]
    raise ValueError("Unknown Pauli " + str(p))

# ---------------------------
# Simulation of measurements
# ---------------------------
def simulate_measurements(state_vec, shots_total=500, seed=None):
    """
    Simulate measurement outcomes (counts) for each measurement setting.
    Returns:
      - measured_counts: dict mapping setting -> list of counts for each outcome (ordered)
      - outcome_projectors: dict mapping setting -> list of 2-qubit projectors (numpy arrays)
      - shots_per_setting: dict mapping setting -> integer number of shots
    """
    rng = np.random.default_rng(seed)
    rho = pure_density_from_state(state_vec)

    settings = all_measurement_settings()
    num_settings = len(settings)
    base_shots = shots_total // num_settings
    remainder = shots_total % num_settings
    shots_per_setting = {}
    measured_counts = {}
    outcome_projectors = {}

    for idx, setting in enumerate(settings):
        nshots = base_shots + (1 if idx < remainder else 0)
        shots_per_setting[setting] = nshots

        p1, p2 = setting
        eigvals1, projs1 = eigenstates_of_single_pauli(p1)
        eigvals2, projs2 = eigenstates_of_single_pauli(p2)

        # Build 2-qubit projectors and list of eigenvalue outcomes (product/or local)
        projs_2q = []
        outcomes = []  # keep track of eigenvalue labels for possible debug
        for i1, P1 in zip(eigvals1, projs1):
            for i2, P2 in zip(eigvals2, projs2):
                projs_2q.append(kron2(P1, P2))
                outcomes.append((i1, i2))
        # compute probabilities
        probs = np.array([np.real_if_close(np.trace(rho @ P)) for P in projs_2q], dtype=float)
        probs = np.maximum(probs, 0.0)
        s = probs.sum()
        if s <= 0:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs = probs / s
        if nshots > 0:
            samples = rng.choice(len(probs), size=nshots, p=probs)
            counts = np.bincount(samples, minlength=len(probs)).astype(int).tolist()
        else:
            counts = [0] * len(projs_2q)

        measured_counts[setting] = counts
        outcome_projectors[setting] = projs_2q

    return measured_counts, outcome_projectors, shots_per_setting

# ---------------------------
# Linear inversion reconstruction
# ---------------------------
def reconstruct_density_from_pauli_expectations_from_counts(measured_counts, outcome_projectors, shots_per_setting):
    """
    From counts for each setting and the corresponding projectors, estimate expectation values
    of the 16 Pauli products and perform linear inversion to get rho_est.
    We'll compute expectation values <σ_i ⊗ σ_j> using the measured counts where possible.
    """
    # We'll compute expectation values for the 15 nontrivial Paulis by mapping outcomes to ±1 products.
    # Map Pauli index: 0:I, 1:X, 2:Y, 3:Z
    sigs = [I, X, Y, Z]
    # Initialize coefficients c_{ij}
    c = np.zeros((4,4), dtype=float)
    c[0,0] = 1.0

    # compute single- and two-qubit expectation values from counts
    # For each setting, derive expectation(s) from counts:
    # - If setting is (p, 'I'), it gives <p ⊗ I>
    # - If setting is ('I', p), it gives <I ⊗ p>
    # - If setting is (p1, p2) with both non-I, it gives <p1 ⊗ p2>
    for setting, counts in measured_counts.items():
        p1, p2 = setting
        projs = outcome_projectors[setting]
        total = sum(counts)
        if total == 0:
            est = 0.0
        else:
            # compute empirical expectation; need mapping from outcome index to +1/-1 product
            # Build eigenvalue lists for p1 and p2
            eig1, projs1 = eigenstates_of_single_pauli(p1)
            eig2, projs2 = eigenstates_of_single_pauli(p2)
            # construct outcomes list aligning with projs order
            vals = []
            for i1 in eig1:
                for i2 in eig2:
                    # For local I we used single eigenvalue 1, interpret product as i1*i2
                    vals.append(i1 * i2)
            vals = np.array(vals, dtype=float)
            counts_arr = np.array(counts, dtype=float)
            est = float(np.sum(counts_arr * vals) / total)
        # assign to c matrix
        if p1 == 'I' and p2 == 'I':
            continue
        if p1 == 'I':
            # <I ⊗ p2> => c[0, j]
            j = PAULI_LABELS.index(p2) + 1
            c[0, j] = est
        elif p2 == 'I':
            i = PAULI_LABELS.index(p1) + 1
            c[i, 0] = est
        else:
            i = PAULI_LABELS.index(p1) + 1
            j = PAULI_LABELS.index(p2) + 1
            c[i, j] = est

    # Now build rho_est = 1/4 sum_{i,j} c_{ij} σ_i ⊗ σ_j
    rho_est = np.zeros((4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            rho_est += c[i, j] * np.kron(sigs[i], sigs[j])
    rho_est = rho_est / 4.0
    rho_est = (rho_est + rho_est.conj().T) / 2.0
    return rho_est

# ---------------------------
# Pure-state MLE via R·psi iteration
# ---------------------------
def mle_pure_state_from_counts(initial_psi, measured_counts, outcome_projectors, max_iters=200, tol=1e-8):
    """
    Iterative ML update adapted to pure states using operator R:
    R = sum_{k,o} (n_{k,o} / p_{k,o}) P_{k,o}
    psi_next = normalize(R psi)
    We clamp p_{k,o} below to eps to avoid divide-by-zero.
    """
    psi = initial_psi.astype(complex)
    psi = psi / np.linalg.norm(psi)
    eps = 1e-9
    last_ll = -np.inf

    for it in range(max_iters):
        # compute p_{k,o} for current psi
        probs = {}
        for setting, projs in outcome_projectors.items():
            p_list = []
            for P in projs:
                val = np.real_if_close(np.vdot(psi, P @ psi))
                # numerical clamp
                if val < 0:
                    val = 0.0
                p_list.append(float(val))
            probs[setting] = p_list

        # compute log-likelihood
        ll = 0.0
        for setting, counts in measured_counts.items():
            ps = probs[setting]
            for n, p in zip(counts, ps):
                if n <= 0:
                    continue
                ll += n * math.log(max(p, eps))
        # small improvement check
        if it > 0 and abs(ll - last_ll) < tol:
            break
        last_ll = ll

        # build R operator
        R = np.zeros((4,4), dtype=complex)
        for setting, counts in measured_counts.items():
            ps = probs[setting]
            projs = outcome_projectors[setting]
            for n, p, P in zip(counts, ps, projs):
                if n == 0:
                    continue
                denom = max(p, eps)
                R += (n / denom) * P

        # apply R to psi and normalize
        psi_new = R @ psi
        norm_new = np.linalg.norm(psi_new)
        if norm_new == 0:
            # numerical problem; break
            break
        psi_new = psi_new / norm_new

        # if change small, stop
        if np.linalg.norm(psi_new - psi) < 1e-9:
            psi = psi_new
            break
        psi = psi_new

    # fix global phase: make first non-zero component real positive
    psi = psi / np.linalg.norm(psi)
    tol_phase = 1e-8
    for k in range(len(psi)):
        if abs(psi[k]) > tol_phase:
            phase = np.angle(psi[k])
            psi = psi * np.exp(-1j * phase)
            break

    return psi

# ---------------------------
# Utilities: projection to pure via top eigenvector
# ---------------------------
def project_to_pure_state(rho):
    vals, vecs = np.linalg.eigh(rho)
    idx = np.argmax(vals)
    psi = vecs[:, idx]
    psi = psi / np.linalg.norm(psi)
    # fix global phase
    tol = 1e-8
    for k in range(len(psi)):
        if abs(psi[k]) > tol:
            phase = np.angle(psi[k])
            psi = psi * np.exp(-1j * phase)
            break
    return psi

# ---------------------------
# Main estimation routine
# ---------------------------
def estimate_state_from_amplitudes(a, b, c, d, shots_total=500, seed=None):
    psi_true = state_vector_from_amplitudes(a, b, c, d)
    measured_counts, outcome_projectors, shots_per_setting = simulate_measurements(psi_true, shots_total=shots_total, seed=seed)

    # Linear inversion reconstruction
    rho_lin = reconstruct_density_from_pauli_expectations_from_counts(measured_counts, outcome_projectors, shots_per_setting)
    psi_init = project_to_pure_state(rho_lin)

    # MLE pure-state refinement
    psi_mle = mle_pure_state_from_counts(psi_init, measured_counts, outcome_projectors, max_iters=500, tol=1e-9)

    return psi_true, psi_init, psi_mle, measured_counts

# ---------------------------
# Formatting and main()
# ---------------------------
def parse_amplitude(s):
    # allow floats or complex strings like "1+0j"
    return complex(s)

def fmt_complex(z):
    # print a b c d as requested: use concise representation, small imag omitted
    re = float(np.real_if_close(z.real))
    im = float(np.real_if_close(z.imag))
    if abs(im) < 1e-10:
        return f"{re:.8f}"
    if abs(re) < 1e-10:
        return f"{im:.8f}j"
    return f"({re:.8f}{im:+.8f}j)"

def main():
    if len(sys.argv) < 2:
        print("Usage: python tomography_mle.py input_file")
        sys.exit(1)
    fname = sys.argv[1]
    with open(fname, 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith('#')]
    ntests = int(lines[0].split()[0])
    idx = 1
    total_fid = 0.0

    for t in range(ntests):
        parts = lines[idx].split()
        idx += 1
        if len(parts) < 4:
            raise ValueError("each test must have 4 amplitudes")
        a, b, c, d = [parse_amplitude(x) for x in parts[:4]]
        psi_true, psi_init, psi_mle, counts = estimate_state_from_amplitudes(a, b, c, d, shots_total=500, seed=1234 + t)

        fidelity = abs(np.vdot(psi_true, psi_mle))**2
        total_fid += fidelity

        # Output estimated amplitudes (a b c d) as requested
        out_line = " ".join(fmt_complex(z) for z in psi_mle)
        print(out_line)
        # Also print fidelities and a short diagnostic line
        print(f"Fidelity: {fidelity:.6f}")
        print()

    avg_fid = total_fid / ntests
    print(f"Average fidelity across all tests = {avg_fid:.6f}")

if __name__ == "__main__":
    main()
