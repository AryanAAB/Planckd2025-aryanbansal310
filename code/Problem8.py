import numpy as np
import scipy.linalg as la
import itertools
import matplotlib.pyplot as plt
from math import log2

# ----------------------------
# Helper linear-algebra tools
# ----------------------------
def kron(*mats):
    out = np.array([[1.0]])
    for M in mats:
        out = np.kron(out, M)
    return out

# Pauli matrices and single-qubit gates
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
plus = np.array([1,1], dtype=complex)/np.sqrt(2)
zero = np.array([1,0], dtype=complex)

def basis_state(index, n_qubits):
    """Return computational basis state |index> in dimension 2^n"""
    vec = np.zeros(2**n_qubits, dtype=complex)
    vec[index] = 1.0
    return vec

def state_to_density(psi):
    return np.outer(psi, psi.conj())

def expm_H(Hmat, t=1.0):
    return la.expm(-1j * Hmat * t)

# ----------------------------
# Build many-qubit operators
# ----------------------------
def single_qubit_operator(op, target, n):
    """Return n-qubit operator with 'op' on qubit 'target' (0-index leftmost)."""
    mats = [I]*n
    mats[target] = op
    return kron(*mats)

def two_qubit_operator(op, target1, target2, n):
    mats = [I]*n
    mats[target1] = op[0]
    mats[target2] = op[1]
    return kron(*mats)

# For CZ between i and i+1, can use diag(1,1,1,-1) in 2-qubit subspace
CZ_2 = np.diag([1,1,1,-1]).astype(complex)
def CZ_on(i, j, n):
    # build operator by placing a 4x4 CZ on qubits (i,j) assuming i<j
    # approach: build using projectors or swap into place
    # simplest: iterate over basis states
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for s in range(dim):
        bits = [(s >> (n-1-k)) & 1 for k in range(n)]
        if bits[i] == 1 and bits[j] == 1:
            U[s,s] = -1
    return U

# ----------------------------
# Amplitude damping channel (single-qubit Kraus)
# ----------------------------
def amplitude_damping_kraus(p):
    # Standard amplitude damping with probability p
    K0 = np.array([[1, 0],[0, np.sqrt(1-p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)], [0,0]], dtype=complex)
    return [K0, K1]

def apply_local_channel_rho(rho, kraus_single, target, n):
    """Apply a single-qubit Kraus channel on 'target' qubit to full density matrix rho."""
    # build full Kraus by tensoring identity on other qubits
    new_rho = np.zeros_like(rho)
    for K in kraus_single:
        K_full = kron(*[K if idx==target else I for idx in range(n)])
        new_rho += K_full @ rho @ K_full.conj().T
    return new_rho

def apply_channel_all_qubits(rho, kraus_single, n):
    """Apply same single-qubit channel independently to every qubit."""
    # Apply per qubit sequentially (equivalent to independent application)
    out = rho.copy()
    for q in range(n):
        out = apply_local_channel_rho(out, kraus_single, q, n)
    return out

# ----------------------------
# Problem 7 (MaxCut on triangle) primitives
# ----------------------------
def H_problem_triangle(n=3):
    # C = 1/2 [ (1 - Z1 Z2) + (1 - Z2 Z3) + (1 - Z3 Z1) ]
    dim = 2**n
    H = np.zeros((dim,dim), dtype=complex)
    # build Z_i Z_j
    Z1Z2 = single_qubit_operator(Z, 0, n) @ single_qubit_operator(Z, 1, n)
    Z2Z3 = single_qubit_operator(Z, 1, n) @ single_qubit_operator(Z, 2, n)
    Z3Z1 = single_qubit_operator(Z, 2, n) @ single_qubit_operator(Z, 0, n)
    H = 0.5 * (3*np.eye(dim) - (Z1Z2 + Z2Z3 + Z3Z1))
    return H.real

def H_mixer(n=3):
    HM = sum(single_qubit_operator(X, q, n) for q in range(n))
    return HM

# expected cost given density matrix
def expected_cost_triangle(rho):
    H = H_problem_triangle()
    return np.real(np.trace(rho @ H))

# ----------------------------
# QAOA circuit builder (density matrix)
# ----------------------------
def qaoa_density(p, gammas, betas, n=3):
    # start |+>^n density
    psi0 = kron(*([plus]*n))
    rho = state_to_density(psi0)
    H_P = H_problem_triangle(n)
    H_M = H_mixer(n)
    for k in range(p):
        # problem unitary
        U_P = la.expm(-1j * gammas[k] * H_P)
        rho = U_P @ rho @ U_P.conj().T
        # mixer
        U_M = la.expm(-1j * betas[k] * H_M)
        rho = U_M @ rho @ U_M.conj().T
    return rho

# ----------------------------
# Trotterized adiabatic unitary (p steps)
# ----------------------------
def trotter_adiabatic_density(p, T, n=3):
    # linear interpolating schedule: H(s) = (1-s) H0 + s H_P
    # we'll approximate evolution by p slices: for k=1..p apply exp(-i dt ((1-sk) H_M + sk H_P))
    # For a rough comparable structure to QAOA, we implement alternating small steps of HM & HP
    H_P = H_problem_triangle(n)
    H_M = H_mixer(n)
    dt = T / p
    rho = state_to_density(kron(*([plus]*n)))
    for k in range(1, p+1):
        s = k / p
        # apply small step with (1-s)*H_M then s*H_P as two exponentials (Trotter)
        U1 = la.expm(-1j * (1-s) * dt * H_M)
        rho = U1 @ rho @ U1.conj().T
        U2 = la.expm(-1j * s * dt * H_P)
        rho = U2 @ rho @ U2.conj().T
    return rho

# ----------------------------
# (a) Simulation routine: QAOA and trotter under amplitude damping
# ----------------------------
def simulate_qaoa_vs_ad(i_p_list=[1,2], pAD_vals=None):
    if pAD_vals is None:
        pAD_vals = np.linspace(0.0, 0.6, 13)
    results = {}
    for p in i_p_list:
        # pick simple (near-optimal) angles for small 3-node problem
        # For production you'd numerically optimize; here use heuristic angles (0.8, 0.3...) or random search
        # We'll attempt a small grid search for gammas/betas to get decent values
        def optimize_angles(p):
            grid = np.linspace(0, np.pi, 9)
            best = (-1e9, None, None)
            for gammas in itertools.product(grid, repeat=p):
                for betas in itertools.product(grid, repeat=p):
                    rho = qaoa_density(p, gammas, betas, n=3)
                    val = expected_cost_triangle(rho)
                    if val > best[0]:
                        best = (val, gammas, betas)
            return best[1], best[2]
        gammas, betas = optimize_angles(p)
        # compute ideal density (no noise)
        rho_ideal = qaoa_density(p, gammas, betas)
        # Now sweep pAD
        qaoa_vals = []
        trotter_vals = []
        for pAD in pAD_vals:
            # apply amplitude damping channel between each layer: approx by applying channel after each unitary block.
            # We'll simulate noise by applying damping after each QAOA layer (problem+mixer)
            # Start fresh
            rho = state_to_density(kron(*([plus]*3)))
            H_P = H_problem_triangle(3)
            H_M = H_mixer(3)
            kraus = amplitude_damping_kraus(pAD)
            for k in range(p):
                U_P = la.expm(-1j * gammas[k] * H_P)
                rho = U_P @ rho @ U_P.conj().T
                rho = apply_channel_all_qubits(rho, kraus, 3)  # damping after problem
                U_M = la.expm(-1j * betas[k] * H_M)
                rho = U_M @ rho @ U_M.conj().T
                rho = apply_channel_all_qubits(rho, kraus, 3)  # damping after mixer
            qaoa_vals.append(expected_cost_triangle(rho))

            # trotterized adiabatic with total time T ~ sum of angles scaled; take T = p (arbitrary scale)
            T = p * 1.0
            rho_t = state_to_density(kron(*([plus]*3)))
            dt = T / p
            for k2 in range(1, p+1):
                s = k2 / p
                U1 = la.expm(-1j * (1-s) * dt * H_M)
                rho_t = U1 @ rho_t @ U1.conj().T
                rho_t = apply_channel_all_qubits(rho_t, kraus, 3)
                U2 = la.expm(-1j * s * dt * H_P)
                rho_t = U2 @ rho_t @ U2.conj().T
                rho_t = apply_channel_all_qubits(rho_t, kraus, 3)
            trotter_vals.append(expected_cost_triangle(rho_t))
        results[p] = (pAD_vals, qaoa_vals, trotter_vals, (gammas, betas))
    return results

# ----------------------------
# Main: run and plot
# ----------------------------
if __name__ == "__main__":
    # (a) QAOA vs adiabatic under amplitude damping
    print("Running (a): QAOA vs Trotter adiabatic under amplitude damping (this may take ~30-90s)...")
    p_list = [1,2]
    pAD_vals = np.linspace(0.0, 0.5, 11)
    res = simulate_qaoa_vs_ad(i_p_list=p_list, pAD_vals=pAD_vals)
    plt.figure(figsize=(6,4))
    for p in p_list:
        x, qvals, tvals, angles = res[p]
        plt.plot(x, qvals, marker='o', label=f"QAOA p={p}")
        plt.plot(x, tvals, marker='x', linestyle='--', label=f"Trotter p={p}")
    plt.xlabel("Amplitude damping pAD")
    plt.ylabel("Expected MaxCut value ⟨C⟩")
    plt.title("QAOA vs Trotter adiabatic under amplitude damping (3-node triangle)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("qaoa_vs_pAD.png", dpi=200)
    print("Saved qaoa_vs_pAD.png")