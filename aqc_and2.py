import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh, norm
import csv

# Basis: |00>, |01>, |10>, |11>
I4 = np.eye(4, dtype=np.complex128)
one = np.ones((4,4), dtype=np.complex128)

psi0 = np.ones(4, dtype=np.complex128) / 2.0  # |psi0> = (1/2) sum_x |x>
P_psi0 = np.outer(psi0, psi0.conj())
e3 = np.zeros(4, dtype=np.complex128); e3[3] = 1.0  # |11>
P_11 = np.outer(e3, e3.conj())

H0 = I4 - P_psi0
HP = I4 - P_11

def H_of_s(s: float) -> np.ndarray:
    J = one
    D = np.diag([1.0, 1.0, 1.0, 1.0 - s])
    return D - ((1.0 - s) / 4.0) * J

def spectrum_and_gap(num_points=1001):
    s_grid = np.linspace(0.0, 1.0, num_points)
    eigvals = np.zeros((num_points, 4), dtype=np.float64)
    gaps = np.zeros(num_points, dtype=np.float64)
    for i, s in enumerate(s_grid):
        Hs = H_of_s(s)
        w, _ = eigh(Hs)
        eigvals[i] = np.sort(w.real)
        gaps[i] = eigvals[i,1] - eigvals[i,0]
    idx_min = np.argmin(gaps)
    s_star = s_grid[idx_min]
    delta_min = gaps[idx_min]
    return s_grid, eigvals, gaps, s_star, delta_min

def adiabatic_coupling_max(s_grid):
    dHds = P_psi0 - P_11
    vals = []
    for s in s_grid:
        Hs = H_of_s(s)
        w, V = eigh(Hs)
        E0_vec = V[:,0]; E1_vec = V[:,1]
        m = np.vdot(E1_vec, dHds @ E0_vec)
        vals.append(abs(m))
    return float(np.max(vals)), np.array(vals)

def evolve_state(T, steps=4000):
    psi = psi0.copy()
    times = np.linspace(0.0, T, steps+1)
    fidelities = np.zeros_like(times, dtype=np.float64)
    for i, t in enumerate(times):
        s = t / T if T > 0 else 0.0
        Hs = H_of_s(s)
        w, V = eigh(Hs)
        E0_vec = V[:,0]
        fidelities[i] = abs(np.vdot(E0_vec, psi))**2
        if i < steps:
            dt = T/steps
            phase = np.exp(-1j * w * dt)
            U = (V * phase) @ V.conj().T
            psi = U @ psi
            psi = psi / norm(psi)
    return times, fidelities, psi

def main():
    import os
    out_dir = "."
    # Spectra and gaps
    s_grid, eigvals, gaps, s_star, delta_min = spectrum_and_gap()
    # Plots
    plt.figure()
    for k in range(4):
        plt.plot(s_grid, eigvals[:,k])
    plt.xlabel("s"); plt.ylabel("Eigenvalues of H(s)"); plt.title("Instantaneous Spectrum of H(s)"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "aqc_eigs.png"), dpi=160); plt.close()
    plt.figure()
    plt.plot(s_grid, gaps); plt.axvline(s_star, linestyle='--')
    plt.xlabel("s"); plt.ylabel("Spectral gap Δ(s) = E1 - E0"); plt.title(f"Spectral Gap; Δ_min ≈ {delta_min:.6f} at s* ≈ {s_star:.4f}")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "aqc_gap.png"), dpi=160); plt.close()
    # Adiabatic condition
    coupling_max, _ = adiabatic_coupling_max(s_grid)
    T_est = 10.0 * coupling_max / (delta_min**2)  # safety factor 10
    # Evolution
    T_eval = float(max(T_est, 5.0))
    times, fidelities, psi_final = evolve_state(T_eval, steps=4000)
    plt.figure()
    plt.plot(times / T_eval, fidelities)
    plt.xlabel("s = t/T"); plt.ylabel("|<E0(s)|ψ(t)>|^2"); plt.title(f"Instantaneous Ground-State Fidelity (T ≈ {T_eval:.3f})"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "aqc_fidelity_Teval.png"), dpi=160); plt.close()
    # Non-adiabatic scaling
    T_values = np.geomspace(0.05, max(10.0, T_est), 40)
    P_success = []
    for T in T_values:
        _, _, psi_T = evolve_state(float(T), steps=2000)
        P_success.append(abs(np.vdot(np.array([0,0,0,1], dtype=np.complex128), psi_T))**2)
    P_success = np.array(P_success)
    plt.figure()
    plt.loglog(T_values, P_success, marker='o', linewidth=1)
    plt.xlabel("T"); plt.ylabel("Success probability P_succ = |<11|ψ(T)>|^2"); plt.title("Non-adiabatic regime: success probability vs T"); plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "aqc_success_vs_T.png"), dpi=160); plt.close()
    # Save summary
    with open(os.path.join(out_dir, "aqc_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s_star_min_gap", s_star])
        w.writerow(["delta_min", delta_min])
        w.writerow(["max_coupling", coupling_max])
        w.writerow(["T_est_adiabatic_heuristic", T_est])
        w.writerow(["T_eval_used_for_fidelity_plot", T_eval])
        P_success_eval = abs(np.vdot(np.array([0,0,0,1], dtype=np.complex128), psi_final))**2
        w.writerow(["P_success_at_T_eval", P_success_eval])
    # Save sample H(s)
    s_samples = [0.0, 0.25, 0.5, 0.75, 1.0]
    with open(os.path.join(out_dir, "Hs_samples.txt"), "w") as f:
        for s in s_samples:
            f.write(f"s = {s}\n")
            f.write(np.array2string(H_of_s(s).real, precision=6, suppress_small=True))
            f.write("\n\n")
    # Print key numbers
    print("Δ_min ~", delta_min, "at s* ~", s_star)
    print("max coupling ~", coupling_max)
    print("T_est (heuristic) ~", T_est)
    print("Done.")

if __name__ == "__main__":
    main()
