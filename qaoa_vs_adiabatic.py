#!/usr/bin/env python3
# QAOA vs Adiabatic on 3-node MaxCut (Triangle)
# Generates plots comparing QAOA, trotterized adiabatic, and continuous adiabatic evolution.

import numpy as np, math, matplotlib.pyplot as plt, json, csv
from numpy.linalg import eigh, norm

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0,1],[1,0]], dtype=np.complex128)
Z = np.array([[1,0],[0,-1]], dtype=np.complex128)

def kron(*ops):
    out = np.array([[1.0+0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def kron_vec(*vecs):
    out = vecs[0]
    for v in vecs[1:]:
        out = np.kron(out, v)
    return out

def Rx(theta):
    c = math.cos(theta/2.0)
    s = -1j*math.sin(theta/2.0)
    return np.array([[c, s],[s, c]], dtype=np.complex128)

def Xi(i,n): return kron(*[X if q==i else I2 for q in range(n)])
def ZiZj(i,j,n): return kron(*[Z if (q==i or q==j) else I2 for q in range(n)])

# Triangle (n=3)
n = 3
dim = 2**n
ZZ12 = ZiZj(0,1,n); ZZ23 = ZiZj(1,2,n); ZZ31 = ZiZj(2,0,n)
H_P = 0.5*((np.eye(dim)-ZZ12)+(np.eye(dim)-ZZ23)+(np.eye(dim)-ZZ31))
H_M = Xi(0,n)+Xi(1,n)+Xi(2,n)
H_target = -H_P

evals_HP, evecs_HP = eigh(H_P)
def U_P(gamma):
    phase = np.exp(-1j * gamma * evals_HP)
    return (evecs_HP * phase) @ evecs_HP.conj().T

def U_M(beta):
    R = Rx(2.0*beta)
    return kron(R,R,R)

plus = (1/np.sqrt(2))*np.array([1,1],dtype=np.complex128)
psi_plus = kron_vec(plus,plus,plus)

def exp_val(state,H): return float(np.real(np.vdot(state,H@state)))

def qaoa_state(gammas,betas):
    psi = psi_plus.copy()
    for g,b in zip(gammas,betas):
        psi = U_P(g) @ psi
        psi = U_M(b) @ psi
    return psi

def success_prob(state):
    probs = np.abs(state)**2
    succ = 0.0
    for idx in range(dim):
        wt = bin(idx).count('1')
        if wt==1 or wt==2: succ += probs[idx]
    return float(succ)

rng = np.random.default_rng(1)
def optimize_qaoa(p,budget=8000):
    best = {"val":-1e9}
    for _ in range(budget):
        gammas = rng.uniform(0,math.pi,size=p)
        betas = rng.uniform(0,math.pi/2,size=p)
        psi = qaoa_state(gammas,betas)
        val = exp_val(psi,H_P)
        if val>best.get("val",-1e9):
            best={"val":val,"gammas":gammas.copy(),"betas":betas.copy()}
    return best

def H_of_s(s): return (1-s)*H_M + s*H_target
def evolve_continuous(T,steps=4000):
    dt=T/steps; psi=psi_plus.copy()
    for i in range(steps):
        t=(i+0.5)*dt; s=t/T
        Hs=H_of_s(s); w,V=eigh(Hs)
        U=(V*np.exp(-1j*w*dt))@V.conj().T
        psi=U@psi; psi/=norm(psi)
    return psi

def trotter_adiabatic(p,T):
    dt=T/p; psi=psi_plus.copy()
    for k in range(1,p+1):
        s=k/p
        psi=U_M((1-s)*dt)@psi
        psi=U_P(-s*dt)@psi
    return psi

# Run
results_qaoa={p:optimize_qaoa(p,budget=(6000 if p==1 else 12000)) for p in [1,2,3]}
qaoa_p=list(results_qaoa.keys())
qaoa_vals=[]; qaoa_succ=[]
for p in qaoa_p:
    psi=qaoa_state(results_qaoa[p]["gammas"],results_qaoa[p]["betas"])
    qaoa_vals.append(exp_val(psi,H_P))
    qaoa_succ.append(success_prob(psi))

T_cont=20.0
psi_ad=evolve_continuous(T_cont,steps=4000)
C_ad=exp_val(psi_ad,H_P); Psucc_ad=success_prob(psi_ad)
p_list=[1,2,3,4,5,6]
trott_vals=[]; trott_succ=[]
for p in p_list:
    psi_t=trotter_adiabatic(p,T_cont)
    trott_vals.append(exp_val(psi_t,H_P))
    trott_succ.append(success_prob(psi_t))

plt.figure()
plt.plot(qaoa_p,qaoa_vals,'o-',label='QAOA (optimized)')
plt.plot(p_list,trott_vals,'s-',label='Trotterized adiabatic')
plt.axhline(C_ad,linestyle='--',label=f'Continuous adiabatic (T={T_cont})')
plt.xlabel('Depth p'); plt.ylabel('<C>'); plt.title('Triangle MaxCut: <C> vs p')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig('qaoa_vs_adiabatic_cost.png',dpi=160)

plt.figure()
plt.plot(qaoa_p,qaoa_succ,'o-',label='QAOA (optimized)')
plt.plot(p_list,trott_succ,'s-',label='Trotterized adiabatic')
plt.axhline(Psucc_ad,linestyle='--',label='Continuous adiabatic')
plt.xlabel('Depth p'); plt.ylabel('Success probability')
plt.title('Triangle MaxCut: success vs p')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig('qaoa_vs_adiabatic_success.png',dpi=160)

json.dump({p:{"C_best":results_qaoa[p]["val"],"gammas":list(map(float,results_qaoa[p]["gammas"])),"betas":list(map(float,results_qaoa[p]["betas"]))} for p in qaoa_p},open('qaoa_triangle_params.json','w'),indent=2)
csv.writer(open('qaoa_triangle_summary.csv','w',newline='')).writerows([["<C>_continuous",C_ad],["Psucc_continuous",Psucc_ad]])
print("All done; plots and data written locally.")
