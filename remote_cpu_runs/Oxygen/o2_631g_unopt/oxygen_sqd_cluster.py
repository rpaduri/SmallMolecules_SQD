# MODIFIED — Cluster-only script. Runs on Kosambi via SLURM. No IBM/ffsim/internet needed.
# Loads sqd_input.pkl (produced by the laptop notebook), runs the heavy SQD diagonalization
# plus a full-molecule CCSD reference (same two-step structure as the master
# propane_sqd_combined pipeline's 02_sqd_cluster.py), saves sqd_results.pkl + sqd_energy_plot.png.
import os
import pickle
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")  # MODIFIED — compute node has no display; render straight to a file
import matplotlib.pyplot as plt
import pyscf
import pyscf.cc

from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch

# MODIFIED — use all cores SLURM gave this job for PySCF's internal (OpenMP) threading,
# matching 02_sqd_cluster.py's behavior/output line.
n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
pyscf.lib.num_threads(n_cpus)
print(f"Using {n_cpus} threads for PySCF")

# --- Load the handoff file from the laptop sampling run ---
with open("sqd_input.pkl", "rb") as f:
    data = pickle.load(f)

hcore = data["hcore"]
eri = data["eri"]
nuclear_repulsion_energy = data["nuclear_repulsion_energy"]
num_orbitals = data["num_orbitals"]
nelec = data["nelec"]
exact_energy = data["exact_energy"]  # laptop-side value, kept as a fallback below
bit_array = BitArray(data["meas_array"], data["meas_num_bits"])  # rebuild samples object

# --- SQD options (same values as the notebook) ---
energy_tol = 1e-3
occupancies_tol = 1e-3
max_iterations = 5
num_batches = 3
samples_per_batch = 300
symmetrize_spin = False  # O2 is open-shell: unequal alpha/beta electron counts
carryover_threshold = 1e-4
max_cycle = 200

sci_solver = partial(solve_sci_batch, spin_sq=2.0, max_cycle=max_cycle)  # triplet ground state, S=1

result_history = []

def callback(results):
    result_history.append(results)
    iteration = len(result_history)
    print(f"Iteration {iteration}", flush=True)
    for i, result in enumerate(results):
        energy = result.energy + nuclear_repulsion_energy
        dim = np.prod(result.sci_state.amplitudes.shape)
        print(f"\tSubsample {i}: Energy={energy}  Subspace dim={dim}", flush=True)

result = diagonalize_fermionic_hamiltonian(
    hcore,
    eri,
    bit_array,
    samples_per_batch=samples_per_batch,
    norb=num_orbitals,
    nelec=nelec,
    num_batches=num_batches,
    energy_tol=energy_tol,
    occupancies_tol=occupancies_tol,
    max_iterations=max_iterations,
    sci_solver=sci_solver,
    symmetrize_spin=symmetrize_spin,
    carryover_threshold=carryover_threshold,
    callback=callback,
    seed=12345,
)

# MODIFIED — recompute the reference energy on the cluster itself, same as
# 02_sqd_cluster.py's Part 2 does, instead of only trusting the laptop-supplied value.
# Falls back to the laptop value if the pkl predates the geometry/basis/spin keys.
if "geometry" in data and "basis" in data:
    mol = pyscf.gto.Mole()
    mol.build(atom=data["geometry"], basis=data["basis"], symmetry="C1", spin=data.get("spin", 0), charge=0)
    scf_ref = pyscf.scf.RHF(mol).run() if mol.spin == 0 else pyscf.scf.ROHF(mol).run()
    ccsd_solver = pyscf.cc.CCSD(scf_ref)
    ccsd_solver.run()
    exact_energy = ccsd_solver.e_tot
    print(f"Reference CCSD Energy: {exact_energy:.6f} Ha", flush=True)
else:
    print(f"No geometry/basis in pkl — using laptop-supplied exact_energy: {exact_energy:.6f} Ha", flush=True)

# --- Best result across all subsamples (uses the exact_energy finalized above) ---
best = None
for i, iter_results in enumerate(result_history):
    for j, res in enumerate(iter_results):
        energy = res.energy + nuclear_repulsion_energy
        error = abs(energy - exact_energy)
        if best is None or error < best["error"]:
            best = {
                "energy": energy,
                "error": error,
                "iteration": i,
                "subsample": j,
                "dim": int(np.prod(res.sci_state.amplitudes.shape)),
            }

print("\nBest SQD energy (Ha):", best["energy"], flush=True)
print("Best energy error (Ha):", best["error"], flush=True)
print("Iteration index:", best["iteration"], "| Subsample index:", best["subsample"], flush=True)

# --- Save results ---
min_e = [
    min(r, key=lambda x: x.energy).energy + nuclear_repulsion_energy
    for r in result_history
]
with open("sqd_results.pkl", "wb") as f:
    pickle.dump(
        {
            "best": best,
            "min_energy_per_iter": min_e,
            "orbital_occupancies": result.orbital_occupancies,
            "exact_energy": exact_energy,
        },
        f,
    )

# --- Plot to PNG (no GUI on a compute node) ---
x1 = range(len(result_history))
e_diff = [abs(e - exact_energy) for e in min_e]
y2 = np.sum(result.orbital_occupancies, axis=0)
x2 = range(len(y2))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(x1, e_diff, marker="o", label="energy error")
axs[0].set_yscale("log")
axs[0].axhline(y=0.001, color="#BF5700", linestyle="--", label="chemical accuracy")
axs[0].set_title("Energy Error vs SQD Iterations")
axs[0].set_xlabel("Iteration Index")
axs[0].set_ylabel("Energy Error (Ha)")
axs[0].legend()
axs[1].bar(x2, y2, width=0.8)
axs[1].set_title("Avg Occupancy per Spatial Orbital")
axs[1].set_xlabel("Orbital Index")
axs[1].set_ylabel("Avg Occupancy")
plt.tight_layout()
plt.savefig("sqd_energy_plot.png", dpi=150)

print("\nWrote sqd_results.pkl and sqd_energy_plot.png", flush=True)
