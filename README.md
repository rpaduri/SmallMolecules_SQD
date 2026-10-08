# Basis set and geometry effects in sample-based quantum diagonalization

Code, data and figures accompanying the paper *Basis set and geometry effects in sample-based quantum diagonalization*.

## Overview

This repository studies how accurately IBM's **Sample-based Quantum Diagonalization (SQD)** algorithm recovers molecular ground-state energies on real quantum hardware, and how that accuracy depends on two choices made *before* any quantum circuit is run:

- **Basis set** — STO-3G, 6-31G and cc-pVDZ
- **Molecular geometry** — unoptimized vs. optimized (Gaussian, HF and B3LYP)

Molecules studied: N₂, O₂, H₂O, CO₂, propane (C₃H₈), cyclopropane (C₃H₆) and benzene (C₆H₆).

### Pipeline

1. **Geometry optimization** (`geometry_optimization/`) — Gaussian input/output files (`.gjf`, `.log`) for HF and B3LYP optimizations at each basis set.
2. **Hamiltonian construction** — PySCF builds the active-space Hamiltonian (`hcore`, `eri`, nuclear repulsion) for each molecule/basis/geometry.
3. **Quantum sampling** (`notebooks/`) — a LUCJ ansatz is built with `ffsim`, transpiled to an IBM Heron backend (≥133 qubits, e.g. `ibm_torino`, `ibm_brisbane`) and sampled through Qiskit Runtime.
4. **Classical post-processing** — `qiskit-addon-sqd` performs configuration recovery on the noisy bitstrings, projects the Hamiltonian into the sampled subspace and diagonalizes it. For the large active spaces (propane, cyclopropane, benzene, CO₂, cc-pVDZ O₂) this step is run on a SLURM CPU cluster from a saved sampling result (`sqd_input.pkl`); see `remote_cpu_runs/`.
5. **Benchmarking** — SQD energies are compared against classical HF/SCF and CCSD references.

## Repository layout

```
.
├── README.md
├── requirements.txt              # pinned Python dependencies
├── ibm_account_config_script.py  # saves IBM credentials locally from .env
├── geometry_optimization/        # Gaussian .gjf inputs and .log outputs (N2, O2, propane)
├── notebooks/                    # hardware-sampling + SQD notebooks (N2, O2)
│   ├── nitrogen/                 #   <molecule>_<opt|unopt>_<basis>.ipynb
│   └── oxygen/
├── remote_cpu_runs/              # heavy classical SQD runs on the cluster
│   └── <Molecule>/<molecule>_<basis>_<opt|unopt>/
│       ├── sqd_input.pkl         #   hardware samples + Hamiltonian (input to the cluster job)
│       ├── *_cluster.py / submit_*.sh   # SQD script and SLURM submission script
│       └── sqd_<jobid>.out/.err  #   job logs
```

---

## 1. Environment setup

Tested with **Python 3.12** on macOS. Python 3.10–3.12 should work; use the same interpreter for the notebooks and scripts.

```bash
# clone and enter the repo
git clone <REPO_URL> SQD
cd SQD

# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install everything (Qiskit, qiskit-addon-sqd, ffsim, PySCF, qiskit-ibm-runtime, Jupyter, ...)
pip install --upgrade pip
pip install -r requirements.txt
```

Key pinned packages (full list in `requirements.txt`):

| Purpose | Package |
|---|---|
| Quantum SDK | `qiskit==2.5.0`, `qiskit-aer==0.17.2` |
| Hardware access | `qiskit-ibm-runtime==0.43.1` |
| SQD algorithm | `qiskit-addon-sqd==0.12.0` |
| LUCJ ansatz / fermionic simulation | `ffsim==0.0.80` |
| Quantum chemistry | `pyscf==2.13.1`, `qiskit-nature==0.8.0`, `cclib==1.8.1` |
| Credentials | `python-dotenv==1.2.1` |
| Notebooks / plotting | `jupyterlab`, `ipykernel`, `matplotlib`, `pandas`, `seaborn` |

Notes:

- `requirements.txt` was frozen on macOS. `appnope` is macOS-only; if `pip install` fails on Linux/Windows, delete that line and retry.
- PySCF builds/wheels can be slow on some platforms; on Apple Silicon and modern Linux, prebuilt wheels are available.
- Register the venv as a Jupyter kernel so notebooks use it:

  ```bash
  python -m ipykernel install --user --name sqd-env --display-name "Python (sqd-env)"
  jupyter lab
  ```

Sanity check:

```bash
python -c "import qiskit, qiskit_ibm_runtime, qiskit_addon_sqd, ffsim, pyscf; print('qiskit', qiskit.__version__, '| pyscf', pyscf.__version__)"
```

Gaussian (used only for the geometry optimizations) is **not** required to reproduce SQD runs; optimized geometries are in `geometry_optimization/*.log`.

---

## 2. IBM Quantum Platform setup

Running the notebooks on real hardware needs an IBM Quantum Platform account, an **instance** (identified by a **CRN**), and an **API key**. Everything except the hardware-sampling notebooks (SQD post-processing and the cluster runs from the supplied `sqd_input.pkl`) works without an account.

### 2.1 Create an account

1. Go to <https://quantum.cloud.ibm.com> and sign up / log in with an IBMid.
2. Accept the terms and complete the account setup.

### 2.2 Create an instance and get its CRN

1. On the IBM Quantum Platform dashboard, click **Create instance** (or **Instances → Create instance**).
2. Choose a plan:
   - **Open Plan** — free, limited QPU time per month on Heron-class devices (sufficient for small experiments).
   - **Pay-as-you-go / Flex / Premium** — for larger runs.
3. Select the QPU(s) you want the instance to access (the notebooks need a device with ≥133 qubits, such as `ibm_torino`), give the instance a name, and confirm.
4. Open **Instances** in the left menu, click your new instance, and copy its **CRN** (Cloud Resource Name). It looks like:

   ```
   crn:v1:bluemix:public:quantum-computing:us-east:a/<account-id>:<instance-id>::
   ```

### 2.3 Create an API key

1. On the dashboard, find the **API key** panel (top right, or **Manage → API keys**) and click **Create +**.
2. Give the key a name, click **Create**, then **Copy** it or **Download** the JSON file. **The key is shown only once** — if you lose it, create a new one.
3. Never commit the key or the JSON file to version control.

### 2.4 Configure credentials locally (`.env`)

Create a file named `.env` in the **repository root** (it is already git-ignored) with exactly these variable names:

```dotenv
IBM_QUANTUM_API_KEY=<your-api-key>
IBM_QUANTUM_CRN=<your-instance-crn>
```

Example (fake values):

```dotenv
IBM_QUANTUM_API_KEY=aBcD1234eFgH5678iJkL9012mNoP3456qRsT7890uVwX
IBM_QUANTUM_CRN=crn:v1:bluemix:public:quantum-computing:us-east:a/0123456789abcdef0123456789abcdef:01234567-89ab-cdef-0123-456789abcdef::
```

Format rules:

- One `KEY=value` per line; **no spaces around `=`, no quotes**, no trailing comments on the same line.
- Variable names are case-sensitive and must be exactly `IBM_QUANTUM_API_KEY` and `IBM_QUANTUM_CRN` — these are the names read by `ibm_account_config_script.py` and the notebooks.

### 2.5 Save the account and verify

```bash
python ibm_account_config_script.py
```

This reads `.env`, calls `QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=..., instance=..., overwrite=True)` (stored in `~/.qiskit/qiskit-ibm.json`), reloads it, and prints the instances and backends you can access. Expected output:

```
Account saved successfully.

Available instances:
[...]

Available backends:
- ibm_torino
- ...
```

If it raises `IBM_QUANTUM_API_KEY not found in .env` or `IBM_QUANTUM_CRN not found in .env`, the `.env` file is missing, misnamed, or not in the directory you ran the script from (or one of its parents).

The notebooks then connect with the saved account:

```python
import os
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv()
service = QiskitRuntimeService(instance=os.environ["IBM_QUANTUM_CRN"])
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=133)
```

---

## 3. Running the experiments

### Hardware sampling + SQD (notebooks)

```bash
source .venv/bin/activate
jupyter lab
```

Open a notebook under `notebooks/nitrogen/` or `notebooks/oxygen/` (named `<molecule>_<opt|unopt>_<basis>.ipynb`), select the `sqd-env` kernel, and run all cells. Each notebook: builds the molecular Hamiltonian with PySCF, constructs the LUCJ circuit with ffsim, transpiles it, submits it to the backend, and runs SQD on the returned samples. **Each run consumes QPU time on your instance.**

**Geometry file paths (O₂ optimized notebooks).** The optimized-geometry O₂ notebooks read their coordinates from a Gaussian `.log` file. Before running, set the `filename = "..."` line in the geometry cell to the corresponding file in `geometry_optimization/Oxygen/` (see table below), as a path relative to the notebook, for example `../../geometry_optimization/Oxygen/O2-HF-STO-3G.log`.

| Notebook | Log file in `geometry_optimization/Oxygen/` |
|---|---|
| `oxygen_opt_sto-3g.ipynb` | `O2-HF-STO-3G.log` |
| `oxygen_opt_6-31g.ipynb` | `O2-HF-6-31G(D).log` |
| `oxygen_opt_ccpvdz.ipynb` | `O2-HF-CC-PVDZ.log` |

The N₂ optimized notebooks define their coordinates inline and need no path setup.

### Classical SQD from saved samples (cluster / large active spaces)

For large active spaces the diagonalization is run offline from `sqd_input.pkl` (hardware samples + Hamiltonian; no IBM access needed):

```bash
cd remote_cpu_runs/Propane/propane_sto3g_opt
python propane_opt_sto3g_sqd_cluster.py        # local, or:
sbatch submit_propane_opt_sto3g_sqd.sh         # SLURM (edit partition / module / venv path for your cluster)
```

The scripts write `sqd_results.pkl` and an energy plot next to the input. Budget memory and wall time generously: propane subspaces reach millions of determinants per iteration.

---

## 4. Compute resources: local machines and the Kosambi cluster

The workload was split across three kinds of compute according to problem size.

| Stage | Where | Systems |
|---|---|---|
| QPU sampling (LUCJ circuit → bitstrings) | IBM Quantum hardware, submitted from a local machine | all systems |
| Full pipeline (Hamiltonian, sampling, SQD post-processing) | **Local machines** (laptops, Jupyter notebooks) | N₂ and O₂ (`notebooks/`) |
| Heavy classical SQD post-processing | **Kosambi** SLURM cluster | CO₂, propane, cyclopropane, benzene, plus heavier O₂ configurations (`remote_cpu_runs/`) |
| Classical reference energies (HF/CCSD/CCSD(T)) | Local machines | all systems |

### 4.1 Local machines: N₂ and O₂

The diatomics have small enough active spaces that the whole pipeline (PySCF Hamiltonian → ffsim LUCJ circuit → hardware sampling → SQD diagonalization) ran end to end inside the notebooks on ordinary local machines. These are the notebooks in `notebooks/nitrogen/` and `notebooks/oxygen/`, one per basis set (STO-3G, 6-31G, cc-pVDZ) and geometry (optimized / unoptimized).

### 4.2 Remote CPU cluster: Kosambi

The larger molecules (CO₂, propane, cyclopropane, benzene) and the largest basis sets have much bigger active spaces. Their SQD subspaces reach hundreds of thousands to millions of determinants per iteration, and the diagonalization plus the full-molecule CCSD reference need far more memory and wall time than a laptop provides. These runs were executed on **Kosambi**, the CPU cluster of the Computer Science & Information Systems department at BITS Pilani, Goa. Their inputs, job scripts and logs are in `remote_cpu_runs/` (organized as `<Molecule>/<molecule>_<basis>_<opt|unopt>/`).

**Cluster and job configuration**

| Item | Value |
|---|---|
| Scheduler | SLURM |
| Partition | `tcp-normal` |
| Resources per job | 1 node, 1 task, **8 CPU cores** (each `tcp-normal` node has 8 CPUs; `--cpus-per-task` is not raised above 8) |
| Nodes excluded | `node5` (`--exclude=node5`) |
| Wall time | 4 h for the diatomics; 12 h for propane (20-orbital active space) |
| Operating system | CentOS 7 (glibc 2.17, gcc 4.8.5) |
| Python | `anaconda3/anaconda3-python3.9.13` module, with a virtualenv at `~/sqd-env` |
| Threading | `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`; PySCF is pointed at the same thread count. `ulimit -s unlimited` and `ulimit -l unlimited` are set in the job |
| Network | Compute nodes have **no internet access** |

> CPU model, clock speed and per-node RAM: **[fill in from the cluster documentation or `sinfo -p tcp-normal -o "%P %l %D %c %m"`]**.

Because compute nodes have no internet, no IBM access is needed or possible on the cluster. The workflow is split in two:

1. **Locally** (with internet and IBM credentials): build the Hamiltonian, run the circuit on the QPU, and save everything the classical step needs into `sqd_input.pkl` (`hcore`, `eri`, nuclear repulsion energy, number of orbitals, electron counts, the measured bitstring array, and a reference energy).
2. **On Kosambi**: copy `sqd_input.pkl` and the `*_cluster.py` script to the cluster, then submit the job. The script rebuilds the samples, runs `qiskit-addon-sqd` (configuration recovery, subspace diagonalization), computes a full-molecule CCSD reference, and writes `sqd_results.pkl` and an energy plot. The scripts are basis- and geometry-agnostic because they read everything from the pickle.

```bash
# from a local machine
scp sqd_input.pkl <user>@<kosambi-login-node>:~/<run_dir>/
scp *_cluster.py submit_*.sh <user>@<kosambi-login-node>:~/<run_dir>/

# on the Kosambi login node
cd ~/<run_dir>
sbatch submit_<name>.sh
squeue -u $USER                 # monitor
```

Things worth knowing when reproducing on a similar cluster:

- `module load` must appear **inside** the batch script; it does not carry over from the login shell.
- Wall time scales steeply with the number of orbitals; check the partition limit with `sinfo` before raising `--time`.
- The pinned `requirements.txt` targets a modern Python (3.12) on a modern OS. On an old CentOS 7 system such as Kosambi, new numpy/qiskit wheels (which need glibc ≥ 2.28) do not install, so the cluster environment was built separately from what the module provides. The numerical algorithm is the same; only the interpreter and package versions differ from the local environment.

## Security

`.env`, API keys and `~/.qiskit/qiskit-ibm.json` contain secrets. Do not commit or share them. If a key is exposed, revoke it in the IBM Quantum Platform API keys page and create a new one.