#%%
"""
Example script demonstrating the cRPA analyzer package
"""
import numpy as np
import matplotlib.pyplot as plt
from crpa_analyzer import ChiCalculator, InteractionParser, plot_bands, plot_chi_comparison
import tqdm

#%%

"""Main example function"""
material ="nio"
basis = "dp"

# File paths (adjust as needed)
hr_file = f"../materials/cRPA_{material}_{basis}/{material}.2_hr.dat"
w_file =  f"../materials/cRPA_{material}_{basis}/dir-intW/dat.Wmat"
j_file =  f"../materials/cRPA_{material}_{basis}/dir-intJ/dat.Jmat"

# Chemical potential
mu = 14.5257
beta = 40.0 # 40
nw = 100 #150
kgrid = (7,7,7)


# High symmetry points
high_sym = {
    "G": [0.0, 0.0, 0.0],
    "N": [0.0, 0.5, 0.0], 
    "X": [0.0, 0.0, 0.5],
    "M": [0.5, 0.5, 0.5],
}
path_labels = ['G', 'N', 'X', 'G', 'M']
path_tick_labels = [r'$\Gamma$', r'$N$', r'$X$', r'$\Gamma$', r'$M$']

# Initialize calculator
print("Setting up tight-binding model...")
calc = ChiCalculator(hr_file, mu=mu, beta=beta, n_max=nw)

# Compute band structure
print("Computing band structure...")
k_vecs, k_plot, k_ticks = calc.get_kpath(high_sym, path_labels)
# calc.setup_kmesh(n_k=(10, 10, 10))
# calc.setup_kmesh(n_k=(11, 11, 11))
calc.setup_kmesh(n_k=kgrid)
bands = calc.compute_bands(k_vecs)

# Plot bands
plot_bands(k_plot, bands, k_ticks, path_tick_labels, mu=mu)

# Compute bare susceptibility
print("Computing bare susceptibility...")
chi00_wk, chi00_wr = calc.compute_bare_susceptibility()


chi00_trace = np.einsum('aabb', chi00_wr.data[0, 0, 0:5, 0:5, 0:5, 0:5] * 2)
print(f"Bare susceptibility trace: {chi00_trace:.6f}")

#%%
# --- Manual W/J Fourier transform demonstration (from utils) ---
from crpa_analyzer.utils import read_WJ, manual_fourier_W

print("\n--- Manual W/J Fourier transform ---")
W_tensors_manual, r_list, q_list = read_WJ(w_file, j_file, calc.n_orb, kmesh_shape=kgrid, R_cutoff=5)
print(f"Read {len(r_list)} R-vectors, {len(q_list)} q-vectors")
W_q_tensors_manual = manual_fourier_W(W_tensors_manual, r_list, q_list)

#%% Show a sample block for q=0
qpoint = 0
U_q0 = np.zeros((calc.n_orb, calc.n_orb), dtype=complex)
U_r0 = np.zeros((calc.n_orb, calc.n_orb), dtype=complex)
for i in range(calc.n_orb):
    for j in range(calc.n_orb):
        U_q0[i, j] = W_q_tensors_manual[qpoint, i, i, j, j]
        U_r0[i, j] = W_tensors_manual[0, i, i, j, j]
with np.printoptions(precision=3, suppress=True):
    print("[Manual FT] U_q0 (first 5x5 block):")
    print(U_q0[0:5, 0:5])
    print(U_q0[5:10, 5:10])

    print("\n[Manual FT] U_r0 (first 5x5 block):")
    print(U_r0[0:5, 0:5])
    print(U_r0[5:10, 5:10])
    print(U_r0[10:13, 10:13])
    print(U_r0[13:16, 13:16])
#%%
chid_wr_manual, chid_wk_manual = calc.compute_rpa_manually(W_q_tensors_manual, threshold=1e-5)
chid_trace = np.einsum('aabb', chid_wr_manual.data[0, 0, 0:5, 0:5, 0:5, 0:5])
print(f"Manual RPA susceptibility trace: {chid_trace:.6f}")
#%% print ligand and TM
chi0_TM_trace = np.einsum('aabb', -chi00_wr.data[0, 0, 0:5, 0:5, 0:5, 0:5] * 2)
chid_TM_trace = np.einsum('aabb', chid_wr_manual.data[0, 0, 0:5, 0:5, 0:5, 0:5])

chi0_TM_trace2 = np.einsum('aabb', -chi00_wr.data[0, 0, 5:10, 5:10, 5:10, 5:10] * 2)
chid_TM2_trace = np.einsum('aabb', chid_wr_manual.data[0, 0, 5:10, 5:10, 5:10, 5:10])

chi0_p_trace = np.einsum('aabb', -chi00_wr.data[0, 0, 10:13, 10:13, 10:13, 10:13] * 2)
chid_p_trace = np.einsum('aabb', chid_wr_manual.data[0, 0, 10:13, 10:13, 10:13, 10:13])

chi0_p2_trace = np.einsum('aabb', -chi00_wr.data[0, 0, 13:16, 13:16, 13:16, 13:16] * 2)
chid_p2_trace = np.einsum('aabb', chid_wr_manual.data[0, 0, 13:16, 13:16, 13:16, 13:16])

print(f"Bare susceptibility trace (TM): {chi0_TM_trace:.6f}")
print(f"Manual RPA susceptibility trace (TM): {chid_TM_trace:.6f}")

print(f"Bare susceptibility trace (TM2): {chi0_TM_trace2:.6f}")
print(f"Manual RPA susceptibility trace (TM2): {chid_TM2_trace:.6f}")

print(f"Bare susceptibility trace (F): {chi0_p_trace:.6f}")
print(f"Manual RPA susceptibility trace (F): {chid_p_trace:.6f}")

print(f"Bare susceptibility trace (F2): {chi0_p2_trace:.6f}")
print(f"Manual RPA susceptibility trace (F2): {chid_p2_trace:.6f}")

#%%
atom_types = ['Ni1', 'Ni2', 'O', 'O']
n_orb_per_atom = [5, 5, 3, 3]
atom_indices = [ list(range(sum(n_orb_per_atom[:i]), sum(n_orb_per_atom[:i+1]))) for i in range(len(n_orb_per_atom))]

# trace over chi0
nr = len(chi00_wr.mesh[1])
n_atoms = len(atom_types)
chi0r_isotropic = np.zeros((nr, n_atoms, n_atoms), dtype=complex)
chidr_isotropic = np.zeros((nr, n_atoms, n_atoms), dtype=complex)

for ir in range(nr):
    for i in range(n_atoms):
        for j in range(n_atoms):
            # Extract the submatrix for atoms i and j using np.ix_ for proper indexing
            submatrix_0 = chi00_wr.data[0, ir][np.ix_(atom_indices[i], atom_indices[i], atom_indices[j], atom_indices[j])]
            chi0r_isotropic[ir, i, j] = np.einsum('aabb', -submatrix_0 * 2)
            submatrix_d = chid_wr_manual.data[0, ir][np.ix_(atom_indices[i], atom_indices[i], atom_indices[j], atom_indices[j])]
            chidr_isotropic[ir, i, j] = np.einsum('aabb', submatrix_d)

with np.printoptions(precision=6, suppress=True):
    print("\nBare susceptibility isotropic R = 0,0,0")
    print(chi0r_isotropic[0, :, :].real)
    print("\nFull susceptibility isotropic R = 0,0,0")
    print(chidr_isotropic[0, :, :].real)


#%%
from crpa_analyzer.parse_Wmat_k import forward_transform, backward_transform

r_list_tprf = [ v.value.value for v in chi00_wr.mesh[1]]

chi0q_isotropic = forward_transform(chi0r_isotropic, r_list_tprf, q_list)
chidq_isotropic = forward_transform(chidr_isotropic, r_list_tprf, q_list)

# invert at every q and backtransform to get U_lrt
U_lrtq = np.zeros_like(chi0q_isotropic)

for iq in range(len(q_list)):
    chi0_inv = np.linalg.pinv(chi0q_isotropic[iq], rcond=1e-5)
    chid_inv = np.linalg.pinv(chidq_isotropic[iq], rcond=1e-5)
    U_lrtq[iq] = chi0_inv - chid_inv

U_lrtr = backward_transform(U_lrtq, q_list, r_list_tprf)
#%% print U_lrt
with np.printoptions(precision=3, suppress=True):
    print("\nU_lrt at R=0,0,0")
    print(U_lrtr[0, :, :])


# %%
import pandas as pd
# save to csv the isotropic components at R = 0 for chis and U,
#  use the atoms as rows, chi and U as columns
# for now save only diagonal components

df = pd.DataFrame(columns = ['chi0_diag', 'chid_diag', 'U_diag'])
for i in range(n_atoms):
    df.loc[atom_types[i]] = [chi0r_isotropic[0, i, i].real, chidr_isotropic[0, i, i].real, U_lrtr[0, i, i].real]
df.to_csv(f"{material}_{basis}_isotropic_chis_R0.csv")

# %%

r_list_tprf = [ v.value.value for v in chi00_wr.mesh[1]]
# --- QE comparison using library function only ---
from crpa_analyzer.qe_compare import get_chis_qe_convention

qe_supercell = [3, 3, 3]
crpa_supercell = [7, 7, 7]

# from importlib import reload
# import crpa_analyzer.qe_compare
# reload(crpa_analyzer.qe_compare)
# from crpa_analyzer.qe_compare import get_chis_qe_convention, hp_r_points_loops, wrap_point

# chi0_isotropic_qe, chid_isotropic_qe, rr_indices = get_chis_qe_convention(
#     chi0r_isotropic, chidr_isotropic, n_atoms,
#     qe_supercell=qe_supercell, crpa_supercell=crpa_supercell, r_list_tprf=r_list_tprf,
#     debug=True
# )

from importlib import reload
import crpa_analyzer.qe_compare
reload(crpa_analyzer.qe_compare)
from crpa_analyzer.qe_compare import get_chis_qe_convention, wrap_point, hp_r_points_loops

chi0_isotropic_qe, chid_isotropic_qe, rr_indices = get_chis_qe_convention(
    chi0r_isotropic, chidr_isotropic, n_atoms,
    qe_supercell=qe_supercell, crpa_supercell=crpa_supercell, r_list_tprf=r_list_tprf,
    debug=False,
)

#%%


inv_chi0_qe = np.linalg.pinv(chi0_isotropic_qe.real, rcond=0.0001)
inv_chid_qe = np.linalg.pinv(chid_isotropic_qe.real - np.sum(chid_isotropic_qe[0].real)/np.product(qe_supercell), rcond=0.01) # crudely restore charge neutrality somehow
U_qe = inv_chi0_qe - inv_chid_qe

with np.printoptions(precision=5, suppress=True):
    print("chi0 isotropic QE R points (first few elements):")
    print(chi0_isotropic_qe[:8, :8].real)
    
    print("chid isotropic QE R points (first few elements):")
    print(chid_isotropic_qe[:8, :8].real)

    print("U isotropic QE R points (first few elements):")
    print(U_qe[:8, :8].real)




#%%
