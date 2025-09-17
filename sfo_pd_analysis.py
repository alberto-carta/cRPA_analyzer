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
material ="sfo"
basis = "dp"

# File paths (adjust as needed)
hr_file = f"../materials/cRPA_{material}_{basis}/{material}.2_hr.dat"
w_file =  f"../materials/cRPA_{material}_{basis}/dir-intW/dat.Wmat"
j_file =  f"../materials/cRPA_{material}_{basis}/dir-intJ/dat.Jmat"

# Chemical potential
# mu = 5.8787
mu = 9.2173
beta = 40.0 # 40
nw = 120 #150
kgrid = (7, 7, 7)


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
W_tensors_manual, r_list, q_list = read_WJ(w_file, j_file, calc.n_orb, kmesh_shape=kgrid)
print(f"Read {len(r_list)} R-vectors, {len(q_list)} q-vectors")
W_q_tensors_manual = manual_fourier_W(W_tensors_manual, r_list, q_list)

# Show a sample block for q=0
qpoint = 0
U_q0 = np.zeros((calc.n_orb, calc.n_orb), dtype=complex)
for i in range(calc.n_orb):
    for j in range(calc.n_orb):
        U_q0[i, j] = W_q_tensors_manual[qpoint, i, i, j, j]
with np.printoptions(precision=3, suppress=True):
    print("[Manual FT] U_q0 (first 5x5 block):")
    print(U_q0[0:5, 0:5])
#%%
from crpa_analyzer.utils import invert_tensor
from triqs_tprf.lattice import chi_wr_from_chi_wk
chi0_inv_wk = chi00_wk.copy() * 0.0
chid_inv_wk = chi00_wk.copy() * 0.0
chid_manual_wk = chi00_wk.copy()*0.0
nk = len(chi00_wk.mesh[1])
for ik in range(nk):
    chi0_inv_wk.data[0, ik] = invert_tensor(-chi00_wk.data[0, ik]*2, threshold=1e-5)

    # chid_inv_wk.data[0, ik] = -chi0_inv_wk.data[0, ik] - WcRPA_wk.data[0, ik]
    chid_inv_wk.data[0, ik] = chi0_inv_wk.data[0, ik] - W_q_tensors_manual[ik]

    chid_manual_wk.data[0, ik] = invert_tensor(chid_inv_wk.data[0, ik], threshold=1e-5)

chid_wr_manual = chi_wr_from_chi_wk(chid_manual_wk)


# chid_wr, chid_wk = calc.compute_rpa_manually(W_q_tensors_manual, threshold=1e-3)
# print bare and RPA traces
chid_trace = np.einsum('aabb', chid_wr_manual.data[0, 0, 0:5, 0:5, 0:5, 0:5])
print(f"Manual RPA susceptibility trace: {chid_trace:.6f}")

# now with the function
chid_wr_manual, chid_wk_manual = calc.compute_rpa_manually(W_q_tensors_manual, threshold=1e-8)
chid_trace = np.einsum('aabb', chid_wr_manual.data[0, 0, 0:5, 0:5, 0:5, 0:5])
print(f"Manual RPA susceptibility trace: {chid_trace:.6f}")
#%% print ligand and TM
chi0_TM_trace = np.einsum('aabb', -chi00_wr.data[0, 0, 0:5, 0:5, 0:5, 0:5] * 2)
chid_TM_trace = np.einsum('aabb', chid_wr_manual.data[0, 0, 0:5, 0:5, 0:5, 0:5])

chi0_p_trace = np.einsum('aabb', -chi00_wr.data[0, 0, 5:8, 5:8, 5:8, 5:8] * 2)
chid_p_trace = np.einsum('aabb', chid_wr_manual.data[0, 0, 5:8, 5:8, 5:8, 5:8])

chi0_2p_trace = np.einsum('aabb', -chi00_wr.data[0, 0, 11:14, 11:14, 11:14, 11:14] * 2)
chi_nonlocal_oxygen2 = np.einsum('aabb', chid_wr_manual.data[0, 0, 11:14, 11:14, 11:14, 11:14])

print(f"Bare susceptibility trace (TM): {chi0_TM_trace:.6f}")
print(f"Manual RPA susceptibility trace (TM): {chid_TM_trace:.6f}")

print(f"Bare susceptibility trace (Oxygen): {chi0_p_trace:.6f}")
print(f"Manual RPA susceptibility trace (Oxygen): {chid_p_trace:.6f}")

print(f"Bare susceptibility trace (Oxygen2): {chi0_2p_trace:.6f}")   
print(f"Non-local RPA trace (Oxygen2): {chi_nonlocal_oxygen2:.6f}")

#%%
atom_types = ['Fe', 'O1', 'O1', 'O2', 'O2']
n_orb_per_atom = [5, 3, 3, 3, 3]
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

with np.printoptions(precision=4, suppress=True):
    print("\nBare susceptibility isotropic R = 0,0,0")
    print(chi0r_isotropic[0, :, :])
    print("\nFull susceptibility isotropic R = 0,0,0")
    print(chidr_isotropic[0, :, :])
    print(chidr_isotropic[48, :, :]) # this is vector 0,-1,-1 


#%%
from crpa_analyzer.parse_Wmat_k import forward_transform, backward_transform

r_list_tprf = [ v.value.value for v in chi00_wr.mesh[1]]

chi0q_isotropic = forward_transform(chi0r_isotropic, r_list_tprf, q_list)
chidq_isotropic = forward_transform(chidr_isotropic, r_list_tprf, q_list)

# invert at every q and backtransform to get U_lrt
U_lrtq = np.zeros_like(chi0q_isotropic)

for iq in range(len(q_list)):
    chi0_inv = np.linalg.pinv(chi0q_isotropic[iq], rcond=1e-8)
    chid_inv = np.linalg.pinv(chidq_isotropic[iq], rcond=1e-8)
    U_lrtq[iq] = chi0_inv - chid_inv

U_lrtr = backward_transform(U_lrtq, q_list, r_list_tprf)
#%% print U_lrt
with np.printoptions(precision=3, suppress=True):
    print("\nU_lrt at R=0,0,0")
    print(U_lrtr[0, :, :])


# %%
r_list_tprf = [ v.value.value for v in chi00_wr.mesh[1]]
# --- QE comparison using library function only ---
from crpa_analyzer.qe_compare import get_chis_qe_convention

qe_supercell = [3, 3, 3]
crpa_supercell = [7, 7, 7]
debug = False


from importlib import reload
import crpa_analyzer.qe_compare
reload(crpa_analyzer.qe_compare)
from crpa_analyzer.qe_compare import get_chis_qe_convention, wrap_point, hp_r_points_loops

chi0_isotropic_qe, chid_isotropic_qe, rr_indices = get_chis_qe_convention(
    chi0r_isotropic, chidr_isotropic, n_atoms,
    qe_supercell=qe_supercell, crpa_supercell=crpa_supercell, r_list_tprf=r_list_tprf,
    debug=True,
)

#%%


inv_chi0_qe = np.linalg.pinv(chi0_isotropic_qe, rcond=0.0001)
inv_chid_qe = np.linalg.pinv(chid_isotropic_qe - np.sum(chid_isotropic_qe[0])/np.product(qe_supercell), rcond=0.0001) # crudely restore charge neutrality somehow
U_qe = inv_chi0_qe - inv_chid_qe

with np.printoptions(precision=5, suppress=True):
    print("chi0 isotropic QE R points (first few elements):")
    print(chi0_isotropic_qe[:5, :5])
    
    print("chid isotropic QE R points (first few elements):")
    print(chid_isotropic_qe[:5, :5])

    print("U isotropic QE R points (first few elements):")
    print(U_qe[:5, :5])
    

# %%
