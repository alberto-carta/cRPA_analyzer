#%%
from triqs_tprf.wannier90 import parse_hopping_from_wannier90_hr_dat
from triqs_tprf.wannier90 import parse_lattice_vectors_from_wannier90_wout
from triqs_tprf.lattice_utils import k_space_path
import numpy as np
import matplotlib.pyplot as plt

hopp_dict, n_orb = parse_hopping_from_wannier90_hr_dat("../materials/cRPA_kcuf_frontier/kcuf.2_hr.dat")
# hopp_dict, n_orb = parse_hopping_from_wannier90_hr_dat("sfo.2_hr.dat")
# hopp_dict, n_orb = parse_hopping_from_wannier90_hr_dat("cRPA_sfo_dp/sfo.2_hr.dat")
# units = parse_lattice_vectors_from_wannier90_wout("kcuf.2.wout")
#use identity matrices for units
units = [(1,0,0), (0,1,0), (0,0,1)]
# %%
from triqs_tprf.tight_binding import TBLattice

# Create single-spin tight-binding lattice (spinless calculation)
tb_lattice = TBLattice(
    units=units,
    hopping=hopp_dict,
    orbital_positions=[(0,0,0)]*n_orb,
    orbital_names=[str(i) for i in range(n_orb)],
)

#%%
# --- Define k-path (from your kpoint_path section) ---
high_sym = {
    "G": [0.0, 0.0, 0.0],
    "N": [0.0, 0.5, 0.0],
    "X": [0.0, 0.0, 0.5],
    "M": [0.5, 0.5, 0.5],
}

path_labels = ['G', 'N', 'X', 'G', 'M']
# Define corresponding LaTeX labels for plotting
path_tick_labels = [r'$\Gamma$', r'$N$', r'$X$', r'$\Gamma$', r'$M$']

paths = [(high_sym[path_labels[i]], high_sym[path_labels[i+1]]) for i in range(len(path_labels)-1)]

# --- Get k-points along path ---
k_vecs, k_plot, k_ticks = k_space_path(paths, bz=tb_lattice.bz, num=200)

# --- Get Fourier transform of the tight-binding Hamiltonian ---
# kmesh = tb_lattice.get_kmesh(n_k=(8, 8, 8))  # smaller mesh for computational efficiency
kmesh = tb_lattice.get_kmesh(n_k=(11, 11, 11))  # smaller mesh for computational efficiency
# kmesh = tb_lattice.get_kmesh(n_k=(7, 7, 7))  # smaller mesh for computational efficiency
# kmesh reciprocal space distance is 2*np.pi / (kgrid1,kgrid2,kgrid3)
e_k = tb_lattice.fourier(kmesh)

# --- Calculate bands along the k-path ---
bands = np.array([np.linalg.eigvalsh(e_k(k)) for k in k_vecs])  # shape: (nk, n_orb)
# --- Set Fermi level (chemical potential) ---
mu = 5.8787
# mu = 9.2173

# --- Plotting ---
plt.figure(figsize=(7,5))
plt.plot(k_plot, bands, color='k', lw=1)
#add horizontal line at fermi level
plt.axhline(y=mu, color='r', linestyle='--', label='Fermi level')
plt.legend()
plt.xticks(k_ticks, path_tick_labels)
plt.xlabel("k-path")
plt.ylabel("Energy (eV)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Compute bare susceptibility using imtime bubble method
from triqs_tprf.lattice import lattice_dyson_g0_wk
from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk
from triqs.gf import MeshImFreq, Idx

# Define frequency mesh
wmesh = MeshImFreq(beta=40.0, S='Fermion', n_max=100)

# Compute lattice Green's function with Fermi level mu = 5.8787
g0_wk = lattice_dyson_g0_wk(mu=mu, e_k=e_k, mesh=wmesh)

# Compute bare susceptibility using imtime bubble
chi00_wk = imtime_bubble_chi0_wk(g0_wk, nw=1)


# %%
# Define helper functions for susceptibility analysis
def chi_charge_contraction(chi, spin_factor=2.0):
    """ Computes the trace susceptibility for charge channel 
    Args:
        chi: bare susceptibility 
        spin_factor: factor to account for spin degeneracy (default=2.0)
    """
    # Use einsum to sum over orbital indices for charge susceptibility
    chi_charge = chi[0, 0, 0, 0].copy()
    chi_charge.data[:] = spin_factor * np.einsum('wqabab->wq', chi.data)[:, :]
    chi_charge = chi_charge[Idx(0), :]
    return chi_charge

def chi_magnetic_contraction(chi, spin_factor=2.0):
    """ Computes magnetic susceptibility for spinless calculation
    For spinless calculation, magnetic susceptibility is zero by construction
    """
    chi_mag = chi[0, 0, 0, 0].copy()
    chi_mag.data[:] = 0.0 * np.einsum('wqabab->wq', chi.data)[:, :]
    chi_mag = chi_mag[Idx(0), :]
    return chi_mag

def interpolate_chi(chi, k_vecs):
    assert( k_vecs.shape[1] == 3 )
    chi_interp = np.zeros(
        [k_vecs.shape[0]] + list(chi.target_shape), dtype=complex)

    for kidx, (kx, ky, kz) in enumerate(k_vecs):
        chi_interp[kidx] = chi((kx, ky, kz))

    return chi_interp

def plot_chi_1D(chi, k_vecs, k_plot, k_ticks, tick_labels, label=None, channel='charge', spin_factor=2.0):
    if channel == 'charge':
        chi_contracted = chi_charge_contraction(chi, spin_factor)
        title = r'Charge susceptibility $\chi_{charge}(\mathbf{q}, \omega=0)$'
    elif channel == 'magnetic':
        chi_contracted = chi_magnetic_contraction(chi, spin_factor)
        title = r'Magnetic susceptibility $\chi_{S_z S_z}(\mathbf{q}, \omega=0)$'
    else:
        raise ValueError("Channel must be 'charge' or 'magnetic'")
        
    chi_interp = interpolate_chi(chi_contracted, k_vecs)
    plt.plot(k_plot, chi_interp.real, label=label)
    plt.grid()
    plt.xticks(ticks=k_ticks, labels=tick_labels)
    plt.title(title)

# %%
# Plot bare susceptibilities along k-path
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Charge susceptibility (doubled for spin degeneracy)
plt.sca(ax1)
plot_chi_1D(chi00_wk, k_vecs, k_plot, k_ticks, path_tick_labels, label=r'$U=0$ (bare)', channel='charge', spin_factor=2.0)
plt.ylabel(r'$\chi_{charge}(\mathbf{q}, \omega=0)$')
plt.legend()

# Magnetic susceptibility (zero for spinless calculation)
plt.sca(ax2)
plot_chi_1D(chi00_wk, k_vecs, k_plot, k_ticks, path_tick_labels, label=r'$U=0$ (bare)', channel='magnetic', spin_factor=2.0)
plt.ylabel(r'$\chi_{S_z S_z}(\mathbf{q}, \omega=0)$')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Transform susceptibility to real space
from triqs_tprf.lattice_utils import fourier_wk_to_wr
from triqs_tprf.lattice import chi_wr_from_chi_wk
#%%
# Transform chi00_wk from k-space to real space
chi00_wr = chi_wr_from_chi_wk(chi00_wk)

print(f"Shape of chi00_wr: {chi00_wr.data.shape}")
print(f"Available R-vectors: {len(chi00_wr.mesh[1])}")

# %%

chi00_wr_R0 = chi00_wr.data[0,0]*2 # factor of 2 for spin degeneracy

chi_trace = -np.einsum('aabb', chi00_wr_R0[0:5,0:5,0:5,0:5])
print(chi_trace)
# manual trace
# np.sum(np.einsum('qaabb',chi00_wk.data[0]/3375))*2
#%% create a chi0 with only iijj and ijji elements
chi00_wr_rel = chi00_wr.copy() *0.0
for ik in range(len(chi00_wr.mesh[1])):
    for i in range(n_orb):
        for j in range(n_orb):
            chi00_wr_rel.data[0, ik, i, i, j, j] = chi00_wr.data[0, ik, i, i, j, j]
            chi00_wr_rel.data[0, ik, i, j, j, i] = chi00_wr.data[0, ik, i, j, j, i]

# compare the two with euclidean norm 
kpoint = 0
diffs = []
for ikpoint in range(len(chi00_wr.mesh[1])):
    diff = np.linalg.norm(chi00_wr_rel.data[0,ikpoint] 
                        - chi00_wr.data[0,ikpoint])
    diffs.append(diff)
print(f"Difference : {np.sum(diffs):.9f}")

plt.plot(diffs)



# %% add W from crpa here
# Load local interaction tensors
from parse_Wmat_k import parse_wmat_block
import numpy as np

# Extract local W tensor (Coulomb interactions) at R=[0,0,0]
print("Loading W matrix...")
W_file = "../materials/cRPA_kcuf_frontier/dir-intW/dat.Wmat"
# W_file = "cRPA_sfo/dir-intW/dat.Wmat"
# W_file = "cRPA_sfo_dp/dir-intW/dat.Wmat"
W_local = parse_wmat_block(W_file, triplet=(0,0,0))
print(f"W_local shape: {W_local.shape}")
print(f"W_local max imaginary part: {np.max(np.abs(W_local.imag))}")

# Extract local J tensor (Exchange interactions) at R=[0,0,0] 
print("Loading J matrix...")
J_file = "../materials/cRPA_kcuf_frontier/dir-intJ/dat.Jmat"
# J_file = "cRPA_sfo/dir-intJ/dat.Jmat"
# J_file = "cRPA_sfo_dp/dir-intJ/dat.Jmat"
J_local = parse_wmat_block(J_file, triplet=(0,0,0))
print(f"J_local shape: {J_local.shape}")
print(f"J_local max imaginary part: {np.max(np.abs(J_local.imag))}")


print(f"Local W interaction tensor loaded with shape {W_local.shape}")
print(f"Local J interaction tensor loaded with shape {J_local.shape}")

V_abcd = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)
for i in range(n_orb):
    for j in range(n_orb):
                V_abcd[i, i, j, j] = W_local[i, j,0]
                V_abcd[i, j, j, i] = J_local[i, j,0]
# %%
# from triqs_tprf.lattice import solve_rpa_PH
# chid_wk = solve_rpa_PH(-chi00_wk, V_abcd)

# chid_wr = chi_wr_from_chi_wk(chid_wk)

# chid_wr_R0 = chid_wr.data[0,0]*2 # factor of 2 for spin degeneracy
# chid_trace = np.einsum('aabb', chid_wr_R0)
# print(chid_trace)

#%% manually compute RPA in k-space
print("Computing RPA manually in k-space...")

# Since V is local (R=[0,0,0]), it's constant for every k-point
# V_abcd is already the local interaction tensor, 

# CODE IS WRONG, CHECK
V_local = V_abcd  # Shape: (norb, norb, norb, norb)

# Get dimensions
nw, nk, norb, _,_,_ = chi00_wk.data.shape
print(f"chi00_wk shape: {chi00_wk.data.shape}")
print(f"V_local shape: {V_local.shape}")
    
from parse_Wmat_k import (list_triplets, parse_wmat_block_real, 
                            generate_dual_q_grid, forward_transform, 
                            backward_transform)
#%%
from crpa_analyzer.utils import read_WJ, manual_fourier_W
# === Manual Fourier Transform of W tensors (separate implementation) ===
# def read_WJ(W_file, J_file, norb, kmesh_shape=(11,11,11), R_cutoff=500):
#     """
#     Manually compute W_q_tensors by explicit Fourier transform, using q_list and r_list.
#     Returns: W_q_tensors, r_list, q_list
#     """
#     import numpy as np
#     r_list = list_triplets(W_file)
#     r_list.sort(key=lambda t: np.linalg.norm(t))
#     q_list = generate_dual_q_grid(kmesh_shape)
#     U_matrices = np.zeros((len(r_list), norb, norb))
#     J_matrices = np.zeros((len(r_list), norb, norb))
#     W_tensors = np.zeros((len(r_list), norb, norb, norb, norb), dtype=complex)
#     for i, triplet in enumerate(r_list):
#         U_matrices[i] = parse_wmat_block_real(W_file, triplet=triplet, orbmax=norb)
#     J_matrices[0] = parse_wmat_block_real(J_file, triplet=r_list[0], orbmax=norb)
#     for k in range(len(r_list)):
#         if np.linalg.norm(r_list[k]) < R_cutoff:
#             for i in range(norb):
#                 for j in range(norb):
#                     W_tensors[k, i, i, j, j] = U_matrices[k, i, j]
#     for i in range(norb):
#         for j in range(norb):
#             W_tensors[0, i, j, j, i] = J_matrices[0, i, j]
#     # Fourier transform to k-space
#     return W_tensors, r_list, q_list

# def manual_fourier_W(W_tensors, r_list, q_list):

#     W_q_tensors = forward_transform(W_tensors, r_list, q_list)
#     retransformed = backward_transform(W_q_tensors, q_list, r_list)
#     diff = np.linalg.norm(W_tensors - retransformed)
#     # error must be zero, if not the grid is wrong
#     print(f"[Manual FT] Transform round-trip error (screened): {diff:.9f}")
#     return W_q_tensors

# --- Usage example: manual Fourier transform of W ---
print("\n--- Reading W and J matrices and performing manual Fourier transform ---")
W_file = "../materials/cRPA_kcuf_frontier/dir-intW/dat.Wmat"
J_file = "../materials/cRPA_kcuf_frontier/dir-intJ/dat.Jmat"
kmesh_shape = (11, 11, 11)
W_tensors_manual, r_list, q_list = read_WJ(W_file, J_file, n_orb, kmesh_shape)
print("\n--- Manual Fourier Transform of W tensors (separate from tprf) ---")
W_q_tensors_manual = manual_fourier_W(W_tensors_manual, r_list, q_list)

#%%

W_q_tensors = W_q_tensors_manual
W_invs = np.zeros_like(W_q_tensors)

norb = W_q_tensors.shape[1]

qpoint = 0
print(f"\nDirect part (U) of W_q at q={qpoint}: {q_list[qpoint]}")
print("Screened case:")
U_q0 = np.zeros((norb, norb), dtype=complex)
for i in range(norb):
    for j in range(norb):
        U_q0[i, j] = W_q_tensors[qpoint, i, i, j, j]#+0.001
with np.printoptions(precision=3, suppress=True):
    print(U_q0[0:5, 0:5])
    # print(np.linalg.inv(U_q0)[0:5, 0:5])


for iq, q in enumerate(q_list):
    W_invs[iq] = np.linalg.tensorinv(W_q_tensors[iq], ind=2)
    # W_invs[iq] = np.linalg.pinv(W_q_tensors[iq].reshape(norb*norb, norb*norb), rcond=1e-5).reshape(norb, norb, norb, norb)
    # W_invs[iq] = np.linalg.pinv(W_q_tensors[iq].reshape(norb*norb, norb*norb), rcond=1e-5).reshape(norb, norb, norb, norb)

U_invs = np.zeros((norb, norb), dtype=complex)
for i in range(norb):
    for j in range(norb):
        U_invs[i, j] = W_invs[qpoint, i, i, j, j]
with np.printoptions(precision=3, suppress=True):
    print("cRPA inverse U matrix at q=0:")
    print(U_invs[0:5, 0:5])

print(np.sum(U_invs[0:5,0:5].real))

# trace over W_invs
W_invs_trace = np.einsum('qaabb->q', W_invs[:,0:5,0:5,0:5,0:5])
print(f"Trace of W_invs (first 5x5 block): {np.sum(W_invs_trace)/len(r_list)}")

#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
#%% solve RPA with WcRPA_wk
from crpa_analyzer.utils import invert_tensor
chi0_inv_wk = chi00_wk.copy() * 0.0
chid_inv_wk = chi00_wk.copy() * 0.0
chid_manual_wk = chi00_wk.copy()*0.0
nk = len(chi00_wk.mesh[1])
for ik in range(nk):
    chi0_inv_wk.data[0, ik] = invert_tensor(-chi00_wk.data[0, ik]*2, threshold=1e-3)

    # chid_inv_wk.data[0, ik] = -chi0_inv_wk.data[0, ik] - WcRPA_wk.data[0, ik]
    chid_inv_wk.data[0, ik] = chi0_inv_wk.data[0, ik] - W_q_tensors[ik]

    chid_manual_wk.data[0, ik] = invert_tensor(chid_inv_wk.data[0, ik], threshold=1e-3)



chid_wr_manual = chi_wr_from_chi_wk(chid_manual_wk)

chid_wr_R0_manual = chid_wr_manual.data[0,0] # factor of 2 for spin degeneracy
chid_trace_manual = np.einsum('aabb', chid_wr_R0_manual[0:5,0:5,0:5,0:5])
print(chid_trace_manual)


# %%
