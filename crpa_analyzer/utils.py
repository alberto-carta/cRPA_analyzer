# === Manual W/J reading and Fourier transform (separate, as in working_prototype.py) ===
from crpa_analyzer.parse_Wmat_k import forward_transform, backward_transform
import numpy as np

def read_WJ(W_file, J_file, norb, kmesh_shape=(11,11,11), R_cutoff=500):
    """
    Read W and J matrices, build W_tensors, r_list, q_list (no FT).
    Returns: W_tensors, r_list, q_list
    """
    from crpa_analyzer.parse_Wmat_k import list_triplets, parse_wmat_block_real, generate_dual_q_grid
    import numpy as np
    r_list = list_triplets(W_file)
    r_list.sort(key=lambda t: np.linalg.norm(t))
    q_list = generate_dual_q_grid(kmesh_shape)
    U_matrices = np.zeros((len(r_list), norb, norb))
    J_matrices = np.zeros((len(r_list), norb, norb))
    W_tensors = np.zeros((len(r_list), norb, norb, norb, norb), dtype=complex)
    for i, triplet in enumerate(r_list):
        U_matrices[i] = parse_wmat_block_real(W_file, triplet=triplet, orbmax=norb)
    J_matrices[0] = parse_wmat_block_real(J_file, triplet=r_list[0], orbmax=norb)
    for k in range(len(r_list)):
        if np.linalg.norm(r_list[k]) < R_cutoff:
            for i in range(norb):
                for j in range(norb):
                    W_tensors[k, i, i, j, j] = U_matrices[k, i, j]
    for i in range(norb):
        for j in range(norb):
            W_tensors[0, i, j, j, i] = J_matrices[0, i, j]
    return W_tensors, r_list, q_list

def manual_fourier_W(W_tensors, r_list, q_list):
    """
    Perform the forward and backward Fourier transform, print round-trip error, return W_q_tensors.
    """
    W_q_tensors = forward_transform(W_tensors, r_list, q_list)
    retransformed = backward_transform(W_q_tensors, q_list, r_list)
    diff = np.linalg.norm(W_tensors - retransformed)
    print(f"[Manual FT] Transform round-trip error (screened): {diff:.9f}")
    return W_q_tensors
"""
Utility functions for susceptibility calculations
"""
import numpy as np
from triqs.gf import Idx


def chi_charge_contraction(chi, spin_factor=2.0):
    """Computes the trace susceptibility for charge channel 
    
    Args:
        chi: bare susceptibility 
        spin_factor: factor to account for spin degeneracy (default=2.0)
    """
    chi_charge = chi[0, 0, 0, 0].copy()
    chi_charge.data[:] = spin_factor * np.einsum('wqabab->wq', chi.data)[:, :]
    chi_charge = chi_charge[Idx(0), :]
    return chi_charge


def chi_magnetic_contraction(chi, spin_factor=2.0):
    """Computes magnetic susceptibility for spinless calculation
    
    For spinless calculation, magnetic susceptibility is zero by construction
    """
    chi_mag = chi[0, 0, 0, 0].copy()
    chi_mag.data[:] = 0.0 * np.einsum('wqabab->wq', chi.data)[:, :]
    chi_mag = chi_mag[Idx(0), :]
    return chi_mag


def interpolate_chi(chi, k_vecs):
    """Interpolate susceptibility on arbitrary k-points"""
    assert k_vecs.shape[1] == 3
    chi_interp = np.zeros([k_vecs.shape[0]] + list(chi.target_shape), dtype=complex)

    for kidx, (kx, ky, kz) in enumerate(k_vecs):
        chi_interp[kidx] = chi((kx, ky, kz))

    return chi_interp


def invert_tensor(tensor, threshold=1e-5):
    """Invert 4-index tensor using pseudo-inverse with threshold"""
    norb = tensor.shape[0]
    tensor_reshaped = np.reshape(tensor, (norb*norb, norb*norb))
    tensor_inv_reshaped = np.linalg.pinv(tensor_reshaped, rcond=threshold)
    tensor_inv = np.reshape(tensor_inv_reshaped, (norb, norb, norb, norb))
    return tensor_inv
