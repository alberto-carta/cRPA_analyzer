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
