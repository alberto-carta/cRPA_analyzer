"""
Simple script to extract key interaction parameters
"""
import numpy as np
from crpa_analyzer import InteractionParser


def extract_hubbard_parameters(w_file, j_file, l_d=2):
    """Extract Hubbard U and J parameters from cRPA calculations
    
    Args:
        w_file: path to dat.Wmat file
        j_file: path to dat.Jmat file  
        l_d: d-orbital quantum number (default: 2 for d orbitals)
        
    Returns:
        U, J: Hubbard parameters in eV
    """
    parser = InteractionParser(w_file, j_file)
    
    # Load local interactions at R=(0,0,0)
    try:
        from crpa_analyzer.parse_Wmat_k import parse_wmat_block
        W_local = parse_wmat_block(w_file, triplet=(0, 0, 0))
        J_local = parse_wmat_block(j_file, triplet=(0, 0, 0))
    except ImportError:
        # Fallback for direct script usage
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from crpa_analyzer.parse_Wmat_k import parse_wmat_block
        W_local = parse_wmat_block(w_file, triplet=(0, 0, 0))
        J_local = parse_wmat_block(j_file, triplet=(0, 0, 0))
    
    # Extract d-orbital block (assuming first 2*l_d+1 orbitals are d-orbitals)
    n_d = 2 * l_d + 1
    U_dd = W_local[:n_d, :n_d, 0]  # real part
    J_dd = J_local[:n_d, :n_d, 0]  # real part
    
    # Calculate average U and effective U-J
    U = np.mean(np.diag(U_dd))
    U_minus_J = np.sum(U_dd - J_dd) / (n_d * (n_d - 1))
    J = U - U_minus_J
    
    print(f"Hubbard U = {U:.3f} eV")
    print(f"Hund's J = {J:.3f} eV") 
    print(f"U - J = {U_minus_J:.3f} eV")
    
    return U, J


if __name__ == "__main__":
    # Example usage
    w_file = "cRPA_kcuf/dir-intW/dat.Wmat"
    j_file = "cRPA_kcuf/dir-intJ/dat.Jmat"
    
    U, J = extract_hubbard_parameters(w_file, j_file)
