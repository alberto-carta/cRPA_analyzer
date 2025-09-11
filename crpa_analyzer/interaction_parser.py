"""
Parse interaction matrices from cRPA calculations
"""
import numpy as np
from .parse_Wmat_k import parse_wmat_block, list_triplets


class InteractionParser:
    """Parser for cRPA interaction matrices"""
    
    def __init__(self, w_file=None, j_file=None):
        """Initialize parser with file paths
        
        Args:
            w_file: path to dat.Wmat file
            j_file: path to dat.Jmat file  
        """
        self.w_file = w_file
        self.j_file = j_file
        self._w_triplets = None
        self._j_triplets = None
    
    def get_available_triplets(self, matrix_type='W'):
        """Get list of available R-vector triplets"""
        if matrix_type == 'W' and self.w_file:
            if self._w_triplets is None:
                self._w_triplets = list_triplets(self.w_file)
            return self._w_triplets
        elif matrix_type == 'J' and self.j_file:
            if self._j_triplets is None:
                self._j_triplets = list_triplets(self.j_file)
            return self._j_triplets
        else:
            return []
    
    def load_local_interactions(self, n_orb, triplet=(0, 0, 0)):
        """Load local interaction tensors at given R-vector
        
        Args:
            n_orb: number of orbitals
            triplet: R-vector triplet (default: (0,0,0))
            
        Returns:
            V_abcd: 4-index interaction tensor
        """
        if not (self.w_file and self.j_file):
            raise ValueError("Both W and J files must be specified")
            
        W_local = parse_wmat_block(self.w_file, triplet=triplet)
        J_local = parse_wmat_block(self.j_file, triplet=triplet)
        
        V_abcd = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)
        for i in range(n_orb):
            for j in range(n_orb):
                V_abcd[i, i, j, j] = W_local[i, j, 0]  # real part
                V_abcd[i, j, j, i] = J_local[i, j, 0]  # real part
        
        return V_abcd
    
    def load_nonlocal_interactions(self, n_orb, rmax, r_cutoff=5.1, chi_wr_template=None, debug=False):
        """Load non-local interactions within cutoff radius
        
        Args:
            n_orb: number of orbitals
            r_cutoff: cutoff radius for R-vectors
            chi_wr_template: template chi_wr object to copy structure from
            
        Returns:
            WcRPA_wr: interaction tensor in real space (proper TRIQS object)
        """
        if chi_wr_template is None:
            raise ValueError("chi_wr_template required for non-local interactions")
            
        # Find R-vectors within cutoff
        # periodic boundary conditions to get back to RESPACK convention
        # kmax is composed of the maximum absolute values of Rx, Ry, Rz found in RESPACK Wmat and Jmat
        # RESPACK uses the convention that R goes from  -rmax to +rmax in each direction
        # while tprf goes from 0 to 2*rmax+1

        # this part of the code wraps the R-vectors to account for this
        # if Rx > rmax[0] then Rx = Rx - (2*rmax[0]+1)
        # else do nothing
        wrapped_R_vectors = [
            np.array(R) - np.rint(np.array(R) / (2 * np.array(rmax) + 1)) * (2 * np.array(rmax) + 1)
            for R in chi_wr_template.mesh[1]
        ]

        masked_indices = [
            iR for iR, R in enumerate(chi_wr_template.mesh[1]) 
            if np.linalg.norm(R.value.value) <= r_cutoff
        ]
        
        # Initialize interaction tensor using template structure
        WcRPA_wr = chi_wr_template.copy() * 0.0
        
        J_local = parse_wmat_block(self.j_file, triplet=(0, 0, 0))
        
        for iR in masked_indices:
            R_vec = tuple(chi_wr_template.mesh[1][iR].value.value.astype(int))
            try:
                W_ir = parse_wmat_block(self.w_file, triplet=R_vec)
            except:
                print(f"Warning: Could not find W matrix for R={R_vec}. Using zeros.")
                continue
                
            for i in range(n_orb):
                for j in range(n_orb):
                    WcRPA_wr.data[0, iR, i, i, j, j] = W_ir[i, j, 0]
                    WcRPA_wr.data[0, iR, i, j, j, i] = J_local[i, j, 0]
        
        return WcRPA_wr
