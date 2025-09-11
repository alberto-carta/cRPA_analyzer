"""
Main susceptibility calculator class
"""
import numpy as np
from triqs_tprf.wannier90 import parse_hopping_from_wannier90_hr_dat
from triqs_tprf.tight_binding import TBLattice
from triqs_tprf.lattice_utils import k_space_path, imtime_bubble_chi0_wk
from triqs_tprf.lattice import lattice_dyson_g0_wk, solve_rpa_PH, chi_wr_from_chi_wk, chi_wk_from_chi_wr
from triqs.gf import MeshImFreq
from .utils import invert_tensor


class ChiCalculator:
    """Calculator for bare and dressed susceptibilities"""
    
    def __init__(self, hr_file, units=None, orbital_positions=None, mu=None, 
                 beta=40.0, n_max=100):
        """Initialize with tight-binding data
        
        Args:
            hr_file: path to Wannier90 hr.dat file
            units: lattice unit vectors (default: identity)
            orbital_positions: orbital positions (default: origin)
            mu: chemical potential/Fermi level
            beta: inverse temperature (default: 40.0)
            n_max: maximum Matsubara frequency index (default: 100)
        """
        self.hr_file = hr_file
        self.hopp_dict, self.n_orb = parse_hopping_from_wannier90_hr_dat(hr_file)
        
        if units is None:
            units = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        if orbital_positions is None:
            orbital_positions = [(0, 0, 0)] * self.n_orb
            
        self.units = units
        self.orbital_positions = orbital_positions
        self.mu = mu
        self.beta = beta
        self.n_max = n_max
        
        # Create tight-binding lattice
        self.tb_lattice = TBLattice(
            units=units,
            hopping=self.hopp_dict,
            orbital_positions=orbital_positions,
            orbital_names=[str(i) for i in range(self.n_orb)],
        )
        
        self.e_k = None
        self.g0_wk = None
        self.chi00_wk = None
        self.chi00_wr = None
    
    def setup_kmesh(self, n_k=(15, 15, 15)):
        """Setup k-mesh and Fourier transform Hamiltonian"""
        kmesh = self.tb_lattice.get_kmesh(n_k=n_k)
        self.e_k = self.tb_lattice.fourier(kmesh)
        return kmesh
    
    def setup_frequency_mesh(self, beta=None, n_max=None):
        """Setup frequency mesh for Green's functions"""
        if beta is None:
            beta = self.beta
        if n_max is None:
            n_max = self.n_max
        self.wmesh = MeshImFreq(beta=beta, S='Fermion', n_max=n_max)
        return self.wmesh
    
    def compute_bare_susceptibility(self, beta=None, n_max=None, n_k=(15, 15, 15)):
        """Compute bare susceptibility chi00"""
        print("Setting up k-mesh and frequency mesh...")
        if self.e_k is None:
            self.setup_kmesh(n_k)
        if not hasattr(self, 'wmesh'):
            self.setup_frequency_mesh(beta, n_max)
        if self.mu is None:
            raise ValueError("Chemical potential mu must be set")
            
        print("Computing lattice Green's function...")
        # Compute Green's function
        self.g0_wk = lattice_dyson_g0_wk(mu=self.mu, e_k=self.e_k, mesh=self.wmesh)
        
        print("Computing bare susceptibility bubble...")
        # Compute bare susceptibility
        self.chi00_wk = imtime_bubble_chi0_wk(self.g0_wk, nw=1)
        
        print("Transforming bare susceptibility to real space...")
        # Transform to real space
        self.chi00_wr = chi_wr_from_chi_wk(self.chi00_wk)
        print("Bare susceptibility computation completed!")
        
        return self.chi00_wk, self.chi00_wr
    
    def compute_rpa_susceptibility(self, V_local):
        """Compute RPA susceptibility with local interactions"""
        if self.chi00_wk is None:
            raise ValueError("Must compute bare susceptibility first")
            
        # Solve RPA equation
        chi_rpa_wk = solve_rpa_PH(-self.chi00_wk, V_local)
        chi_rpa_wr = chi_wr_from_chi_wk(chi_rpa_wk)
        
        return chi_rpa_wk, chi_rpa_wr
    
    def compute_rpa_susceptibility_nonlocal(self, WcRPA_wr, threshold=1e-3):
        """Compute RPA susceptibility with non-local interactions"""
        if self.chi00_wk is None:
            raise ValueError("Must compute bare susceptibility first")
            
        print("Phase 1: Transforming non-local interactions to k-space...")
        # Transform non-local interactions to k-space
        WcRPA_wk = chi_wk_from_chi_wr(WcRPA_wr)
        
        print("Phase 2: Computing RPA susceptibility at each k-point...")
        # Manual RPA solution in k-space
        nw, nk = self.chi00_wk.data.shape[:2]
        chi_rpa_wk = self.chi00_wk.copy() * 0.0
        
        # Progress tracking
        try:
            from tqdm import tqdm
            k_iterator = tqdm(range(nk), desc="k-points", ncols=80)
        except ImportError:
            k_iterator = range(nk)
            print(f"Processing {nk} k-points...")
        
        for ik in k_iterator:
            if not hasattr(k_iterator, 'update'):  # no tqdm available
                if ik % max(1, nk // 10) == 0:  # print every 10%
                    print(f"  k-point {ik+1}/{nk} ({100*(ik+1)/nk:.1f}%)")
            
            chi0_inv = invert_tensor(self.chi00_wk.data[0, ik], threshold)
            chi_inv = -chi0_inv - WcRPA_wk.data[0, ik]
            chi_rpa_wk.data[0, ik] = invert_tensor(chi_inv, threshold)
        
        print("Phase 3: Back-transforming to real space...")
        chi_rpa_wr = chi_wr_from_chi_wk(chi_rpa_wk)
        print("RPA computation completed!")
        
        return chi_rpa_wk, chi_rpa_wr
    
    def get_kpath(self, high_sym_points, path_labels, num=200):
        """Get k-point path for band structure plotting"""
        paths = [(high_sym_points[path_labels[i]], high_sym_points[path_labels[i+1]]) 
                for i in range(len(path_labels)-1)]
        k_vecs, k_plot, k_ticks = k_space_path(paths, bz=self.tb_lattice.bz, num=num)
        return k_vecs, k_plot, k_ticks
    
    def compute_bands(self, k_vecs):
        """Compute bands along k-path"""
        if self.e_k is None:
            raise ValueError("Must setup k-mesh first")
        bands = np.array([np.linalg.eigvalsh(self.e_k(k)) for k in k_vecs])
        return bands
