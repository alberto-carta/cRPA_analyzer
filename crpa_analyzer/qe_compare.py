import numpy as np

def hp_r_points_loops(nq1, nq2, nq3):
    """
    Generates lattice vectors using a loop structure similar to Fortran.
    Returns shape (3, nq1*nq2*nq3)
    This is translated from the Fortran code in Quantum ESPRESSO.
    in HP/src/hp_R_points.f90

    """
    nqsh = nq1 * nq2 * nq3
    if nqsh == 1:
        return np.zeros((3, 1), dtype=int)
    r_vect = np.zeros((3, nqsh))
    a1 = np.array([1,0,0])
    a2 = np.array([0,1,0])
    a3 = np.array([0,0,1])
    icell = 0
    for i in range(nq1):
        for j in range(nq2):
            for k in range(nq3):
                r_vect[:, icell] = i * a1 + j * a2 + k * a3 
                icell += 1
    return r_vect

def wrap_point(R, max_Rs):
    """
    Moves from tprf convention to QE convention by wrapping points 
    If supercell size is 5 5 5 then it does:
    0 0 0 -> 0 0 0
    1 0 0 -> 1 0 0
    2 0 0 -> 2 0 0
    4 0 0 -> -1 0 0
    3 0 0 -> -2 0 0
    """
    Reff = R.copy()
    for icoord in range(3):
        periodicity = max_Rs[icoord]
        half = periodicity //2
        Reff[icoord] = (R[icoord]+half)%periodicity-half
    Reff = [int(r) for r in Reff]
    return Reff

def get_chis_qe_convention(
    chi0r_isotropic, chidr_isotropic, n_atoms, qe_supercell=(3,3,3), crpa_supercell=(7,7,7), r_list_tprf=None,
    debug=False
    ):
    """
    Compare isotropic susceptibility and interaction matrices with Quantum ESPRESSO format.
    Returns: chi0_isotropic_qe, chid_isotropic_qe, U_qe, indices, rr_indices
    """
    if r_list_tprf is None:
        raise ValueError("r_list_tprf must be provided (list of R points from cRPA)")
    # Generate QE R points
    r_list_qe = hp_r_points_loops(*qe_supercell).T.tolist()
    r_list_wrapped = [wrap_point(R, qe_supercell) for R in r_list_qe]

    # tprf R points are read from RESPACK cRPA calculation
    r_list_tprf_wrapped = [wrap_point(R, crpa_supercell) for R in r_list_tprf]

    # match R points between QE and TPRF
    indices = [r_list_tprf_wrapped.index(R) for R in r_list_wrapped]

    # build a matrix of relative R points in QE convention
    rr_matrix = [[wrap_point(np.array(R2) - np.array(R1), qe_supercell) for R2 in r_list_wrapped] for R1 in r_list_wrapped]
    rr_indices = [[indices[r_list_wrapped.index(rr_matrix[i][j])] for j in range(len(r_list_wrapped))] for i in range(len(r_list_wrapped))]

    n_R = len(indices)
    n_total = n_R * n_atoms
    # Create matrices directly in flattened format (n_total, n_total)
    chid_isotropic_qe = np.zeros((n_total, n_total), dtype=complex)
    chi0_isotropic_qe = np.zeros((n_total, n_total), dtype=complex)
    # for i in range(len(indices)):
    #     for j in range(len(indices)):
    #         for iatom in range(n_atoms):
    #             for jatom in range(n_atoms):
    #                 flat_i = i * n_atoms + iatom
    #                 flat_j = j * n_atoms + jatom
    #                 if debug:
    #                     print(f"cell {i} atom {iatom} to cell {j} atom {jatom} -> flat {flat_i} to flat {flat_j}")
    #                 chid_isotropic_qe[flat_i, flat_j] = chidr_isotropic[rr_indices[i][j], iatom, jatom]
    #                 chi0_isotropic_qe[flat_i, flat_j] = chi0r_isotropic[rr_indices[i][j], iatom, jatom]
    
    chid_blocks = [[chidr_isotropic[rr_indices[i][j], :, :].T for j in range(len(indices))] for i in range(len(indices))]
    chi0_blocks = [[chi0r_isotropic[rr_indices[i][j], :, :].T for j in range(len(indices))] for i in range(len(indices))]
    chid_isotropic_qe = np.block(chid_blocks)
    chi0_isotropic_qe = np.block(chi0_blocks)

    return chi0_isotropic_qe, chid_isotropic_qe, rr_indices
