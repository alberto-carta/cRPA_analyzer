import numpy as np
from typing import Iterable, List, Optional, Tuple


def _is_triplet_line(line: str) -> Optional[Tuple[int, int, int]]:
    """Return (a, b, c) if the line consists of exactly three integers, else None."""
    s = line.strip()
    if not s or s.startswith('#'):
        return None
    parts = s.split()
    if len(parts) != 3:
        return None
    try:
        a, b, c = (int(p) for p in parts)
        return a, b, c
    except ValueError:
        return None


def list_triplets(filename: str) -> List[Tuple[int, int, int]]:
    """
    Scan a W/J-matrix file and list all available 3-integer triplet headers.

    Returns a list of (t1, t2, t3) tuples in file order (may contain duplicates if present).
    """
    triplets: List[Tuple[int, int, int]] = []
    with open(filename, 'r') as f:
        for line in f:
            t = _is_triplet_line(line)
            if t is not None:
                triplets.append(t)
    return triplets


def parse_wmat_block(
    filename: str,
    *,
    triplet: Iterable[int] = (0, 0, 0),
    orbmax: Optional[int] = None,
) -> np.ndarray:
    """
    Parse a single block (for a given 3-integer triplet) from a W/J matrix file.

    Parameters:
    - filename: path to dat.Wmat or dat.Jmat
    - triplet: 3-integer header to match, e.g., (0, 0, 0) or (-3, -3, -3)
    - orbmax: optional. If None, inferred from the first block line by max(i,j) across the block.

    Returns:
    - matrix: shape (orbmax, orbmax, 2), where last dim is [real, imag]

    Raises:
    - ValueError if the triplet is not found or if the file layout is inconsistent.
    """
    t_req = tuple(int(x) for x in triplet)

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the start index of the requested triplet
    start_idx: Optional[int] = None
    for i, line in enumerate(lines):
        t = _is_triplet_line(line)
        if t is not None and t == t_req:
            start_idx = i + 1
            break

    if start_idx is None:
        available = list_triplets(filename)
        raise ValueError(
            f"Triplet {t_req} not found in {filename}. Available: {available[:10]}" +
            (" ..." if len(available) > 10 else "")
        )

    # Collect subsequent data lines until we have orbmax**2 entries (or infer orbmax)
    data: List[Tuple[int, int, float, float]] = []
    for line in lines[start_idx:]:
        s = line.strip()
        if not s or s.startswith('#'):
            # stop at next header if encountered before finishing
            t_next = _is_triplet_line(line)
            if t_next is not None and len(data) > 0:
                break
            continue
        parts = s.split()
        if len(parts) < 4:
            # Likely the start of next block, bail out if we already collected some
            t_next = _is_triplet_line(line)
            if t_next is not None and len(data) > 0:
                break
            # Otherwise skip
            continue
        try:
            i_idx = int(parts[0])
            j_idx = int(parts[1])
            re_val = float(parts[2])
            im_val = float(parts[3])
            data.append((i_idx, j_idx, re_val, im_val))
        except ValueError:
            # Non-data line; if it's another triplet and we already have data, stop
            t_next = _is_triplet_line(line)
            if t_next is not None and len(data) > 0:
                break
            # else ignore
            continue

        # If orbmax is known, we can stop early when we've read enough
        if orbmax is not None and len(data) >= orbmax * orbmax:
            break

    if not data:
        raise ValueError(f"No data lines found after triplet {t_req} in {filename}")

    # Infer orbmax if not provided
    if orbmax is None:
        max_idx = 0
        for i_idx, j_idx, *_ in data:
            if i_idx > max_idx:
                max_idx = i_idx
            if j_idx > max_idx:
                max_idx = j_idx
        orbmax = int(max_idx)

        # Ensure we have enough lines; if not, try to continue reading more
        needed = orbmax * orbmax
        if len(data) < needed:
            # Try to continue from where we left off
            cursor = start_idx + len(data)
            for line in lines[cursor:]:
                s = line.strip()
                if not s or s.startswith('#'):
                    t_next = _is_triplet_line(line)
                    if t_next is not None and len(data) > 0:
                        break
                    continue
                parts = s.split()
                if len(parts) < 4:
                    t_next = _is_triplet_line(line)
                    if t_next is not None and len(data) > 0:
                        break
                    continue
                try:
                    i_idx = int(parts[0])
                    j_idx = int(parts[1])
                    re_val = float(parts[2])
                    im_val = float(parts[3])
                    data.append((i_idx, j_idx, re_val, im_val))
                except ValueError:
                    t_next = _is_triplet_line(line)
                    if t_next is not None and len(data) > 0:
                        break
                    continue
                if len(data) >= needed:
                    break

    # Final sanity check
    needed = orbmax * orbmax
    if len(data) < needed:
        raise ValueError(
            f"Incomplete block: got {len(data)} entries but need {needed} for orbmax={orbmax}. "
            f"Triplet={t_req} file={filename}"
        )

    # Build the matrix (1-indexed i,j in files)
    mat = np.zeros((orbmax, orbmax, 2), dtype=float)
    for i_idx, j_idx, re_val, im_val in data[:needed]:
        mat[i_idx - 1, j_idx - 1, 0] = re_val
        mat[i_idx - 1, j_idx - 1, 1] = im_val

    return mat


def parse_wmat_block_real(
    filename: str, *, triplet: Iterable[int] = (0, 0, 0), orbmax: Optional[int] = None
) -> np.ndarray:
    """Convenience wrapper that returns only the real part matrix of shape (orbmax, orbmax)."""
    m = parse_wmat_block(filename, triplet=triplet, orbmax=orbmax)
    return m[..., 0]


def generate_dual_q_grid(grid_dims):
    """
    Generates a Gamma-centered reciprocal-space grid dual to a real-space grid.

    Args:
        grid_dims (tuple of int): The dimensions of the R-grid, e.g., (7, 7, 7).

    Returns:
        list of np.ndarray: A list of q-vectors in fractional coordinates.
    """
    N1, N2, N3 = grid_dims
    
    # Create the fractional coordinates along each axis
    q1_coords = np.arange(N1) / N1
    q2_coords = np.arange(N2) / N2
    q3_coords = np.arange(N3) / N3
    
    # Use meshgrid to create all combinations
    q1_grid, q2_grid, q3_grid = np.meshgrid(q1_coords, q2_coords, q3_coords, indexing='ij')
    
    # Stack and reshape to get a list of (q1, q2, q3) vectors
    q_list = np.stack([q1_grid.ravel(), q2_grid.ravel(), q3_grid.ravel()], axis=1)
    
    return [np.array(q) for q in q_list]


# def extract_kpoints_from_nnkp(filename: str) -> List[Tuple[float, float, float]]:
#     """
#     Extract k-points from a Wannier90 .nnkp file.
    
#     Parameters:
#     - filename: path to the .nnkp file (e.g., 'nio.nnkp')
    
#     Returns:
#     - List of k-points as (kx, ky, kz) tuples in fractional coordinates
    
#     The function looks for the 'begin kpoints' section and extracts all k-point
#     coordinates until 'end kpoints'.
#     """
#     kpoints: List[Tuple[float, float, float]] = []
    
#     with open(filename, 'r') as f:
#         lines = f.readlines()
    
#     # Find the start of kpoints section
#     start_idx = None
#     for i, line in enumerate(lines):
#         if line.strip() == 'begin kpoints':
#             start_idx = i + 1
#             break
    
#     if start_idx is None:
#         raise ValueError(f"'begin kpoints' section not found in {filename}")
    
#     # The first line after 'begin kpoints' should contain the number of k-points
#     if start_idx >= len(lines):
#         raise ValueError(f"Unexpected end of file after 'begin kpoints' in {filename}")
    
#     try:
#         num_kpoints = int(lines[start_idx].strip())
#     except ValueError:
#         raise ValueError(f"Could not parse number of k-points from line: {lines[start_idx].strip()}")
    
#     # Read the k-points
#     for i in range(start_idx + 1, min(start_idx + 1 + num_kpoints, len(lines))):
#         line = lines[i].strip()
        
#         # Check for end of kpoints section
#         if line == 'end kpoints':
#             break
            
#         # Skip empty lines and comments
#         if not line or line.startswith('#'):
#             continue
            
#         # Parse k-point coordinates
#         parts = line.split()
#         if len(parts) >= 3:
#             try:
#                 kx = float(parts[0])
#                 ky = float(parts[1])
#                 kz = float(parts[2])
#                 kpoints.append((kx, ky, kz))
#             except ValueError:
#                 continue  # Skip lines that can't be parsed as floats
    
#     if len(kpoints) != num_kpoints:
#         print(f"Warning: Expected {num_kpoints} k-points but found {len(kpoints)}")
    
#     return kpoints


def forward_transform(W_r_list, R_list, q_list):
    """
    Performs a Fourier transform on a list of 4-index tensors.
    
    Parameters:
    - W_r_list: List of tensors W(R) in real space
    - R_list: List of R vectors (lattice vectors)
    - q_list: List of q vectors (k-points) in reciprocal space
    
    Returns:
    - W_q_list: List of tensors W(q) in reciprocal space
    """
    tensor_shape = W_r_list[0].shape
    n_q = len(q_list)
    
    # Initialize output array for tensors at each q-point
    W_q_list = np.zeros((n_q, *tensor_shape), dtype=complex)

    for iq, q in enumerate(q_list):
        W_q = np.zeros(tensor_shape, dtype=complex)
        for W_R, R in zip(W_r_list, R_list):
            # Phase factor for Fourier transform
            phase = np.exp(-2j * np.pi * np.dot(q, R))
            W_q += W_R * phase
        
        W_q_list[iq] = W_q

    return W_q_list


def backward_transform(W_q_list, q_list, R_list):
    """
    Performs an inverse Fourier transform on a list of 4-index tensors.
    
    Parameters:
    - W_q_list: List of tensors W(q) in reciprocal space
    - q_list: List of q vectors (k-points) in reciprocal space
    - R_list: List of R vectors (lattice vectors)
    
    Returns:
    - W_r_list: List of tensors W(R) in real space
    """
    tensor_shape = W_q_list[0].shape
    n_r = len(R_list)
    n_q = len(q_list)
    
    W_r_list = np.zeros((n_r, *tensor_shape), dtype=complex)

    for ir, R in enumerate(R_list):
        W_R = np.zeros(tensor_shape, dtype=complex)
        for W_q, q in zip(W_q_list, q_list):
            # Inverse transform uses a positive sign in the exponent
            phase = np.exp(2j * np.pi * np.dot(q, R))
            W_R += W_q * phase
        
        # Normalize by the number of q-points for the inverse transform
        W_r_list[ir] = W_R / n_q

    return W_r_list


def create_W_tensor(U_matrices, J_matrices, norb):
    """
    Create 4-component interaction tensors from U and J matrices.
    
    Parameters:
    - U_matrices: Array of U matrices [n_k, norb, norb]
    - J_matrices: Array of J matrices [n_k, norb, norb] (typically only on-site)
    - norb: Number of orbitals
    
    Returns:
    - W_tensors: 4-index interaction tensors [n_k, norb, norb, norb, norb]
                 where W[k,i,i,j,j] = U[k,i,j] and W[0,i,j,j,i] = J[0,i,j]
    """
    n_k = len(U_matrices)
    W_tensors = np.zeros((n_k, norb, norb, norb, norb), dtype=complex)
    
    for k in range(n_k):
        for i in range(norb):
            for j in range(norb):
                # Density-density interaction: W_iijj = U_ij
                W_tensors[k, i, i, j, j] = U_matrices[k, i, j]
                
                # Exchange interaction: W_ijji = J_ij (only for k=0, on-site)
                if k == 0:
                    W_tensors[0, i, j, j, i] = J_matrices[0, i, j]
    
    return W_tensors


#%%

#%% Testing and demonstration
if __name__ == "__main__":
    print("Testing parse_wmat_k functions with arbitrary triplets (k-points)")
    print("=" * 60)
    
    # Test k-point extraction from nnkp file
    print("\n0. Testing k-point extraction from nio.nnkp file:")
    try:
        nnkp_file = './dp/nio.nnkp'
        kpoints = extract_kpoints_from_nnkp(nnkp_file)
        print(f"Found {len(kpoints)} k-points in {nnkp_file}")
        print("First 10 k-points:")
        for i, kpt in enumerate(kpoints[:10]):
            print(f"  {i+1:3d}: ({kpt[0]:8.5f}, {kpt[1]:8.5f}, {kpt[2]:8.5f})")
        if len(kpoints) > 10:
            print(f"  ... and {len(kpoints) - 10} more")
    except Exception as e:
        print(f"Error reading k-points from nnkp file: {e}")
    
    # Test with dp directory files
    try:
        filename_U = './dp/dir-intW/dat.Wmat'
        filename_J = './dp/dir-intJ/dat.Jmat'
        
        print(f"\n1. Listing available triplets in {filename_U}:")
        triplets_U = list_triplets(filename_U)
        print(f"Found {len(triplets_U)} triplets:")
        for i, triplet in enumerate(triplets_U[:10]):  # Show first 10
            print(f"  {i+1}: {triplet}")
        if len(triplets_U) > 10:
            print(f"  ... and {len(triplets_U) - 10} more")
        
        print(f"\n2. Testing with default triplet (0, 0, 0):")
        U_matrix = parse_wmat_block(filename_U, triplet=(0, 0, 0), orbmax=16)
        J_matrix = parse_wmat_block(filename_J, triplet=(0, 0, 0), orbmax=16)
        
        print(f"U matrix shape: {U_matrix.shape}")
        print(f"J matrix shape: {J_matrix.shape}")
        
        # Print the result, only the real part for the first 5x5 matrix
        l_d = 2
        with np.printoptions(precision=3, suppress=True):
            print("\nU matrix (first 5x5, real part):")
            print(U_matrix[:5, :5, 0])
            print("\nJ matrix (first 5x5, real part):")
            print(J_matrix[:5, :5, 0])

        U = np.mean(U_matrix[:5, :5, 0])
        U_minus_J = np.sum(U_matrix[:5, :5, 0] - J_matrix[:5, :5, 0])/((2*l_d+1)*(2*l_d))
        
        print(f"\nU = {U:.3f} eV")
        print(f"U - J = {U_minus_J:.3f} eV")
        
        # Test with different triplets if available
        if len(triplets_U) > 1:
            test_triplet = triplets_U[1]  # Use second triplet
            print(f"\n3. Testing with arbitrary triplet {test_triplet}:")
            U_matrix_arb = parse_wmat_block(filename_U, triplet=test_triplet, orbmax=16)
            print(f"Successfully parsed matrix for triplet: {test_triplet}")
            print(f"Matrix shape: {U_matrix_arb.shape}")
            
            with np.printoptions(precision=3, suppress=True):
                print("U matrix (first 3x3, real part) for this triplet:")
                print(U_matrix_arb[:3, :3, 0])
                
            # Compare with (0,0,0) triplet
            diff_norm = np.linalg.norm(U_matrix_arb[:5, :5, 0] - U_matrix[:5, :5, 0])
            print(f"Difference norm from (0,0,0) triplet: {diff_norm:.6f}")
        
        # Test with convenience function for real part only
        print(f"\n4. Testing convenience function parse_wmat_block_real:")
        U_real = parse_wmat_block_real(filename_U, triplet=(0, 0, 0), orbmax=16)
        print(f"Real part matrix shape: {U_real.shape}")
        print("First 3x3 elements:")
        with np.printoptions(precision=3, suppress=True):
            print(U_real[:3, :3])

    except Exception as e:
        print(f"Error testing dp directory: {e}")
        import traceback
        traceback.print_exc()

    # Test with donly directory
    print("\n" + "="*60)
    print("Testing with donly directory:")

    try:
        filename_U = './donly/try/dir-intW/dat.Wmat'
        filename_J = './donly/try/dir-intJ/dat.Jmat'
        
        print(f"\nListing available triplets in {filename_U}:")
        triplets_U = list_triplets(filename_U)
        print(f"Found {len(triplets_U)} triplets:")
        for i, triplet in enumerate(triplets_U[:10]):  # Show first 10
            print(f"  {i+1}: {triplet}")
        
        # Test with default triplet (0, 0, 0)
        U_matrix = parse_wmat_block(filename_U, triplet=(0, 0, 0), orbmax=10)
        J_matrix = parse_wmat_block(filename_J, triplet=(0, 0, 0), orbmax=10)
        
        # Print the result
        l_d = 2
        with np.printoptions(precision=3, suppress=True):
            print("\nU matrix (first 5x5, real part):")
            print(U_matrix[:5, :5, 0])
            print("\nJ matrix (first 5x5, real part):")
            print(J_matrix[:5, :5, 0])

        U = np.mean(U_matrix[:5, :5, 0])
        U_minus_J = np.sum(U_matrix[:5, :5, 0] - J_matrix[:5, :5, 0])/((2*l_d+1)*(2*l_d))
        
        print(f"\nU = {U:.3f} eV")
        print(f"U - J = {U_minus_J:.3f} eV")

    except Exception as e:
        print(f"Error testing donly directory: {e}")
        import traceback
        traceback.print_exc()

    # Demonstrate parsing multiple triplets
    print("\n" + "="*60)
    print("Testing multiple arbitrary triplets:")

    try:
        filename_U = './dp/dir-intW/dat.Wmat'
        triplets = list_triplets(filename_U)
        
        # Test first 3 triplets (or all if less than 3)
        test_triplets = triplets[:min(3, len(triplets))]
        
        for i, triplet in enumerate(test_triplets):
            print(f"\nTriplet {i+1}: {triplet}")
            try:
                U_matrix = parse_wmat_block(filename_U, triplet=triplet, orbmax=16)
                print(f"  Shape: {U_matrix.shape}")
                print(f"  Sample value at [0,0]: {U_matrix[0, 0, 0]:.6f} + {U_matrix[0, 0, 1]:.6f}i")
                print(f"  Max real value: {np.max(U_matrix[:, :, 0]):.6f}")
                print(f"  Max imag value: {np.max(np.abs(U_matrix[:, :, 1])):.6f}")
            except Exception as e:
                print(f"  Failed: {e}")

    except Exception as e:
        print(f"Error in multiple triplets test: {e}")

    print("\n" + "="*60)
    print("Testing completed!")
    print("Use parse_wmat_block(filename, triplet=(a,b,c), orbmax=N) for arbitrary triplets")
    print("Use list_triplets(filename) to see all available triplets in a file")


def forward_transform(W_r_list, R_list, q_list):
    """
    Performs a Fourier transform on a list of 4-index tensors from R-space to k-space.
    
    Parameters:
    W_r_list: List of 4-index tensors in real space
    R_list: List of R-vectors corresponding to the tensors
    q_list: List of q-points for the Fourier transform
    
    Returns:
    W_q_list: List of 4-index tensors in k-space
    """
    tensor_shape = W_r_list[0].shape
    n_q = len(q_list)
    
    # Initialize output array for tensors at each q-point
    W_q_list = np.zeros((n_q, *tensor_shape), dtype=complex)

    for iq, q in enumerate(q_list):
        W_q = np.zeros(tensor_shape, dtype=complex)
        for W_R, R in zip(W_r_list, R_list):
            # Phase factor for Fourier transform
            phase = np.exp(-2j * np.pi * np.dot(q, R))
            W_q += W_R * phase
        
        W_q_list[iq] = W_q

    return W_q_list


def backward_transform(W_q_list, q_list, R_list):
    """
    Performs an inverse Fourier transform on a list of 4-index tensors from k-space to R-space.
    
    Parameters:
    W_q_list: List of 4-index tensors in k-space
    q_list: List of q-points corresponding to the tensors
    R_list: List of R-vectors for the inverse transform
    
    Returns:
    W_r_list: List of 4-index tensors in real space
    """
    tensor_shape = W_q_list[0].shape
    n_r = len(R_list)
    n_q = len(q_list)
    
    W_r_list = np.zeros((n_r, *tensor_shape), dtype=complex)

    for ir, R in enumerate(R_list):
        W_R = np.zeros(tensor_shape, dtype=complex)
        for W_q, q in zip(W_q_list, q_list):
            # Inverse transform uses a positive sign in the exponent
            phase = np.exp(2j * np.pi * np.dot(q, R))
            W_R += W_q * phase
        
        # Normalize by the number of q-points for the inverse transform
        W_r_list[ir] = W_R / n_q

    return W_r_list
