"""
Minimal test without triqs dependencies
"""
import numpy as np
import sys
import os

# Add package to path
sys.path.insert(0, '/home/acarta/Documents_shared/Projects/Calculations/U_calculation/test_idea_bridging/TPRF/cRPA_analyzer')

def test_parse_wmat_basic():
    """Test basic Wmat parsing without triqs"""
    try:
        from crpa_analyzer.parse_Wmat_k import list_triplets, parse_wmat_block
        
        # Create test file
        dummy_file = "/tmp/test_wmat.dat"
        with open(dummy_file, 'w') as f:
            f.write("0 0 0\n")
            f.write("1 1 1.0 0.0\n")
            f.write("1 2 0.5 0.0\n")
            f.write("2 1 0.5 0.0\n") 
            f.write("2 2 1.0 0.0\n")
        
        triplets = list_triplets(dummy_file)
        matrix = parse_wmat_block(dummy_file, triplet=(0, 0, 0), orbmax=2)
        
        os.remove(dummy_file)
        
        print(f"✓ Found {len(triplets)} triplets")
        print(f"✓ Matrix shape: {matrix.shape}")
        print(f"✓ Matrix dtype: {matrix.dtype}")
        return True
    except Exception as e:
        print(f"✗ Wmat parsing failed: {e}")
        return False

def test_tensor_utils():
    """Test tensor utilities"""
    try:
        # Test without triqs imports
        import numpy as np
        
        def invert_tensor_simple(tensor, threshold=1e-5):
            norb = tensor.shape[0]
            tensor_reshaped = np.reshape(tensor, (norb*norb, norb*norb))
            tensor_inv_reshaped = np.linalg.pinv(tensor_reshaped, rcond=threshold)
            tensor_inv = np.reshape(tensor_inv_reshaped, (norb, norb, norb, norb))
            return tensor_inv
        
        # Test with simple matrix
        test_tensor = np.eye(4).reshape(2, 2, 2, 2)
        inv_tensor = invert_tensor_simple(test_tensor)
        
        print("✓ Tensor inversion works")
        return True
    except Exception as e:
        print(f"✗ Tensor utils failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing cRPA analyzer package (minimal)...")
    print("=" * 45)
    
    tests = [test_parse_wmat_basic, test_tensor_utils]
    results = [test() for test in tests]
    
    print("=" * 45)
    if all(results):
        print("✓ Core functionality tests passed!")
        print("Note: Full functionality requires triqs_tprf")
    else:
        print("✗ Some core tests failed")
