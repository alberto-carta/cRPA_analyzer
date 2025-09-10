"""
Quick test of the cRPA analyzer package
"""
import numpy as np
import sys
import os

# Add package to path for testing
sys.path.insert(0, '/home/acarta/Documents_shared/Projects/Calculations/U_calculation/test_idea_bridging/TPRF/cRPA_analyzer')

def test_import():
    """Test basic imports"""
    try:
        from crpa_analyzer import ChiCalculator, InteractionParser
        from crpa_analyzer.utils import chi_charge_contraction, invert_tensor
        from crpa_analyzer.plotting import plot_bands
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_parse_wmat():
    """Test interaction matrix parsing"""
    try:
        from crpa_analyzer.parse_Wmat_k import list_triplets, parse_wmat_block
        
        # Test with dummy data
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
        print("✓ Wmat parsing works")
        return True
    except Exception as e:
        print(f"✗ Wmat parsing failed: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    try:
        from crpa_analyzer.utils import invert_tensor
        
        # Test tensor inversion
        test_tensor = np.random.random((2, 2, 2, 2)) + 1e-3 * np.eye(2)[:, :, None, None]
        inv_tensor = invert_tensor(test_tensor)
        
        print("✓ Utilities work")
        return True
    except Exception as e:
        print(f"✗ Utilities failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing cRPA analyzer package...")
    print("=" * 40)
    
    tests = [test_import, test_parse_wmat, test_utilities]
    results = [test() for test in tests]
    
    print("=" * 40)
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
