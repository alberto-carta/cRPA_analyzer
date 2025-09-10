# cRPA_analyzer

Tools for analyzing cRPA calculations and computing susceptibilities.

## Installation

```bash
pip install -e .
```

## Usage

### Basic Example

```python
from crpa_analyzer import ChiCalculator, InteractionParser

# Initialize calculator
calc = ChiCalculator("kcuf.2_hr.dat", mu=5.8787)

# Compute bare susceptibility  
chi00_wk, chi00_wr = calc.compute_bare_susceptibility()

# Load cRPA interactions
parser = InteractionParser("dir-intW/dat.Wmat", "dir-intJ/dat.Jmat")
V_local = parser.load_local_interactions(calc.n_orb)

# Compute RPA susceptibility
chi_rpa_wk, chi_rpa_wr = calc.compute_rpa_susceptibility(V_local)
```

### Scripts

- `scripts/extract_parameters.py`: Extract Hubbard U and J parameters
- `examples/kcuf_example.py`: Complete workflow example

## Modules

- `ChiCalculator`: Main susceptibility calculator
- `InteractionParser`: Parse cRPA interaction matrices  
- `plotting`: Visualization utilities
- `utils`: Helper functions
