#%%
"""
Example script demonstrating the cRPA analyzer package
"""
import numpy as np
import matplotlib.pyplot as plt
from crpa_analyzer import ChiCalculator, InteractionParser, plot_bands, plot_chi_comparison

#%%
"""Main example function"""
material ="sfo"

# File paths (adjust as needed)
hr_file = f"./cRPA_{material}_frontier/{material}.2_hr.dat"
w_file =  f"./cRPA_{material}_frontier/dir-intW/dat.Wmat"
j_file =  f"./cRPA_{material}_frontier/dir-intJ/dat.Jmat"

# Chemical potential
# mu = 5.8787
mu = 9.2173


# High symmetry points
high_sym = {
    "G": [0.0, 0.0, 0.0],
    "N": [0.0, 0.5, 0.0], 
    "X": [0.0, 0.0, 0.5],
    "M": [0.5, 0.5, 0.5],
}
path_labels = ['G', 'N', 'X', 'G', 'M']
path_tick_labels = [r'$\Gamma$', r'$N$', r'$X$', r'$\Gamma$', r'$M$']

# Initialize calculator
print("Setting up tight-binding model...")
calc = ChiCalculator(hr_file, mu=mu)

# Compute band structure
print("Computing band structure...")
k_vecs, k_plot, k_ticks = calc.get_kpath(high_sym, path_labels)
calc.setup_kmesh(n_k=(15, 15, 15))
bands = calc.compute_bands(k_vecs)

# Plot bands
plot_bands(k_plot, bands, k_ticks, path_tick_labels, mu=mu)

# Compute bare susceptibility
print("Computing bare susceptibility...")
chi00_wk, chi00_wr = calc.compute_bare_susceptibility()


chi00_trace = np.einsum('aabb', chi00_wr.data[0, 0] * 2)
print(f"Bare susceptibility trace: {chi00_trace:.6f}")

# Load interactions
print("Loading cRPA interactions...")
parser = InteractionParser(w_file, j_file)
V_local = parser.load_local_interactions(calc.n_orb)

# Compute RPA susceptibility with local interactions
print("Computing local RPA susceptibility...")
chi_local_wk, chi_local_wr = calc.compute_rpa_susceptibility(V_local)

#%%
# Compute RPA with non-local interactions
print("Computing non-local RPA susceptibility...")
Rcut = 3.1
WcRPA_wr = parser.load_nonlocal_interactions(calc.n_orb, r_cutoff=Rcut, 
                                           chi_wr_template=chi00_wr)
chi_nonlocal_wk, chi_nonlocal_wr = calc.compute_rpa_susceptibility_nonlocal(WcRPA_wr, threshold=1e-2)
#%%
# Plot susceptibilities

# Print local traces
chi00_trace = np.einsum('aabb', chi00_wr.data[0, 0] * 2)
chi_local_trace = np.einsum('aabb', chi_local_wr.data[0, 0] * 2)
chi_nonlocal_trace = np.einsum('aabb', chi_nonlocal_wr.data[0, 0] * 2)

print(f"Bare susceptibility trace: {chi00_trace:.6f}")
print(f"Local RPA trace: {chi_local_trace:.6f}")
print(f"Non-local RPA trace: {chi_nonlocal_trace:.6f}")



# %%
