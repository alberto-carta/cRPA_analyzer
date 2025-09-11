#%%
"""
Example script demonstrating the cRPA analyzer package
"""
import numpy as np
import matplotlib.pyplot as plt
from crpa_analyzer import ChiCalculator, InteractionParser, plot_bands, plot_chi_comparison
import tqdm

#%%
"""Main example function"""
material ="kcuf"
basis = "frontier"

# File paths (adjust as needed)
hr_file = f"./cRPA_{material}_{basis}/{material}.2_hr.dat"
w_file =  f"./cRPA_{material}_{basis}/dir-intW/dat.Wmat"
j_file =  f"./cRPA_{material}_{basis}/dir-intJ/dat.Jmat"

# Chemical potential
mu = 5.8787
# mu = 9.2173
beta = 20.0 # 40
nw = 120 #150


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
calc = ChiCalculator(hr_file, mu=mu, beta=beta, n_max=nw)

# Compute band structure
print("Computing band structure...")
k_vecs, k_plot, k_ticks = calc.get_kpath(high_sym, path_labels)
# calc.setup_kmesh(n_k=(10, 10, 10))
calc.setup_kmesh(n_k=(10, 10, 10))
bands = calc.compute_bands(k_vecs)

# Plot bands
plot_bands(k_plot, bands, k_ticks, path_tick_labels, mu=mu)

# Compute bare susceptibility
print("Computing bare susceptibility...")
chi00_wk, chi00_wr = calc.compute_bare_susceptibility()


chi00_trace = np.einsum('aabb', chi00_wr.data[0, 0, 0:5, 0:5, 0:5, 0:5] * 2)
print(f"Bare susceptibility trace: {chi00_trace:.6f}")


#%%
# Load interactions
print("Loading cRPA interactions...")
parser = InteractionParser(w_file, j_file)

# Compute RPA with non-local interactions
print("Computing non-local RPA susceptibility...")
# Rcut = 5.1 # kcuf
# Rcut = 4.0 # sfo
Rcut = 3.1 # sfo
WcRPA_wr = parser.load_nonlocal_interactions(calc.n_orb, r_cutoff=Rcut, 
                                           chi_wr_template=chi00_wr)
chi_nonlocal_wk, chi_nonlocal_wr = calc.compute_rpa_susceptibility_nonlocal(WcRPA_wr, threshold=1e-3)
#%%
# Plot susceptibilities

# Print local traces
chi00_trace = np.einsum('aabb', chi00_wr.data[0, 0, 0:5,0:5,0:5,0:5] * 2)
# chi_local_trace = np.einsum('aabb', chi_local_wr.data[0, 0] * 2)
chi_nonlocal_trace = np.einsum('aabb', chi_nonlocal_wr.data[0, 0, 0:5, 0:5, 0:5, 0:5] * 2)

print(f"Bare susceptibility trace: {chi00_trace:.6f}")
# print(f"Local RPA trace: {chi_local_trace:.6f}")
print(f"Non-local RPA trace: {chi_nonlocal_trace:.6f}")



# %%
chi00_oxygen = np.einsum('aabb', chi00_wr.data[0, 0, 5:8,5:8,5:8,5:8] * 2)
print(f"Bare susceptibility trace (Oxygen): {chi00_oxygen:.6f}")
chi_nonlocal_oxygen = np.einsum('aabb', chi_nonlocal_wr.data[0, 0, 5:8,5:8,5:8,5:8] * 2)
print(f"Non-local RPA trace (Oxygen): {chi_nonlocal_oxygen:.6f}")


# %%
