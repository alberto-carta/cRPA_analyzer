"""
Plotting utilities for bands and susceptibilities
"""
import numpy as np
import matplotlib.pyplot as plt
from .utils import chi_charge_contraction, chi_magnetic_contraction, interpolate_chi


def plot_bands(k_plot, bands, k_ticks, tick_labels, mu=None, title="Band Structure"):
    """Plot band structure along k-path"""
    plt.figure(figsize=(7, 5))
    plt.plot(k_plot, bands, color='k', lw=1)
    
    if mu is not None:
        plt.axhline(y=mu, color='r', linestyle='--', label='Fermi level')
        plt.legend()
    
    plt.xticks(k_ticks, tick_labels)
    plt.xlabel("k-path")
    plt.ylabel("Energy (eV)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_chi_1D(chi, k_vecs, k_plot, k_ticks, tick_labels, label=None, 
                channel='charge', spin_factor=2.0):
    """Plot 1D susceptibility along k-path"""
    if channel == 'charge':
        chi_contracted = chi_charge_contraction(chi, spin_factor)
        title = r'Charge susceptibility $\chi_{charge}(\mathbf{q}, \omega=0)$'
    elif channel == 'magnetic':
        chi_contracted = chi_magnetic_contraction(chi, spin_factor)
        title = r'Magnetic susceptibility $\chi_{S_z S_z}(\mathbf{q}, \omega=0)$'
    else:
        raise ValueError("Channel must be 'charge' or 'magnetic'")
        
    chi_interp = interpolate_chi(chi_contracted, k_vecs)
    plt.plot(k_plot, chi_interp.real, label=label)
    plt.grid()
    plt.xticks(ticks=k_ticks, labels=tick_labels)
    plt.title(title)


def plot_chi_comparison(chi_bare, chi_dressed, k_vecs, k_plot, k_ticks, tick_labels,
                       channels=['charge', 'magnetic'], spin_factor=2.0):
    """Plot comparison of bare vs dressed susceptibilities"""
    nchannels = len(channels)
    fig, axes = plt.subplots(nchannels, 1, figsize=(8, 5*nchannels))
    if nchannels == 1:
        axes = [axes]
    
    for i, channel in enumerate(channels):
        plt.sca(axes[i])
        plot_chi_1D(chi_bare, k_vecs, k_plot, k_ticks, tick_labels, 
                   label=r'$U=0$ (bare)', channel=channel, spin_factor=spin_factor)
        plot_chi_1D(chi_dressed, k_vecs, k_plot, k_ticks, tick_labels,
                   label=r'cRPA dressed', channel=channel, spin_factor=spin_factor)
        plt.ylabel(f'$\\chi_{{{channel}}}$')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
