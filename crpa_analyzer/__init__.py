"""
cRPA Analyzer Package
Tools for analyzing cRPA calculations and susceptibilities
"""

from .chi_calculator import ChiCalculator
from .interaction_parser import InteractionParser
from .plotting import plot_bands, plot_chi_1D, plot_chi_comparison
from .utils import chi_charge_contraction, chi_magnetic_contraction

__version__ = "0.1.0"
__all__ = [
    "ChiCalculator", 
    "InteractionParser", 
    "plot_bands", 
    "plot_chi_1D",
    "plot_chi_comparison",
    "chi_charge_contraction",
    "chi_magnetic_contraction"
]
