"""PTMC-GPU: GPU-accelerated Parallel-Tempering / Population-Annealing
Monte Carlo for protein adsorption-orientation prediction on solid surfaces.

Units (GROMACS convention): length nm, energy kJ/mol, charge e, temperature K.
Numerical core: JAX (jit + lax.scan), float32. Batch axis = number of chains.
"""
__version__ = "0.0.0"
