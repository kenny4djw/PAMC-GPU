# Units and Non-Bonded Interaction Conventions

## Base units (GROMACS convention)

| Quantity   | Unit     |
|------------|----------|
| length     | nm       |
| energy     | kJ/mol   |
| charge     | e        |
| temperature| K        |
| mass       | amu      |

## Physical constants

| Symbol   | Value (GROMACS units)          | Description                |
|----------|--------------------------------|----------------------------|
| k_B      | 8.314462618 × 10⁻³           | Boltzmann constant (kJ/mol/K) |
| f        | 138.935458                    | Coulomb factor 1/(4πε₀) (kJ/mol·nm/e²) |

See `ptmc.config` for the authoritative definitions (`BOLTZMANN_KJ_PER_MOL_K`,
`COULOMB_FACTOR_KJ_NM_PER_E2`).

## Lennard-Jones

### Geometric combination (legacy, comb-rule 1/3)

```
C6_ij  = sqrt(C6_i  × C6_j)
C12_ij = sqrt(C12_i × C12_j)
```

Used by the reference (direct-sum) energy and when the topology file
specifies C6/C12 columns directly.

### Lorentz-Berthelot (comb-rule 2)

```
σ_ij    = (σ_i + σ_j) / 2
eps_ij  = sqrt(eps_i × eps_j)

C6_ij   = 4 × eps_ij × σ_ij⁶
C12_ij  = 4 × eps_ij × σ_ij¹²
```

Used by the AMBER99SB-ILDN force field. The `parse_topology` module
handles this conversion from epsilon-sigma to C6/C12 internally.

## Electrostatics

### Screened Coulomb (Debye-Hückel)

```
U_Coul(r) = f × q_i × q_j / r × exp(-r / λ_D)
```

where λ_D is the Debye screening length.

**Vacuum Coulomb is forbidden** for the physical model — all
electrostatics must be screened.

### Debye length

For a 1:1 electrolyte in water at 25 °C (Israelachvili):

```
λ_D [nm] = 0.304 / sqrt(I [mol/L])
```

Default ionic strength: 0.15 M → λ_D ≈ 0.785 nm.

### Continuum surface electrostatics

For the `continuum` and `patterned` surface types, the electrostatic
potential follows linearized Poisson-Boltzmann:

```
ψ(z) = ψ₀ × exp(-z / λ_D)
```

where ψ₀ is the surface potential at z = 0 (in kJ/mol/e).

## Steele 9-3 potential

For the homogeneous continuum surface (vdW part):

```
U_93(z) = 2π × ρ_s × [ (2/15) × C12 × z⁻⁹ - (1/3) × C6 × z⁻³ ]
```

where ρ_s is the surface atom number density (nm⁻³).

For the ε-σ path:

```
C6  = 4 × ε × σ⁶
C12 = 4 × ε × σ¹²
```

## Grid factorization

The energy field U(x, y, z, q) is precomputed on a 3D grid with
trilinear interpolation:

```
U = sum_i [ G12(x_i, y_i, z_i) + q_i × φ(x_i, y_i, z_i) ]
```

where:
- G12 field encodes LJ (C6 + C12) per unit (C6_i, C12_i)
- φ field encodes electrostatic potential per unit charge
