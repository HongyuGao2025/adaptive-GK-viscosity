## Molecular Dynamics Data

This directory contains pressure tensor data from equilibrium molecular dynamics (EMD) simulations of 2,2,4,4,6,8,8-heptamethylnonane (HMN) at 100 °C and 146 MPa, used for Green-Kubo viscosity calculations.

### Data Organization

The dataset contains 20 independently initialized NVE production replicas, organized in separate folders (`01_146MPa_nve` to `20_146MPa_nve`).

### Data Format

Each data file contains seven columns:

Step  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz

- **MD timestep:** 1 fs
- **Stress averaging interval:** 100 timesteps
- **Output interval:** 100 fs (0.1 ps)
- **Pressure units:** atm

The pressure tensor components are averaged over consecutive, non-overlapping intervals of 100 MD timesteps.

Simulation inputs and analysis scripts are provided in the `input/` and `code/` directories, respectively.
