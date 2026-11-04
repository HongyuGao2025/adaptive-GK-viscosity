## Molecular Dynamics Data

This directory contains pressure tensor data from equilibrium molecular dynamics (EMD) simulations used for Green-Kubo viscosity calculations.

### Data Organization

Data are organized by pressure and replica. 

Each pressure condition contains 20 independently initialized NVE replicas (e.g., `01_146MPa_nve` to `20_146MPa_nve`).

Each replica folder contains three consecutive stress files: `stress1.dat`, `stress2.dat`, and `stress3.dat`.

### Data Format

Each file contains time-averaged pressure tensor components:

```text
# Time-averaged data for fix 2
# TimeStep v_pxx v_pyy v_pzz v_pxy v_pxz v_pyz
100 741.063 1787.78 1427.06 -112.772 -66.7364 -147.76
200 784.453 1853.44 1291.4 -162.181 202.958 -292.934
```

- **Columns:** MD timestep, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz
- **MD timestep:** 1 fs
- **Stress averaging interval:** 100 timesteps
- **Output interval:** 100 fs (0.1 ps)
- **Pressure units:** bar
- **Temperature:** 373.15 K (100 °C)

The pressure tensor values are averages over consecutive, non-overlapping 100-timestep intervals, rather than instantaneous measurements.

Simulation inputs and analysis scripts are provided in the `input/` and `code/` directories, respectively.
