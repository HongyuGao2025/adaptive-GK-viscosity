
## Example LAMMPS Simulation Input

This directory provides a representative LAMMPS input for the NVE production simulation of 2,2,4,4,6,8,8-heptamethylnonane (HMN) at 100 °C and 146 MPa.

The starting configuration was obtained following NPT and NVT equilibration. The supplied restart file corresponds to replica 1 and is provided as an example of the production setup.

### Files

| File | Description |
|---|---|
| `in.input` | LAMMPS input script for NVE production and stress output. |
| `force_field.system` | Force-field parameters and interaction settings. |
| `nvt_100C_146MPa_1.restart` | Equilibrated starting configuration for replica 1. |

### Simulation Details

- **LAMMPS version:** 22 Jul 2025
- **Production ensemble:** NVE
- **MD timestep:** 1 fs
- **Production length:** 10 ns
- **Stress output interval:** 100 fs (averaged over 100 timesteps)

### Running the Simulation

From this directory, run:

```bash
lmp -in in.input
```

The simulation generates the pressure tensor data in `stress.dat`. The corresponding 20-replica dataset is provided separately in the `data/` directory.
