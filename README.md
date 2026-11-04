
## Adaptive Green-Kubo Viscosity Analysis

A Python implementation of an adaptive Green-Kubo (GK) framework for viscosity calculation and convergence assessment from equilibrium molecular dynamics simulations.

The method determines the required sampling from the relaxation dynamics rather than prescribing a fixed trajectory length. It combines viscosity-buildup analysis, stretched-exponential fitting, replica-level bootstrap uncertainty quantification, and separate temporal and statistical convergence criteria.

### Main Features

- Green-Kubo viscosity analysis from stress trajectories
- Five-dimensional deviatoric-stress estimator
- Stretched-exponential analysis of viscosity buildup
- Adaptive assessment of temporal and statistical convergence
- Replica-level bootstrap uncertainty quantification
- Multiple fitting-window convergence tests
- Estimation of characteristic relaxation times (t50, t90, t95)

### Repository Structure

| Directory | Description |
|---|---|
| [`code/`](code/) | Python implementation, configuration files, and instructions for running the analysis. |
| [`data/`](data/) | Example pressure tensor trajectories from 20 independent HMN replicas at 146 MPa. |
| [`input/`](input/) | Example LAMMPS production input, force-field settings, and equilibrated restart file. |

Each directory contains a README with additional information.

### Citation

If you use this software, please cite the associated publication and the archived software release:

**Software DOI:** https://doi.org/10.5281/zenodo.22671587
