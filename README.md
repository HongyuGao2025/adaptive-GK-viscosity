
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
| [`code/`](code/) | Python implementation and analysis configuration. |
| [`data/`](data/) | Representative pressure tensor trajectories for testing the analysis. |
| [`input/`](input/) | Example LAMMPS production input and equilibrated restart file. |

### Data and Software Availability

The complete pressure tensor trajectories for HMN at six pressures
(0.1, 34.5, 69, 146, 224, and 301 MPa) are available on Zenodo.

- **Simulation dataset:** https://doi.org/10.5281/zenodo.23965298
- **Analysis software:** https://doi.org/10.5281/zenodo.22671587
