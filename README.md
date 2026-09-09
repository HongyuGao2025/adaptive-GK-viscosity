# Adaptive Green–Kubo Viscosity Analysis

A Python implementation of an adaptive Green–Kubo (GK) framework for
viscosity calculation and convergence assessment from equilibrium molecular
dynamics simulations.

The method determines the required sampling from the relaxation dynamics
rather than prescribing a fixed trajectory length. It combines viscosity-buildup
analysis, stretched-exponential fitting, replica-level bootstrap uncertainty
quantification, and separate temporal and statistical convergence criteria.

## Main features

- Green–Kubo viscosity analysis from stress trajectories
- Five-dimensional deviatoric-stress estimator
- Stretched-exponential analysis of viscosity buildup
- Adaptive assessment of temporal and statistical convergence
- Replica-level bootstrap uncertainty quantification
- Multiple fitting-window convergence tests
- Estimation of characteristic relaxation times (t50, t90, t95)
