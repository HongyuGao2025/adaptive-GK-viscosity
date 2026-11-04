This directory contains molecular dynamics (MD) simulation data supporting the study:
An Adaptive Green–Kubo Framework for Equilibrium Viscosity: Convergence and Relaxation-Time Scaling

The data are used to calculate shear viscosity from equilibrium stress autocorrelation functions (SACFs) and to evaluate the adaptive Green-Kubo convergence procedure.

1. Simulation system
  Liquid: 2,2,4,4,6,8,8-Heptamethylnonane (HMN)
  Temperature: 100 °C (373.15 K)
  Pressure: 146 MPa
  Number of molecules: 100
  Force field: OPLS-AA
  MD software: LAMMPS
  Production ensemble: NVE
  Number of replicas: 20
**Thereplicas** were independently initialized and used to estimate statistical uncertainty through replica-level bootstrap resampling.

2. Data organization

data/
├── README.md
└── 146MPa/
    ├── 01_146MPa_nve/
    ├── 02_146MPa_nve/
    ├── 03_146MPa_nve/
    ...
    └── 20_146MPa_nve/
