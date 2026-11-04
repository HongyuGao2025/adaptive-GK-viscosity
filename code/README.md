## Adaptive Green-Kubo Analysis Code

This directory contains the Python implementation of the adaptive Green-Kubo framework for equilibrium viscosity calculations.

### Files

| File | Description |
|---|---|
| `main.py` | Main script for running the analysis. |
| `analysis_pipeline.py` | Functions for stress autocorrelation, viscosity calculation, and convergence analysis. |
| `pipeline_config.py` | Analysis parameters and configuration settings. |
| `input.json` | User-editable input file specifying the data and analysis settings. |

### Configuration

Before running the analysis, edit `input.json` to specify the input data locations and analysis settings.

Additional parameters can be adjusted in `pipeline_config.py` if necessary.

### Running the Analysis

Open a terminal in the `code/` directory and run:

```bash
python main.py
```

The program processes the specified pressure tensor data and performs the Green-Kubo viscosity and convergence analysis.

Example pressure tensor data are provided in the `data/` directory.
