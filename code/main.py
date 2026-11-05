"""Command-line entry point for the adaptive Green-Kubo viscosity analysis.

Usage:
    python main.py
    python main.py path/to/input.json

By default, the configuration is loaded from input.json in the same
directory as this script. 
"""

import sys
from pathlib import Path

from analysis_pipeline import run_pipeline
from pipeline_config import build_config, resolve_path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "input.json"
    if len(sys.argv) > 1:
        config_path = resolve_path(script_dir, sys.argv[1])
    config = build_config(config_path)
    run_pipeline(config)


if __name__ == "__main__":
    main()
