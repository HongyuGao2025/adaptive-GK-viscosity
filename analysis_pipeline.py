from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

from pipeline_config import (
    AverageConfig,
    BootstrapConfig,
    PipelineConfig,
    ViscosityConfig,
)


def autocorrelation_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    n = len(x)

    c = fftconvolve(x, x[::-1], mode="full")
    c = c[n - 1 :]
    c /= np.arange(n, 0, -1)

    return c


def make_log_indices(t_ps: np.ndarray, points_per_decade: int = 100) -> np.ndarray:
    t_ps = np.asarray(t_ps)

    if len(t_ps) <= 1:
        return np.arange(len(t_ps))

    positive = t_ps[t_ps > 0.0]
    if len(positive) == 0:
        return np.array([0], dtype=int)

    tmin = positive[0]
    tmax = positive[-1]
    n_decades = np.log10(tmax) - np.log10(tmin)
    n_log = int(np.ceil(points_per_decade * n_decades)) + 1

    t_log = np.logspace(np.log10(tmin), np.log10(tmax), n_log)
    idx = np.searchsorted(t_ps, t_log)
    idx = idx[idx < len(t_ps)]
    idx = np.unique(np.concatenate(([0], idx)))

    return idx


def combine_stress_files(
    folder: Path, stress_file_pattern: str, output_name: str
) -> Path | None:
    files = sorted(folder.glob(stress_file_pattern))
    if not files:
        print(f"Skip {folder}: no {stress_file_pattern}")
        return None

    output_file = folder / output_name
    with output_file.open("w", encoding="utf-8") as wf:
        for file_name in files:
            print(f"Reading {file_name}")
            with file_name.open("r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    wf.write(line + "\n")

    print(f"Wrote {output_file}")
    return output_file


def process_one_folder(folder: Path, config: ViscosityConfig) -> Path | None:
    input_file = folder / config.input_name
    output_file = folder / config.output_name

    if not input_file.exists():
        print(f"Skip {folder}: no {config.input_name}")
        return None

    print(f"\nProcessing {folder}")

    data = np.loadtxt(input_file)
    dt_s = config.dt_fs * config.step_interval * 1e-15
    sample_ps = dt_s * 1e12
    prefactor = (
        (config.box_length * 1e-10) ** 3 / (1.3806504e-23 * config.temp) * (101325.0**2)
    )

    max_points = round((config.max_time_ns * 1000.0) / sample_ps)
    data = data[:max_points]

    n = len(data)
    block_size_raw = n // config.n_blocks

    if block_size_raw < 10:
        print(f"Skip {folder}: block too small")
        return None

    print(f"Raw sampling interval = {sample_ps:.3f} ps")
    print(f"Using {config.n_blocks} blocks")
    print(f"Block size = {block_size_raw} points = {block_size_raw * sample_ps:.1f} ps")

    eta_all: list[np.ndarray] = []

    for b in range(config.n_blocks):
        block = data[b * block_size_raw : (b + 1) * block_size_raw]

        pxx = block[:, 1]
        pyy = block[:, 2]
        pzz = block[:, 3]
        pxy = block[:, 4]
        pxz = block[:, 5]
        pyz = block[:, 6]

        dxy = 0.5 * (pxx - pyy)
        dyz = 0.5 * (pyy - pzz)
        dzx = 0.5 * (pzz - pxx)

        cxy = autocorrelation_fft(pxy)
        cxz = autocorrelation_fft(pxz)
        cyz = autocorrelation_fft(pyz)
        cdxy = autocorrelation_fft(dxy)
        cdyz = autocorrelation_fft(dyz)
        cdzx = autocorrelation_fft(dzx)

        c = (cxy + cxz + cyz + (2.0 / 3.0) * (cdxy + cdyz + cdzx)) / 5.0

        eta_tc = np.zeros(len(c))
        eta_tc[1:] = np.cumsum(0.5 * (c[1:] + c[:-1]) * dt_s) * prefactor
        eta_all.append(eta_tc)

    # FIXME:
    eta_all = np.array(eta_all)
    n_tau = eta_all.shape[1]

    tcorr_ps = np.arange(n_tau) * sample_ps
    tcorr_ns = tcorr_ps * 1e-3
    eta_mean = np.mean(eta_all, axis=0)

    if config.n_blocks > 1:
        eta_std = np.std(eta_all, axis=0, ddof=1)
        eta_err = eta_std / np.sqrt(config.n_blocks)
    else:
        eta_err = np.zeros_like(eta_mean)

    log_idx = make_log_indices(tcorr_ps, points_per_decade=config.points_per_decade)

    with output_file.open("w", encoding="utf-8") as wf:
        wf.write("# tcorr(ns) eta(Pa.s) err(Pa.s)\n")
        wf.write("# ACF and GK integration performed on uniform raw grid\n")
        wf.write("# Output written on approximately log-spaced time grid\n")
        wf.write(f"# raw_dt(ps) = {sample_ps:.6f}\n")
        wf.write(f"# points_per_decade = {config.points_per_decade}\n")

        for i in log_idx:
            wf.write(f"{tcorr_ns[i]:.8e}\t{eta_mean[i]:.8e}\t{eta_err[i]:.8e}\n")

    print(f"Saved {output_file}")
    print(f"Output points: {len(log_idx)} / {n_tau}")
    return output_file


def average_replica_curves(project_root: Path, config: AverageConfig) -> Path:
    files = sorted(project_root.glob(f"{config.folder_pattern}/{config.input_name}"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found with pattern: {config.folder_pattern}/{config.input_name}"
        )

    print(f"Found {len(files)} files")
    for file_name in files:
        print(f"  {file_name}")

    times: list[np.ndarray] = []
    etas: list[np.ndarray] = []

    for file_name in files:
        data = np.loadtxt(file_name, comments="#")
        times.append(data[:, 0])
        etas.append(data[:, 1])

    times_arr = np.array(times)
    etas_arr = np.array(etas)

    for i in range(1, len(files)):
        if not np.allclose(times_arr[0], times_arr[i]):
            raise ValueError(f"Time grid mismatch in {files[i]}")

    t = times_arr[0]
    eta_mean = np.mean(etas_arr, axis=0)
    eta_std = np.std(etas_arr, axis=0, ddof=1)
    eta_sem = eta_std / np.sqrt(len(files))

    output_file = project_root / config.output_name
    with output_file.open("w", encoding="utf-8") as wf:
        wf.write("# tcorr(ns) eta_mean(Pa.s) eta_std(Pa.s) eta_sem(Pa.s)\n")
        for ti, mean, std, sem in zip(t, eta_mean, eta_std, eta_sem):
            wf.write(f"{ti:.6f}\t{mean:.6e}\t{std:.6e}\t{sem:.6e}\n")

    print(f"Done. Averaged {len(files)} replicas.")
    print(f"Output: {output_file}")
    return output_file


def stretched_exp(t: np.ndarray, eta_inf: float, tau: float, beta: float) -> np.ndarray:
    return eta_inf * (1.0 - np.exp(-((t / tau) ** beta)))


def summarize(results: np.ndarray) -> tuple[list[str], dict[str, dict[str, float]]]:
    names = ["eta_inf", "tau", "beta", "t90", "t95"]

    if len(results) == 0:
        return ["# no valid results\n"], {}

    lines = ["# parameter median p2.5 p97.5 mean std\n"]
    stats: dict[str, dict[str, float]] = {}

    for i, name in enumerate(names):
        vals = results[:, i]
        median = np.percentile(vals, 50)
        p025 = np.percentile(vals, 2.5)
        p975 = np.percentile(vals, 97.5)
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)

        stats[name] = {
            "median": float(median),
            "p2.5": float(p025),
            "p97.5": float(p975),
            "mean": float(mean),
            "std": float(std),
        }

        lines.append(
            f"{name}\t{median:.8e}\t{p025:.8e}\t{p975:.8e}\t{mean:.8e}\t{std:.8e}\n"
        )

    return lines, stats


def bootstrap_fit(project_root: Path, config: BootstrapConfig) -> Path:
    files = sorted(project_root.glob(f"{config.folder_pattern}/{config.input_name}"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found with pattern: {config.folder_pattern}/{config.input_name}"
        )

    print(f"Found {len(files)} files")

    times: list[np.ndarray] = []
    etas: list[np.ndarray] = []

    for file_name in files:
        data = np.loadtxt(file_name, comments="#")
        times.append(data[:, 0])
        etas.append(data[:, 1])

    times_arr = np.array(times)
    etas_arr = np.array(etas)

    for i in range(1, len(files)):
        if not np.allclose(times_arr[0], times_arr[i]):
            raise ValueError(f"Time grid mismatch in {files[i]}")

    t = times_arr[0]
    n_rep = etas_arr.shape[0]
    run_length = t[-1]
    rng = np.random.default_rng(config.random_seed)
    fit_summary: list[dict[str, float | bool]] = []

    output_file = project_root / config.output_name
    with output_file.open("w", encoding="utf-8") as wf:
        wf.write(
            "# Bootstrap stretched-exponential fitting over multiple fit fractions\n"
        )
        wf.write("# Data are fitted on the stored log-spaced time grid.\n")
        wf.write("# No log(t) transformation is applied in the model.\n")
        wf.write(
            "# Valid-fit filters: eta_inf > 0, tau > 0, finite values, beta bounds.\n"
        )
        wf.write(f"# pattern = {config.folder_pattern}/{config.input_name}\n")
        wf.write(f"# n_boot = {config.n_boot}\n")
        wf.write(f"# run_length(ns) = {run_length:.8e}\n")
        wf.write(f"# n_rep = {n_rep}\n")
        wf.write(f"# fit_fractions = {config.fit_fractions}\n")
        wf.write(f"# beta_min = {config.beta_min}\n")
        wf.write(f"# beta_max = {config.beta_max}\n")
        wf.write(f"# ci_eta_cutoff = {config.ci_eta_cutoff:.3f}\n\n")

        for fit_fraction in config.fit_fractions or []:
            tmax_fit = fit_fraction * run_length
            mask = (t > 0.0) & (t <= tmax_fit)
            t_fit = t[mask]

            if len(t_fit) < 5:
                wf.write(f"\n# FIT_FRACTION {fit_fraction:.2f}: too few points\n")
                continue

            valid_results: list[list[float]] = []
            n_fail_fit = 0
            n_invalid_basic = 0

            for _ in range(config.n_boot):
                idx = rng.integers(0, n_rep, size=n_rep)
                eta_boot = np.mean(etas_arr[idx, :], axis=0)
                y_fit = eta_boot[mask]

                eta_obs_max = np.max(y_fit)
                eta_guess = max(eta_obs_max, 1e-12)
                tau_guess = max(0.02, 0.1 * tmax_fit)
                beta_guess = 0.5

                try:
                    popt, _pcov = curve_fit(
                        stretched_exp,
                        t_fit,
                        y_fit,
                        p0=[eta_guess, tau_guess, beta_guess],
                        bounds=(
                            [0.0, config.tau_min, config.beta_min],
                            [np.inf, np.inf, config.beta_max],
                        ),
                        maxfev=20000,
                    )
                except (RuntimeError, ValueError, FloatingPointError):
                    n_fail_fit += 1
                    continue

                eta_inf, tau, beta = popt
                if not np.all(np.isfinite([eta_inf, tau, beta])):
                    n_invalid_basic += 1
                    continue

                if eta_inf <= 0 or tau <= 0 or beta <= 0:
                    n_invalid_basic += 1
                    continue

                t90 = tau * (-np.log(0.10)) ** (1.0 / beta)
                t95 = tau * (-np.log(0.05)) ** (1.0 / beta)

                if not np.all(np.isfinite([t90, t95])):
                    n_invalid_basic += 1
                    continue

                valid_results.append([eta_inf, tau, beta, t90, t95])

            valid_results_arr = np.array(valid_results)
            lines, stats = summarize(valid_results_arr)

            wf.write("\n")
            wf.write("# ============================================================\n")
            wf.write(f"# FIT_FRACTION = {fit_fraction:.2f}\n")
            wf.write(f"# tmax_fit(ns) = {tmax_fit:.8e}\n")
            wf.write(f"# n_fit_points = {len(t_fit)}\n")
            wf.write(f"# valid_fits = {len(valid_results_arr)}\n")
            wf.write(f"# failed_fits = {n_fail_fit}\n")
            wf.write(f"# invalid_basic = {n_invalid_basic}\n")

            if len(valid_results_arr) > 0:
                eta_med = stats["eta_inf"]["median"]
                eta_p025 = stats["eta_inf"]["p2.5"]
                eta_p975 = stats["eta_inf"]["p97.5"]
                t95_med = stats["t95"]["median"]
                t95_p975 = stats["t95"]["p97.5"]

                ci_eta = (eta_p975 - eta_p025) / eta_med
                temporal_limit = fit_fraction * run_length
                pass_temporal = t95_p975 < temporal_limit
                pass_stat = ci_eta < config.ci_eta_cutoff
                pass_both = pass_temporal and pass_stat

                wf.write(f"# CI_eta_relative = {ci_eta:.8e}\n")
                wf.write(f"# temporal_limit(ns) = ffit * tblk = {temporal_limit:.8e}\n")
                wf.write(f"# t95_median(ns) = {t95_med:.8e}\n")
                wf.write(f"# t95_p97.5(ns) = {t95_p975:.8e}\n")
                wf.write(f"# pass_temporal = {pass_temporal}\n")
                wf.write(f"# pass_statistical = {pass_stat}\n")
                wf.write(f"# pass_both = {pass_both}\n")

                fit_summary.append(
                    {
                        "fit_fraction": fit_fraction,
                        "ci_eta": ci_eta,
                        "eta_med": eta_med,
                        "t95_med": t95_med,
                        "t95_p975": t95_p975,
                        "temporal_limit": temporal_limit,
                        "pass_temporal": pass_temporal,
                        "pass_stat": pass_stat,
                        "pass_both": pass_both,
                    }
                )

            wf.write("\n# VALID_FITS\n")
            for line in lines:
                wf.write(line)

        wf.write("\n")
        wf.write("# ============================================================\n")
        wf.write("# FINAL_SELECTION\n")

        accepted = [x for x in fit_summary if x["pass_both"]]
        if len(accepted) == 0:
            wf.write(
                "# No fitting fraction satisfied both temporal and statistical criteria.\n"
            )

            temporal_ok = [x for x in fit_summary if x["pass_temporal"]]
            if len(temporal_ok) == 0:
                wf.write(
                    "# Recommended next step: increase t_blk while keeping n_blk fixed.\n"
                )
            else:
                wf.write(
                    "# Recommended next step: keep t_blk fixed and increase n_blk / t_rep.\n"
                )
        else:
            best = min(accepted, key=lambda x: x["ci_eta"])
            wf.write(f"# selected_fit_fraction = {best['fit_fraction']:.2f}\n")
            wf.write(f"# selected_eta_inf_median = {best['eta_med']:.8e}\n")
            wf.write(f"# selected_CI_eta_relative = {best['ci_eta']:.8e}\n")
            wf.write(f"# selected_t95_median(ns) = {best['t95_med']:.8e}\n")
            wf.write(f"# selected_t95_p97.5(ns) = {best['t95_p975']:.8e}\n")

    print(f"Output: {output_file}")
    print("\nSummary:")
    for x in fit_summary:
        print(
            f"ffit={x['fit_fraction']:.2f}, "
            f"eta={x['eta_med']:.4e}, "
            f"CI_eta={100 * x['ci_eta']:.1f}%, "
            f"t95={x['t95_med']:.3f} ns, "
            f"t95_p97.5={x['t95_p975']:.3f} ns, "
            f"limit={x['temporal_limit']:.3f} ns, "
            f"temporal={x['pass_temporal']}, "
            f"stat={x['pass_stat']}, "
            f"both={x['pass_both']}"
        )

    return output_file


def run_pipeline(config: PipelineConfig) -> None:
    project_root = config.project_root
    replica_dirs = sorted(
        p for p in project_root.glob(config.viscosity.folder_pattern) if p.is_dir()
    )

    print(f"Project root: {project_root}")
    print(f"Replica folders: {len(replica_dirs)}")

    if config.combine.enabled:
        for folder in replica_dirs:
            combine_stress_files(
                folder, config.combine.stress_file_pattern, config.combine.output_name
            )

    if config.viscosity.enabled:
        for folder in replica_dirs:
            process_one_folder(folder, config.viscosity)

    if config.average.enabled:
        average_replica_curves(project_root, config.average)

    if config.bootstrap.enabled:
        bootstrap_fit(project_root, config.bootstrap)
