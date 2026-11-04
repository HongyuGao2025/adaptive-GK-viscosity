from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

from pipeline_config import PipelineConfig, ViscosityConfig, BootstrapConfig

KB = 1.3806504e-23
ATM_TO_PA = 101325.0


def autocorrelation_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    c = fftconvolve(x, x[::-1], mode="full")[n - 1:]
    c /= np.arange(n, 0, -1)
    return c


def make_log_indices(t_ps: np.ndarray, points_per_decade: int = 100) -> np.ndarray:
    t_ps = np.asarray(t_ps)
    if len(t_ps) <= 1:
        return np.arange(len(t_ps))
    positive = t_ps[t_ps > 0.0]
    if len(positive) == 0:
        return np.array([0], dtype=int)
    tmin, tmax = positive[0], positive[-1]
    n_decades = np.log10(tmax) - np.log10(tmin)
    n_log = max(2, int(np.ceil(points_per_decade * n_decades)) + 1)
    idx = np.searchsorted(t_ps, np.logspace(np.log10(tmin), np.log10(tmax), n_log))
    idx = idx[idx < len(t_ps)]
    return np.unique(np.concatenate(([0], idx)))


def combine_stress_files(folder: Path, pattern: str, output_name: str) -> Path | None:
    files = sorted(folder.glob(pattern))
    if not files:
        print(f"Skip {folder}: no {pattern}")
        return None
    output = folder / output_name
    with output.open("w", encoding="utf-8") as wf:
        for src in files:
            with src.open("r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        wf.write(line + "\n")
    return output


def _load_replica_data(replica_dirs: list[Path], config: ViscosityConfig) -> list[np.ndarray]:
    data = []
    for folder in replica_dirs:
        path = folder / config.input_name
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        arr = np.loadtxt(path)
        if arr.ndim != 2 or arr.shape[1] < 7:
            raise ValueError(f"{path} must contain at least 7 columns: step/time, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz")
        data.append(arr)
    return data


def _replica_viscosity_curve(data: np.ndarray, config: ViscosityConfig,
                             t_blk_ns: float, n_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    dt_s = config.dt_fs * config.step_interval * 1e-15
    sample_ps = dt_s * 1e12
    block_points = int(round(t_blk_ns * 1000.0 / sample_ps))
    required = block_points * n_blocks
    if block_points < 10:
        raise ValueError("Block contains fewer than 10 stored stress points.")
    if len(data) < required:
        raise ValueError(f"Need {required} stress samples but replica contains only {len(data)}.")

    prefactor = (config.box_length * 1e-10) ** 3 / (KB * config.temp) * ATM_TO_PA**2
    eta_blocks = []
    for b in range(n_blocks):
        block = data[b * block_points:(b + 1) * block_points]
        pxx, pyy, pzz = block[:, 1], block[:, 2], block[:, 3]
        pxy, pxz, pyz = block[:, 4], block[:, 5], block[:, 6]
        dxy = 0.5 * (pxx - pyy)
        dyz = 0.5 * (pyy - pzz)
        dzx = 0.5 * (pzz - pxx)
        c = (
            autocorrelation_fft(pxy) + autocorrelation_fft(pxz) + autocorrelation_fft(pyz)
            + (2.0 / 3.0) * (
                autocorrelation_fft(dxy) + autocorrelation_fft(dyz) + autocorrelation_fft(dzx)
            )
        ) / 5.0
        eta = np.zeros(len(c))
        eta[1:] = np.cumsum(0.5 * (c[1:] + c[:-1]) * dt_s) * prefactor
        eta_blocks.append(eta)

    eta_mean = np.mean(np.asarray(eta_blocks), axis=0)
    t_ns = np.arange(block_points) * sample_ps * 1e-3
    idx = make_log_indices(t_ns * 1000.0, config.points_per_decade)
    return t_ns[idx], eta_mean[idx]


def stretched_exp(t: np.ndarray, eta_inf: float, tau: float, beta: float) -> np.ndarray:
    return eta_inf * (1.0 - np.exp(-((t / tau) ** beta)))


def _bootstrap_one_fraction(t: np.ndarray, etas: np.ndarray, fraction: float, t_blk_ns: float,
                            config: BootstrapConfig, rng: np.random.Generator) -> dict:
    tmax = fraction * t[-1]
    mask = (t > 0.0) & (t <= tmax)
    tfit = t[mask]
    if len(tfit) < 5:
        return {"fraction": fraction, "valid": 0, "pass_early": False, "pass_bound": False,
                "pass_statistical": False, "pass_all": False, "pass_both": False,
                "error": "too few fit points"}

    rows = []
    failures = 0
    for _ in range(config.n_boot):
        idx = rng.integers(0, etas.shape[0], size=etas.shape[0])
        y = np.mean(etas[idx], axis=0)[mask]
        try:
            popt, _ = curve_fit(
                stretched_exp, tfit, y,
                p0=[max(float(np.max(y)), 1e-12), max(0.02, 0.1 * tmax), 0.5],
                bounds=([0.0, config.tau_min, config.beta_min], [np.inf, np.inf, config.beta_max]),
                maxfev=20000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            failures += 1
            continue
        eta_inf, tau, beta = popt
        if not np.all(np.isfinite(popt)) or eta_inf <= 0 or tau <= 0 or beta <= 0:
            failures += 1
            continue
        t90 = tau * (-np.log(0.10)) ** (1.0 / beta)
        t95 = tau * (-np.log(0.05)) ** (1.0 / beta)
        if np.isfinite(t90) and np.isfinite(t95):
            rows.append((eta_inf, tau, beta, t90, t95))
        else:
            failures += 1

    if not rows:
        return {"fraction": fraction, "valid": 0, "failed": failures,
                "pass_early": False, "pass_bound": False, "pass_statistical": False,
                "pass_all": False, "pass_both": False,
                "error": "no valid bootstrap fits"}

    arr = np.asarray(rows)
    def stat(col: int) -> tuple[float, float, float]:
        return tuple(float(x) for x in np.percentile(arr[:, col], [50, 2.5, 97.5]))

    eta_med, eta_lo, eta_hi = stat(0)
    tau_med, tau_lo, tau_hi = stat(1)
    beta_med, beta_lo, beta_hi = stat(2)
    t90_med, t90_lo, t90_hi = stat(3)
    t95_med, t95_lo, t95_hi = stat(4)
    ci_relative = (eta_hi - eta_lo) / eta_med
    # Three convergence criteria used in the paper, evaluated against the
    # requested block length t_blk.
    early_limit = config.early_t95_fraction * t_blk_ns
    bound_limit = fraction * t_blk_ns
    pass_early = t95_med < early_limit
    pass_bound = t95_hi < bound_limit
    pass_statistical = ci_relative < config.ci_eta_cutoff
    pass_all = pass_statistical and pass_early and pass_bound

    return {
        "fraction": fraction, "valid": len(arr), "failed": failures,
        "eta_med": eta_med, "eta_lo": eta_lo, "eta_hi": eta_hi,
        "tau_med": tau_med, "tau_lo": tau_lo, "tau_hi": tau_hi,
        "beta_med": beta_med, "beta_lo": beta_lo, "beta_hi": beta_hi,
        "t90_med": t90_med, "t90_lo": t90_lo, "t90_hi": t90_hi,
        "t95_med": t95_med, "t95_lo": t95_lo, "t95_hi": t95_hi,
        "ci_relative": ci_relative,
        "early_limit": early_limit, "bound_limit": bound_limit,
        "pass_early": pass_early, "pass_bound": pass_bound,
        "pass_statistical": pass_statistical, "pass_all": pass_all,
        # Backward-compatible aliases for older downstream scripts.
        "temporal_limit": bound_limit, "pass_temporal": pass_bound, "pass_both": pass_all,
    }


def _evaluate_tblk(t: np.ndarray, etas: np.ndarray, t_blk_ns: float,
                   config: BootstrapConfig) -> list[dict]:
    # Reinitialize with the manuscript-analysis seed at each t_blk. Within a
    # t_blk evaluation the same RNG stream is consumed sequentially by the
    # 20/30/40/50% fits, reproducing the behavior of the original bootstrap
    # script when the same viscosity curves are supplied.
    rng = np.random.default_rng(config.random_seed)
    return [
        _bootstrap_one_fraction(t, etas, f, t_blk_ns, config, rng)
        for f in config.fit_fractions
    ]


def _write_curves(outdir: Path, t_blk_ns: float, t: np.ndarray, etas: np.ndarray) -> None:
    tag = f"tblk_{t_blk_ns:g}ns"
    d = outdir / tag
    d.mkdir(parents=True, exist_ok=True)
    for i, eta in enumerate(etas, start=1):
        np.savetxt(d / f"replica_{i:03d}_viscosity.dat", np.column_stack([t, eta]),
                   header="tcorr(ns) eta(Pa.s)")
    mean = np.mean(etas, axis=0)
    std = np.std(etas, axis=0, ddof=1) if len(etas) > 1 else np.zeros_like(mean)
    sem = std / np.sqrt(len(etas)) if len(etas) > 1 else np.zeros_like(mean)
    np.savetxt(d / "viscosity_average.dat", np.column_stack([t, mean, std, sem]),
               header="tcorr(ns) eta_mean(Pa.s) eta_std(Pa.s) eta_sem(Pa.s)")


def _fmt_pass(v: bool) -> str:
    return "PASS" if v else "FAIL"


def _attempt_state(results: list[dict], expected: int) -> tuple[bool, bool, bool]:
    complete = len(results) == expected and all(r.get("valid", 0) > 0 for r in results)
    temporal_all = complete and all(r.get("pass_early", False) and r.get("pass_bound", False) for r in results)
    statistical_all = complete and all(r.get("pass_statistical", False) for r in results)
    return complete and temporal_all and statistical_all, temporal_all, statistical_all


def _required_tblk_from_temporal(results: list[dict], early_fraction: float) -> float:
    requirements = []
    for r in results:
        if r.get("valid", 0) == 0:
            continue
        requirements.append(r["t95_med"] / early_fraction)
        requirements.append(r["t95_hi"] / r["fraction"])
    return max(requirements) if requirements else 0.0


def _snap_up(value: float, resolution: float) -> float:
    return float(np.ceil((value - 1e-12) / resolution) * resolution)


def _snap_midpoint(low: float, high: float, resolution: float) -> float | None:
    # Return a grid point strictly inside (low, high), biased toward the midpoint.
    lo_i = int(np.floor(low / resolution + 1e-12))
    hi_i = int(np.ceil(high / resolution - 1e-12))
    if hi_i - lo_i <= 1:
        return None
    mid_i = (lo_i + hi_i) // 2
    if mid_i <= lo_i:
        mid_i = lo_i + 1
    if mid_i >= hi_i:
        mid_i = hi_i - 1
    return mid_i * resolution


def _write_report(path: Path, attempts: list[dict], final: dict | None,
                  n_rep: int, n_blocks: int, config: BootstrapConfig,
                  max_available_tblk: float, search_note: str = "") -> None:
    with path.open("w", encoding="utf-8") as wf:
        wf.write("=" * 76 + "\n")
        wf.write("ADAPTIVE GREEN-KUBO VISCOSITY CONVERGENCE REPORT\n")
        wf.write("=" * 76 + "\n\n")
        wf.write(f"Independent replicas : {n_rep}\n")
        wf.write(f"Blocks per replica   : {n_blocks}\n")
        wf.write(f"Fit fractions        : {', '.join(f'{100*f:.0f}%' for f in config.fit_fractions)}\n")
        wf.write(f"Bootstrap samples    : {config.n_boot}\n")
        wf.write(f"Stat. criterion      : CI = (eta97.5-eta2.5)/eta50 < {100*config.ci_eta_cutoff:.1f}%\n")
        wf.write(f"Early criterion      : t95 < {config.early_t95_fraction:.2f} * t_blk\n")
        wf.write("Bound criterion      : t95,97.5 < f_fit * t_blk\n")
        wf.write("Convergence rule     : ALL fit fractions must pass ALL THREE criteria\n")
        if search_note:
            wf.write(f"Search strategy      : {search_note}\n")
        wf.write("\n")

        for attempt in attempts:
            wf.write("-" * 76 + "\n")
            wf.write(f"t_blk = {attempt['t_blk_ns']:.3f} ns   "
                     f"t_rep used = {attempt['t_rep_ns']:.3f} ns per replica\n")
            wf.write("f_fit   eta_inf(Pa.s)   CI(%)   t95(ns)  0.2tblk(ns)  t95_97.5(ns)  ffit*tblk(ns)   Stat.   Early   Bound   ALL\n")
            for r in attempt["results"]:
                if r.get("valid", 0) == 0:
                    wf.write(f"{100*r['fraction']:5.0f}%   --              --      --       --           --            --             FAIL    FAIL    FAIL    FAIL\n")
                else:
                    wf.write(f"{100*r['fraction']:5.0f}%   {r['eta_med']:12.6e}   "
                             f"{100*r['ci_relative']:6.2f}   {r['t95_med']:7.4f}   "
                             f"{r['early_limit']:11.4f}   {r['t95_hi']:12.4f}   "
                             f"{r['bound_limit']:13.4f}   {_fmt_pass(r['pass_statistical']):5s}   "
                             f"{_fmt_pass(r['pass_early']):5s}   {_fmt_pass(r['pass_bound']):5s}   "
                             f"{_fmt_pass(r['pass_all']):5s}\n")
            wf.write(f"ALL FOUR WINDOWS / ALL THREE CRITERIA: {_fmt_pass(attempt['converged'])}\n\n")

        wf.write("=" * 76 + "\n")
        if final is not None:
            selected = final["results"][-1]  # largest f_fit; all have passed
            wf.write("FINAL STATUS: CONVERGED AND CONFIRMED\n" if final.get("confirmed") else "FINAL STATUS: CONVERGED\n")
            wf.write("=" * 76 + "\n")
            wf.write(f"Recommended t_blk    : {final['t_blk_ns']:.3f} ns\n")
            if final.get("confirmation_t_blk_ns") is not None:
                wf.write(f"Confirmation t_blk   : {final['confirmation_t_blk_ns']:.3f} ns\n")
            wf.write(f"Trajectory used      : {final['t_rep_ns']:.3f} ns per replica\n")
            wf.write(f"Selected fit fraction: {100*selected['fraction']:.0f}%\n")
            wf.write(f"eta_inf median       : {selected['eta_med']:.8e} Pa.s\n")
            wf.write(f"eta_inf 95% CI       : [{selected['eta_lo']:.8e}, {selected['eta_hi']:.8e}] Pa.s\n")
            wf.write(f"CI relative width    : {100*selected['ci_relative']:.2f}%\n")
            wf.write(f"tau_0 median         : {selected['tau_med']:.6f} ns\n")
            wf.write(f"beta median          : {selected['beta_med']:.6f}\n")
            wf.write(f"beta 95% CI          : [{selected['beta_lo']:.6f}, {selected['beta_hi']:.6f}]\n")
            wf.write(f"t95 median           : {selected['t95_med']:.4f} ns\n")
            wf.write(f"t95 97.5% bound      : {selected['t95_hi']:.4f} ns\n")
            wf.write(f"Stat. criterion      : {_fmt_pass(selected['pass_statistical'])}\n")
            wf.write(f"Early criterion      : {_fmt_pass(selected['pass_early'])}\n")
            wf.write(f"Bound criterion      : {_fmt_pass(selected['pass_bound'])}\n")
        else:
            last = attempts[-1] if attempts else None
            wf.write("FINAL STATUS: NOT CONVERGED WITH AVAILABLE DATA\n")
            wf.write("=" * 76 + "\n")
            wf.write(f"Maximum available t_blk: {max_available_tblk:.3f} ns\n")
            if last is not None:
                early_all = all(r.get("pass_early", False) for r in last["results"])
                bound_all = all(r.get("pass_bound", False) for r in last["results"])
                statistical_all = all(r.get("pass_statistical", False) for r in last["results"])
                if not early_all or not bound_all:
                    wf.write("Reason: one or both relaxation-resolution/temporal-coverage criteria fail for at least one fit window.\n")
                    wf.write("Recommended next step: extend each replica and repeat with a larger t_blk while keeping n_blocks fixed.\n")
                elif not statistical_all:
                    wf.write("Reason: temporal convergence passes, but statistical precision is insufficient.\n")
                    wf.write("Recommended next step: increase independent statistical sampling.\n")
                else:
                    wf.write("Reason: convergence criteria could not be established from the available data.\n")


def run_pipeline(config: PipelineConfig) -> None:
    root = config.project_root
    replica_dirs = sorted(p for p in root.glob(config.viscosity.folder_pattern) if p.is_dir())
    if not replica_dirs:
        raise FileNotFoundError(f"No replica folders found: {root / config.viscosity.folder_pattern}")

    print("=" * 78)
    print("Adaptive Green-Kubo viscosity analysis")
    print("=" * 78)
    print(f"Project root          : {root}")
    print(f"Replica folders       : {len(replica_dirs)}")

    if config.combine.enabled:
        print("\n[1/3] Combining stress files")
        for folder in replica_dirs:
            combine_stress_files(folder, config.combine.stress_file_pattern, config.combine.output_name)

    if not config.viscosity.enabled or not config.bootstrap.enabled or not config.adaptive.enabled:
        raise ValueError("This release expects viscosity, bootstrap, and adaptive stages to be enabled.")

    print("\n[2/3] Loading replica stress data")
    raw = _load_replica_data(replica_dirs, config.viscosity)
    sample_ps = config.viscosity.dt_fs * config.viscosity.step_interval * 1e-3
    available_ns = min(len(x) for x in raw) * sample_ps * 1e-3
    max_available_tblk = available_ns / config.adaptive.n_blocks
    max_tblk = max_available_tblk
    if config.adaptive.max_t_blk_ns is not None:
        max_tblk = min(max_tblk, config.adaptive.max_t_blk_ns)

    resolution = config.adaptive.refinement_resolution_ns
    max_tblk = np.floor(max_tblk / resolution + 1e-12) * resolution
    outdir = root / config.adaptive.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Shortest replica       : {available_ns:.3f} ns")
    print(f"Blocks per replica     : {config.adaptive.n_blocks}")
    print(f"Initial t_blk          : {config.adaptive.initial_t_blk_ns:g} ns")
    print(f"Refinement resolution  : {resolution:g} ns")
    print(f"Maximum growth factor  : {config.adaptive.max_growth_factor:g}x")
    print(f"Maximum usable t_blk   : {max_tblk:.3f} ns")
    print(f"Fit fractions          : {', '.join(f'{100*f:.0f}%' for f in config.bootstrap.fit_fractions)}")
    print("Rule                   : every fit fraction must pass Stat. + Early + Bound")

    attempts: list[dict] = []
    cache: dict[float, dict] = {}
    expected = len(config.bootstrap.fit_fractions)

    def evaluate(t_blk: float, phase: str) -> dict:
        t_blk = round(float(t_blk), 12)
        if t_blk in cache:
            return cache[t_blk]
        if t_blk > max_tblk + 1e-12:
            raise ValueError(f"Requested t_blk={t_blk:g} ns exceeds available maximum {max_tblk:g} ns.")
        print(f"\n--- {phase}: t_blk = {t_blk:.3f} ns ---")
        curves = [_replica_viscosity_curve(x, config.viscosity, t_blk, config.adaptive.n_blocks) for x in raw]
        t = curves[0][0]
        if any(not np.allclose(t, x[0]) for x in curves[1:]):
            raise ValueError("Replica time grids do not match.")
        etas = np.asarray([x[1] for x in curves])
        _write_curves(outdir, t_blk, t, etas)
        results = _evaluate_tblk(t, etas, t_blk, config.bootstrap)
        converged, temporal_all, statistical_all = _attempt_state(results, expected)
        attempt = {
            "t_blk_ns": t_blk,
            "t_rep_ns": t_blk * config.adaptive.n_blocks,
            "results": results,
            "converged": converged,
            "temporal_all": temporal_all,
            "statistical_all": statistical_all,
            "phase": phase,
        }
        cache[t_blk] = attempt
        attempts.append(attempt)

        for r in results:
            if r.get("valid", 0):
                print(
                    f"f_fit={100*r['fraction']:4.0f}% | eta={r['eta_med']:.5e} Pa.s | "
                    f"beta={r['beta_med']:.3f} | CI={100*r['ci_relative']:5.1f}% | "
                    f"t95={r['t95_med']:.3f} ns | t95_97.5={r['t95_hi']:.3f} ns | "
                    f"Stat.={_fmt_pass(r['pass_statistical'])} | "
                    f"Early={_fmt_pass(r['pass_early'])} | "
                    f"Bound={_fmt_pass(r['pass_bound'])} | ALL={_fmt_pass(r['pass_all'])}"
                )
            else:
                print(f"f_fit={100*r['fraction']:4.0f}% | invalid ({r.get('error', 'fit failure')})")
        print(f"RESULT: {_fmt_pass(converged)}")
        return attempt

    print("\n[3/3] Adaptive convergence search")
    first = _snap_up(config.adaptive.initial_t_blk_ns, resolution)
    if first > max_tblk + 1e-12:
        raise ValueError("initial_t_blk_ns is larger than the maximum block length allowed by the data.")

    # Phase A: coarse adaptive search. Temporal failures increase t_blk using
    # the measured relaxation requirement, but no single jump may exceed the
    # configured growth factor. A statistical-only failure stops the t_blk
    # search because the paper prescribes more independent sampling instead.
    current = first
    last_temporal_fail: float | None = None
    first_pass: float | None = None
    stopped_for_statistics = False

    while current <= max_tblk + 1e-12:
        a = evaluate(current, "COARSE SEARCH")
        if a["converged"]:
            first_pass = current
            break
        if a["temporal_all"] and not a["statistical_all"]:
            stopped_for_statistics = True
            print("Temporal criteria pass for all windows, but Stat. fails: stop increasing t_blk.")
            print("Recommended action: increase independent statistical sampling at fixed t_blk.")
            break

        last_temporal_fail = current
        required = _required_tblk_from_temporal(a["results"], config.bootstrap.early_t95_fraction)
        proposed = max(current + resolution, _snap_up(required, resolution))
        capped = min(proposed, current * config.adaptive.max_growth_factor)
        next_t = _snap_up(capped, resolution)
        if next_t <= current + 1e-12:
            next_t = current + resolution
        if next_t > max_tblk + 1e-12:
            if current < max_tblk - 1e-12:
                next_t = max_tblk
            else:
                break
        print(f"Temporal requirement suggests t_blk >= {required:.3f} ns; next test = {next_t:.3f} ns.")
        current = next_t

    final: dict | None = None
    candidate: dict | None = None

    # Phase B: once a passing upper bound is found, refine the bracket to find
    # the smallest passing t_blk on the user-selected resolution grid.
    if first_pass is not None:
        high = first_pass
        low = last_temporal_fail if last_temporal_fail is not None else max(0.0, first - resolution)
        print(f"\nPASS bracket found: ({low:.3f}, {high:.3f}] ns. Refining...")
        while True:
            mid = _snap_midpoint(low, high, resolution)
            if mid is None:
                break
            a = evaluate(mid, "BRACKET REFINEMENT")
            if a["converged"]:
                high = mid
            else:
                # If temporal criteria are already satisfied but statistics fail,
                # do not interpret the point as evidence that a larger t_blk is needed.
                if a["temporal_all"] and not a["statistical_all"]:
                    print("Refinement point is temporally converged but statistically insufficient; "
                          "it cannot define a temporal lower bound.")
                    low = mid
                else:
                    low = mid
        candidate = cache[round(high, 12)]
        print(f"\nCandidate minimum converged t_blk = {high:.3f} ns")

        # Phase C: confirmation. Require the next grid point to pass as well.
        # If it fails, continue upward until two consecutive passing t_blk values
        # are observed; report the first member of that stable PASS/PASS pair.
        if config.adaptive.confirm_next:
            prev = candidate
            next_t = high + resolution
            if next_t <= max_tblk + 1e-12:
                while next_t <= max_tblk + 1e-12:
                    a = evaluate(next_t, "CONFIRMATION")
                    if prev["converged"] and a["converged"]:
                        final = prev
                        final["confirmation_t_blk_ns"] = next_t
                        final["confirmed"] = True
                        print(f"Confirmed: {prev['t_blk_ns']:.3f} and {next_t:.3f} ns both PASS.")
                        break
                    prev = a
                    next_t += resolution
                if final is None:
                    print("No consecutive PASS/PASS pair could be confirmed within available data.")
            else:
                print("Candidate is at the maximum available t_blk; candidate+resolution cannot be checked.")
        else:
            final = candidate
            final["confirmed"] = False

    # Preserve chronological/search order in report, but avoid duplicate evaluations.
    search_note = (
        f"adaptive coarse growth (<= {config.adaptive.max_growth_factor:g}x), bracket refinement to "
        f"{resolution:g} ns resolution, then candidate + {resolution:g} ns confirmation"
    )
    report = outdir / config.adaptive.report_name
    _write_report(report, attempts, final, len(replica_dirs), config.adaptive.n_blocks,
                  config.bootstrap, max_available_tblk, search_note)

    print("\n" + "=" * 78)
    if final is not None:
        selected = final["results"][-1]
        print("FINAL STATUS: CONVERGED AND CONFIRMED" if final.get("confirmed") else "FINAL STATUS: CONVERGED")
        print(f"recommended t_blk : {final['t_blk_ns']:.3f} ns")
        if final.get("confirmation_t_blk_ns") is not None:
            print(f"confirmation t_blk: {final['confirmation_t_blk_ns']:.3f} ns")
        print(f"t_rep used        : {final['t_rep_ns']:.3f} ns per replica")
        print(f"fit fraction      : {100*selected['fraction']:.0f}%")
        print(f"eta_inf           : {selected['eta_med']:.6e} Pa.s")
        print(f"95% CI            : [{selected['eta_lo']:.6e}, {selected['eta_hi']:.6e}] Pa.s")
        print(f"tau_0             : {selected['tau_med']:.6f} ns")
        print(f"beta              : {selected['beta_med']:.6f}")
        print(f"t95               : {selected['t95_med']:.6f} ns")
        print(f"t95 97.5%         : {selected['t95_hi']:.6f} ns")
        print(f"Stat./Early/Bound : {_fmt_pass(selected['pass_statistical'])} / "
              f"{_fmt_pass(selected['pass_early'])} / {_fmt_pass(selected['pass_bound'])}")
    elif stopped_for_statistics:
        print("FINAL STATUS: TEMPORALLY CONVERGED, STATISTICALLY INSUFFICIENT")
        print("Recommended next step: increase independent statistical sampling at fixed t_blk.")
    elif candidate is not None:
        print("FINAL STATUS: CANDIDATE CONVERGENCE FOUND BUT NOT CONFIRMED")
        print(f"candidate t_blk   : {candidate['t_blk_ns']:.3f} ns")
        print("More trajectory length is needed to test the next block length.")
    else:
        print("FINAL STATUS: NOT CONVERGED WITH AVAILABLE DATA")
        print("See the report for the diagnosed failure mode and recommended next step.")
    print(f"Report            : {report}")
    print("=" * 78)
