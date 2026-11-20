"""Audit the steady case's extrema, ray batching, boundary sampling and energy settings."""

import argparse
import csv
from functools import cache
from pathlib import Path
from time import perf_counter

import numpy as np

from thermal_opd.convergence import continuous_aperture_pv
from thermal_opd.experiments import experiment_configuration
from thermal_opd.heat_source import gaussian_disk_power
from thermal_opd.metrics import circular_aperture_mask
from thermal_opd.model import ModalResolution, ThermalModel
from thermal_opd.opd import transmitted_ray_mask
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.run_io import (
    create_run, data_directory, file_sha256, provenance, validation_check,
    validation_report, write_json, write_summary,
)
from thermal_opd.steady_validation import steady_boundary_residuals, steady_energy_balance
from thermal_opd.validation_common import validate_saved_evidence_run


repository_root = Path(__file__).resolve().parents[2]
angles_deg = (0., 30., 60.)

# The OPD target follows the steady finite-reference checks. The boundary and
# energy bounds follow the saved steady-case validator.
pv_relative_tolerance = 1e-3
boundary_absolute_tolerance_W_per_m2 = 1e-7
energy_absolute_tolerance_W = 1e-9
energy_relative_tolerance = 1e-7
chunk_rtol = 1e-12
chunk_atol_m = 1e-20
sweep_counts = {
    "num_pv_seed_radius": (5, 9, 13, 17),
    "num_pv_seed_phi": (24, 48, 64, 96),
    "opd_point_chunk_size": (128, 256, 512, 1024, 2048, 4096),
    "num_boundary_z": (33, 65, 129, 257),
    "num_energy_r": (20, 40, 80, 160),
    "num_energy_z": (16, 32, 64, 128),
    "num_energy_phi": (45, 90, 180, 360),
}


def audit_angle(p, angle, defaults, counts, rows, checks, evidence):
    """Reuse one prepared modal model for all settings at this angle."""
    resolution = ModalResolution(
        defaults["max_azimuthal_order"], defaults["num_radial_modes"],
        defaults["num_projection_r"], defaults["num_projection_phi"],
        defaults["num_opd_ray_points"],
    )
    prepared = ThermalModel(p, angle, resolution).prepare()
    solution = prepared.solve_steady()
    a = defaults["aperture_radius_ratio"] * p.radius
    internal = prepared.refraction_angle
    chunk = defaults["opd_point_chunk_size"]
    optical_at = solution.prepare_opd(point_chunk_size=chunk)
    radial = np.linspace(0., p.radius, defaults["num_temperature_r"])
    angular = np.linspace(0., 2*np.pi, defaults["num_temperature_phi"], endpoint=False)
    thermal = (prepared.coefficients, prepared.roots, p.radius, p.conductivity,
               p.thickness, p.h_eff, p.h_eff, p.h_eff)

    @cache
    def extrema(nr, nphi):
        start = perf_counter()
        result = continuous_aperture_pv(
            optical_at, radius=p.radius, thickness=p.thickness, refraction_angle=internal,
            aperture_radius=a, num_seed_radius=nr, num_seed_phi=nphi,
        )
        values = {key+"_m": result[key] for key in ("minimum", "maximum", "pv")}
        return values, perf_counter()-start

    @cache
    def losses(nr, nz, nphi):
        start = perf_counter()
        values = steady_energy_balance(*thermal, num_r=nr, num_z=nz, num_phi=nphi)
        values["prescribed_disk_input_W"] = gaussian_disk_power(
            p.radius, p.beam_radius, np.deg2rad(angle), p.absorbed_power,
            num_r=nr, num_phi=nphi,
        )
        return values, perf_counter()-start

    def side(nz):
        start = perf_counter()
        values = steady_boundary_residuals(radial, angular, np.linspace(0., p.thickness, nz), *thermal)
        return {"side_W_per_m2": values["side_W_per_m2"]}, perf_counter()-start

    def record(name, cases, default, evaluate, limit, scope, absolute=(), *, all_cases=False):
        reference, _ = evaluate(cases[-1])
        ratios = []
        for settings in cases:
            values, seconds = evaluate(settings)
            used_for_gate = all_cases or settings == default
            for metric, value in values.items():
                difference = abs(value-reference[metric])
                threshold = limit(metric, reference)
                bounded = metric in absolute
                error = max(abs(value), abs(reference[metric])) if bounded else difference
                ratio = error / threshold
                rows.append(dict(
                    angle_deg=angle, parameter=name, settings=str(settings),
                    reference_settings=str(cases[-1]), metric=metric, value=value,
                    reference=reference[metric], absolute_difference=difference,
                    criterion="absolute_bound" if bounded else "change_vs_reference",
                    threshold=threshold, fraction_of_limit=ratio,
                    status="passed" if ratio <= 1 else "failed",
                    is_default=settings == default, used_for_gate=used_for_gate,
                    elapsed_seconds=seconds,
                ))
                if used_for_gate:
                    ratios.append(ratio)
        key = f"theta{angle:g}_{name}"
        evidence[key] = np.asarray(ratios)
        worst = float(np.max(evidence[key]))
        checks.append(validation_check(
            key, "passed" if worst <= 1 else "failed",
            metrics={"max_error": worst}, threshold=1., scope=scope, evidence=key,
        ))

    def vary(keys, evaluate, limit, scope, absolute=()):
        default = tuple(defaults[key] for key in keys)
        dense = tuple(counts[key][-1] for key in keys)
        for i, name in enumerate(keys):
            cases = [default[:i]+(count,)+default[i+1:] for count in counts[name]]
            record(name, cases, default, lambda settings: evaluate(*settings),
                   limit, scope, absolute)
        return default, dense

    def pv_limit(metric, reference):
        return pv_relative_tolerance * reference["pv_m"]

    pv_scope = "Change in numerical minimum, maximum and PV vs finer seeds, normalized by reference PV."
    pv_default, pv_dense = vary(
        ("num_pv_seed_radius", "num_pv_seed_phi"), extrema, pv_limit, pv_scope,
    )
    record("pv_seeds_joint", [pv_default, pv_dense], pv_default,
           lambda settings: extrema(*settings), pv_limit, pv_scope)

    axis = np.linspace(-a, a, defaults["num_opd_xy"])
    x, y = np.meshgrid(axis, axis, indexing="xy")
    mask = circular_aperture_mask(x, y, a) & transmitted_ray_mask(x, y, p.radius, p.thickness, internal)
    x, y = x[mask], y[mask]
    largest_chunk = counts["opd_point_chunk_size"][-1]
    start = perf_counter()
    reference_opd = solution.opd(x, y, point_chunk_size=largest_chunk)
    reference_seconds = perf_counter()-start
    chunk_scale = chunk_atol_m + chunk_rtol*np.abs(reference_opd)

    def batching(size):
        if size == largest_chunk:
            return {"max_pointwise_tolerance_ratio": 0., "max_absolute_difference_m": 0.}, reference_seconds
        tick = perf_counter()
        values = solution.opd(x, y, point_chunk_size=size)
        delta = np.abs(values-reference_opd)
        return {
            "max_pointwise_tolerance_ratio": float(np.max(delta/chunk_scale)),
            "max_absolute_difference_m": float(np.max(delta)),
        }, perf_counter()-tick

    record("opd_point_chunk_size", counts["opd_point_chunk_size"], chunk, batching,
           lambda metric, _: 1. if metric == "max_pointwise_tolerance_ratio" else float(chunk_scale.max()),
           "Pointwise OPD batching consistency on identical rays. Runtime is diagnostic.",
           ("max_pointwise_tolerance_ratio", "max_absolute_difference_m"), all_cases=True)
    record("num_boundary_z", counts["num_boundary_z"], defaults["num_boundary_z"], side,
           lambda *_: boundary_absolute_tolerance_W_per_m2,
           "Default and refined sidewall samples must both satisfy the absolute Robin residual bound.",
           ("side_W_per_m2",))

    def energy_limit(metric, reference):
        return energy_absolute_tolerance_W + abs(reference["modal_input_W"])*energy_relative_tolerance

    energy_scope = "Separate heat losses and prescribed disk input compared with refined quadrature; also check absolute modal energy imbalance."
    energy_default, energy_dense = vary(
        ("num_energy_r", "num_energy_z", "num_energy_phi"), losses, energy_limit,
        energy_scope, ("balance_absolute_error_W",),
    )
    record("energy_quadrature_joint", [energy_default, energy_dense], energy_default,
           lambda settings: losses(*settings), energy_limit, energy_scope,
           ("balance_absolute_error_W",))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", type=float, nargs="+", choices=angles_deg, default=angles_deg)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if len(set(args.angles)) != len(args.angles):
        parser.error("--angles must be distinct")

    production = experiment_configuration("steady")
    defaults = production["defaults"]
    p = PhysicalParameters(**production["physical_parameters_SI"])
    counts = {name: sorted({*values, defaults[name], max(values[-1], 2*defaults[name]-1)})
              for name, values in sweep_counts.items()}
    directory = create_run(repository_root, "numerical_settings_audit", args.output_dir)
    config = dict(
        angles_deg=list(args.angles), physical_parameters_SI=production["physical_parameters_SI"],
        steady_defaults=defaults, steady_source_sha256=file_sha256(production["script"]),
        sweep_counts=counts, pv_relative_tolerance=pv_relative_tolerance,
        boundary_absolute_tolerance_W_per_m2=boundary_absolute_tolerance_W_per_m2,
        energy_absolute_tolerance_W=energy_absolute_tolerance_W,
        energy_relative_tolerance=energy_relative_tolerance, chunk_rtol=chunk_rtol,
        chunk_atol_m=chunk_atol_m,
        scope="Fixed steady modal settings; finite numerical references, no new PDE or FEM validation.",
    )
    summary = dict(kind="scientific_verification", verification_type="steady_numerical_settings",
                   status="RUNNING", config=config, provenance=provenance(repository_root, __file__))
    write_summary(directory, summary)
    print(f"Run: {directory}", flush=True)

    rows, checks, evidence = [], [], {}
    try:
        for angle in args.angles:
            start = perf_counter()
            audit_angle(p, angle, defaults, counts, rows, checks, evidence)
            print(f"Angle {angle:g} deg: {perf_counter()-start:.2f} s", flush=True)
        data = data_directory(directory)
        with (data/"comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(data/"verification_fields.npz", **evidence)
        report = validation_report(checks, config=config, run_directory=directory)
        write_json(directory/"verification.json", report)
        summary.update(status="COMPLETE", verification_status=report["status"],
                       artifacts={"comparison": "data/comparison.csv", "verification_fields": "data/verification_fields.npz"})
        write_summary(directory, summary)
        write_json(directory/"check_report.json", validate_saved_evidence_run(directory))
    except (Exception, KeyboardInterrupt) as exc:
        write_summary(directory, {**summary, "status": "FAILED", "error": str(exc)})
        raise

    print(f"Status: COMPLETE\nResults: {directory}\nScientific status: {report['status']}", flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
