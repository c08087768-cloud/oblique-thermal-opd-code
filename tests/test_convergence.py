import numpy as np
import pytest

from thermal_opd import convergence


def test_field_metrics_weight_temperature_and_remove_only_opd_piston():
    # Replacing weighted errors by ordinary means or retaining piston must fail.
    error = convergence.field_errors(
        np.array([[2., 6.]]), np.array([[2., 4.]]), np.array([1., 3.]),
        np.array([11., 14., 17.]), np.array([1., 3., 5.]),
    )
    assert error['temperature_relative_l2'] == pytest.approx([np.sqrt(12 / 52)])
    assert error['temperature_relative_linf'] == pytest.approx([0.5])
    assert error['opd_piston_removed_rms_over_reference_pv'] == pytest.approx(np.sqrt(2/3)/4)
    assert error['opd_piston_removed_linf_over_reference_pv'] == pytest.approx(0.25)
    assert error['opd_relative_pv'] == pytest.approx(0.5)


def test_joint_acceptance_checks_every_field_and_plane():
    good = {'temperature_relative_l2': [0., 0.0001, 0.],
            'temperature_relative_linf': [0., 0., 0.],
            'opd_piston_removed_rms_over_reference_pv': 0.,
            'opd_piston_removed_linf_over_reference_pv': 0., 'opd_relative_pv': 0.}
    assert convergence.within_tolerance(good, 0.001)
    assert not convergence.within_tolerance({**good, 'temperature_relative_linf': [0., 0.002, 0.]}, 0.001)
    assert not convergence.within_tolerance({**good, 'opd_relative_pv': np.nan}, 0.001)


def test_continuous_pv_recovers_quadratic_extrema_between_grid_points():
    result = convergence.continuous_aperture_pv(
        lambda x, y: (x - 0.13)**2 + 2*(y + 0.09)**2,
        radius=1., thickness=0.2, refraction_angle=0., aperture_radius=0.9,
        num_seed_radius=7, num_seed_phi=24,
    )
    assert result['minimum'] == pytest.approx(0., abs=1e-10)
    # Independently maximize the boundary's one-variable polynomial by dense scan.
    phi = np.linspace(0, 2*np.pi, 200001)
    want = np.max((0.9*np.cos(phi)-0.13)**2 + 2*(0.9*np.sin(phi)+0.09)**2)
    assert result['pv'] == pytest.approx(want, rel=1e-8)


@pytest.mark.parametrize('angle', [np.arctan(0.3), -np.arctan(0.3)])
def test_continuous_pv_enforces_rear_transmission_for_signed_angle(angle):
    # R=1,L=1,d=+-0.3 clips y at +-0.7, leaving the other end at +-0.9.
    result = convergence.continuous_aperture_pv(
        lambda x, y: y, radius=1., thickness=1., refraction_angle=angle,
        aperture_radius=0.9, num_seed_radius=7, num_seed_phi=24,
    )
    assert result['pv'] == pytest.approx(1.6, abs=1e-8)
    assert result['maximum'] == pytest.approx(0.7 if angle > 0 else 0.9, abs=1e-8)


def test_settings_audit_rejects_a_wrong_nondefault_batch(monkeypatch):
    from scripts.convergence import check_steady_numerical_settings as audit
    from thermal_opd.experiments import experiment_configuration
    from thermal_opd.parameters import PhysicalParameters

    defaults = {**experiment_configuration("steady")["defaults"], "num_opd_xy": 9}
    counts = {name: (defaults[name], defaults[name]+1) for name in audit.sweep_counts}
    counts["opd_point_chunk_size"] = (128, defaults["opd_point_chunk_size"], 4096)

    class PlantedModel:
        coefficients = {}
        roots = {}
        refraction_angle = 0.

        def prepare(self):
            return self

        def solve_steady(self):
            return self

        def prepare_opd(self, *, point_chunk_size):
            return lambda x, y: x

        def opd(self, x, y, *, point_chunk_size):
            # Only the small batch is wrong. The production default is correct.
            return np.full_like(x, 2e-7 if point_chunk_size == 128 else 1e-7)

    monkeypatch.setattr(audit, "ThermalModel", lambda *args: PlantedModel())
    monkeypatch.setattr(audit, "steady_boundary_residuals", lambda *args: {"side_W_per_m2": 0.})
    monkeypatch.setattr(audit, "steady_energy_balance",
                        lambda *args, **kwargs: {"modal_input_W": 1., "balance_absolute_error_W": 0.})
    monkeypatch.setattr(audit, "gaussian_disk_power", lambda *args, **kwargs: 1.)

    rows, checks, evidence = [], [], {}
    audit.audit_angle(PhysicalParameters(), 0., defaults, counts, rows, checks, evidence)
    assert len(checks) == 9
    assert {check["name"] for check in checks if check["status"] == "failed"} == {
        "theta0_opd_point_chunk_size",
    }
    assert all(row["used_for_gate"] for row in rows if row["parameter"] == "opd_point_chunk_size")
