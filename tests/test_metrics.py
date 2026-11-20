import numpy as np
import pytest

from thermal_opd.metrics import circular_aperture_mask, opd_peak_to_valley


def test_circular_aperture_includes_edge_and_excludes_corners():
    x = np.array([-1.0, 0.0, 1.0])[:, None]
    y = np.array([-1.0, 0.0, 1.0])[None, :]
    expected = np.array([[False, True, False], [True, True, True], [False, True, False]])

    assert np.array_equal(circular_aperture_mask(x, y, 1.0), expected)


def test_pv_uses_only_aperture_points_including_nonfinite_values_outside():
    opd = np.array([2e-9, 7e-9, 999e-9, np.nan])
    mask = np.array([True, True, False, False])

    assert np.isclose(opd_peak_to_valley(opd, mask), 5e-9, rtol=1e-12, atol=0.0)


def test_pv_is_unchanged_by_constant_piston():
    opd = np.array([2e-9, 7e-9, 4e-9])
    mask = np.ones(3, dtype=bool)

    assert np.isclose(opd_peak_to_valley(opd + 100e-9, mask), 5e-9, rtol=1e-12, atol=0.0)


def test_pv_rejects_empty_aperture():
    with pytest.raises(ValueError, match="empty"):
        opd_peak_to_valley(np.array([1e-9]), np.array([False]))


def test_pv_rejects_nonfinite_values_inside_aperture():
    with pytest.raises(ValueError, match="finite"):
        opd_peak_to_valley(np.array([1e-9, np.nan]), np.array([True, True]))


def test_common_optical_metrics_separate_piston_and_shape():
    from thermal_opd.metrics import opd_metrics
    values=np.array([0.,1.,2.,np.nan])*1e-9
    mask=np.array([True,True,True,False])
    metrics=opd_metrics(values,mask)
    shifted=opd_metrics(values+100e-9,mask)
    assert metrics['piston_removed_rms_m']==pytest.approx(np.sqrt(2/3)*1e-9)
    assert shifted['pv_m']==pytest.approx(metrics['pv_m'])
    assert shifted['piston_removed_rms_m']==pytest.approx(metrics['piston_removed_rms_m'])
    assert shifted['mean_opd_m']==pytest.approx(metrics['mean_opd_m']+100e-9)
