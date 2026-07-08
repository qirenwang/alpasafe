import numpy as np

from safeworld.geometry.trajectory import accelerations, as_xy_array, progress, speeds


def test_constant_speed_trajectory() -> None:
    trajectory = [[float(i), 0.0] for i in range(8)]
    arr = as_xy_array(trajectory)
    assert np.allclose(speeds(arr), 10.0)
    assert np.allclose(accelerations(arr), 0.0)
    assert progress(arr) == 7.0

