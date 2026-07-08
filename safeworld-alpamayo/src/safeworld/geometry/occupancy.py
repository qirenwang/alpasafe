from __future__ import annotations

import numpy as np


def path_aligned_occupancy(trajectory: list[list[float]], grid_size: int = 16, resolution_m: float = 1.5) -> list[list[int]]:
    grid = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    for x, y, *_ in trajectory:
        gx = int(round(float(x) / resolution_m))
        gy = int(round(float(y) / resolution_m)) + center
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            grid[gx, gy] = 1
    return grid.tolist()

