"""2D geometry helpers: arc-length parameterised polylines and Bezier turns.

WHY arc length: every vehicle's longitudinal state is a single number ``s``
(metres travelled along its current lane). Car-following becomes 1-D maths,
and (x, y, heading) is only computed when a renderer asks for it. That keeps
the physics simple and fast, and fully separate from drawing.
"""

from __future__ import annotations

import numpy as np


class Path:
    """A polyline with precomputed cumulative arc length."""

    __slots__ = ("points", "cum", "length", "_seg", "_seglen", "_heading")

    def __init__(self, points: np.ndarray | list) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            raise ValueError("Path needs an (n>=2, 2) array of points")
        seg = np.diff(pts, axis=0)
        seglen = np.hypot(seg[:, 0], seg[:, 1])
        if np.any(seglen <= 1e-9):
            raise ValueError("Path has duplicate consecutive points")
        self.points = pts
        self._seg = seg
        self._seglen = seglen
        self.cum = np.concatenate([[0.0], np.cumsum(seglen)])
        self.length = float(self.cum[-1])
        self._heading = np.arctan2(seg[:, 1], seg[:, 0])

    def poses_at(self, s: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised (x, y, heading) at arc lengths ``s`` (clamped to the path)."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        i = np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self._seglen) - 1)
        t = (s - self.cum[i]) / self._seglen[i]
        xy = self.points[i] + t[..., None] * self._seg[i]
        return xy[..., 0], xy[..., 1], self._heading[i]

    def pose_at(self, s: float) -> tuple[float, float, float]:
        x, y, h = self.poses_at(s)
        return float(x), float(y), float(h)

    def sample(self, spacing: float = 0.5) -> np.ndarray:
        """Evenly spaced points along the path, as an (n, 2) array (used for conflict tests)."""
        n = max(2, int(np.ceil(self.length / spacing)) + 1)
        x, y, _ = self.poses_at(np.linspace(0.0, self.length, n))
        return np.stack([x, y], axis=1)

    def total_turn(self) -> float:
        """Absolute heading change from start to end (radians)."""
        d = np.diff(self._heading)
        d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap into [-pi, pi)
        return float(abs(d.sum()))


def quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 24) -> np.ndarray:
    """Points on a quadratic Bezier. Its tangents at the ends point along p0->p1 and
    p1->p2, so a turn leaves the stop line straight and joins the exit lane straight."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def line_intersection(p: np.ndarray, d: np.ndarray, q: np.ndarray, e: np.ndarray) -> np.ndarray | None:
    """Intersection of lines p + t*d and q + u*e, or None if they are parallel."""
    m = np.array([[d[0], -e[0]], [d[1], -e[1]]])
    if abs(np.linalg.det(m)) < 1e-9:
        return None
    t, _ = np.linalg.solve(m, q - p)
    return p + t * d


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between two point sets (brute force; fine for ~50x50 points)."""
    diff = a[:, None, :] - b[None, :, :]
    return float(np.sqrt((diff ** 2).sum(axis=2)).min())
