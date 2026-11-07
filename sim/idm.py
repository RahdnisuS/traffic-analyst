"""Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000).

IDM gives each driver an acceleration from three things: their own speed v,
the gap s to the vehicle ahead, and the approach rate dv = v - v_leader:

    a_IDM = a * [ 1 - (v / v0)^delta - (s*(v, dv) / s)^2 ]

    s*(v, dv) = s0 + max(0,  v*T + v*dv / (2*sqrt(a*b)) )

* ``1 - (v/v0)^delta``: the free-road term. Accelerate until you reach v0.
* ``(s*/s)^2``: the interaction term. s* is the gap you would like to have.
  If the real gap s is smaller, this term dominates and you brake.
* ``v*T``: keep a time headway of T seconds.
* ``v*dv/(2*sqrt(ab))``: the "intelligent" part. If you are closing in fast,
  start braking early, so a typical stop needs only about b of deceleration.

All functions are vectorised over NumPy arrays. One call computes the
accelerations of every vehicle in the network.
"""

from __future__ import annotations

import numpy as np

_MIN_GAP = 0.01  # guards against divide-by-zero if two bumpers touch


def desired_gap(v, dv, a, b, T, s0):
    """s*(v, dv): the gap the driver wants to keep."""
    return s0 + np.maximum(0.0, v * T + v * dv / (2.0 * np.sqrt(a * b)))


def idm_acceleration(v, v0, gap, dv, a, b, T, s0, delta):
    """IDM acceleration. ``gap`` may be ``np.inf`` (no leader); the interaction term is then 0."""
    v = np.asarray(v, dtype=float)
    gap = np.maximum(np.asarray(gap, dtype=float), _MIN_GAP)
    free = 1.0 - (v / v0) ** delta
    interaction = (desired_gap(v, dv, a, b, T, s0) / gap) ** 2
    return a * (free - interaction)


def ballistic_step(v, acc, dt):
    """Advance speed and position by one step with constant acceleration.

    WHY ballistic (not plain Euler): Treiber recommends it for IDM. Position uses
    the exact constant-acceleration formula, and if a vehicle would reverse within
    the step we stop it at the exact point where v reaches 0. Without that, vehicles
    queued at a red light would creep backwards.
    Returns (distance travelled, new speed).
    """
    v = np.asarray(v, dtype=float)
    acc = np.asarray(acc, dtype=float)
    v_new = v + acc * dt
    ds = v * dt + 0.5 * acc * dt * dt
    stops = v_new < 0.0
    if np.any(stops):
        # Comes to rest mid-step: distance = v^2 / (2|a|). ``acc`` is negative here.
        ds = np.where(stops, -0.5 * v * v / np.minimum(acc, -1e-9), ds)
        v_new = np.where(stops, 0.0, v_new)
    return ds, v_new


def equilibrium_gap(v, v0, a, b, T, s0, delta):
    """Steady-state gap at speed v when following a leader at the same speed (dv=0).
    Setting a_IDM = 0 gives s_e = s*(v,0) / sqrt(1 - (v/v0)^delta). Used in tests."""
    return desired_gap(v, 0.0, a, b, T, s0) / np.sqrt(1.0 - (v / v0) ** delta)
