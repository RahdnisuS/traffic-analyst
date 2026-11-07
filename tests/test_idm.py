import numpy as np

from sim.idm import ballistic_step, equilibrium_gap, idm_acceleration

P = dict(a=1.5, b=2.0, T=1.5, s0=2.0, delta=4.0)


def test_free_road_from_rest_gives_max_acceleration():
    assert np.isclose(idm_acceleration(0.0, 13.9, np.inf, 0.0, **P), P["a"])


def test_at_desired_speed_on_free_road_acceleration_is_zero():
    assert np.isclose(idm_acceleration(13.9, 13.9, np.inf, 0.0, **P), 0.0)


def test_equilibrium_gap_gives_zero_acceleration():
    v = 10.0
    s_e = equilibrium_gap(v, 13.9, P["a"], P["b"], P["T"], P["s0"], P["delta"])
    assert abs(idm_acceleration(v, 13.9, s_e, 0.0, **P)) < 1e-9


def test_closing_fast_brakes_harder_than_matching_speed():
    calm = idm_acceleration(10.0, 13.9, 30.0, 0.0, **P)
    closing = idm_acceleration(10.0, 13.9, 30.0, 5.0, **P)
    assert closing < calm


def test_vehicle_stops_behind_stationary_obstacle_without_collision():
    """Drive at 13.9 m/s towards a wall 120 m away: it must stop before the wall, close to s0."""
    x, v, wall, dt = 0.0, 13.9, 120.0, 0.1
    for _ in range(2000):
        a = max(idm_acceleration(v, 13.9, wall - x, v, **P), -8.0)
        ds, v = ballistic_step(np.array([v]), np.array([a]), dt)
        x, v = x + float(ds[0]), float(v[0])
        assert x < wall
    assert v < 0.01
    assert 0.5 < wall - x <= P["s0"] + 0.5


def test_ballistic_step_never_reverses():
    ds, v = ballistic_step(np.array([1.0]), np.array([-8.0]), 0.5)
    assert v[0] == 0.0
    assert np.isclose(ds[0], 1.0 / 16.0)  # v^2 / (2|a|)
