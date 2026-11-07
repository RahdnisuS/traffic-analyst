"""Pedestrians: the social force model (Helbing & Molnar, 1995), simplified.

Each walker i is a disc of radius r moving in 2D. Its acceleration is a sum of "forces"
(really accelerations: everything is divided by body mass, so mass never appears):

    dv_i/dt = (v0_i * e_i - v_i) / tau                        (1) driving force
            + sum_j  A * exp((2r - d_ij) / B) * w_ij * n_ij    (2) other people
            + sum_W  A_w * exp((r - d_iW) / B_w) * n_iW        (3) walls (kerbs, crosswalk edges)

(1) Pull towards the goal: e_i is the unit vector to the goal and v0_i the walker's preferred
    speed. If the actual velocity v_i differs, the walker corrects it within about tau seconds.
(2) Push away from each neighbour j. n_ij points from j to i, d_ij is the distance between
    centres, so (2r - d_ij) is how much the two discs would overlap. The push is strong at
    contact and fades exponentially with range B. We use the ELLIPTICAL specification
    (Johansson, Helbing & Shukla 2007): d_ij is replaced by an effective distance b_ij that
    shrinks when j is closing in, so people sidestep an oncoming walker early instead of
    braking at the last moment (the plain circular version is poor at head-on encounters).
    w_ij is the anisotropy factor:
        w_ij = lambda + (1 - lambda) * (1 + cos(phi_ij)) / 2
    where phi_ij is the angle between where i is heading and where j is. Someone straight ahead
    gets weight 1; someone directly behind gets lambda (0.35). People react to what they see.
(3) The same exponential push from a wall, along the wall's normal.

What emerges without being programmed: people slow and spread out in a crowd, keep a personal
distance while waiting at the kerb, and two opposing flows on a crosswalk split into lanes
(walkers fall in behind someone going their way because that costs the least repulsion).

Simplifications (be ready to defend these): no physical contact/friction forces (those matter for
panic crushes, not for a crossing), walls are straight half-planes, and vehicles are not force
sources. Vehicles and walkers are kept apart by RULES instead: walkers only start on WALK and
only when no vehicle is in their crosswalk; drivers yield to anyone still crossing. A safety rule
is easier to verify than a force balance.

Life of a walker: APPROACH (walk along the sidewalk to the kerb) -> WAIT (press the button) ->
CROSS (on WALK) -> LEAVE (walk off along the far sidewalk) -> despawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .config import PedestrianSettings
from .junction import ARM_VECTORS, Crosswalk, Junction

ARRIVE_M = 0.6        # close enough to the waiting point to stop
CROWD_STOP_M = 3.0    # ... or within this and blocked by the crowd (speed below CROWD_SPEED)
CROWD_SPEED = 0.25
KERB_BACK_M = 0.5     # waiting point sits this far back from the kerb edge
DESPAWN_M = 1.0
HURRY = 1.15          # walkers speed up a little on flashing DON'T WALK
ALONG_JITTER = 1.0    # m: spread along the kerb, so a crowd doesn't aim at one point


class PedState(IntEnum):
    APPROACH = 0
    WAIT = 1
    CROSS = 2
    LEAVE = 3


@dataclass(slots=True, eq=False)
class Pedestrian:
    id: int
    arm: str                  # which crosswalk
    side: int                 # +1 starts on the "hi" side of the crosswalk, -1 on the "lo" side
    desired_speed: float
    along: float              # this walker's lane along the crosswalk (u-coordinate)
    spawn_time: float
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    state: PedState = PedState.APPROACH
    kerb_time: float | None = None
    cross_start: float | None = None
    cross_end: float | None = None


@dataclass(slots=True, frozen=True)
class PedTripRecord:
    """One finished pedestrian trip.
    wait_time     = cross_start - kerb_time (time standing at the kerb: the signal's cost to walkers)
    crossing_time = cross_end - cross_start (time on the carriageway)
    """
    id: int
    crosswalk: str
    from_side: str
    spawn_time: float
    kerb_time: float
    cross_start: float
    cross_end: float
    exit_time: float
    wait_time: float
    crossing_time: float
    travel_time: float


# ------------------------------------------------------------------ the model itself
def social_force(pos, vel, desired_vel, p: PedestrianSettings, wall_n, wall_off, wall_mask):
    """Social-force acceleration for every walker at once (vectorised, O(N^2) pairs).

    pos, vel, desired_vel: (N, 2). wall_n: (N, K, 2) unit normals pointing INTO the walkable
    side; wall_off: (N, K) so that a walker is on the walkable side when n . x >= off;
    wall_mask: (N, K) which walls apply to which walker.
    """
    acc = (desired_vel - vel) / p.relaxation_time                                  # (1)
    n = len(pos)
    if n > 1:
        diff = pos[:, None, :] - pos[None, :, :]                                   # d_ij = x_i - x_j
        d = np.hypot(diff[..., 0], diff[..., 1])
        np.fill_diagonal(d, np.inf)
        safe = np.maximum(d, 1e-9)
        d_hat = np.where((d > 1e-9)[..., None], diff / safe[..., None], 0.0)
        # Elliptical specification (anticipation): y_ij = (v_j - v_i) * dt_a is how the gap will
        # change over the next dt_a seconds. The effective distance
        #     b = 0.5 * sqrt((|d| + |d - y|)^2 - |y|^2)
        # is smaller than |d| for someone closing in, so walkers react BEFORE a head-on
        # encounter instead of only braking at the last moment. y = 0 gives b = |d| (circular).
        y = (vel[None, :, :] - vel[:, None, :]) * p.anticipation_s
        dy = diff - y
        ndy = np.hypot(dy[..., 0], dy[..., 1])
        ny = np.hypot(y[..., 0], y[..., 1])
        s_sum = np.minimum(d, 1e6) + ndy
        b = 0.5 * np.sqrt(np.maximum(s_sum ** 2 - ny ** 2, 1e-12))
        b = np.where(np.isfinite(d), b, np.inf)
        dy_hat = np.where((ndy > 1e-9)[..., None], dy / np.maximum(ndy, 1e-9)[..., None], d_hat)
        # Force = -grad of the potential A*B*exp(-b/B); grad_d b = (|d|+|d-y|)/(4b) * (d_hat + dy_hat)
        grad = (s_sum / (4.0 * np.maximum(b, 1e-6)))[..., None] * (d_hat + dy_hat)
        grad = np.where(np.isfinite(d)[..., None], grad, 0.0)
        n_ij = d_hat
        d = b
        # Heading e_i: where the walker wants to go (or is going, if standing still).
        speed_des = np.hypot(desired_vel[:, 0], desired_vel[:, 1])
        speed = np.hypot(vel[:, 0], vel[:, 1])
        e = np.where((speed_des > 1e-6)[:, None], desired_vel / np.maximum(speed_des, 1e-9)[:, None],
                     np.where((speed > 1e-6)[:, None], vel / np.maximum(speed, 1e-9)[:, None], 0.0))
        cos_phi = -(n_ij * e[:, None, :]).sum(axis=2)  # 1 when j is straight ahead of i
        w = p.anisotropy + (1.0 - p.anisotropy) * (1.0 + cos_phi) / 2.0
        mag = p.repulsion_strength * np.exp(np.minimum((2 * p.radius - d) / p.repulsion_range, 20.0)) * w
        acc = acc + (mag[..., None] * grad).sum(axis=1)                            # (2)
    if wall_n.shape[1]:
        dw = (wall_n * pos[:, None, :]).sum(axis=2) - wall_off                     # distance to each wall
        mag_w = p.wall_strength * np.exp(np.minimum((p.radius - dw) / p.wall_range, 20.0)) * wall_mask
        acc = acc + (mag_w[..., None] * wall_n).sum(axis=1)                        # (3)
    return acc


# ------------------------------------------------------------------ the population
class PedestrianSystem:
    """Spawns walkers, runs the social force model, and applies the crossing rules."""

    def __init__(self, junction: Junction, settings: PedestrianSettings, rng: np.random.Generator):
        self.junction = junction
        self.s = settings
        self.rng = rng
        self.crosswalks: dict[str, Crosswalk] = junction.crosswalks
        self.arms = sorted(self.crosswalks)          # sorted: fixed RNG draw order -> determinism
        self.sidewalk = junction.spec.sidewalk_width
        self.peds: list[Pedestrian] = []             # always ordered by id
        self._next_id = 0
        self.spawned = 0
        self.completed = 0
        self.crossing_arms: frozenset[str] = frozenset()  # crosswalks with someone on the road
        self.recent_trips: list[PedTripRecord] = []       # finished this step (observers read)
        self.recent_starts: list[tuple[float, float]] = []  # (time, wait) of crossings started this step
        self.walk_log: list[dict] = []                    # one entry per WALK interval served
        self._prev_signal = "dont_walk"

    # ------------------------------------------------------------------ geometry helpers
    def _kerb(self, cw: Crosswalk, side: int) -> float:
        return cw.hi if side > 0 else cw.lo

    def _wait_point(self, ped: Pedestrian) -> np.ndarray:
        cw = self.crosswalks[ped.arm]
        return cw.point(ped.along, self._kerb(cw, ped.side) + ped.side * KERB_BACK_M)

    def _far_point(self, ped: Pedestrian) -> np.ndarray:
        cw = self.crosswalks[ped.arm]
        return cw.point(ped.along, self._kerb(cw, -ped.side) - ped.side * KERB_BACK_M)

    def _exit_point(self, ped: Pedestrian) -> np.ndarray:
        cw = self.crosswalks[ped.arm]
        return cw.point(cw.centre_u + self.s.sidewalk_approach_m,
                        self._kerb(cw, -ped.side) - ped.side * self.sidewalk / 2)

    def side_name(self, arm: str, side: int) -> str:
        """Compass name of a crosswalk end, e.g. walkers on the N crosswalk start 'E' or 'W'."""
        v = self.crosswalks[arm].n * side
        return next(k for k, u in ARM_VECTORS.items() if np.allclose(u, v))

    # ------------------------------------------------------------------ demand
    def set_rate(self, rate_ph: float) -> None:
        if rate_ph < 0:
            raise ValueError("pedestrian rate must be >= 0")
        self.s = self.s.model_copy(update={"rate_ph": float(rate_ph)})

    def _spawn(self, now: float, dt: float) -> None:
        if not self.arms:
            return
        rate = self.s.rate_ph / 3600.0 / (2 * len(self.arms))  # per crosswalk end
        for arm in self.arms:
            cw = self.crosswalks[arm]
            for side in (1, -1):
                for _ in range(int(self.rng.poisson(rate * dt))):
                    v0 = float(np.clip(self.rng.normal(self.s.desired_speed_mean, self.s.desired_speed_std),
                                       0.7, 2.0))
                    along = cw.centre_u + float(self.rng.uniform(-ALONG_JITTER, ALONG_JITTER))
                    start_along = cw.centre_u + self.s.sidewalk_approach_m + float(self.rng.uniform(-2, 2))
                    across = self._kerb(cw, side) + side * float(self.rng.uniform(0.6, self.sidewalk - 0.6))
                    self.peds.append(Pedestrian(self._next_id, arm, side, v0, along, now,
                                                cw.point(start_along, across)))
                    self._next_id += 1
                    self.spawned += 1

    # ------------------------------------------------------------------ one step
    def update(self, sim, now: float, dt: float) -> None:
        self.recent_trips = []
        self.recent_starts = []
        signals = sim.signals
        light = signals.ped_signal()
        self._spawn(now, dt)

        if light == "walk" and self._prev_signal != "walk":
            waiting = [p for p in self.peds if p.state is PedState.WAIT]
            self.walk_log.append({"t": round(now, 1), "waiting": len(waiting),
                                  "max_wait": round(max((now - p.kerb_time for p in waiting), default=0.0), 1)})
        self._prev_signal = light

        # Crosswalks with a vehicle on (or committed to) a path over them: don't step off.
        active = sim.active_connectors()
        busy = {a for a in self.arms if self.junction.crosswalk_connectors[a] & active}
        for p in self.peds:
            if p.state is PedState.APPROACH:
                dist = float(np.hypot(*(self._wait_point(p) - p.pos)))
                if dist < ARRIVE_M or (dist < CROWD_STOP_M and float(np.hypot(*p.vel)) < CROWD_SPEED):
                    p.state, p.kerb_time = PedState.WAIT, now
            if p.state is PedState.WAIT and light == "walk" and p.arm not in busy:
                p.state, p.cross_start = PedState.CROSS, now
                self.recent_starts.append((now, now - p.kerb_time))

        if self.peds:
            self._move(dt, light)

        keep = []
        for p in self.peds:
            cw = self.crosswalks[p.arm]
            if p.state is PedState.CROSS and p.side * (float(cw.n @ p.pos) - self._kerb(cw, -p.side)) < 0:
                p.state, p.cross_end = PedState.LEAVE, now  # reached the far sidewalk
            if p.state is PedState.LEAVE and float(np.hypot(*(self._exit_point(p) - p.pos))) < DESPAWN_M:
                self._finish(p, now)
                continue
            keep.append(p)
        self.peds = keep

        if any(p.state is PedState.WAIT for p in self.peds):
            signals.call_pedestrian()
        self.crossing_arms = frozenset(p.arm for p in self.peds if p.state is PedState.CROSS)

    def _move(self, dt: float, light: str) -> None:
        peds, s = self.peds, self.s
        n = len(peds)
        pos = np.array([p.pos for p in peds])
        vel = np.array([p.vel for p in peds])
        v0 = np.array([p.desired_speed for p in peds])
        goal = np.empty((n, 2))
        speed_scale = np.ones(n)
        wall_n = np.zeros((n, 2, 2))
        wall_off = np.zeros((n, 2))
        wall_mask = np.zeros((n, 2))
        for i, p in enumerate(peds):
            cw = self.crosswalks[p.arm]
            if p.state is PedState.CROSS:
                goal[i] = self._far_point(p)
                # Stay inside the zebra strip: walls on both long edges.
                wall_n[i, 0], wall_off[i, 0] = cw.u, cw.u0
                wall_n[i, 1], wall_off[i, 1] = -cw.u, -cw.u1
                wall_mask[i] = 1.0
                if light == "flash":
                    speed_scale[i] = HURRY
            else:
                side = p.side if p.state is not PedState.LEAVE else -p.side
                goal[i] = self._wait_point(p) if p.state is not PedState.LEAVE else self._exit_point(p)
                # The kerb: stay on the sidewalk, off the carriageway.
                wall_n[i, 0], wall_off[i, 0] = cw.n * side, side * self._kerb(cw, side)
                wall_mask[i, 0] = 1.0
                if p.state is PedState.WAIT:
                    speed_scale[i] = 0.0  # standing: desired velocity is zero
        to_goal = goal - pos
        dist = np.hypot(to_goal[:, 0], to_goal[:, 1])
        e = to_goal / np.maximum(dist, 1e-9)[:, None]
        # Ease off in the last metre so nobody overshoots their waiting point.
        desired = e * (v0 * speed_scale * np.minimum(1.0, dist))[:, None]

        acc = social_force(pos, vel, desired, s, wall_n, wall_off, wall_mask)
        vel = vel + acc * dt
        speed = np.hypot(vel[:, 0], vel[:, 1])
        vmax = v0 * s.max_speed_factor * np.maximum(speed_scale, 1.0)
        vel = vel * np.minimum(1.0, vmax / np.maximum(speed, 1e-9))[:, None]
        pos = pos + vel * dt
        # Hard constraint backing up the soft wall force: never end a step inside a wall,
        # and drop the velocity component pointing into it.
        dw = (wall_n * pos[:, None, :]).sum(axis=2) - wall_off
        push = np.where(wall_mask > 0, np.maximum(0.0, s.radius - dw), 0.0)
        pos = pos + (push[..., None] * wall_n).sum(axis=1)
        into = np.where((push > 0), np.minimum(0.0, (wall_n * vel[:, None, :]).sum(axis=2)), 0.0)
        vel = vel - (into[..., None] * wall_n).sum(axis=1)
        for i, p in enumerate(peds):
            p.pos, p.vel = pos[i], vel[i]

    def _finish(self, p: Pedestrian, now: float) -> None:
        self.completed += 1
        self.recent_trips.append(PedTripRecord(
            id=p.id, crosswalk=p.arm, from_side=self.side_name(p.arm, p.side),
            spawn_time=round(p.spawn_time, 3), kerb_time=round(p.kerb_time, 3),
            cross_start=round(p.cross_start, 3), cross_end=round(p.cross_end, 3),
            exit_time=round(now, 3), wait_time=round(p.cross_start - p.kerb_time, 3),
            crossing_time=round(p.cross_end - p.cross_start, 3),
            travel_time=round(now - p.spawn_time, 3)))

    # ------------------------------------------------------------------ queries
    def count(self, state: PedState) -> int:
        return sum(1 for p in self.peds if p.state is state)

    def longest_wait(self, now: float) -> float:
        return max((now - p.kerb_time for p in self.peds if p.state is PedState.WAIT), default=0.0)

    def max_crossing_length(self) -> float:
        return max((cw.length for cw in self.crosswalks.values()), default=0.0)
