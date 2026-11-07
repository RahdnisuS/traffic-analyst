"""Vehicle state. Pure data: all behaviour lives in the engine and the IDM module."""

from __future__ import annotations

from dataclasses import dataclass

from .config import VehicleTypeParams


@dataclass(slots=True, eq=False)
class Vehicle:
    id: int
    vtype: str
    params: VehicleTypeParams
    route: list[str]              # [inbound, connector, outbound] edge ids
    origin: str                   # arm the vehicle arrived from
    movement: str                 # left | straight | right
    desired_speed: float          # personal v0 (type v0 x driver speed factor)
    arrival_time: float           # when it joined the demand queue (s)
    insert_time: float            # when it physically entered the road (s)
    route_idx: int = 0
    s: float = 0.0                # front-bumper position along current edge (m)
    v: float = 0.0                # speed (m/s)
    a: float = 0.0                # last acceleration (m/s^2)
    committed: bool = False       # has decided to enter the junction box (can't take it back)
    commit_signal: str = ""       # signal state ('G' or 'Y') at the moment it committed
    wait_time: float = 0.0        # accumulated time spent nearly stopped (s)
    free_flow_time: float = 0.0   # time the route would take at desired speed, no interference
    stops: int = 0                # number of separate times the vehicle came to a halt
    is_stopped: bool = False

    @property
    def edge_id(self) -> str:
        return self.route[self.route_idx]

    @property
    def length(self) -> float:
        return self.params.length


@dataclass(slots=True, frozen=True)
class TripRecord:
    """One completed trip. Metric definitions (used everywhere downstream):

    travel_time = exit_time - arrival_time (includes time waiting to enter the road)
    entry_delay = insert_time - arrival_time (time stuck off-network because the lane was full)
    wait_time   = time on the road with speed < 0.1 m/s
    delay       = travel_time - free_flow_time ("control delay": the time lost to the junction).
                  Free-flow ignores accelerating and braking, so even an unimpeded car shows ~1-3 s.
    """
    id: int
    vtype: str
    origin: str
    movement: str
    exit_arm: str
    arrival_time: float
    insert_time: float
    exit_time: float
    travel_time: float
    entry_delay: float
    wait_time: float
    delay: float
    stops: int
