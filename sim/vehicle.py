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

    @property
    def edge_id(self) -> str:
        return self.route[self.route_idx]

    @property
    def length(self) -> float:
        return self.params.length
