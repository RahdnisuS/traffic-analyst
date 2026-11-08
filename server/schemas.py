"""WebSocket command schemas: browser -> server.

WHY a discriminated union: every message names its ``cmd``, Pydantic picks the
matching model and validates its fields, and anything malformed is rejected with
a readable error before it can reach the simulation.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SPEEDS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
VOLUME_MAX_VPH = 1500.0
GREEN_MAX_S = 90.0
PED_RATE_MAX = 1200.0
MixWeight = Annotated[float, Field(ge=0, le=100)]


class _Cmd(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlayCmd(_Cmd):
    cmd: Literal["play"]


class PauseCmd(_Cmd):
    cmd: Literal["pause"]


class SpeedCmd(_Cmd):
    cmd: Literal["speed"]
    value: float


class ResetCmd(_Cmd):
    cmd: Literal["reset"]
    seed: int | None = Field(None, ge=0, le=2**31 - 1)


class SettingsCmd(_Cmd):
    cmd: Literal["settings"]
    volume_vph: float | None = Field(None, ge=0, le=VOLUME_MAX_VPH)
    aggressiveness: float | None = Field(None, ge=0, le=1)
    green_s: float | None = Field(None, ge=0, le=GREEN_MAX_S)
    # Relative weights per vehicle type (e.g. percentages); the sim normalises them.
    vehicle_mix: dict[str, MixWeight] | None = Field(None, max_length=8)
    ped_rate_ph: float | None = Field(None, ge=0, le=PED_RATE_MAX)


class StressCmd(_Cmd):
    """Start or stop the live stress test (demand ramp until breakdown)."""
    cmd: Literal["stress"]
    action: Literal["start", "stop"]


class ControllerCmd(_Cmd):
    """AI on/off: which policy drives the live run. Takes over from the current state."""
    cmd: Literal["controller"]
    value: Literal["fixed", "rl"]


class CompareCmd(_Cmd):
    """Start or stop the side-by-side comparison (fixed timer vs AI on the same seed)."""
    cmd: Literal["compare"]
    action: Literal["start", "stop"]
    # On stop: which side keeps running ("fixed" | "rl"). None keeps the side the AI switch was on.
    keep: Literal["fixed", "rl"] | None = None


Command = Annotated[Union[PlayCmd, PauseCmd, SpeedCmd, ResetCmd, SettingsCmd, StressCmd, ControllerCmd,
                          CompareCmd],
                    Field(discriminator="cmd")]
command_adapter: TypeAdapter = TypeAdapter(Command)
