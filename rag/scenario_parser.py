"""Plain-English scenario -> a validated settings proposal the user confirms.

"Evening rush, 60% bikes, school crossing" becomes: volume 520 veh/h per approach, mix with
bikes at 60%, pedestrians 900 per hour, plus the assumptions behind each value. The model only
PROPOSES: the result is validated by ``ScenarioProposal`` (Pydantic: types, ranges, the same
limits as the dashboard sliders, a mix that sums to 100%), shown to the user as a table of
current -> proposed values with the assumptions, and applied only when the user presses Apply
(through the same validated ``settings`` / ``controller`` commands the sliders send).

WHY the model may choose numbers here although the analyst may never invent them in answers:
these are SETTINGS the user asked for and confirms, not claims about what happened. Each one
comes with its assumption in words ("evening rush: near capacity, ~450 veh/h per approach"), and
the context the model gets (current settings, slider limits, the measured capacity) is real.

If validation fails, the model gets one retry with the validation errors; then the request fails
with a readable message instead of applying something half-valid.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from server.schemas import GREEN_MAX_S, PED_RATE_MAX, VOLUME_MAX_VPH

from .llm import GroqClient, LLMError

VEHICLE_TYPES = ("auto", "bike", "bus", "car")
LABELS = {"volume_vph": "Traffic volume (veh/h per approach)", "vehicle_mix": "Vehicle mix (%)",
          "ped_rate_ph": "Pedestrians per hour", "aggressiveness": "Driver aggressiveness (0-1)",
          "green_s": "Fixed-timer green (s)", "controller": "Signal control"}


class ScenarioProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., min_length=1, max_length=80)
    volume_vph: float | None = Field(None, ge=0, le=VOLUME_MAX_VPH)
    vehicle_mix: dict[Literal["auto", "bike", "bus", "car"], float] | None = None
    ped_rate_ph: float | None = Field(None, ge=0, le=PED_RATE_MAX)
    aggressiveness: float | None = Field(None, ge=0, le=1)
    green_s: float | None = Field(None, le=GREEN_MAX_S)
    controller: Literal["fixed", "rl"] | None = None
    assumptions: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("vehicle_mix")
    @classmethod
    def _mix(cls, v):
        if v is None:
            return v
        if any(x < 0 for x in v.values()):
            raise ValueError("vehicle_mix percentages must be >= 0")
        total = sum(v.values())
        if not 95 <= total <= 105:
            raise ValueError(f"vehicle_mix percentages must sum to 100 (got {total:g})")
        full = {t: float(v.get(t, 0.0)) for t in VEHICLE_TYPES}
        return {t: round(100 * x / total, 1) for t, x in full.items()}  # exactly 100 after rounding noise


def _system(current: dict, limits: dict, context: dict) -> str:
    return f"""You turn a plain-English traffic scenario into settings for a junction simulator
(4-way junction, left-hand Indian traffic: cars, bikes = motorcycles/scooters, autos = auto-rickshaws, buses;
pedestrians cross in an exclusive pedestrian phase).

Return a JSON object with these keys (use null for anything the request does not imply; never change
what the request does not mention):
  "title": short name for the scenario,
  "volume_vph": mean traffic per approach in vehicles/hour (0-{limits['volume_vph'][1]:g}),
  "vehicle_mix": percentages for car, bike, auto, bus summing to 100 (only if the mix is implied),
  "ped_rate_ph": pedestrians per hour across all crossings (0-{limits['ped_rate_ph'][1]:g}),
  "aggressiveness": driver aggressiveness 0 (timid) to 1 (aggressive), 0.5 = normal,
  "green_s": fixed-timer green per phase in seconds ({limits['green_s'][0]:g}-{limits['green_s'][1]:g}),
  "controller": "rl" if the user wants the AI controlling the signals, "fixed" for the fixed timer, else null,
  "assumptions": one short sentence per value you set, saying how you read the request.

Facts about this junction you may use:
- Current settings: {json.dumps(current)}
- Measured capacity: {context.get('capacity')}. Quiet traffic is ~150-250 veh/h per approach, normal
  ~350-450, a rush hour is near or above capacity.
- Default vehicle mix: car 35, bike 45, auto 15, bus 5 (%).
- Typical pedestrian rates: quiet 100-200/h, busy street 300-500/h, school or market crossing 800-1200/h.
If a value is stated explicitly ("60% bikes"), use it exactly; scale the other mix shares to keep
their proportions."""


def parse_scenario(llm: GroqClient, request: str, current: dict, limits: dict, context: dict) -> dict:
    """{"proposal": ScenarioProposal as dict, "changes": [...], "model": ...}; raises LLMError."""
    messages = [{"role": "system", "content": _system(current, limits, context)},
                {"role": "user", "content": request}]
    last_error = None
    for _attempt in range(2):
        data, reply = llm.chat_json(messages, temperature=0.1, max_tokens=700)
        data = {k: v for k, v in data.items() if k in ScenarioProposal.model_fields}
        try:
            p = ScenarioProposal.model_validate(data)
            if p.green_s is not None and p.green_s < limits["green_s"][0]:
                raise ValueError(f"green_s must be at least {limits['green_s'][0]:g} s (minimum green)")
            break
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            messages += [{"role": "assistant", "content": reply.text},
                         {"role": "user", "content": f"That is invalid: {last_error}. Return corrected JSON."}]
    else:
        raise LLMError(f"Could not turn that into valid settings ({last_error}). Try rephrasing it.")
    return {"proposal": p.model_dump(), "changes": diff(p, current), "model": reply.model}


def diff(p: ScenarioProposal, current: dict) -> list[dict]:
    """Rows for the confirmation card: only settings that would actually change."""
    rows = []
    for key in ("volume_vph", "vehicle_mix", "ped_rate_ph", "aggressiveness", "green_s", "controller"):
        new = getattr(p, key)
        if new is None:
            continue
        old = current.get(key)
        if key == "vehicle_mix":
            old = {t: round(float((old or {}).get(t, 0.0)), 1) for t in VEHICLE_TYPES}
            if all(abs(old[t] - new[t]) < 0.5 for t in VEHICLE_TYPES):
                continue
        elif isinstance(new, float) and old is not None and abs(float(old) - new) < 1e-6:
            continue
        elif new == old:
            continue
        rows.append({"key": key, "label": LABELS[key], "current": old, "proposed": new})
    return rows
