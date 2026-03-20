"""Fixed S1 single-shot layouts used by the Phase 1 bare core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pooltool as pt

S1PresetName = Literal["five_point_straight_in", "eight_ball_reference"]


@dataclass(frozen=True)
class S1Layout:
    """Container for a fixed S1 layout."""

    preset_name: S1PresetName
    layout_id: str
    cue_ball_pos: tuple[float, float]
    object_ball_pos: tuple[float, float]
    target_pocket_id: str
    target_pocket_pos: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "preset_name": self.preset_name,
            "layout_id": self.layout_id,
            "cue_ball_pos": self.cue_ball_pos,
            "object_ball_pos": self.object_ball_pos,
            "target_pocket_id": self.target_pocket_id,
            "target_pocket_pos": self.target_pocket_pos,
        }


def _get_table(table: pt.Table | None) -> pt.Table:
    return table if table is not None else pt.Table.default()


def make_five_point_straight_in_layout(table: pt.Table | None = None) -> S1Layout:
    """Build the fixed `five_point_straight_in` preset.

    This keeps a long corner-pocket straight-in drill while ensuring all generated
    coordinates stay on the table.
    """

    resolved_table = _get_table(table)
    target_pocket_id = "rb"
    target_pocket = resolved_table.pockets[target_pocket_id]
    pocket_x, pocket_y = float(target_pocket.center[0]), float(target_pocket.center[1])

    # Object ball in the right-lower half, leaving room for a longer cue distance.
    obj_x = resolved_table.w * 0.72
    obj_y = resolved_table.l * 0.68

    # Direction from object ball toward target pocket.
    vec_x = pocket_x - obj_x
    vec_y = pocket_y - obj_y
    norm = (vec_x**2 + vec_y**2) ** 0.5
    if norm == 0:
        vec_x, vec_y = 0.0, -1.0
        norm = 1.0
    unit_x = vec_x / norm
    unit_y = vec_y / norm

    # Place cue ball behind the object ball along the opposite direction.
    cue_distance = 0.35
    cue_x = obj_x - unit_x * cue_distance
    cue_y = obj_y - unit_y * cue_distance

    # Clamp inside playable area to avoid invalid out-of-table coordinates.
    ball_radius = pt.Ball.create("tmp").params.R
    cue_x = float(min(max(cue_x, ball_radius), resolved_table.w - ball_radius))
    cue_y = float(min(max(cue_y, ball_radius), resolved_table.l - ball_radius))

    return S1Layout(
        preset_name="five_point_straight_in",
        layout_id="five_point_straight_in_long_corner_v2",
        cue_ball_pos=(cue_x, cue_y),
        object_ball_pos=(float(obj_x), float(obj_y)),
        target_pocket_id=target_pocket_id,
        target_pocket_pos=(pocket_x, pocket_y),
    )


def make_eight_ball_reference_layout(table: pt.Table | None = None) -> S1Layout:
    """Build the fixed `eight_ball_reference` preset.

    Coordinate choices (8-ball reference semantics):
    - Target pocket fixed to `rt` (right-top corner pocket).
    - Object ball is placed on the table foot-spot reference (`x=w/2, y=3l/4`).
    - Cue ball is placed behind the head-string reference (`x=w/2, y=0.23l`), which
      approximates a legal break-area reference without modeling a full rack.
    """

    resolved_table = _get_table(table)
    target_pocket_id = "rt"
    target_pocket = resolved_table.pockets[target_pocket_id]
    target_x, target_y = float(target_pocket.center[0]), float(target_pocket.center[1])

    # Foot-spot-inspired object-ball reference point.
    obj_x = resolved_table.w * 0.50
    obj_y = resolved_table.l * 0.75

    # Head-string-behind cue reference point for a fixed opening position.
    cue_x = resolved_table.w * 0.50
    cue_y = resolved_table.l * 0.23

    return S1Layout(
        preset_name="eight_ball_reference",
        layout_id="eight_ball_reference_fixed_v1",
        cue_ball_pos=(float(cue_x), float(cue_y)),
        object_ball_pos=(float(obj_x), float(obj_y)),
        target_pocket_id=target_pocket_id,
        target_pocket_pos=(target_x, target_y),
    )


def build_s1_layout(
    preset_name: str,
    table: pt.Table | None = None,
) -> S1Layout:
    """Dispatch preset name to one of the fixed S1 layout builders."""

    if preset_name == "five_point_straight_in":
        return make_five_point_straight_in_layout(table=table)

    if preset_name == "eight_ball_reference":
        return make_eight_ball_reference_layout(table=table)

    raise ValueError(
        "Unsupported preset_name="
        f"{preset_name!r}. Expected 'five_point_straight_in' or "
        "'eight_ball_reference'."
    )
