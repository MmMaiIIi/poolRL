"""Tests for fixed S1 layout generation."""

from __future__ import annotations

import pooltool as pt

from MLAgent.envs.layout_generators import (
    S1Layout,
    make_eight_ball_reference_layout,
    make_five_point_straight_in_layout,
)


def _assert_layout_valid(layout: S1Layout, expected_preset: str) -> None:
    table = pt.Table.default()

    assert layout.preset_name == expected_preset
    assert isinstance(layout.layout_id, str)
    assert isinstance(layout.target_pocket_id, str)
    assert layout.target_pocket_id in table.pockets

    cue_x, cue_y = layout.cue_ball_pos
    obj_x, obj_y = layout.object_ball_pos
    assert 0.0 < cue_x < table.w
    assert 0.0 < cue_y < table.l
    assert 0.0 < obj_x < table.w
    assert 0.0 < obj_y < table.l

    as_dict = layout.as_dict()
    assert as_dict["preset_name"] == expected_preset
    assert "cue_ball_pos" in as_dict
    assert "object_ball_pos" in as_dict
    assert "target_pocket_id" in as_dict
    assert "layout_id" in as_dict


def test_make_five_point_straight_in_layout() -> None:
    layout = make_five_point_straight_in_layout()
    _assert_layout_valid(layout, expected_preset="five_point_straight_in")


def test_make_eight_ball_reference_layout() -> None:
    layout = make_eight_ball_reference_layout()
    _assert_layout_valid(layout, expected_preset="eight_ball_reference")

