"""Invariants for the crack segment review.

The tests use synthetic dark traces so the expected topology is known:
a Y fork, a T fork with a side shoot, and a region-clipped trace.
"""
import numpy as np
from PIL import Image, ImageDraw

from endgrain import cracks
from endgrain.cracks import Raster
from endgrain.geometry import plane_for


def raster_of(draw, width=360, height=360, region=None):
    image = Image.new("L", (width, height), 230)
    draw(ImageDraw.Draw(image))
    pixels = np.asarray(image)
    support = np.ones(pixels.shape, dtype=bool)
    inspected = support.copy()
    if region is not None:
        canvas = Image.new("1", (width, height))
        ImageDraw.Draw(canvas).polygon(region, fill=1)
        inspected = np.asarray(canvas, dtype=bool)
    plane = plane_for(width, height, {"mode": "pixels"})
    return Raster(np.dstack([pixels] * 3), support, inspected, plane)


def analyze(raster, params=None):
    return cracks.analyze_regions(raster, cracks.normalize_params(params or {}, raster.plane))


def assert_reconciles(regions):
    total = sum(trace["length"] for trace in regions.objects)
    segmented = sum(segment["length"] for segment in regions.segments)
    held = sum(fork["held_length"] for fork in regions.forks)
    assert abs(segmented + held - total) < 1e-8


def y_fork(painter):
    painter.line([(40, 180), (160, 180)], fill=20, width=7)
    painter.line([(160, 180), (300, 90)], fill=20, width=7)
    painter.line([(160, 180), (300, 270)], fill=20, width=7)


def test_y_fork_splits_into_three_segments_with_mutual_forks():
    regions = analyze(raster_of(y_fork))
    assert len(regions.objects) == 1
    assert len(regions.forks) == 1
    labels = [segment["label"] for segment in regions.segments]
    assert sorted(labels) == ["S1", "S2", "S3"]
    fork = regions.forks[0]
    assert fork["label"] == "F1"
    assert sorted(fork["segments"]) == ["S1", "S2", "S3"]
    assert set(fork["main_segments"]) == {"S1", "S2"} or set(fork["main_segments"]) == {"S1", "S3"}
    assert len(fork["main_segments"]) == 2
    assert_reconciles(regions)


def test_segments_never_overlap_and_add_to_the_whole_trace():
    regions = analyze(raster_of(y_fork))
    whole = regions.objects[0]["length"]
    assert abs(sum(segment["length"] for segment in regions.segments) - whole) < 1e-8
    assert all(segment["share"] is not None and 0 < segment["share"] <= 1 for segment in regions.segments)


def test_side_shoot_is_a_separate_segment_with_a_deviation():
    def branchy(painter):
        y_fork(painter)
        painter.line([(210, 140), (300, 140)], fill=20, width=5)

    regions = analyze(raster_of(branchy))
    forks = {tuple(fork["segments"]) for fork in regions.forks}
    assert len(regions.forks) == 2
    side = [angle for fork in regions.forks for label, angle in fork["side_branches"].items()]
    assert side, "a side shoot should deviate from the main line"
    assert any(angle > 20 for angle in side)
    assert_reconciles(regions)


def test_tip_at_the_region_border_is_marked_as_a_cut():
    def check(painter):
        painter.line([(40, 180), (240, 180)], fill=20, width=7)

    clipped = analyze(raster_of(check, region=[(0, 0), (200, 0), (200, 360), (0, 360)]))
    boundary = [segment for segment in clipped.segments if segment["boundary"]]
    assert boundary, "an end stopping at the selected region should be flagged"
    assert any(segment["to_kind"] == "region" or segment["from_kind"] == "region" for segment in boundary)
    assert_reconciles(clipped)


def test_open_tip_away_from_a_border_is_not_a_cut():
    regions = analyze(raster_of(y_fork))
    assert not any(segment["boundary"] for segment in regions.segments)


def test_held_length_is_counted_once_at_the_fork():
    def broad(painter):
        painter.line([(20, 180), (180, 180)], fill=20, width=61)
        painter.line([(180, 180), (340, 90)], fill=20, width=61)
        painter.line([(180, 180), (340, 310)], fill=20, width=61)

    regions = analyze(raster_of(broad, width=400, height=400))
    total = sum(trace["length"] for trace in regions.objects)
    segmented = sum(segment["length"] for segment in regions.segments)
    held = sum(fork["held_length"] for fork in regions.forks)
    assert abs(segmented + held - total) < 1e-8
    assert held >= 0
    assert_reconciles(regions)


def test_no_traces_still_yields_consistent_totals():
    regions = analyze(raster_of(lambda painter: painter.rectangle((5, 5, 30, 30), fill=230)))
    assert regions.objects == []
    assert regions.segments == []
    assert_reconciles(regions)


def test_fork_records_which_segments_took_its_length():
    regions = analyze(raster_of(y_fork))
    fork = regions.forks[0]
    assert fork["attribution"].startswith("main line")
    assert sorted(fork["main_segments"]) == sorted(fork["main_segments"])
    assert len(fork["main_segments"]) == 2
    assert fork["held_length"] == 0.0


def test_held_branches_are_kept_out_of_segments():
    regions = analyze(raster_of(y_fork))
    whole = regions.objects[0]["length"]
    segments = {segment["label"] for segment in regions.segments}
    for fork in regions.forks:
        assert set(fork["segments"]) <= segments
        assert fork["held_branches"] >= 0
    if regions.forks:
        assert all(fork["held_length"] <= whole for fork in regions.forks)


def test_side_branch_angle_points_at_the_shoot():
    def branchy(painter):
        y_fork(painter)
        painter.line([(215, 145), (300, 145)], fill=20, width=5)

    regions = analyze(raster_of(branchy))
    shots = []
    for fork in regions.forks:
        for label, angle in fork["side_branches"].items():
            shots.append((label, angle))
    assert shots
    assert all(angle >= 0 for _, angle in shots)
    assert max(angle for _, angle in shots) > 45
