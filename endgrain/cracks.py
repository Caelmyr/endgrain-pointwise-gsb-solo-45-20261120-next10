from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi

from .geometry import Plane, Raster
from .imaging import luminance, masked_smooth, safe_ratio, symmetric_overlay
from .validation import document, fields, integer, number

SQUARE = np.ones((3, 3), dtype=bool)
CROSS = ndi.generate_binary_structure(2, 1)
NEIGHBORS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

SEGMENT_PALETTE = (
    (252, 211, 112), (149, 212, 190), (240, 160, 120), (169, 189, 240),
    (215, 180, 235), (200, 220, 140), (243, 191, 121), (140, 205, 224),
)
FORK_TONE = (255, 151, 94)
CUT_TONE = (126, 223, 209)
SEGMENT_LIST_LIMIT = 600
FORK_LIST_LIMIT = 300
SEGMENT_LABEL_LIMIT = 90
PANEL_LIMIT = 1300

# A free skeleton end within this many pixels of the inspected-region rim or a source
# gap is treated as cut off rather than as a natural crack tip. The tolerance widens with
# the local trace width, but stays below the usual gap between separate checks.
CUT_TOLERANCE_MIN = 1.5
CUT_TOLERANCE_MAX = 6.0
CUT_TOLERANCE_WIDTH = 0.75


@dataclass(frozen=True)
class CrackRegion:
    mask: np.ndarray
    skeleton: np.ndarray
    strength: np.ndarray
    labels: np.ndarray
    objects: list[dict]
    endpoints: np.ndarray
    junctions: np.ndarray
    segments: list[dict] = field(default_factory=list)
    forks: list[dict] = field(default_factory=list)
    paths: list[np.ndarray] = field(default_factory=list)
    fork_pixels: np.ndarray | None = None
    cut_endpoints: np.ndarray | None = None
    trace_points: list[tuple[int, float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class TraceReview:
    segments: list[dict]
    forks: list[dict]
    paths: list[np.ndarray]
    fork_skeleton: np.ndarray
    cut_marks: np.ndarray
    next_segment: int
    next_fork: int


def normalize_params(raw: Any, plane: Plane) -> dict:
    body = document(raw, "crack settings")
    fields(body, {"contrast", "scale", "minimum_length", "bridge", "edge_margin"}, "crack")
    return {
        "contrast": number(body.get("contrast", 0.14), "crack contrast", 0.012, 0.8),
        "scale": number(body.get("scale", 6), "background radius", 2, 44),
        "minimum_length": number(body.get("minimum_length", 13), "minimum trace length", 3, max(13, min(600, max(plane.width, plane.height)))),
        "bridge": integer(body.get("bridge", 1), "gap closing radius", 0, 4),
        "edge_margin": number(body.get("edge_margin", 2), "edge contact tolerance", 0.5, 12),
    }


def shifted_neighbors(binary: np.ndarray) -> list[np.ndarray]:
    padded = np.pad(binary, 1)
    height, width = binary.shape
    return [
        padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        for dy, dx in NEIGHBORS
    ]


def thin(binary: np.ndarray, maximum: int = 130) -> np.ndarray:
    current = binary.copy()
    for _ in range(maximum):
        changing = False
        for phase in (0, 1):
            around = shifted_neighbors(current)
            count = np.sum(around, axis=0, dtype=np.uint8)
            transitions = np.zeros(binary.shape, dtype=np.uint8)
            for left, right in zip(around, around[1:] + around[:1], strict=True):
                transitions += ~left & right
            n, e, s, w = around[0], around[2], around[4], around[6]
            if phase == 0:
                triplets = ~(n & e & s) & ~(e & s & w)
            else:
                triplets = ~(n & e & w) & ~(n & s & w)
            drop = current & (count >= 2) & (count <= 6) & (transitions == 1) & triplets
            if drop.any():
                current[drop] = False
                changing = True
        if not changing:
            break
    return current


def trace_masks(raster: Raster, params: dict) -> tuple[np.ndarray, np.ndarray]:
    image = luminance(raster.rgb)
    allowed = raster.support
    background, confidence = masked_smooth(image, allowed, params["scale"])
    strength = np.maximum(background - image, 0)
    strong = (strength >= params["contrast"]) & allowed & (confidence >= 0.55)
    if params["bridge"]:
        gap = params["bridge"]
        closed = ndi.binary_closing(strong, structure=ndi.iterate_structure(CROSS, gap), border_value=0)
        strong |= closed & allowed
    return strong, strength


def degrees(skeleton: np.ndarray) -> np.ndarray:
    near = shifted_neighbors(skeleton)
    cardinal = near[0::2]
    count = np.sum(cardinal, axis=0, dtype=np.uint8)
    for index in range(4):
        diagonal = near[index * 2 + 1]
        count += diagonal & ~cardinal[index] & ~cardinal[(index + 1) % 4]
    return count * skeleton


def edge_lengths(skeleton: np.ndarray, dx: float, dy: float) -> np.ndarray:
    near = shifted_neighbors(skeleton)
    right, below = near[2], near[4]
    southeast, southwest = near[3], near[5]
    diagonal = np.hypot(dx, dy)
    corners = southeast & ~right & ~below
    corners_other = southwest & ~near[6] & ~below
    total = right * dx + below * dy + (corners.astype(np.uint8) + corners_other) * diagonal
    return np.where(skeleton, total, 0)


def outside_edge(support: np.ndarray) -> np.ndarray:
    return support & ~ndi.binary_erosion(support, structure=SQUARE, border_value=0)


def diameter_map(component: np.ndarray, plane: Plane) -> np.ndarray:
    padded = np.pad(component, 1)
    step = (plane.mm_y or 1, plane.mm_x or 1)
    radius = ndi.distance_transform_edt(padded, sampling=step)[1:-1, 1:-1]
    return radius * 2


def object_shape(x_indices: np.ndarray, y_indices: np.ndarray, plane: Plane) -> tuple[float, float]:
    if len(x_indices) < 3:
        return 0.0, 1.0
    coords = np.stack([x_indices * (plane.mm_x or 1), y_indices * (plane.mm_y or 1)], axis=1)
    cov = np.cov(coords, rowvar=False, bias=True)
    vals, vectors = np.linalg.eigh(cov)
    angle = float(np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])) % 180)
    elongation = float(np.sqrt(max(vals[-1], 0.01) / max(vals[0], 0.01)))
    return angle, elongation


def skeleton_graph(
    skeleton: np.ndarray, plane: Plane,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[float], list[tuple[int, int]]]:
    """Pixel adjacency of the thinned trace, using the same edge model as edge_lengths.

    Returns the skeleton coordinates, adjacency as (edge index, neighbour) pairs, the
    measuring length of every edge and the edge endpoints. Cardinal steps use the plane
    scale and diagonals its hypotenuse, so the edge lengths sum to the whole trace length.
    """
    ys, xs = np.nonzero(skeleton)
    lookup = np.full(skeleton.shape, -1, dtype=np.int64)
    lookup[ys, xs] = np.arange(len(ys))
    height, width = skeleton.shape
    ends: list[tuple[int, int]] = []
    lengths: list[float] = []
    dx, dy = plane.mm_x or 1, plane.mm_y or 1
    diagonal = float(np.hypot(dx, dy))
    for y, x in zip(ys.tolist(), xs.tolist()):
        right = x + 1 < width and skeleton[y, x + 1]
        below = y + 1 < height and skeleton[y + 1, x]
        left = x - 1 >= 0 and skeleton[y, x - 1]
        if right:
            ends.append((int(lookup[y, x]), int(lookup[y, x + 1])))
            lengths.append(dx)
        if below:
            ends.append((int(lookup[y, x]), int(lookup[y + 1, x])))
            lengths.append(dy)
        if y + 1 < height and x + 1 < width and skeleton[y + 1, x + 1] and not right and not below:
            ends.append((int(lookup[y, x]), int(lookup[y + 1, x + 1])))
            lengths.append(diagonal)
        if y + 1 < height and x - 1 >= 0 and skeleton[y + 1, x - 1] and not below and not left:
            ends.append((int(lookup[y, x]), int(lookup[y + 1, x - 1])))
            lengths.append(diagonal)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(len(ys))]
    for index, (a, b) in enumerate(ends):
        adjacency[a].append((index, b))
        adjacency[b].append((index, a))
    return np.stack([ys, xs], axis=1), adjacency, lengths, ends


def walk_branches(
    adjacency: list[list[tuple[int, int]]], lengths: list[float], ends: list[tuple[int, int]],
) -> tuple[list[dict], np.ndarray]:
    """Partition skeleton edges into branches between junctions or free ends.

    Every edge belongs to exactly one branch, so branch lengths never overlap and together
    they add up to the whole-trace length. A branch that leaves a junction and returns to
    the same pixel is reported as a closed loop under that junction's fork entry.
    """
    degree = np.array([len(neighbours) for neighbours in adjacency])
    turning = degree != 2
    visited = np.zeros(len(lengths), dtype=bool)
    branches: list[dict] = []

    def follow(start: int, edge: int, neighbour: int) -> dict:
        path = [start, neighbour]
        total = lengths[edge]
        visited[edge] = True
        current = neighbour
        closed = False
        while not turning[current]:
            steps = [item for item in adjacency[current] if item[0] != edge]
            if not steps:
                break
            edge, following = steps[0]
            if visited[edge]:
                closed = True
                break
            path.append(following)
            total += lengths[edge]
            visited[edge] = True
            current = following
        if closed and current == start:
            # The branch leaves and re-enters the same junction pixel, forming a loop.
            closed = False
        return {"nodes": path, "length": float(total), "closed": closed, "start": start, "end": current}

    for start in range(len(adjacency)):
        if not turning[start]:
            continue
        for edge, neighbour in adjacency[start]:
            if not visited[edge]:
                branches.append(follow(start, edge, neighbour))
    for edge, (a, _b) in enumerate(ends):
        if not visited[edge]:
            branches.append(follow(a, edge, ends[edge][1]))
    return branches, degree


def prune_labels(raw: np.ndarray, minimum_px: int) -> np.ndarray:
    labels, count = ndi.label(raw, structure=SQUARE)
    numerators = np.bincount(labels.ravel(), minlength=count + 1)
    keep = numerators >= minimum_px
    keep[0] = False
    return keep[labels]


def boundary_distances(raster: Raster) -> tuple[np.ndarray, np.ndarray]:
    """Distance from every plane pixel to the inspected-region rim and to missing pixels."""
    rim = raster.inspected & ~ndi.binary_erosion(raster.inspected, structure=SQUARE, border_value=0)
    region_distance = ndi.distance_transform_edt(~rim) if rim.any() else np.full(rim.shape, np.inf)
    missing = raster.inspected & ~raster.supported
    if not missing.any():
        return region_distance, np.full(rim.shape, np.inf)
    return region_distance, ndi.distance_transform_edt(~missing)


def mean_pixel_step(plane: Plane) -> float:
    return ((plane.mm_x if plane.mm_x is not None else 1) + (plane.mm_y if plane.mm_y is not None else 1)) / 2


def cut_tolerance(width_mm: float, edge_margin: float, plane: Plane) -> float:
    width_px = width_mm / max(mean_pixel_step(plane), 1e-9)
    return float(max(edge_margin, min(CUT_TOLERANCE_MAX, CUT_TOLERANCE_WIDTH * width_px)))


def cut_mark_map(
    skeleton: np.ndarray, region_distance: np.ndarray, source_distance: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Mark free skeleton ends which stop at a border rather than at a natural tip.

    Values are 0 inside, 1 at the inspected-region rim and 2 at missing source pixels.
    """
    marks = np.zeros(skeleton.shape, dtype=np.uint8)
    free = skeleton & (degrees(skeleton) == 1)
    if not free.any():
        return marks
    marks[free & (region_distance <= tolerance) & (source_distance > tolerance)] = 1
    marks[free & (source_distance <= tolerance)] = 2
    return marks


def end_kind(cluster: int, mark: int, closed: bool) -> str:
    if closed:
        return "loop"
    if cluster >= 0:
        return "fork"
    if mark == 1:
        return "region"
    if mark == 2:
        return "source"
    return "open"


def end_label(kind: str, fork: int | None, has_region: bool) -> str:
    if kind == "fork":
        return "Fork F" + str(fork)
    if kind == "region":
        return "Region edge" if has_region else "Image edge"
    if kind == "source":
        return "Source gap"
    if kind == "loop":
        return "Closed loop"
    return "Open end"


def outgoing_vector(
    nodes: list[int], at_start: bool, coords: np.ndarray, plane: Plane, span: int = 6,
) -> np.ndarray:
    """Unit direction leaving a branch near one of its ends, in measuring units."""
    ordered = nodes if at_start else list(reversed(nodes))
    if len(ordered) < 2:
        return np.zeros(2)
    reach = ordered[min(len(ordered) - 1, span)]
    delta = coords[reach].astype(float) - coords[ordered[0]].astype(float)
    scaled = np.array([delta[1] * (plane.mm_x or 1), delta[0] * (plane.mm_y or 1)])
    norm = float(np.linalg.norm(scaled))
    return scaled / norm if norm > 1e-9 else np.zeros(2)


def pair_angle(one: np.ndarray, two: np.ndarray) -> float:
    if np.linalg.norm(one) < 1e-9 or np.linalg.norm(two) < 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(one, two), -1, 1))
    return float(np.degrees(np.arccos(cosine)))


def fork_roles(vectors: dict[str, np.ndarray]) -> tuple[list[str], dict[str, float]]:
    """Name the main line through a fork and measure each side branch's deviation.

    The pair of adjoining segments leaving most nearly opposite to each other forms the
    main line. Other segments become side branches, and their deviation is measured from
    the closest main-line direction.
    """
    labels = list(vectors)
    if len(labels) < 2:
        return [], {}
    best = (0.0, None, None)
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            angle = pair_angle(vectors[labels[first]], vectors[labels[second]])
            if angle > best[0]:
                best = (angle, labels[first], labels[second])
    main = [best[1], best[2]] if best[1] is not None else labels[:2]
    deviations: dict[str, float] = {}
    for label in labels:
        if label in main:
            deviations[label] = 0.0
        else:
            deviations[label] = round(min(
                pair_angle(vectors[label], vectors[main[0]]),
                pair_angle(vectors[label], vectors[main[1]]),
            ), 1)
    return main, deviations


def sample_path(path: np.ndarray, limit: int = 32) -> list[list[float]]:
    """Thin a pixel path so the browser can pick a curved segment without a huge payload."""
    if len(path) <= limit:
        return [[round(float(x), 1), round(float(y), 1)] for x, y in path]
    positions = np.linspace(0, len(path) - 1, limit)
    picked = path[np.rint(positions).astype(int)]
    return [[round(float(x), 2), round(float(y), 2)] for x, y in picked]


def analyze_trace(
    piece: np.ndarray, individual: np.ndarray, excerpt: tuple[slice, slice],
    plane: Plane, strength: np.ndarray, region_distance: np.ndarray,
    source_distance: np.ndarray, has_region: bool, edge_margin: float,
    trace_id: int, trace_length: float, next_segment: int, next_fork: int,
) -> TraceReview:
    """Split one thinned trace into a non-overlapping set of scored segments and forks.

    Segment length comes from skeleton edges only, so the sum of segments plus the length
    held inside junction clusters equals the whole-trace length reported by the existing
    reading. Nothing is counted twice.
    """
    empty = np.zeros(individual.shape, dtype=bool)
    coords, adjacency, lengths, ends = skeleton_graph(individual, plane)
    if not len(coords):
        return TraceReview([], [], [], empty, np.zeros(individual.shape, np.uint8), next_segment, next_fork)
    branches, degree = walk_branches(adjacency, lengths, ends)
    junction_nodes = degree >= 3
    junction_mask = np.zeros(individual.shape, dtype=bool)
    junction_mask[coords[junction_nodes, 0], coords[junction_nodes, 1]] = True
    cluster_map, cluster_count = ndi.label(junction_mask, structure=SQUARE)
    cluster_of = np.where(junction_nodes, cluster_map[coords[:, 0], coords[:, 1]] - 1, -1)

    widths = diameter_map(piece, plane)
    y0, x0 = excerpt[0].start, excerpt[1].start
    local = coords
    node_widths = widths[local[:, 0], local[:, 1]]
    trace_width = float(np.median(node_widths)) if len(node_widths) else 0.0
    tolerance = cut_tolerance(trace_width, edge_margin, plane)
    cuts = cut_mark_map(individual, region_distance[excerpt], source_distance[excerpt], tolerance)
    node_strength = strength[excerpt][local[:, 0], local[:, 1]]

    centres = []
    for cluster in range(cluster_count):
        nodes = np.flatnonzero(cluster_of == cluster)
        cy, cx = coords[nodes].mean(axis=0)
        centres.append((float(cy), float(cx), cluster, nodes))
    centres.sort(key=lambda item: (item[0], item[1]))
    fork_number = {cluster: next_fork + index for index, (_, _, cluster, _) in enumerate(centres)}
    adjoining: dict[int, list[str]] = {cluster: [] for _, _, cluster, _ in centres}
    directions: dict[int, dict[str, np.ndarray]] = {cluster: {} for _, _, cluster, _ in centres}
    held: dict[int, float] = {cluster: 0.0 for _, _, cluster, _ in centres}
    held_counts: dict[int, int] = {cluster: 0 for _, _, cluster, _ in centres}

    raw_segments: list[tuple[dict, np.ndarray]] = []
    for branch in branches:
        nodes = branch["nodes"]
        start, finish = nodes[0], nodes[-1]
        start_cluster, end_cluster = int(cluster_of[start]), int(cluster_of[finish])
        if (
            start_cluster >= 0
            and start_cluster == end_cluster
            and bool(np.all(cluster_of[nodes] == start_cluster))
        ):
            held[start_cluster] += branch["length"]
            held_counts[start_cluster] += 1
            continue
        from_kind = end_kind(start_cluster, int(cuts[tuple(coords[start])]), branch["closed"])
        to_kind = end_kind(end_cluster, int(cuts[tuple(coords[finish])]), branch["closed"])
        looping = bool(branch["closed"] or (start_cluster >= 0 and start_cluster == end_cluster))
        if looping and start_cluster >= 0:
            from_kind = to_kind = "fork"
        ordered = np.asarray(nodes)
        segment_widths = widths[local[ordered, 0], local[ordered, 1]]
        angle, _ = object_shape(coords[ordered, 1], coords[ordered, 0], plane)
        boundary = "region" if "region" in (from_kind, to_kind) else (
            "source" if "source" in (from_kind, to_kind) else ""
        )
        segment = {
            "trace": trace_id,
            "from": end_label(from_kind, fork_number.get(start_cluster), has_region),
            "to": end_label(to_kind, fork_number.get(end_cluster), has_region),
            "from_kind": from_kind,
            "to_kind": to_kind,
            "start": [float(coords[start][1] + x0), float(coords[start][0] + y0)],
            "end": [float(coords[finish][1] + x0), float(coords[finish][0] + y0)],
            "length": branch["length"],
            "share": safe_ratio(branch["length"], trace_length),
            "median_width": float(np.median(segment_widths)),
            "width_p90": float(np.percentile(segment_widths, 90)),
            "orientation_deg": angle,
            "contrast": float(np.mean(node_strength[ordered])),
            "boundary": boundary,
            "loop": bool(branch["closed"]),
            "from_fork": fork_number.get(start_cluster),
            "to_fork": fork_number.get(end_cluster),
            "_start_cluster": start_cluster,
            "_end_cluster": end_cluster,
            "_vectors": {
                "start": outgoing_vector(nodes, True, coords, plane),
                "end": outgoing_vector(nodes, False, coords, plane),
            },
        }
        path = np.stack([coords[ordered, 1] + x0, coords[ordered, 0] + y0], axis=1).astype(float)
        segment["path"] = sample_path(path)
        raw_segments.append((segment, path))

    raw_segments.sort(key=lambda item: (-item[0]["length"], item[1][0][1] if len(item[1]) else 0.0))
    segments: list[dict] = []
    paths: list[np.ndarray] = []
    for index, (segment, path) in enumerate(raw_segments):
        segment["id"] = next_segment + index
        segment["label"] = "S" + str(segment["id"])
        vectors = segment.pop("_vectors")
        for cluster, side in ((segment.pop("_start_cluster"), "start"), (segment.pop("_end_cluster"), "end")):
            if cluster >= 0:
                if segment["label"] not in adjoining[cluster]:
                    adjoining[cluster].append(segment["label"])
                directions[cluster].setdefault(segment["label"], vectors[side])
        segments.append(segment)
        paths.append(path)

    forks: list[dict] = []
    for _, _, cluster, nodes in centres:
        fork_id = fork_number[cluster]
        fork_widths = widths[local[nodes, 0], local[nodes, 1]]
        main, deviations = fork_roles(directions[cluster])
        side = {label: angle for label, angle in deviations.items() if label not in main}
        if held[cluster] > 0:
            attribution = "held in fork"
        elif main:
            attribution = "main line " + " – ".join(main)
        elif adjoining[cluster]:
            attribution = "single adjoining segment"
        else:
            attribution = "no measurable branch"
        forks.append({
            "id": fork_id,
            "label": "F" + str(fork_id),
            "trace": trace_id,
            "x": float(coords[nodes, 1].mean() + x0),
            "y": float(coords[nodes, 0].mean() + y0),
            "pixels": int(len(nodes)),
            "held_length": float(held[cluster]),
            "share": safe_ratio(held[cluster], trace_length),
            "median_width": float(np.median(fork_widths)),
            "segments": adjoining[cluster],
            "main_segments": main,
            "side_branches": side,
            "deviations_deg": deviations,
            "attribution": attribution,
            "held_branches": held_counts[cluster],
            "boundary": "",
        })
    fork_skeleton = junction_mask
    return TraceReview(
        segments, forks, paths, fork_skeleton, cuts,
        next_segment + len(segments), next_fork + len(centres),
    )


def analyze_regions(raster: Raster, params: dict) -> CrackRegion:
    proposed, strength = trace_masks(raster, params)
    minimum = params["minimum_length"]
    proposed = prune_labels(proposed, max(4, int(minimum)))
    labels, _ = ndi.label(proposed, structure=SQUARE)
    ids = ndi.find_objects(labels)
    accepted = np.zeros(proposed.shape, dtype=bool)
    skeleton = accepted.copy()
    endpoints = accepted.copy()
    junctions = accepted.copy()
    support_edge = outside_edge(raster.support)
    region_distance, source_distance = boundary_distances(raster)
    has_region = not bool(raster.inspected.all())
    fork_pixels = np.zeros(proposed.shape, dtype=bool)
    cut_endpoints = np.zeros(proposed.shape, dtype=bool)
    objects: list[dict] = []
    segments: list[dict] = []
    forks: list[dict] = []
    paths: list[np.ndarray] = []
    trace_points: list[tuple[int, float, float]] = []
    next_segment, next_fork = 1, 1
    for label, excerpt in enumerate(ids, 1):
        if excerpt is None:
            continue
        piece = labels[excerpt] == label
        individual = thin(piece)
        length_px = float(edge_lengths(individual, 1, 1).sum())
        if length_px < minimum:
            continue
        # Compact dark objects are reported by pore measurement, not elongated checks.
        ys, xs = np.nonzero(individual)
        angle, elongation = object_shape(xs, ys, raster.plane)
        widths = diameter_map(piece, raster.plane)[individual]
        local_d = degrees(individual)
        local_end = individual & (local_d == 1)
        local_junction = individual & (local_d >= 3)
        if elongation < 1.38 and not local_junction.any():
            continue
        y0, x0 = excerpt[0].start, excerpt[1].start
        accepted[excerpt] |= piece
        skeleton[excerpt] |= individual
        endpoints[excerpt] |= local_end
        junctions[excerpt] |= local_junction
        length = float(edge_lengths(individual, raster.plane.mm_x or 1, raster.plane.mm_y or 1).sum())
        edge = bool(np.any(piece & support_edge[excerpt]))
        trace_id = len(objects) + 1
        objects.append({
            "id": trace_id,
            "x": float(np.mean(xs) + x0),
            "y": float(np.mean(ys) + y0),
            "bbox": [x0, y0, excerpt[1].stop, excerpt[0].stop],
            "area": int(piece.sum()) * raster.plane.pixel_area,
            "length": length,
            "median_width": float(np.median(widths)),
            "width_p90": float(np.percentile(widths, 90)),
            "orientation_deg": angle,
            "endpoints": int(local_end.sum()),
            "junctions": int(ndi.label(local_junction, structure=SQUARE)[1]),
            "touches_support_edge": edge,
            "contrast": float(np.mean(strength[excerpt][piece])),
            "segments": 0,
            "forks": 0,
            "boundary_segments": 0,
        })
        review = analyze_trace(
            piece, individual, excerpt, raster.plane, strength,
            region_distance, source_distance, has_region, params["edge_margin"],
            trace_id, length, next_segment, next_fork,
        )
        next_segment, next_fork = review.next_segment, review.next_fork
        objects[-1]["segments"] = len(review.segments)
        objects[-1]["forks"] = len(review.forks)
        objects[-1]["boundary_segments"] = sum(1 for item in review.segments if item["boundary"])
        segments.extend(review.segments)
        forks.extend(review.forks)
        paths.extend(review.paths)
        if review.segments:
            trace_points.append((trace_id, objects[-1]["x"], objects[-1]["y"]))
        fork_pixels[excerpt] |= review.fork_skeleton
        cut_endpoints[excerpt] |= review.cut_marks.astype(bool)
    return CrackRegion(
        accepted, skeleton, strength, labels, objects, endpoints, junctions,
        segments, forks, paths, fork_pixels, cut_endpoints, trace_points,
    )


def draw_preview(raster: Raster, regions: CrackRegion) -> np.ndarray:
    composite = symmetric_overlay(
        raster.rgb,
        [
            (regions.mask, (209, 79, 89), 0.60),
            (regions.skeleton, (252, 211, 112), 0.97),
        ],
        support=raster.support,
    )
    output = Image.fromarray(composite)
    pen = ImageDraw.Draw(output)
    radius = max(2, max(raster.plane.width, raster.plane.height) // 310)
    for points, color in ((regions.endpoints, (133, 202, 222)), (regions.junctions, (251, 154, 130))):
        marks, count = ndi.label(points, structure=SQUARE)
        if count:
            for y, x in ndi.center_of_mass(points, marks, np.arange(1, count + 1)):
                pen.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=1)
    for item in sorted(regions.objects, key=lambda candidate: candidate["length"], reverse=True)[:28]:
        x, y = item["x"], item["y"]
        pen.text((x + 5, y + 4), str(item["id"]), fill=(250, 237, 198), stroke_width=2, stroke_fill=(63, 39, 28))
    return np.asarray(output)


def draw_segment_preview(raster: Raster, regions: CrackRegion, legend: bool = True) -> np.ndarray:
    composite = symmetric_overlay(
        raster.rgb,
        [(regions.mask, (150, 67, 70), 0.36)],
        support=raster.support,
    )
    if regions.fork_pixels is not None and regions.fork_pixels.any():
        composite[regions.fork_pixels] = np.rint(
            0.30 * composite[regions.fork_pixels] + 0.70 * np.asarray(FORK_TONE)
        ).astype(np.uint8)
    output = Image.fromarray(composite)
    pen = ImageDraw.Draw(output)
    radius = max(2, max(raster.plane.width, raster.plane.height) // 310)
    width = max(2, radius)
    for index, path in enumerate(regions.paths):
        if len(path) < 2:
            continue
        pen.line([tuple(point) for point in path], fill=SEGMENT_PALETTE[index % len(SEGMENT_PALETTE)], width=width)
    labelled = 0
    for segment, path in zip(regions.segments, regions.paths, strict=True):
        if labelled >= SEGMENT_LABEL_LIMIT or len(path) < 2:
            continue
        point = path[max(0, int(len(path) * 0.58) - 1)]
        pen.text(
            (float(point[0]) + 4, float(point[1]) + 3), segment["label"],
            fill=(255, 251, 230), stroke_width=2, stroke_fill=(31, 36, 29),
        )
        labelled += 1
    for fork in regions.forks[:48]:
        pen.text(
            (fork["x"] + 5, fork["y"] - 11), fork["label"],
            fill=(255, 210, 170), stroke_width=2, stroke_fill=(45, 29, 21),
        )
    if regions.cut_endpoints is not None and regions.cut_endpoints.any():
        marks, count = ndi.label(regions.cut_endpoints, structure=SQUARE)
        for y, x in ndi.center_of_mass(regions.cut_endpoints, marks, np.arange(1, count + 1)):
            pen.ellipse((x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1), outline=CUT_TONE, width=2)
    for trace_id, x, y in regions.trace_points[:40]:
        pen.text(
            (x - 7, y + 8), "T" + str(trace_id),
            fill=(247, 241, 213), stroke_width=2, stroke_fill=(38, 42, 34),
        )
    legend_text = "S segment · F fork (length held) · ○ cut at region/source edge · T trace"
    if legend:
        pen.text((6, 5), legend_text, fill=(248, 242, 219), stroke_width=2, stroke_fill=(29, 33, 26))
    return np.asarray(output)


def review_sheet(raster: Raster, regions: CrackRegion) -> np.ndarray:
    """Side-by-side whole-trace and segment-review panels for the saved overlay."""
    panels = []
    for title, picture in (
        ("A  Whole trace (existing reading)", draw_preview(raster, regions)),
        ("B  Segment review", draw_segment_preview(raster, regions, legend=False)),
    ):
        fitted = ImageOps.contain(Image.fromarray(picture), (PANEL_LIMIT, PANEL_LIMIT), Image.Resampling.LANCZOS)
        panels.append((title, fitted))
    header = 42
    margin = 8
    height = max(item[1].height for item in panels) + header + margin
    width = sum(item[1].width for item in panels) + margin * (len(panels) + 1)
    canvas = Image.new("RGB", (width, height), (36, 41, 35))
    pen = ImageDraw.Draw(canvas)
    unit = raster.plane.unit
    summary = "segments {:,.2f} {} + held at forks {:,.2f} {} = whole-trace total {:,.2f} {}".format(
        sum(item["length"] for item in regions.segments), unit,
        sum(item["held_length"] for item in regions.forks), unit,
        sum(item["length"] for item in regions.objects), unit,
    )
    pen.text((margin + 2, 5), summary, fill=(246, 239, 213))
    pen.text((margin + 2, 23), "S = segment   F = fork (orange, length held here)   ○ cyan = cut by region or image edge   T = trace",
             fill=(198, 209, 179))
    offset = margin
    for title, picture in panels:
        canvas.paste(picture, (offset, header))
        pen.rectangle((offset, header, offset + picture.width - 1, header + picture.height - 1), outline=(83, 92, 74))
        pen.text((offset + 6, header + 5), title, fill=(252, 246, 222), stroke_width=2, stroke_fill=(29, 34, 27))
        offset += picture.width + margin
    return np.asarray(canvas)


def trace_row(item: dict) -> list:
    return [
        item["id"], item["x"], item["y"], item["length"], item["area"],
        item["median_width"], item["width_p90"], item["orientation_deg"],
        item["endpoints"], item["junctions"], "yes" if item["touches_support_edge"] else "no",
        item["contrast"], "trace", item["id"], item["segments"], item["forks"],
        "", "", "", "", "", "",
    ]


def segment_row(item: dict) -> list:
    return [
        item["label"], item["start"][0], item["start"][1], item["length"], "",
        item["median_width"], item["width_p90"], item["orientation_deg"],
        "", "", "", item["contrast"], "segment", item["trace"], "", "",
        item["from"], item["to"], item["share"], "", "", "",
    ]


def fork_row(item: dict) -> list:
    sides = ", ".join(
        f"{label} ({angle:.1f}°)" for label, angle in item.get("side_branches", {}).items()
    )
    return [
        item["label"], item["x"], item["y"], item["held_length"], "",
        item["median_width"], "", "", "", "", "", "", "fork", item["trace"],
        ", ".join(item["segments"]), " – ".join(item.get("main_segments", [])),
        "", "", item["share"], item["pixels"],
        item.get("attribution", ""), sides,
    ]


def summary_rows(result: dict) -> list[list]:
    metrics = result["metrics"]
    segment_total = metrics["segment_length"]
    fork_total = metrics["fork_length"]
    rows = []
    for label, value in (
        ("whole trace total", metrics["total_length"]),
        ("segments total", segment_total),
        ("held at forks total", fork_total),
        ("segments plus forks", segment_total + fork_total),
    ):
        rows.append([label, "", "", value, "", "", "", "", "", "", "", "", "total", "", "", "",
                     "", "", "", "", "", ""])
    return rows


CRACK_COLUMNS = [
    "id", "x", "y", "length", "area", "median_width", "width_p90", "orientation_deg",
    "endpoints", "junctions", "touches_support_edge", "contrast",
    "row_type", "trace", "segments", "main_line", "from", "to", "share",
    "junction_pixels", "attribution", "side_branches",
]


def measure(raster: Raster, params: dict) -> tuple[dict, np.ndarray, list[list]]:
    regions = analyze_regions(raster, params)
    all_regions = regions.objects
    disconnected = Counter("edge" if obj["touches_support_edge"] else "internal" for obj in all_regions)
    total = float(sum(item["length"] for item in all_regions))
    segment_total = float(sum(item["length"] for item in regions.segments))
    fork_total = float(sum(item["held_length"] for item in regions.forks))
    boundary_segments = sum(1 for item in regions.segments if item.get("boundary"))
    unit = raster.plane.unit
    notices = ["Edge-touching traces may continue beyond the inspected support. Widths approximate a binary dark region."]
    if not all_regions:
        notices.append("No elongated dark traces passed the current contrast and length settings.")
    else:
        notices.append(
            "Segment review keeps lengths separate: {:,.3f} {} in segments plus {:,.3f} {} held at forks equals the whole-trace total {:,.3f} {}.".format(
                segment_total, unit, fork_total, unit, total, unit,
            )
        )
    if boundary_segments:
        notices.append(
            "{} segment(s) stop at the inspected-region or source edge, where the trace may continue outside the visible area.".format(
                boundary_segments
            )
        )
    if len(regions.segments) > SEGMENT_LIST_LIMIT:
        notices.append(
            "Only the longest {} review segments are listed; the totals still use every segment.".format(SEGMENT_LIST_LIMIT)
        )
    if len(regions.forks) > FORK_LIST_LIMIT:
        notices.append(
            "Only the first {} fork clusters are listed; the totals still use every fork.".format(FORK_LIST_LIMIT)
        )
    result = {
        "metrics": {
            "traces": len(all_regions),
            "total_length": total,
            "longest_trace": max((item["length"] for item in all_regions), default=None),
            "edge_traces": disconnected["edge"],
            "internal_traces": disconnected["internal"],
            "trace_area_fraction": safe_ratio(int(regions.mask.sum()), int(raster.support.sum())),
            "segments": len(regions.segments),
            "segment_length": segment_total,
            "fork_length": fork_total,
            "boundary_segments": boundary_segments,
        },
        "notices": notices,
        "objects": all_regions[:100],
        "segments": regions.segments[:SEGMENT_LIST_LIMIT],
        "forks": regions.forks[:FORK_LIST_LIMIT],
        "unit": unit,
    }
    rows = [trace_row(item) for item in all_regions]
    rows.extend(segment_row(item) for item in result["segments"])
    rows.extend(fork_row(item) for item in result["forks"])
    rows.extend(summary_rows(result))
    visuals = {
        "traces": draw_preview(raster, regions),
        "segments": draw_segment_preview(raster, regions),
        "review": review_sheet(raster, regions),
    }
    return result, visuals, rows
