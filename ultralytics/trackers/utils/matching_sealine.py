# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

from .matching import fuse_score, linear_assignment, iou_distance


SMALL_TARGET_GEOM_WEIGHTS = np.array([0.40, 0.35, 0.15, 0.10], dtype=np.float32)
LARGE_TARGET_GEOM_WEIGHTS = np.array([0.55, 0.20, 0.15, 0.10], dtype=np.float32)
FIXED_GEOM_WEIGHTS = 0.5 * (SMALL_TARGET_GEOM_WEIGHTS + LARGE_TARGET_GEOM_WEIGHTS)


def _to_tlbr_list(tracks: list) -> list[np.ndarray]:
    """Convert tracker objects or raw arrays into xyxy/xywha boxes."""
    if (tracks and isinstance(tracks[0], np.ndarray)) or (tracks and isinstance(tracks[0], list)):
        return tracks
    return [track.xywha if track.angle is not None else track.xyxy for track in tracks]


def _relative_scale(box: np.ndarray, image_shape: tuple[int, int], eps: float) -> float:
    """Return the relative object scale used by the adaptive IoU module."""
    height, width = image_shape
    w = max(float(box[2] - box[0]), eps)
    h = max(float(box[3] - box[1]), eps)
    return float(np.sqrt((w * h) / max(width * height, eps)))


def _scale_factor(r: float, args, axis_bias: float) -> float:
    """Compute a monotonic decreasing scale factor s(r) = 1 + alpha * exp(-beta * r)."""
    scale = 1.0 + args.adaptive_iou_alpha * np.exp(-args.adaptive_iou_beta * r)
    scale = np.clip(scale, args.adaptive_iou_min_scale, args.adaptive_iou_max_scale)
    return float(scale * axis_bias)


def _expand_box(box: np.ndarray, image_shape: tuple[int, int], scale_x: float, scale_y: float) -> np.ndarray:
    """Expand a box around its center while keeping it inside the image."""
    height, width = image_shape
    x1, y1, x2, y2 = map(float, box[:4])
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * (x2 - x1) * scale_x
    half_h = 0.5 * (y2 - y1) * scale_y
    return np.array(
        [
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(float(width), cx + half_w),
            min(float(height), cy + half_h),
        ],
        dtype=np.float32,
    )


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two axis-aligned boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _adaptive_iou_similarity(track_box: np.ndarray, det_box: np.ndarray, image_shape: tuple[int, int], args) -> tuple[float, float]:
    """Return adaptive IoU similarity together with the relative scale used for weighting."""
    track_scale = _relative_scale(track_box, image_shape, args.adaptive_iou_eps)
    det_scale = _relative_scale(det_box, image_shape, args.adaptive_iou_eps)
    relative_scale = min(track_scale, det_scale)

    lambda_r = relative_scale / (relative_scale + args.adaptive_iou_lambda_tau)
    lambda_r = float(np.clip(lambda_r, 0.0, 1.0))

    scale_x = _scale_factor(relative_scale, args, args.adaptive_iou_x_bias)
    scale_y = _scale_factor(relative_scale, args, args.adaptive_iou_y_bias)

    expanded_track = _expand_box(track_box, image_shape, scale_x, scale_y)
    expanded_det = _expand_box(det_box, image_shape, scale_x, scale_y)

    iou_orig = _bbox_iou(track_box, det_box)
    iou_expanded = _bbox_iou(expanded_track, expanded_det)
    adaptive_iou = lambda_r * iou_orig + (1.0 - lambda_r) * iou_expanded
    return float(adaptive_iou), float(relative_scale)


def _plain_iou_similarity(track_box: np.ndarray, det_box: np.ndarray, image_shape: tuple[int, int] | None, args) -> tuple[float, float]:
    """Return plain IoU similarity together with the relative scale used by the geometry module."""
    relative_scale = 0.0
    if image_shape is not None:
        track_scale = _relative_scale(track_box, image_shape, args.adaptive_iou_eps)
        det_scale = _relative_scale(det_box, image_shape, args.adaptive_iou_eps)
        relative_scale = min(track_scale, det_scale)
    return _bbox_iou(track_box, det_box), float(relative_scale)


def adaptive_iou_distance(atracks: list, btracks: list, image_shape: tuple[int, int] | None, args) -> np.ndarray:
    """Compute the Relative-Scale Adaptive IoU cost matrix for SealineTracker."""
    if not getattr(args, "use_adaptive_iou", True):
        return iou_distance(atracks, btracks)

    atlbrs = _to_tlbr_list(atracks)
    btlbrs = _to_tlbr_list(btracks)

    cost_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return cost_matrix

    if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
        ious = batch_probiou(
            np.ascontiguousarray(atlbrs, dtype=np.float32),
            np.ascontiguousarray(btlbrs, dtype=np.float32),
        ).numpy()
        return 1 - ious

    if image_shape is None:
        ious = bbox_ioa(
            np.ascontiguousarray(atlbrs, dtype=np.float32),
            np.ascontiguousarray(btlbrs, dtype=np.float32),
            iou=True,
        )
        return 1 - ious

    for row, track_box in enumerate(atlbrs):
        for col, det_box in enumerate(btlbrs):
            adaptive_iou, _ = _adaptive_iou_similarity(track_box, det_box, image_shape, args)
            cost_matrix[row, col] = 1.0 - adaptive_iou

    return cost_matrix


def _center_distance_loss(
    track_box: np.ndarray,
    det_box: np.ndarray,
    image_shape: tuple[int, int],
    eps: float,
    center_x_weight: float,
    center_y_weight: float,
) -> float:
    """Normalized direction-aware center distance loss for sea-surface tracking."""
    height, width = image_shape
    tcx = 0.5 * (float(track_box[0]) + float(track_box[2]))
    tcy = 0.5 * (float(track_box[1]) + float(track_box[3]))
    dcx = 0.5 * (float(det_box[0]) + float(det_box[2]))
    dcy = 0.5 * (float(det_box[1]) + float(det_box[3]))
    diag = max(np.sqrt(width * width + height * height), eps)
    dx = tcx - dcx
    dy = tcy - dcy
    return float(np.sqrt(center_x_weight * dx * dx + center_y_weight * dy * dy) / diag)


def _ratio_loss(track_box: np.ndarray, det_box: np.ndarray, eps: float) -> float:
    """Aspect-ratio difference loss compressed into [0, 1]."""
    tw = max(float(track_box[2] - track_box[0]), eps)
    th = max(float(track_box[3] - track_box[1]), eps)
    dw = max(float(det_box[2] - det_box[0]), eps)
    dh = max(float(det_box[3] - det_box[1]), eps)
    return float((2.0 / np.pi) * abs(np.arctan(tw / th) - np.arctan(dw / dh)))


def _scale_loss(track_box: np.ndarray, det_box: np.ndarray, eps: float, width_weight: float, height_weight: float) -> float:
    """Width-height consistency loss with asymmetric sea-surface weighting."""
    tw = max(float(track_box[2] - track_box[0]), eps)
    th = max(float(track_box[3] - track_box[1]), eps)
    dw = max(float(det_box[2] - det_box[0]), eps)
    dh = max(float(det_box[3] - det_box[1]), eps)
    width_loss = abs(tw - dw) / max(tw, dw, eps)
    height_loss = abs(th - dh) / max(th, dh, eps)
    return float(width_weight * width_loss + height_weight * height_loss)


def _scale_aware_weights(relative_scale: float, args) -> np.ndarray:
    """Interpolate between small-target and large-target geometry weights."""
    if not getattr(args, "use_scale_aware_weights", True):
        weights = FIXED_GEOM_WEIGHTS.copy()
        return weights / max(weights.sum(), args.adaptive_iou_eps)

    gamma = float(np.clip(relative_scale / max(args.geom_scale_transition, args.adaptive_iou_eps), 0.0, 1.0))
    weights = (1.0 - gamma) * SMALL_TARGET_GEOM_WEIGHTS + gamma * LARGE_TARGET_GEOM_WEIGHTS
    return weights / max(weights.sum(), args.adaptive_iou_eps)


def _motion_distance_loss(track_box: np.ndarray, det_box: np.ndarray, image_shape: tuple[int, int], eps: float) -> float:
    """Normalized prediction-to-detection center distance for recovery matching."""
    height, width = image_shape
    tcx = 0.5 * (float(track_box[0]) + float(track_box[2]))
    tcy = 0.5 * (float(track_box[1]) + float(track_box[3]))
    dcx = 0.5 * (float(det_box[0]) + float(det_box[2]))
    dcy = 0.5 * (float(det_box[1]) + float(det_box[3]))
    diag = max(np.sqrt(width * width + height * height), eps)
    return float(np.sqrt((tcx - dcx) ** 2 + (tcy - dcy) ** 2) / diag)


def scale_aware_geometric_distance(
    atracks: list,
    btracks: list,
    image_shape: tuple[int, int] | None,
    args,
) -> np.ndarray:
    """Compute the Scale-Aware Geometric Cost (SAGC) matrix for SealineTracker."""
    atlbrs = _to_tlbr_list(atracks)
    btlbrs = _to_tlbr_list(btracks)

    if not getattr(args, "use_sea_geo_cost", True):
        return adaptive_iou_distance(atracks, btracks, image_shape, args)

    cost_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return cost_matrix

    if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
        ious = batch_probiou(
            np.ascontiguousarray(atlbrs, dtype=np.float32),
            np.ascontiguousarray(btlbrs, dtype=np.float32),
        ).numpy()
        return 1 - ious

    if image_shape is None:
        return adaptive_iou_distance(atracks, btracks, image_shape, args)

    for row, track_box in enumerate(atlbrs):
        for col, det_box in enumerate(btlbrs):
            if getattr(args, "use_adaptive_iou", True):
                iou_value, relative_scale = _adaptive_iou_similarity(track_box, det_box, image_shape, args)
            else:
                iou_value, relative_scale = _plain_iou_similarity(track_box, det_box, image_shape, args)

            iou_loss = 1.0 - iou_value
            center_loss = _center_distance_loss(
                track_box,
                det_box,
                image_shape,
                args.adaptive_iou_eps,
                1.0,
                args.geom_center_y_weight if getattr(args, "use_direction_aware_center", True) else 1.0,
            )
            ratio_loss = _ratio_loss(track_box, det_box, args.adaptive_iou_eps)
            height_weight = args.geom_scale_height_weight if getattr(args, "use_asymmetric_scale", True) else 0.5
            scale_loss = _scale_loss(
                track_box,
                det_box,
                args.adaptive_iou_eps,
                max(0.0, 1.0 - height_weight),
                height_weight,
            )
            weights = _scale_aware_weights(relative_scale, args)
            losses = np.array([iou_loss, center_loss, ratio_loss, scale_loss], dtype=np.float32)
            cost_matrix[row, col] = float(np.clip(np.sum(weights * losses), 0.0, 1.0))

    return cost_matrix


def sea_recovery_distance(
    tracks: list,
    detections: list,
    image_shape: tuple[int, int] | None,
    args,
) -> np.ndarray:
    """Compute modular recovery-matching costs for the third association stage."""
    atracks = _to_tlbr_list(tracks)
    btracks = _to_tlbr_list(detections)

    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    if len(atracks) == 0 or len(btracks) == 0:
        return cost_matrix

    if image_shape is None:
        return scale_aware_geometric_distance(tracks, detections, image_shape, args)

    for row, track in enumerate(tracks):
        track_box = atracks[row]
        track_cls = getattr(track, "cls", None)
        for col, det in enumerate(detections):
            det_box = btracks[col]
            det_cls = getattr(det, "cls", None)

            if args.recovery_use_class_gate and track_cls != det_cls:
                cost_matrix[row, col] = 1.0
                continue

            motion_loss = _motion_distance_loss(track_box, det_box, image_shape, args.adaptive_iou_eps)
            if motion_loss > args.recovery_motion_gate:
                cost_matrix[row, col] = 1.0
                continue

            center_loss = _center_distance_loss(
                track_box,
                det_box,
                image_shape,
                args.adaptive_iou_eps,
                1.0,
                args.geom_center_y_weight,
            )
            scale_loss = _scale_loss(
                track_box,
                det_box,
                args.adaptive_iou_eps,
                max(0.0, 1.0 - args.geom_scale_height_weight),
                args.geom_scale_height_weight,
            )

            cost = (
                args.recovery_motion_weight * motion_loss
                + args.recovery_center_weight * center_loss
                + args.recovery_scale_weight * scale_loss
            )
            cost_matrix[row, col] = float(np.clip(cost, 0.0, 1.0))

    return cost_matrix


__all__ = (
    "adaptive_iou_distance",
    "scale_aware_geometric_distance",
    "sea_recovery_distance",
    "fuse_score",
    "linear_assignment",
)
