from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_RAW_MATCH_ROOT = Path(r"E:\Dataset\716\raw-match")


def sorted_image_paths(image_dir: Path) -> list[Path]:
    def frame_key(path: Path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES], key=frame_key)


def sorted_flat_image_paths(image_dir: Path) -> list[Path]:
    def frame_key(path: Path):
        stem = path.stem
        if "_" in stem:
            seq_name, frame_name = stem.split("_", 1)
            if frame_name.isdigit():
                return (seq_name, 0, int(frame_name))
            return (seq_name, 1, frame_name)
        return ("", 1, stem)

    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES], key=frame_key)


def group_images_by_prefix(image_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for image_path in sorted_flat_image_paths(image_dir):
        stem = image_path.stem
        seq_name = stem.split("_", 1)[0] if "_" in stem else "default"
        grouped.setdefault(seq_name, []).append(image_path)
    return grouped


def parse_sequence_names(sequence_names: str | None) -> list[str]:
    if not sequence_names:
        return []
    parsed: list[str] = []
    for item in sequence_names.split(","):
        token = item.strip()
        if not token:
            continue
        parsed.append(token.zfill(4) if token.isdigit() else token)
    return parsed


def resolve_raw_match_sequence_dirs(raw_match_root: Path, sequence_names: list[str]) -> list[Path]:
    sequence_dirs: list[Path] = []
    missing: list[str] = []
    for seq in sequence_names:
        image_dir = raw_match_root / seq / "visible" / "image"
        if image_dir.is_dir():
            sequence_dirs.append(image_dir)
        else:
            missing.append(seq)
    if missing:
        raise FileNotFoundError(
            f"Could not find raw-match visible image directories for sequences: {', '.join(missing)} under {raw_match_root}"
        )
    return sequence_dirs


def discover_all_raw_match_sequence_dirs(raw_match_root: Path) -> list[Path]:
    """Discover all raw-match visible/image directories under the given root."""
    sequence_dirs: list[Path] = []
    for seq_dir in sorted([p for p in raw_match_root.iterdir() if p.is_dir()]):
        image_dir = seq_dir / "visible" / "image"
        if image_dir.is_dir():
            sequence_dirs.append(image_dir)
    if not sequence_dirs:
        raise FileNotFoundError(f"No raw-match visible image directories found under {raw_match_root}")
    return sequence_dirs


def infer_sequence_name_from_dir(sequence_dir: Path) -> str:
    """Infer logical sequence name from an input directory path."""
    if sequence_dir.name.lower() == "image" and sequence_dir.parent.name.lower() == "visible":
        return sequence_dir.parent.parent.name
    return sequence_dir.name


def compute_two_windows(width: int, overlap_ratio: float) -> list[tuple[int, int]]:
    overlap_px = int(round(width * overlap_ratio))
    half = int(np.ceil((width + overlap_px) / 2.0))
    left = (0, min(width, half))
    right = (max(0, width - half), width)
    return [left, right]


def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
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
    return inter / union if union > 0 else 0.0


def classwise_nms(detections: list[tuple[int, float, np.ndarray]], iou_thresh: float) -> list[tuple[int, float, np.ndarray]]:
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    kept: list[tuple[int, float, np.ndarray]] = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        remaining = []
        for det in detections:
            same_cls = det[0] == best[0]
            overlap = box_iou_xyxy(det[2], best[2]) > iou_thresh
            if same_cls and overlap:
                continue
            remaining.append(det)
        detections = remaining
    return kept


def detections_to_results(
    orig_img: np.ndarray,
    image_path: Path,
    names: dict[int, str],
    detections: list[tuple[int, float, np.ndarray]],
) -> Results:
    if detections:
        boxes = []
        for cls_id, score, xyxy in detections:
            x1, y1, x2, y2 = xyxy.tolist()
            boxes.append([x1, y1, x2, y2, score, cls_id])
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    else:
        boxes_tensor = torch.zeros((0, 6), dtype=torch.float32)
    return Results(orig_img=orig_img, path=str(image_path), names=names, boxes=boxes_tensor)


def sanitize_track_row(row: np.ndarray, image_w: int, image_h: int) -> tuple[float, float, float, float, int, float, int] | None:
    """Clip tracker output to image bounds and return a normalized export/visualization tuple."""
    x1, y1, x2, y2, track_id, score, cls_id = row[:7]
    x1 = float(np.clip(x1, 0, image_w))
    y1 = float(np.clip(y1, 0, image_h))
    x2 = float(np.clip(x2, 0, image_w))
    y2 = float(np.clip(y2, 0, image_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2, int(track_id), float(score), int(cls_id)


def draw_tracks(image: np.ndarray, tracks: list[tuple[float, float, float, float, int, float, int]]) -> np.ndarray:
    canvas = image.copy()
    for x1, y1, x2, y2, track_id, score, cls_id in tracks:
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(canvas, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"id:{int(track_id)} cls:{int(cls_id)} {float(score):.2f}",
            (pt1[0], max(0, pt1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


class BaseDetector(ABC):
    """Detector abstraction: implementations must return final detections after pre/postprocess."""

    def __init__(self, weights: str | Path, imgsz: int, conf: float, iou: float, device: str, classes=None) -> None:
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = classes
        self.names = self.model.names

    @abstractmethod
    def infer(self, image: np.ndarray, image_path: Path) -> Results:
        """Return final detector results in original-image coordinates."""


class FullImageUltralyticsDetector(BaseDetector):
    """Detector wrapper whose full-image forward includes Ultralytics pre/postprocess."""

    def infer(self, image: np.ndarray, image_path: Path) -> Results:
        result = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )[0]
        return Results(orig_img=image, path=str(image_path), names=self.names, boxes=result.boxes.data)


class TwoSliceUltralyticsDetector(BaseDetector):
    """Detector wrapper whose forward includes two-slice pre/postprocess and merge."""

    def __init__(
        self,
        weights: str | Path,
        imgsz: int,
        conf: float,
        iou: float,
        device: str,
        overlap_ratio: float = 0.15,
        classes=None,
    ) -> None:
        super().__init__(weights=weights, imgsz=imgsz, conf=conf, iou=iou, device=device, classes=classes)
        self.overlap_ratio = overlap_ratio

    def infer(self, image: np.ndarray, image_path: Path) -> Results:
        height, width = image.shape[:2]
        windows = compute_two_windows(width, self.overlap_ratio)
        merged_detections: list[tuple[int, float, np.ndarray]] = []

        for x_start, x_end in windows:
            crop = image[:, x_start:x_end]
            result = self.model.predict(
                source=crop,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                classes=self.classes,
                verbose=False,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls_id in zip(xyxy, confs, clses):
                mapped = np.array([box[0] + x_start, box[1], box[2] + x_start, box[3]], dtype=np.float32)
                mapped[[0, 2]] = np.clip(mapped[[0, 2]], 0, width)
                mapped[[1, 3]] = np.clip(mapped[[1, 3]], 0, height)
                merged_detections.append((int(cls_id), float(score), mapped))

        kept = classwise_nms(merged_detections, self.iou)
        return detections_to_results(image, image_path, self.names, kept)


class SealineMOTRunner:
    """Detector + tracker wrapper that exports MOT-format tracking results."""

    def __init__(self, detector: BaseDetector, tracker: str | Path, frame_rate: int = 30) -> None:
        tracker_path = check_yaml(tracker)
        cfg = IterableSimpleNamespace(**YAML.load(tracker_path))
        self.tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=frame_rate)
        self.detector = detector
        self.tracker_path = str(tracker_path)

    def run_sequence(
        self,
        image_dir: str | Path | None,
        mot_txt_path: str | Path,
        save_vis_dir: str | Path | None = None,
        image_paths: list[Path] | None = None,
    ) -> dict:
        image_dir = Path(image_dir) if image_dir is not None else None
        mot_txt_path = Path(mot_txt_path)
        images = image_paths if image_paths is not None else sorted_image_paths(image_dir)
        if not images:
            raise FileNotFoundError(f"No images found for sequence input: {image_dir}")

        self.tracker.reset()
        mot_txt_path.parent.mkdir(parents=True, exist_ok=True)
        vis_dir = Path(save_vis_dir) if save_vis_dir else None
        if vis_dir is not None:
            vis_dir.mkdir(parents=True, exist_ok=True)

        mot_lines: list[str] = []
        total_tracks = 0
        for frame_idx, image_path in enumerate(images, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image_h, image_w = image.shape[:2]

            det_result = self.detector.infer(image, image_path)
            det_boxes = det_result.boxes.cpu().numpy() if det_result.boxes is not None else det_result.boxes
            tracks = self.tracker.update(det_boxes, image)
            vis_tracks: list[tuple[float, float, float, float, int, float, int]] = []

            for row in tracks:
                sanitized = sanitize_track_row(row, image_w, image_h)
                if sanitized is None:
                    continue
                x1, y1, x2, y2, track_id, score, cls_id = sanitized
                vis_tracks.append(sanitized)
                w = x2 - x1
                h = y2 - y1
                mot_lines.append(
                    f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.6f},{cls_id},-1"
                )
                total_tracks += 1

            if vis_dir is not None:
                plotted = draw_tracks(image, vis_tracks)
                cv2.imwrite(str(vis_dir / image_path.name), plotted)

        mot_txt_path.write_text("\n".join(mot_lines) + ("\n" if mot_lines else ""), encoding="utf-8")
        return {
            "sequence_dir": str(image_dir) if image_dir is not None else str(images[0].parent),
            "frames": len(images),
            "tracks_written": total_tracks,
            "mot_txt": str(mot_txt_path),
            "tracker": self.tracker_path,
            "detector_mode": self.detector.__class__.__name__,
            "imgsz": self.detector.imgsz,
            "conf": self.detector.conf,
            "iou": self.detector.iou,
            "device": self.detector.device,
        }

    def run_sequences(
        self,
        sequence_dirs: Iterable[str | Path],
        output_root: str | Path,
        save_visuals: bool = False,
    ) -> dict[str, dict]:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        summaries: dict[str, dict] = {}
        for sequence_dir in sequence_dirs:
            seq_path = Path(sequence_dir)
            seq_name = infer_sequence_name_from_dir(seq_path)
            seq_out_dir = output_root / seq_name
            mot_txt = seq_out_dir / f"{seq_name}.txt"
            vis_dir = seq_out_dir / "vis" if save_visuals else None
            summaries[seq_name] = self.run_sequence(seq_path, mot_txt, vis_dir)
        return summaries

    def run_flat_directory_by_prefix(
        self,
        image_dir: str | Path,
        output_root: str | Path,
        save_visuals: bool = False,
    ) -> dict[str, dict]:
        image_dir = Path(image_dir)
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        grouped = group_images_by_prefix(image_dir)
        summaries: dict[str, dict] = {}
        for seq_name, image_paths in grouped.items():
            seq_out_dir = output_root / seq_name
            mot_txt = seq_out_dir / f"{seq_name}.txt"
            vis_dir = seq_out_dir / "vis" if save_visuals else None
            summaries[seq_name] = self.run_sequence(
                image_dir=image_dir,
                mot_txt_path=mot_txt,
                save_vis_dir=vis_dir,
                image_paths=image_paths,
            )
        return summaries


def build_detector(args: argparse.Namespace) -> BaseDetector:
    common = {
        "weights": args.detector,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "classes": None,
    }
    if args.detector_mode == "full":
        return FullImageUltralyticsDetector(**common)
    if args.detector_mode == "two-slice":
        return TwoSliceUltralyticsDetector(**common, overlap_ratio=args.overlap_ratio)
    raise ValueError(f"Unsupported detector mode: {args.detector_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector + SealineTracker and export MOT-format results.")
    parser.add_argument("--detector", required=True, help="Path to detector weights.")
    parser.add_argument(
        "--detector-mode",
        default="full",
        choices=("full", "two-slice"),
        help="Detector wrapper mode. Each mode includes its own pre/postprocess.",
    )
    parser.add_argument(
        "--tracker",
        default=r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\sealinetracker.yaml",
        help="Tracker YAML path.",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Sequence image directory or a root containing multiple sequence dirs. Not required when --sequence-names is used.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"E:\Code\PRCV\ultralytics\experiments_tool\runs\mot_results",
        help="Directory to save MOT txt outputs.",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--overlap-ratio", type=float, default=0.15, help="Overlap ratio used by two-slice detector mode.")
    parser.add_argument("--multi-seq", action="store_true", help="Treat --source as a root folder of multiple sequence dirs.")
    parser.add_argument(
        "--flat-by-prefix",
        action="store_true",
        help="Treat --source as a flat image directory and group frames by filename prefix before the first underscore.",
    )
    parser.add_argument("--save-vis", action="store_true", help="Save per-frame rendered tracking images.")
    parser.add_argument(
        "--raw-match-root",
        default=str(DEFAULT_RAW_MATCH_ROOT),
        help="Root of raw-match dataset. Used together with --sequence-names to locate <seq>/visible/image.",
    )
    parser.add_argument(
        "--sequence-names",
        default="",
        help="Comma-separated sequence names, e.g. 0011,0040,0157. When provided, sequences are read from raw-match.",
    )
    parser.add_argument(
        "--all-raw-match",
        action="store_true",
        help="Run all sequences found under --raw-match-root using <seq>/visible/image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = build_detector(args)
    runner = SealineMOTRunner(detector=detector, tracker=args.tracker)
    sequence_names = parse_sequence_names(args.sequence_names)

    source_path = Path(args.source) if args.source else Path()
    image_files = [p for p in source_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES] if args.source and source_path.is_dir() else []
    child_dirs = [p for p in source_path.iterdir() if p.is_dir()] if args.source and source_path.is_dir() else []
    auto_multi_seq = source_path.is_dir() and not image_files and child_dirs

    if args.all_raw_match:
        raw_match_root = Path(args.raw_match_root)
        sequence_dirs = discover_all_raw_match_sequence_dirs(raw_match_root)
        summary = runner.run_sequences(sequence_dirs, args.out_dir, save_visuals=args.save_vis)
    elif sequence_names:
        raw_match_root = Path(args.raw_match_root)
        sequence_dirs = resolve_raw_match_sequence_dirs(raw_match_root, sequence_names)
        summary = runner.run_sequences(sequence_dirs, args.out_dir, save_visuals=args.save_vis)
    elif args.flat_by_prefix:
        if not args.source:
            raise ValueError("--source is required when using --flat-by-prefix.")
        summary = runner.run_flat_directory_by_prefix(source_path, args.out_dir, save_visuals=args.save_vis)
    elif args.multi_seq or auto_multi_seq:
        if not args.source:
            raise ValueError("--source is required when using --multi-seq or auto-discovery.")
        sequence_dirs = [p for p in sorted(source_path.iterdir()) if p.is_dir()]
        summary = runner.run_sequences(sequence_dirs, args.out_dir, save_visuals=args.save_vis)
    else:
        if not args.source:
            raise ValueError("--source is required for single-sequence mode when --sequence-names is not provided.")
        seq_name = source_path.name
        out_dir = Path(args.out_dir) / seq_name
        out_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = out_dir / "vis" if args.save_vis else None
        summary = {
            seq_name: runner.run_sequence(
                image_dir=source_path,
                mot_txt_path=out_dir / f"{seq_name}.txt",
                save_vis_dir=vis_dir,
            )
        }

    summary_path = Path(args.out_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
