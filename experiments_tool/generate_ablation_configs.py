from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the paper-oriented SeaTrack ablation YAML files.")
    parser.add_argument(
        "--base",
        default=r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\sealinetracker.yaml",
        help="Base tracker YAML.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations",
        help="Output directory for generated tracker YAML files.",
    )
    return parser.parse_args()


def load_flat_yaml(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return lines, values


def apply_overrides(lines: list[str], overrides: dict[str, str]) -> list[str]:
    updated_lines: list[str] = []
    for raw_line in lines:
        prefix, *comment = raw_line.split("#", 1)
        if ":" not in prefix:
            updated_lines.append(raw_line)
            continue
        key, _value = prefix.split(":", 1)
        key = key.strip()
        if key not in overrides:
            updated_lines.append(raw_line)
            continue

        new_line = f"{key}: {overrides[key]}"
        if comment:
            new_line += f" # {comment[0].strip()}"
        updated_lines.append(new_line)
    return updated_lines


def write_variant(base_lines: list[str], out_path: Path, overrides: dict[str, str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_lines = apply_overrides(base_lines, overrides)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    out_dir = Path(args.out_dir)

    base_lines, _base_values = load_flat_yaml(base_path)

    variants = {
        "seatrack_rsa_iou_sgl.yaml": {},
        "seatrack_rsa_iou.yaml": {
            "use_adaptive_iou": "True",
            "use_sea_geo_cost": "False",
            "use_direction_aware_center": "True",
            "use_asymmetric_scale": "True",
            "use_ratio_loss": "True",
            "use_recovery_matching": "False",
        },
        "seatrack_sgl.yaml": {
            "use_adaptive_iou": "False",
            "use_sea_geo_cost": "True",
            "use_direction_aware_center": "True",
            "use_asymmetric_scale": "True",
            "use_ratio_loss": "True",
            "use_recovery_matching": "False",
        },
        "seatrack_sgl_wo_scl.yaml": {
            "use_adaptive_iou": "False",
            "use_sea_geo_cost": "True",
            "use_direction_aware_center": "False",
            "use_asymmetric_scale": "True",
            "use_ratio_loss": "True",
            "use_recovery_matching": "False",
        },
        "seatrack_sgl_wo_srl.yaml": {
            "use_adaptive_iou": "False",
            "use_sea_geo_cost": "True",
            "use_direction_aware_center": "True",
            "use_asymmetric_scale": "True",
            "use_ratio_loss": "False",
            "use_recovery_matching": "False",
        },
        "seatrack_sgl_wo_ssl.yaml": {
            "use_adaptive_iou": "False",
            "use_sea_geo_cost": "True",
            "use_direction_aware_center": "True",
            "use_asymmetric_scale": "False",
            "use_ratio_loss": "True",
            "use_recovery_matching": "False",
        },
    }

    for filename, overrides in variants.items():
        write_variant(base_lines, out_dir / filename, overrides)

    readme = [
        "# SeaTrack Paper Ablation Configs",
        "",
        "Files:",
        "",
        "- `seatrack_rsa_iou_sgl.yaml`: full paper version (`RSA-IoU + SGL`).",
        "- `seatrack_rsa_iou.yaml`: only `RSA-IoU` enabled.",
        "- `seatrack_sgl.yaml`: only `SGL` enabled.",
        "- `seatrack_sgl_wo_scl.yaml`: `SGL` without `SCL`.",
        "- `seatrack_sgl_wo_srl.yaml`: `SGL` without `SRL`.",
        "- `seatrack_sgl_wo_ssl.yaml`: `SGL` without `SSL`.",
        "",
        "Use `bytetrack_baseline.yaml` as the ByteTrack baseline config.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Generated {len(variants)} ablation configs in {out_dir}")


if __name__ == "__main__":
    main()
