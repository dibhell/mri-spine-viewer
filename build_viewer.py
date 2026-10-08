from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageFilter


ROOT = Path(__file__).parent / "20261006"
OUT = Path(__file__).parent / "mri_viewer"


def clean(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")


def position(ds: pydicom.Dataset) -> float:
    try:
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        return float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), normal))
    except (AttributeError, TypeError, ValueError):
        return float(getattr(ds, "InstanceNumber", 0))


def plane(ds: pydicom.Dataset) -> str:
    try:
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        axis = int(np.argmax(np.abs(np.cross(orientation[:3], orientation[3:]))))
        return ("strzałkowa", "czołowa", "osiowa")[axis]
    except (AttributeError, TypeError, ValueError):
        return "nieokreślona"


def pixels(ds: pydicom.Dataset) -> np.ndarray:
    data = ds.pixel_array.astype(np.float32)
    data *= float(getattr(ds, "RescaleSlope", 1))
    data += float(getattr(ds, "RescaleIntercept", 0))
    return data


def enhance_structures(data: np.ndarray) -> np.ndarray:
    """Multi-scale local contrast: fine edges plus broader disc/bone boundaries."""
    base = Image.fromarray(data, mode="L")
    fine = np.asarray(base.filter(ImageFilter.GaussianBlur(1.1)), dtype=np.float32)
    broad = np.asarray(base.filter(ImageFilter.GaussianBlur(5.0)), dtype=np.float32)
    enhanced = data.astype(np.float32) + 1.35 * (data - fine) + 0.55 * (data - broad)
    low, high = np.percentile(enhanced, (0.5, 99.5))
    return np.clip((enhanced - low) * 255 / max(high - low, 1), 0, 255).astype(np.uint8)


def main() -> None:
    groups: dict[str, list[tuple[Path, pydicom.Dataset]]] = defaultdict(list)
    for path in ROOT.rglob("*"):
        if path.is_file():
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            groups[str(ds.SeriesInstanceUID)].append((path, ds))

    image_root = OUT / "images"
    enhanced_root = OUT / "images_enhanced"
    image_root.mkdir(parents=True, exist_ok=True)
    enhanced_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    volume = None

    ordered = sorted(groups.values(), key=lambda items: int(getattr(items[0][1], "SeriesNumber", 0)))
    for items in ordered:
        items.sort(key=lambda item: position(item[1]))
        first = items[0][1]
        number = int(getattr(first, "SeriesNumber", 0))
        folder = image_root / f"{number:04d}"
        enhanced_folder = enhanced_root / f"{number:04d}"
        folder.mkdir(exist_ok=True)
        enhanced_folder.mkdir(exist_ok=True)

        samples = []
        for path, _ in items:
            arr = pixels(pydicom.dcmread(path, force=True))
            samples.append(arr[::8, ::8].ravel())
        low, high = np.percentile(np.concatenate(samples), (1, 99.5))
        if high <= low:
            high = low + 1

        files = []
        enhanced_files = []
        volume_slices = []
        for index, (path, _) in enumerate(items, 1):
            ds = pydicom.dcmread(path, force=True)
            arr = np.clip((pixels(ds) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
            if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
                arr = 255 - arr
            filename = f"{index:03d}.png"
            Image.fromarray(arr, mode="L").save(folder / filename, optimize=True)
            Image.fromarray(enhance_structures(arr), mode="L").save(enhanced_folder / filename, optimize=True)
            files.append(f"images/{number:04d}/{filename}")
            enhanced_files.append(f"images_enhanced/{number:04d}/{filename}")
            if number == 401:
                volume_slices.append(arr)

        spacing = [float(x) for x in getattr(first, "PixelSpacing", [])]
        description = str(getattr(first, "SeriesDescription", f"Seria {number}"))
        manifest.append({
            "number": number,
            "description": description,
            "plane": "lokalizacyjna" if "scout" in description.lower() else plane(first),
            "count": len(files),
            "dimensions": [int(first.Columns), int(first.Rows)],
            "pixelSpacing": spacing,
            "sliceThickness": float(getattr(first, "SliceThickness", 0)),
            "files": files,
            "enhancedFiles": enhanced_files,
        })
        if number == 401:
            volume_data = np.stack(volume_slices)
            (OUT / "volume.raw").write_bytes(volume_data.tobytes())
            physical_spacing = (
                float(getattr(first, "SpacingBetweenSlices", first.SliceThickness)),
                float(spacing[0]),
                float(spacing[1]),
            )
            gradients = np.gradient(volume_data.astype(np.float32) / 255, *physical_spacing)
            edges = np.sqrt(sum(gradient * gradient for gradient in gradients))
            edge_high = float(np.percentile(edges, 99.5)) or 1
            edge_data = np.clip(edges * 255 / edge_high, 0, 255).astype(np.uint8)
            (OUT / "volume_edges.raw").write_bytes(edge_data.tobytes())
            volume = {
                "file": "volume.raw",
                "edgeFile": "volume_edges.raw",
                "contourFile": "volume_contours.raw",
                "width": int(first.Columns),
                "height": int(first.Rows),
                "depth": len(volume_slices),
                "spacing": [
                    float(spacing[1]),
                    float(spacing[0]),
                    float(getattr(first, "SpacingBetweenSlices", first.SliceThickness)),
                ],
            }

    (OUT / "data.js").write_text(
        "window.MRI_SERIES = " + json.dumps(manifest, ensure_ascii=False) + ";\n"
        + "window.MRI_VOLUME = " + json.dumps(volume, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"Gotowe: {len(manifest)} serii, {sum(x['count'] for x in manifest)} obrazów -> {OUT}")


if __name__ == "__main__":
    main()
