import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
from nibabel.processing import resample_from_to, resample_to_output
from scipy.ndimage import binary_erosion, gaussian_filter
from skimage.measure import marching_cubes
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


ROOT = Path(__file__).parent
SOURCE = ROOT / "20261006"
WORK = ROOT / "segmentation"


def series_files(number: int) -> list[str]:
    files = []
    for path in SOURCE.rglob("*"):
        if not path.is_file():
            continue
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        if int(getattr(ds, "SeriesNumber", -1)) == number:
            position = tuple(float(x) for x in getattr(ds, "ImagePositionPatient", (0, 0, getattr(ds, "InstanceNumber", 0))))
            files.append((position, str(path)))
    assert files, f"Brak serii {number}"
    return [path for _, path in sorted(files)]


def main() -> None:
    WORK.mkdir(exist_ok=True)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_files(401))
    image = reader.Execute()
    output = WORK / "series401.nii.gz"
    sitk.WriteImage(image, output)
    print(f"{output}: rozmiar={image.GetSize()}, odstępy={image.GetSpacing()}")


def export_meshes() -> None:
    detailed = WORK / "derivatives_spineps"
    instances = detailed / "mod-series401_seg-vert_msk.nii.gz"
    anatomy = detailed / "mod-series401_seg-spine_msk.nii.gz"
    sources = [("C1", WORK / "vertebrae" / "vertebrae_C1.nii.gz", "bone", None)]
    sources += [(f"C{i}", instances, "bone", i) for i in range(2, 8)]
    sources += [("T1", instances, "bone", 8)]
    sources += [(f"C{i}/{'T1' if i == 7 else 'C' + str(i + 1)}", instances, "disc", i + 100) for i in range(2, 8)]
    sources += [("Rdzeń", anatomy, "cord", 60), ("Kanał", anatomy, "canal", 61)]
    cache = {}
    def source_image(path: Path):
        if path not in cache:
            image = nib.load(path)
            cache[path] = (image, np.asarray(image.dataobj))
        return cache[path]

    # Przytnij rdzeń i kanał do poziomów kręgów widocznych w modelu.
    first, first_data = source_image(WORK / "vertebrae" / "vertebrae_C1.nii.gz")
    last, last_data = source_image(WORK / "vertebrae" / "vertebrae_T1.nii.gz")
    heights = []
    for image, data in ((first, first_data), (last, last_data)):
        coords = np.argwhere(data > 0)
        heights.extend(nib.affines.apply_affine(image.affine, coords)[:, 2])
    zmin, zmax = min(heights) - 3, max(heights) + 3
    meshes = []
    for name, path, kind, label in sources:
        source, data = source_image(path)
        selected = (data > 0) if label is None else (data == label)
        image = nib.Nifti1Image(selected.astype(np.float32), source.affine)
        image = resample_to_output(image, voxel_sizes=(.8, .8, .8), order=1)
        mask = np.asarray(image.dataobj, dtype=np.float32)
        if kind in ("cord", "canal"):
            z = image.affine[2, 2] * np.arange(mask.shape[2]) + image.affine[2, 3]
            mask[:, :, (z < zmin) | (z > zmax)] = 0
        assert mask.max() > 0, f"Pusta maska: {path}"
        # Oś X odpowiada odstępowi 3,3 mm między warstwami; krążki wygładzamy oszczędniej.
        sigma = (1.5, .85, .85) if kind == "bone" else (1.25, .55, .55)
        vertices, faces, _, _ = marching_cubes(gaussian_filter(mask, sigma), .45, step_size=2 if kind == "canal" else 1)
        world = nib.affines.apply_affine(image.affine, vertices)
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(world, dtype=np.float32), deep=True))
        cells = vtk.vtkCellArray()
        cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64).ravel(), deep=True))
        poly.SetPoints(points)
        poly.SetPolys(cells)
        # ponytail: jeden podział Loop; dodatkowe trójkąty nie tworzą brakujących danych MRI.
        subdivide = vtk.vtkLoopSubdivisionFilter()
        subdivide.SetNumberOfSubdivisions(1)
        subdivide.SetInputData(poly)
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputConnection(subdivide.GetOutputPort())
        smooth.SetNumberOfIterations(18 if kind == "bone" else 10)
        smooth.SetPassBand(.12 if kind == "bone" else .22)
        smooth.BoundarySmoothingOff()
        smooth.FeatureEdgeSmoothingOff()
        smooth.NormalizeCoordinatesOn()
        smooth.Update()
        refined = smooth.GetOutput()
        world = vtk_to_numpy(refined.GetPoints().GetData())
        faces = vtk_to_numpy(refined.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
        meshes.append({
            "name": name,
            "kind": kind,
            "center": np.round(world.mean(axis=0), 3).tolist(),
            "vertices": np.round(world, 3).ravel().tolist(),
            "indices": faces.astype(np.uint32).ravel().tolist(),
        })
        print(f"{name}: {len(world)} wierzchołków, {len(faces)} trójkątów")
    output = ROOT / "mri_viewer" / "spine_meshes_detail.json"
    output.write_text(json.dumps(meshes, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"{output}: {output.stat().st_size / 1024 / 1024:.1f} MB")


def export_volume_contours() -> None:
    reference = nib.load(WORK / "series401.nii.gz")
    instances = nib.load(WORK / "derivatives_spineps" / "mod-series401_seg-vert_msk.nii.gz")
    assert reference.shape == instances.shape and np.allclose(reference.affine, instances.affine, atol=0.01)
    labels = np.asarray(instances.dataobj)
    bone = np.isin(labels, range(2, 9))
    c1 = nib.load(WORK / "vertebrae" / "vertebrae_C1.nii.gz")
    bone |= np.asarray(resample_from_to(c1, reference, order=0).dataobj) > 0
    disc = np.isin(labels, range(102, 108))
    contours = np.zeros(reference.shape, dtype=np.uint8)
    for layer in range(reference.shape[2]):
        for mask, value in ((bone, 128), (disc, 255)):
            area = mask[:, :, layer]
            contours[:, :, layer][area & ~binary_erosion(area, iterations=2)] = value
    # PNG i volume.raw są zapisane w odwrotnej kolejności warstw niż NIfTI.
    output = ROOT / "mri_viewer" / "volume_contours.raw"
    output.write_bytes(np.transpose(contours, (2, 1, 0))[::-1].copy().tobytes())
    print(f"{output}: {np.count_nonzero(contours)} pikseli obrysu")


if __name__ == "__main__":
    if (WORK / "vertebrae" / "vertebrae_C1.nii.gz").exists():
        export_meshes()
        export_volume_contours()
    else:
        main()
