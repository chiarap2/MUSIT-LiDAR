"""
lidar_io.py

Shared point-cloud I/O helpers used across the pipeline scripts
(read_lidar_data.py, georeference_point_cloud.py, cluster_objects.py,
track_objects.py, plot_point_cloud.py, plot_clusters_on_map.py), so the PLY
format and per-rotation frame-index logic are defined in exactly one place.
"""

import numpy as np


def read_ply_full(path):
    """
    Reads a binary little-endian PLY file with float32 properties and
    returns the full Nx(n_fields) array plus the list of field names,
    so callers can access any column (e.g. timestamp) by name/index.
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertices = None
        field_names = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                field_names.append(line.split()[-1])
        if n_vertices is None:
            raise ValueError("Could not find 'element vertex' count in PLY header")

        n_fields = len(field_names)
        raw = f.read(n_vertices * 4 * n_fields)
        cloud = np.frombuffer(raw, dtype="<f4").reshape(n_vertices, n_fields)
        return cloud, field_names


def read_ply_xyzi(path):
    """
    Reads a binary little-endian PLY file with float32 properties and
    returns just the first 4 columns (x, y, z, intensity), regardless of
    whether the file has extra columns (e.g. timestamp) after them.
    """
    cloud, _ = read_ply_full(path)
    return cloud[:, :4]


def save_point_cloud_ply(cloud, output_path, field_names=("x", "y", "z", "intensity")):
    """
    Saves an Nx(len(field_names)) array as a binary PLY file.
    This is dramatically faster than text CSV for large point clouds
    (millions of points) and can be opened directly in CloudCompare,
    MeshLab, Open3D, etc. Extra fields beyond x,y,z (e.g. intensity,
    timestamp) are stored as additional PLY vertex properties.
    """
    n = len(cloud)
    property_lines = "".join(f"property float {name}\n" for name in field_names)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"{property_lines}"
        "end_header\n"
    ).encode("ascii")

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(cloud.astype("<f4").tobytes())


def save_point_cloud_csv(cloud, output_path, field_names=("x", "y", "z", "intensity")):
    """
    Saves an Nx(len(field_names)) array as a text CSV file.
    Simple and universally readable, but slow for large point clouds
    (tens of millions of rows can take a long time to write as text).
    Prefer save_point_cloud_ply for large recordings.
    """
    np.savetxt(
        output_path,
        cloud,
        delimiter=",",
        header=",".join(field_names),
        comments="",
        fmt="%.6f",
    )


def derive_frame_index(cloud, wrap_threshold_deg=180.0):
    """
    Derives a frame_index (one per full 360-degree sensor rotation) on the
    fly from the x, y columns already present in the point cloud, without
    needing a dedicated 'frame_index' column.

    Relies on:
      - x, y being in the sensor-local frame (x = r*sin(azimuth), y = r*cos(azimuth)),
        so azimuth = atan2(x, y) recovers each point's original scan angle exactly.
      - points being stored in their original chronological/scan order in the
        file (true for files produced by read_lidar_data.py, as long as they
        haven't been reordered or subsampled).

    NOTE: this only works correctly on LOCAL point clouds. On a georeferenced
    file, x/y have been rotated and translated to UTM coordinates, so
    atan2(x, y) no longer reflects the sensor's scan angle -- derive frames
    from the local point cloud instead, before georeferencing.
    """
    x, y = cloud[:, 0], cloud[:, 1]
    azimuth_deg = np.degrees(np.arctan2(x, y)) % 360.0

    diffs = np.diff(azimuth_deg)
    wraps = diffs < -wrap_threshold_deg
    frame_index = np.concatenate([[0], np.cumsum(wraps)]).astype(np.float32)
    return frame_index
