"""
ground.py

Ground-plane fitting/removal, shared by fit_ground_plane.py,
cluster_objects.py, and track_objects.py.

RANSAC plane fitting is the expensive step in the pipeline (hundreds of
random 3-point samples, each scored against every point). It only needs to
run ONCE per point cloud -- the ground doesn't move between clustering
method comparisons, or between frames of a stationary-sensor capture -- so
fit_ground_plane.py fits it a single time and saves the result to a small
JSON file (save_ground_plane/load_ground_plane) that other stages load
instead of re-fitting.
"""

import json
import numpy as np


def voxel_downsample(cloud, voxel_size):
    """
    Reduces the point cloud by averaging all points that fall in the same
    voxel_size-sized grid cell into a single point. This is the standard fix
    for memory/time blowups in ground-fitting and clustering when a
    stationary sensor has recorded many rotations: without this, the same
    physical surface (e.g. the floor) gets sampled again and again at nearly
    identical coordinates, and a clustering method's neighbor lists for
    those points can explode in size and memory.

    Averages all columns (x, y, z, intensity, timestamp, ...), not just xyz,
    so extra columns are preserved (as an average over the voxel).
    """
    xyz = cloud[:, :3]
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)

    n_voxels = len(counts)
    n_fields = cloud.shape[1]
    sums = np.zeros((n_voxels, n_fields), dtype=np.float64)
    np.add.at(sums, inverse, cloud)
    downsampled = (sums / counts[:, None]).astype(np.float32)
    return downsampled


def fit_ground_plane_ransac(points, n_iterations=200, distance_threshold=0.15, seed=0):
    """
    Fits a plane a*x + b*y + c*z + d = 0 to the points using RANSAC, returning
    (a, b, c, d) for the plane with the most inliers within distance_threshold.

    Simple, dependency-free RANSAC: repeatedly sample 3 random points, form
    the plane through them, count how many points lie within the threshold
    distance, and keep the best one. This is the expensive step (200
    iterations, each scoring every point) -- fit it once per capture with
    fit_ground_plane.py rather than repeating it per clustering run.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    best_inlier_count = -1
    best_plane = None

    for _ in range(n_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = points[idx]

        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue  # degenerate (collinear) sample, skip
        normal = normal / norm_len
        a, b, c = normal
        d = -np.dot(normal, p1)

        distances = np.abs(points @ normal + d)
        inlier_count = int(np.sum(distances < distance_threshold))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = (a, b, c, d)

    return best_plane, best_inlier_count


def split_by_plane(cloud, plane, distance_threshold):
    """
    Splits cloud into (non_ground, ground) using an already-fitted plane
    (a, b, c, d), without re-running RANSAC. This is the cheap part (one
    dot product + threshold per point) -- safe to repeat per frame/per
    clustering run, unlike the RANSAC fit itself.
    """
    a, b, c, d = plane
    normal = np.array([a, b, c])
    distances = np.abs(cloud[:, :3] @ normal + d)
    is_ground = distances < distance_threshold
    return cloud[~is_ground], cloud[is_ground]


def remove_ground(cloud, distance_threshold=0.15, n_iterations=200, seed=0):
    """
    Fits AND removes the ground plane in one call (RANSAC + split). Kept for
    standalone convenience/backward compatibility when no precomputed plane
    is available; prefer fit_ground_plane.py + split_by_plane when running
    many clustering methods or frames against the same cloud.
    Returns (non_ground_cloud, ground_cloud, plane_params).
    """
    xyz = cloud[:, :3]
    plane, inlier_count = fit_ground_plane_ransac(xyz, n_iterations=n_iterations,
                                                     distance_threshold=distance_threshold, seed=seed)
    if plane is None:
        raise RuntimeError("RANSAC failed to find a plane (not enough points, or all collinear).")

    a, b, c, d = plane
    print(f"Ground plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0 "
          f"({inlier_count} inliers out of {len(cloud)} points, {100 * inlier_count / len(cloud):.1f}%)")

    non_ground, ground = split_by_plane(cloud, plane, distance_threshold)
    return non_ground, ground, plane


def save_ground_plane(plane, path, **metadata):
    """Saves a fitted plane (a, b, c, d) plus any fit metadata to a small JSON file."""
    a, b, c, d = plane
    data = {"a": float(a), "b": float(b), "c": float(c), "d": float(d), **metadata}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_ground_plane(path):
    """Loads a plane (a, b, c, d) previously saved by save_ground_plane."""
    with open(path) as f:
        data = json.load(f)
    return (data["a"], data["b"], data["c"], data["d"])
