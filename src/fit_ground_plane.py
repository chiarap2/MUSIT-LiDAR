"""
fit_ground_plane.py

Fits the RANSAC ground plane ONCE on a point cloud and saves it to a small
JSON file. RANSAC (hundreds of random 3-point samples, each scored against
every point) is the expensive part of the pipeline -- the ground doesn't
change between clustering method comparisons or between frames of a
stationary-sensor capture, so fit it once here and pass the result to
cluster_objects.py / track_objects.py via --ground-plane instead of letting
each of them re-fit it from scratch.

Usage:
    python fit_ground_plane.py point_cloud.ply --output ground_plane.json

    # then reuse it across as many clustering runs as you like, with no RANSAC refit:
    python cluster_objects.py point_cloud.ply --ground-plane ground_plane.json --method dbscan
    python cluster_objects.py point_cloud.ply --ground-plane ground_plane.json --method hdbscan
    python track_objects.py point_cloud.ply --ground-plane ground_plane.json

Dependencies:
    pip install numpy
"""

import argparse

from lidar_io import read_ply_full
from ground import voxel_downsample, fit_ground_plane_ransac, save_ground_plane


def main():
    parser = argparse.ArgumentParser(description="Fit the RANSAC ground plane once and save it for reuse.")
    parser.add_argument("input", help="Input PLY point cloud file (LOCAL frame, not georeferenced)")
    parser.add_argument("--ground-threshold", type=float, default=0.15,
                         help="Max distance (m) from the fitted plane to count as a RANSAC inlier. Default 0.15. "
                              "(cluster_objects.py/track_objects.py can still use a different threshold to split "
                              "ground/non-ground with this same plane -- only the plane orientation is fixed here.)")
    parser.add_argument("--ransac-iterations", type=int, default=200,
                         help="Number of RANSAC iterations. Default 200.")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                         help="Voxel downsampling size (m) applied before fitting, so a multi-rotation capture's "
                              "repeated ground samples don't make RANSAC slower than it needs to be. Default 0.05. "
                              "Set to 0 to disable.")
    parser.add_argument("--output", default="ground_plane.json", help="Output JSON path. Default ground_plane.json.")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    cloud, _ = read_ply_full(args.input)
    print(f"Loaded {len(cloud)} points")

    if args.voxel_size > 0:
        n_before = len(cloud)
        cloud = voxel_downsample(cloud, args.voxel_size)
        print(f"Voxel downsampling ({args.voxel_size}m): {n_before} -> {len(cloud)} points "
              f"({100 * len(cloud) / n_before:.1f}% kept)")

    print(f"Fitting ground plane (RANSAC, {args.ransac_iterations} iterations) ...")
    plane, inlier_count = fit_ground_plane_ransac(
        cloud[:, :3], n_iterations=args.ransac_iterations, distance_threshold=args.ground_threshold
    )
    if plane is None:
        raise RuntimeError("RANSAC failed to find a plane (not enough points, or all collinear).")

    a, b, c, d = plane
    print(f"Ground plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0 "
          f"({inlier_count} inliers out of {len(cloud)} points, {100 * inlier_count / len(cloud):.1f}%)")

    save_ground_plane(
        plane, args.output,
        ground_threshold=args.ground_threshold, ransac_iterations=args.ransac_iterations,
        voxel_size=args.voxel_size, n_points_fit=len(cloud), n_inliers=inlier_count,
        source=args.input,
    )
    print(f"Saved ground plane to {args.output}")


if __name__ == "__main__":
    main()
