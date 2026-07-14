"""
cluster_objects.py

Simple object detection on a LiDAR point cloud via clustering:
  1. Remove the ground plane (RANSAC plane fit)
  2. Cluster the remaining points (DBSCAN) -> each cluster is a candidate object
  3. Compute a bounding box + stats for each cluster
  4. Plot the result (top-down view, clusters colored, bounding boxes drawn)

IMPORTANT: run this on the LOCAL point cloud (x, y, z relative to the sensor),
not on a georeferenced one. Georeferenced files store UTM coordinates (huge
numbers, ~300000+) as float32, which loses ~0.5m of precision at that
magnitude -- enough to corrupt small object bounding boxes. Cluster first,
georeference the results afterwards if you need real-world coordinates for
the detected objects.

Usage:
    python cluster_objects.py point_cloud.ply
    python cluster_objects.py point_cloud.ply --eps 0.4 --min-samples 15
    python cluster_objects.py point_cloud.ply --frame 0   # only one full rotation

RANSAC ground-plane fitting is the slow part of step 1, and doesn't change
between clustering method comparisons. If you're comparing multiple
--method runs (or also running track_objects.py) on the same point cloud,
fit it once instead:
    python fit_ground_plane.py point_cloud.ply --output ground_plane.json
    python cluster_objects.py point_cloud.ply --ground-plane ground_plane.json --method dbscan
    python cluster_objects.py point_cloud.ply --ground-plane ground_plane.json --method hdbscan

Dependencies:
    pip install numpy matplotlib scikit-learn
"""

import sys
import csv
import argparse
import numpy as np
import matplotlib
if __name__ == "__main__":
    # Only force the non-interactive Agg backend for standalone CLI runs -- if this module
    # is imported instead (e.g. from a notebook), leave whatever backend is already active
    # (Agg has no display, so plt.show()/interactive widgets would silently render nothing).
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN

from lidar_io import read_ply_full, derive_frame_index
from ground import voxel_downsample, remove_ground, split_by_plane, load_ground_plane


def cluster_dbscan(xyz, eps=0.5, min_samples=10):
    """
    Density-based clustering: a point is a "core point" if it has >= min_samples
    neighbors within eps; clusters are formed by chaining core points together
    (and their neighbors). Points that end up in no cluster are noise (-1).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    return db.labels_


def cluster_hdbscan(xyz, min_cluster_size=10, min_samples=None):
    """
    Hierarchical density-based clustering. Unlike DBSCAN, there's no single
    global eps: it builds a hierarchy of clusters over a range of densities
    and extracts the most stable ones, so it copes better with a scene that
    has both dense (near) and sparse (far) objects. min_samples defaults to
    min_cluster_size (sklearn's own default) when not given.
    """
    from sklearn.cluster import HDBSCAN
    db = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(xyz)
    return db.labels_


def cluster_voxel_cc(xyz, voxel_size=0.5, min_samples=10):
    """
    Connected-components clustering on a voxel grid: snap points to a
    voxel_size grid, then flood-fill (union-find) 26-connected occupied
    voxels into components. Much faster than a neighbor-radius search on
    dense clouds since adjacency is a dict lookup rather than a tree query,
    at the cost of clusters being grid-aligned rather than smoothly radius-based.
    Components with fewer than min_samples points become noise (-1).
    """
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)
    unique_voxels, inverse, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)
    n_voxels = len(unique_voxels)
    voxel_to_id = {tuple(v): i for i, v in enumerate(unique_voxels)}

    parent = list(range(n_voxels))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if not (dx == 0 and dy == 0 and dz == 0)]
    for i, v in enumerate(unique_voxels):
        vt = tuple(v)
        for off in offsets:
            neighbor = (vt[0] + off[0], vt[1] + off[1], vt[2] + off[2])
            j = voxel_to_id.get(neighbor)
            if j is not None:
                union(i, j)

    # sum point counts per component (root), then keep only components with enough points
    roots = np.array([find(i) for i in range(n_voxels)])
    voxel_point_counts = counts  # points per voxel, aligned with unique_voxels/roots
    component_sizes = {}
    for r, c in zip(roots, voxel_point_counts):
        component_sizes[r] = component_sizes.get(r, 0) + c

    valid_roots = [r for r, size in component_sizes.items() if size >= min_samples]
    root_to_label = {r: label for label, r in enumerate(valid_roots)}

    voxel_labels = np.array([root_to_label.get(r, -1) for r in roots])
    return voxel_labels[inverse]


def cluster_euclidean(xyz, eps=0.5, min_samples=10):
    """
    Classic "Euclidean cluster extraction" (as in PCL): build a KD-tree, then
    for each unvisited point do a BFS/region-growing radius search (eps),
    marking all reachable points as visited. A grown region becomes a cluster
    if it has >= min_samples points, otherwise its points are noise (-1).
    Unlike DBSCAN, there's no core-point requirement -- any connectivity
    within eps joins two points -- so results can differ slightly at sparse
    cluster edges/tails.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    n = len(xyz)
    visited = np.zeros(n, dtype=bool)
    labels = np.full(n, -1, dtype=np.int64)
    next_label = 0

    for seed in range(n):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [seed]
        region = [seed]
        while stack:
            idx = stack.pop()
            for neighbor in tree.query_ball_point(xyz[idx], eps):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
                    region.append(neighbor)
        if len(region) >= min_samples:
            labels[region] = next_label
            next_label += 1

    return labels


def run_clustering(xyz, method="dbscan", eps=0.5, min_samples=10,
                    hdbscan_min_cluster_size=None, cc_voxel_size=None):
    """
    Dispatches to one of the clustering methods above by name, filling in
    sensible defaults so all methods can share the same --eps/--min-samples
    CLI options unless the caller wants to override a method-specific knob.
    """
    if method == "dbscan":
        return cluster_dbscan(xyz, eps=eps, min_samples=min_samples)
    elif method == "hdbscan":
        min_cluster_size = hdbscan_min_cluster_size if hdbscan_min_cluster_size is not None else min_samples
        return cluster_hdbscan(xyz, min_cluster_size=min_cluster_size)
    elif method == "voxel-cc":
        voxel_size = cc_voxel_size if cc_voxel_size is not None else eps
        return cluster_voxel_cc(xyz, voxel_size=voxel_size, min_samples=min_samples)
    elif method == "euclidean":
        return cluster_euclidean(xyz, eps=eps, min_samples=min_samples)
    else:
        raise ValueError(f"Unknown clustering method: {method!r}")


def summarize_clusters(non_ground_cloud, labels):
    """
    Computes per-cluster stats: point count, centroid, axis-aligned bounding
    box dimensions, and horizontal distance from the origin (sensor).
    Returns a list of dicts, one per cluster (excluding noise, label -1).
    """
    summaries = []
    unique_labels = sorted(l for l in np.unique(labels) if l != -1)

    for label in unique_labels:
        mask = labels == label
        pts = non_ground_cloud[mask, :3]
        intensity = non_ground_cloud[mask, 3]

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        centroid = pts.mean(axis=0)
        size = maxs - mins
        horizontal_distance = float(np.hypot(centroid[0], centroid[1]))

        summaries.append({
            "cluster_id": int(label),
            "n_points": int(mask.sum()),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "size_x": float(size[0]),
            "size_y": float(size[1]),
            "size_z": float(size[2]),
            "min_x": float(mins[0]), "max_x": float(maxs[0]),
            "min_y": float(mins[1]), "max_y": float(maxs[1]),
            "min_z": float(mins[2]), "max_z": float(maxs[2]),
            "horizontal_distance_m": horizontal_distance,
            "mean_intensity": float(intensity.mean()),
        })

    return summaries


def save_summary_csv(summaries, output_path):
    """Writes the per-cluster summary table to a CSV file."""
    if not summaries:
        print("No clusters found -- nothing to save.")
        return
    fieldnames = list(summaries[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def plot_clusters(non_ground_cloud, labels, ground_cloud, summaries, output_path,
                   point_size=1.5, max_ground_points=50_000, method=None):
    """
    Top-down plot: ground points in light gray, clustered points colored by
    cluster id, noise points (label -1) in dark gray, and a bounding box +
    id label drawn around each detected cluster.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # ground, subsampled for speed/clarity
    if len(ground_cloud) > 0:
        g = ground_cloud
        if len(g) > max_ground_points:
            idx = np.random.choice(len(g), size=max_ground_points, replace=False)
            g = g[idx]
        ax.scatter(g[:, 0], g[:, 1], s=point_size * 0.5, c="lightgray", label="ground")

    # noise (unclustered) points
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(non_ground_cloud[noise_mask, 0], non_ground_cloud[noise_mask, 1],
                   s=point_size, c="dimgray", label="noise")

    # clustered points + bounding boxes
    cmap = plt.get_cmap("tab20")
    for i, summary in enumerate(summaries):
        mask = labels == summary["cluster_id"]
        color = cmap(i % 20)
        ax.scatter(non_ground_cloud[mask, 0], non_ground_cloud[mask, 1], s=point_size, color=color)

        rect = Rectangle(
            (summary["min_x"], summary["min_y"]),
            summary["size_x"], summary["size_y"],
            fill=False, edgecolor=color, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.annotate(
            f"#{summary['cluster_id']} ({summary['n_points']}pts)",
            (summary["min_x"], summary["max_y"]),
            fontsize=8, color=color, va="bottom",
        )

    ax.set_aspect("equal")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    method_suffix = f" [{method}]" if method else ""
    ax.set_title(f"Detected objects: {len(summaries)} clusters (top-down view){method_suffix}")
    ax.legend(loc="upper right", markerscale=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Detect objects in a LiDAR point cloud via ground removal + clustering.")
    parser.add_argument("input", help="Input PLY point cloud file (LOCAL frame, not georeferenced)")
    parser.add_argument("--frame", type=int, default=None,
                         help="If set, only use points from this single full-rotation frame "
                              "(derived on the fly from x/y). Default: use all points in the file.")
    parser.add_argument("--ground-threshold", type=float, default=0.15,
                         help="Max distance (m) from the ground plane to be considered ground. Default 0.15.")
    parser.add_argument("--ransac-iterations", type=int, default=200,
                         help="Number of RANSAC iterations for ground plane fitting. Only used if --ground-plane "
                              "is not given. Default 200.")
    parser.add_argument("--ground-plane", default=None,
                         help="Path to a ground_plane.json from fit_ground_plane.py. If given, skips the RANSAC "
                              "fit entirely and reuses this plane -- much faster when comparing multiple --method "
                              "runs, or when a matching track_objects.py run should use the exact same ground.")
    parser.add_argument("--method", choices=["dbscan", "hdbscan", "voxel-cc", "euclidean"], default="dbscan",
                         help="Clustering algorithm to use. Default dbscan.")
    parser.add_argument("--eps", type=float, default=0.5,
                         help="Neighborhood radius (m) used by dbscan/euclidean: points within this distance are "
                              "considered connected. Smaller = stricter clusters, larger = merges nearby objects. "
                              "Also used as the default voxel size for --method voxel-cc. Default 0.5.")
    parser.add_argument("--min-samples", type=int, default=10,
                         help="Minimum points to form a cluster (dbscan/euclidean/voxel-cc) or, unless overridden "
                              "by --hdbscan-min-cluster-size, the min_cluster_size for hdbscan. Default 10. Lower "
                              "it if small/distant objects (fewer LiDAR returns) are being missed.")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=None,
                         help="min_cluster_size for --method hdbscan. Default: same as --min-samples.")
    parser.add_argument("--cc-voxel-size", type=float, default=None,
                         help="Voxel size (m) for --method voxel-cc's connected-components grid. Default: same as --eps.")
    parser.add_argument("--output-prefix", default="", help="Prefix for output filenames")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                         help="Voxel grid downsampling size in meters, applied before ground removal "
                              "and clustering (averages points within each cell). This is critical if "
                              "the point cloud accumulates many rotations of a stationary sensor -- "
                              "without it, near-duplicate points can make RANSAC/DBSCAN extremely slow "
                              "or blow up memory. Default 0.05 (5cm). Set to 0 to disable.")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    cloud, field_names = read_ply_full(args.input)
    print(f"Loaded {len(cloud)} points with fields: {field_names}")

    if args.frame is not None:
        frame_ids = derive_frame_index(cloud)
        mask = frame_ids == args.frame
        if not mask.any():
            print(f"Error: frame {args.frame} not found (max frame index is {int(frame_ids.max())}).")
            sys.exit(1)
        cloud = cloud[mask]
        print(f"Filtered to frame {args.frame}: {len(cloud)} points.")

    if args.voxel_size > 0:
        n_before = len(cloud)
        cloud = voxel_downsample(cloud, args.voxel_size)
        print(f"Voxel downsampling ({args.voxel_size}m): {n_before} -> {len(cloud)} points "
              f"({100 * len(cloud) / n_before:.1f}% kept)")

    if args.ground_plane:
        print(f"Loading precomputed ground plane from {args.ground_plane} (skipping RANSAC) ...")
        plane = load_ground_plane(args.ground_plane)
        non_ground, ground = split_by_plane(cloud, plane, args.ground_threshold)
    else:
        print("Fitting ground plane (RANSAC) ...")
        non_ground, ground, plane = remove_ground(
            cloud, distance_threshold=args.ground_threshold, n_iterations=args.ransac_iterations
        )
    print(f"Non-ground points: {len(non_ground)} (removed {len(ground)} ground points)")

    print(f"Clustering with {args.method} (eps={args.eps}, min_samples={args.min_samples}) ...")
    labels = run_clustering(
        non_ground[:, :3], method=args.method, eps=args.eps, min_samples=args.min_samples,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size, cc_voxel_size=args.cc_voxel_size,
    )
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"Found {n_clusters} clusters ({n_noise} noise points, not assigned to any cluster)")

    summaries = summarize_clusters(non_ground, labels)
    for s in summaries:
        print(f"  Cluster #{s['cluster_id']}: {s['n_points']} points, "
              f"centroid=({s['centroid_x']:.2f}, {s['centroid_y']:.2f}, {s['centroid_z']:.2f}), "
              f"size=({s['size_x']:.2f} x {s['size_y']:.2f} x {s['size_z']:.2f}) m, "
              f"distance={s['horizontal_distance_m']:.2f}m")

    csv_path = f"{args.output_prefix}detected_objects.csv"
    save_summary_csv(summaries, csv_path)
    print(f"Saved cluster summary to {csv_path}")

    plot_path = f"{args.output_prefix}detected_objects.png"
    plot_clusters(non_ground, labels, ground, summaries, plot_path, method=args.method)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()