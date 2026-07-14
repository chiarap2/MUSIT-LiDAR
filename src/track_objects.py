"""
track_objects.py

Tracks detected objects across the multiple rotations ("frames") recorded by
a stationary LiDAR sensor, answering: is a cluster detected in frame N the
same physical object as a cluster detected in frame N+1, N+2, ... -- and if
so, is it moving or just a static object being re-detected every rotation?

Pipeline, per frame:
  1. Ground plane is fit ONCE (RANSAC) on the whole point cloud and reused for
     every frame -- the sensor is stationary, so ground geometry doesn't
     change between rotations, and fitting once is both faster and more
     stable than refitting per frame. Pass --ground-plane a JSON file from
     fit_ground_plane.py to skip this fit entirely (e.g. reusing the exact
     same ground as a cluster_objects.py run on the same point cloud).
  2. Each frame is voxel-downsampled and clustered independently (see
     cluster_objects.py for the available --method options).
  3. This frame's cluster centroids are matched against the previous frame's
     active tracks via greedy nearest-centroid matching (closest pairs first,
     one-to-one, within --max-match-distance). Matched clusters keep their
     track id; unmatched clusters start a new one. A track survives up to
     --max-missed-frames without a match before it's retired, so a brief
     rotation where an object drops below --min-samples doesn't split its
     track in two.
  4. Per track, aggregate stats (path length, net displacement, max spread
     around its own mean position) classify it as "static" (re-detected in
     roughly the same spot) or "moving".

IMPORTANT: same caveat as cluster_objects.py -- run this on the LOCAL point
cloud (x, y, z relative to the sensor), not a georeferenced one.

Usage:
    python track_objects.py point_cloud.ply
    python track_objects.py point_cloud.ply --frame-start 0 --frame-end 5
    python track_objects.py point_cloud.ply --method hdbscan --max-match-distance 0.8

Dependencies:
    pip install numpy matplotlib scikit-learn scipy
"""

import argparse
import numpy as np
import matplotlib
if __name__ == "__main__":
    # Only force the non-interactive Agg backend for standalone CLI runs -- if this module
    # is imported instead (e.g. from a notebook), leave whatever backend is already active
    # (Agg has no display, so plt.show()/interactive widgets would silently render nothing).
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lidar_io import read_ply_full, derive_frame_index
from ground import voxel_downsample, fit_ground_plane_ransac, split_by_plane, load_ground_plane
from cluster_objects import run_clustering, summarize_clusters, save_summary_csv


def fit_ground_plane_once(cloud, voxel_size, ground_threshold, ransac_iterations):
    """Fits the ground plane on a (voxel-downsampled, for speed) copy of the whole cloud."""
    plane_fit_cloud = voxel_downsample(cloud, voxel_size) if voxel_size > 0 else cloud
    plane, inlier_count = fit_ground_plane_ransac(
        plane_fit_cloud[:, :3], n_iterations=ransac_iterations, distance_threshold=ground_threshold
    )
    if plane is None:
        raise RuntimeError("RANSAC failed to find a plane (not enough points, or all collinear).")
    a, b, c, d = plane
    print(f"Ground plane (fit once, reused across all frames): {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0 "
          f"({inlier_count} inliers out of {len(plane_fit_cloud)} points)")
    return plane


def match_tracks(active_centroids, cluster_centroids, max_match_distance):
    """
    Greedy nearest-centroid matching: considers every (cluster, track) pair
    within max_match_distance, sorted by distance ascending, and assigns them
    one-to-one, closest first. Returns {cluster_index: track_id} for matches;
    unmatched cluster indices are simply absent from the result.
    """
    track_ids = list(active_centroids.keys())
    if not track_ids or len(cluster_centroids) == 0:
        return {}

    track_xyz = np.array([active_centroids[t] for t in track_ids])
    diffs = cluster_centroids[:, None, :] - track_xyz[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)  # (n_clusters, n_tracks)

    pairs = [
        (dists[ci, ti], ci, ti)
        for ci in range(dists.shape[0])
        for ti in range(dists.shape[1])
        if dists[ci, ti] <= max_match_distance
    ]
    pairs.sort(key=lambda p: p[0])

    assignment = {}
    matched_clusters, matched_tracks = set(), set()
    for _, ci, ti in pairs:
        if ci in matched_clusters or ti in matched_tracks:
            continue
        assignment[ci] = track_ids[ti]
        matched_clusters.add(ci)
        matched_tracks.add(ti)
    return assignment


def plot_tracks(track_history, output_path, static_threshold):
    """
    Top-down plot of every track's centroid path across frames. Static tracks
    (re-detected near the same spot) are drawn muted gray; moving tracks are
    colored and labeled, so real motion is visible at a glance.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab20")
    color_i = 0

    for track_id in sorted(track_history):
        history = sorted(track_history[track_id], key=lambda h: h[0])
        centroids = np.array([h[1] for h in history])
        mean_centroid = centroids.mean(axis=0)
        max_dist_from_mean = float(np.linalg.norm(centroids - mean_centroid, axis=1).max())
        is_static = max_dist_from_mean < static_threshold

        if is_static:
            color, zorder = "lightgray", 1
        else:
            color, zorder = cmap(color_i % 20), 2
            color_i += 1

        if len(centroids) > 1:
            ax.plot(centroids[:, 0], centroids[:, 1], "-o", color=color,
                     markersize=4, linewidth=1.2, zorder=zorder)
        else:
            ax.scatter(centroids[:, 0], centroids[:, 1], color=color, s=30, zorder=zorder)

        if not is_static:
            ax.annotate(f"#{track_id}", (centroids[-1, 0], centroids[-1, 1]),
                         fontsize=8, color=color, zorder=zorder + 1)

    ax.set_aspect("equal")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_title(f"Object tracks: {len(track_history)} tracks (top-down view)\n"
                 f"gray = static (<{static_threshold}m spread), colored = moving")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster each rotation of a LiDAR point cloud and track objects across frames."
    )
    parser.add_argument("input", help="Input PLY point cloud file (LOCAL frame, not georeferenced)")
    parser.add_argument("--frame-start", type=int, default=None, help="First frame to process. Default: 0.")
    parser.add_argument("--frame-end", type=int, default=None,
                         help="Last frame to process (inclusive). Default: last frame in the file.")
    parser.add_argument("--ground-threshold", type=float, default=0.15,
                         help="Max distance (m) from the fitted ground plane to be considered ground. Default 0.15.")
    parser.add_argument("--ransac-iterations", type=int, default=200,
                         help="Number of RANSAC iterations for the one-time ground plane fit. Only used if "
                              "--ground-plane is not given. Default 200.")
    parser.add_argument("--ground-plane", default=None,
                         help="Path to a ground_plane.json from fit_ground_plane.py. If given, skips the "
                              "one-time RANSAC fit entirely and reuses this plane.")
    parser.add_argument("--method", choices=["dbscan", "hdbscan", "voxel-cc", "euclidean"], default="dbscan",
                         help="Clustering algorithm applied to each frame. Default dbscan.")
    parser.add_argument("--eps", type=float, default=0.5,
                         help="Neighborhood radius (m) for dbscan/euclidean, and default voxel size for voxel-cc. Default 0.5.")
    parser.add_argument("--min-samples", type=int, default=10,
                         help="Minimum points to form a cluster (dbscan/euclidean/voxel-cc), and default "
                              "min_cluster_size for hdbscan. Default 10.")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=None,
                         help="min_cluster_size for --method hdbscan. Default: same as --min-samples.")
    parser.add_argument("--cc-voxel-size", type=float, default=None,
                         help="Voxel size (m) for --method voxel-cc. Default: same as --eps.")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                         help="Voxel downsampling size (m), applied independently within each frame "
                              "(and once for the ground-plane fit). Default 0.05. Set to 0 to disable.")
    parser.add_argument("--max-match-distance", type=float, default=1.0,
                         help="Max centroid distance (m) between frames to consider two clusters the same "
                              "object. Default 1.0.")
    parser.add_argument("--max-missed-frames", type=int, default=2,
                         help="Number of consecutive frames a track can go unmatched before it's retired. "
                              "Default 2 (tolerates a rotation occasionally missing the object).")
    parser.add_argument("--static-threshold", type=float, default=0.3,
                         help="Max distance (m) a track's centroid can wander from its own mean position "
                              "and still be classified 'static' rather than 'moving'. Default 0.3.")
    parser.add_argument("--output-prefix", default="", help="Prefix for output filenames")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    cloud, field_names = read_ply_full(args.input)
    print(f"Loaded {len(cloud)} points with fields: {field_names}")

    print("Deriving per-rotation frame index ...")
    frame_ids = derive_frame_index(cloud)
    max_frame = int(frame_ids.max())
    frame_start = args.frame_start if args.frame_start is not None else 0
    frame_end = args.frame_end if args.frame_end is not None else max_frame
    print(f"File spans frames 0..{max_frame}; processing frames {frame_start}..{frame_end}")

    if args.ground_plane:
        print(f"Loading precomputed ground plane from {args.ground_plane} (skipping RANSAC) ...")
        plane = load_ground_plane(args.ground_plane)
    else:
        plane = fit_ground_plane_once(cloud, args.voxel_size, args.ground_threshold, args.ransac_iterations)

    active_tracks = {}   # track_id -> {"centroid": np.array(3,), "missed": int}
    track_history = {}   # track_id -> list of (frame_id, centroid, n_points)
    per_frame_rows = []
    next_track_id = 0

    for frame_id in range(frame_start, frame_end + 1):
        frame_mask = frame_ids == frame_id
        if not frame_mask.any():
            continue
        frame_cloud = cloud[frame_mask]

        if args.voxel_size > 0:
            frame_cloud = voxel_downsample(frame_cloud, args.voxel_size)

        non_ground, _ground = split_by_plane(frame_cloud, plane, args.ground_threshold)

        if len(non_ground) > 0:
            labels = run_clustering(
                non_ground[:, :3], method=args.method, eps=args.eps, min_samples=args.min_samples,
                hdbscan_min_cluster_size=args.hdbscan_min_cluster_size, cc_voxel_size=args.cc_voxel_size,
            )
            summaries = summarize_clusters(non_ground, labels)
        else:
            summaries = []

        cluster_centroids = np.array(
            [[s["centroid_x"], s["centroid_y"], s["centroid_z"]] for s in summaries]
        ) if summaries else np.zeros((0, 3))

        active_centroids = {tid: t["centroid"] for tid, t in active_tracks.items()}
        assignment = match_tracks(active_centroids, cluster_centroids, args.max_match_distance)

        matched_this_frame = set()
        for ci, summary in enumerate(summaries):
            track_id = assignment.get(ci)
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
            matched_this_frame.add(track_id)

            centroid = cluster_centroids[ci]
            active_tracks[track_id] = {"centroid": centroid, "missed": 0}
            track_history.setdefault(track_id, []).append((frame_id, centroid, summary["n_points"]))

            row = dict(summary)
            row["cluster_id_in_frame"] = row.pop("cluster_id")
            row["frame"] = frame_id
            row["track_id"] = track_id
            per_frame_rows.append(row)

        for track_id in list(active_tracks):
            if track_id not in matched_this_frame:
                active_tracks[track_id]["missed"] += 1
                if active_tracks[track_id]["missed"] > args.max_missed_frames:
                    del active_tracks[track_id]

        print(f"  Frame {frame_id}: {len(summaries)} clusters, {len(active_tracks)} active tracks")

    track_summaries = []
    for track_id in sorted(track_history):
        history = sorted(track_history[track_id], key=lambda h: h[0])
        frames = [h[0] for h in history]
        centroids = np.array([h[1] for h in history])
        n_points_list = [h[2] for h in history]

        mean_centroid = centroids.mean(axis=0)
        max_dist_from_mean = float(np.linalg.norm(centroids - mean_centroid, axis=1).max())

        if len(centroids) > 1:
            total_path_length = float(np.linalg.norm(np.diff(centroids, axis=0), axis=1).sum())
            net_displacement = float(np.linalg.norm(centroids[-1] - centroids[0]))
        else:
            total_path_length = 0.0
            net_displacement = 0.0

        classification = "static" if max_dist_from_mean < args.static_threshold else "moving"

        track_summaries.append({
            "track_id": track_id,
            "n_frames": len(history),
            "first_frame": frames[0],
            "last_frame": frames[-1],
            "mean_n_points": float(np.mean(n_points_list)),
            "mean_centroid_x": float(mean_centroid[0]),
            "mean_centroid_y": float(mean_centroid[1]),
            "mean_centroid_z": float(mean_centroid[2]),
            "total_path_length_m": total_path_length,
            "net_displacement_m": net_displacement,
            "max_displacement_from_mean_m": max_dist_from_mean,
            "classification": classification,
        })

    n_moving = sum(1 for t in track_summaries if t["classification"] == "moving")
    print(f"Found {len(track_summaries)} tracks ({n_moving} moving, {len(track_summaries) - n_moving} static)")

    per_frame_csv = f"{args.output_prefix}tracks_per_frame.csv"
    save_summary_csv(per_frame_rows, per_frame_csv)
    print(f"Saved per-frame cluster/track rows to {per_frame_csv}")

    summary_csv = f"{args.output_prefix}tracks_summary.csv"
    save_summary_csv(track_summaries, summary_csv)
    print(f"Saved per-track summary to {summary_csv}")

    plot_path = f"{args.output_prefix}tracks_plot.png"
    plot_tracks(track_history, plot_path, static_threshold=args.static_threshold)
    print(f"Saved trajectory plot to {plot_path}")


if __name__ == "__main__":
    main()
