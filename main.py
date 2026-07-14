"""
main.py

Pipeline driver: chains the processing stages end to end.

  1. read           pcap -> local point cloud (src/read_lidar_data.py)
  2. fit ground     local point cloud -> ground_plane.json, ONE RANSAC fit (src/fit_ground_plane.py)
  3. cluster        local point cloud + ground plane -> detected objects, one snapshot (src/cluster_objects.py)
  4. track          local point cloud + ground plane -> objects tracked across frames (src/track_objects.py)
  5. georeference   detected objects -> real-world map (src/plot_clusters_on_map.py)

IMPORTANT: clustering and tracking run on the LOCAL point cloud, not a
georeferenced one -- see src/cluster_objects.py and src/track_objects.py for
why (float32 UTM coordinates lose too much precision for small object boxes).
Georeferencing only happens at the end, projecting the already-detected
objects' centroids/boxes to real-world coordinates for the map. If you want
a fully georeferenced point cloud for GIS tools instead, run
src/georeference_point_cloud.py directly -- it's a standalone utility, not
part of this pipeline.

The ground plane (RANSAC, the expensive step) is fit exactly ONCE and shared
by both the cluster and track stages -- see src/ground.py and
src/fit_ground_plane.py -- instead of each stage re-fitting it from scratch.

This is a thin orchestrator: each stage is just the existing CLI script run
as a subprocess, so every stage stays independently runnable/debuggable with
its own full flag set (run `python src/<stage>.py --help` for everything
this driver doesn't expose). A stage is skipped (with a log message) if its
output already exists, since reading a large pcap or fitting the ground
plane on a full multi-rotation cloud can each take minutes -- pass --force
to re-run anyway.

Usage:
    python main.py data/marina-giannis-test2-20260513-133011.pcap \
        --lat 37.430785059983215 --lon 24.941538092360716 --heading 280 \
        --output-dir outputs/

    # skip the final map step (e.g. no internet access for OSM tiles)
    python main.py data/capture.pcap --skip-map

Dependencies: same as the individual stage scripts (see requirements.txt).
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"


def run_stage(description, output_marker, cmd, force=False, allow_failure=False):
    """
    Runs one pipeline stage as a subprocess, skipping it if output_marker
    already exists (unless force). Returns True if the stage's output is
    available afterwards (either just produced or already there).
    """
    if output_marker.exists() and not force:
        print(f"[skip] {description}: {output_marker} already exists (use --force to re-run)")
        return True

    print(f"[run]  {description}")
    print("       " + " ".join(str(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[fail] {description}: {e}")
        if allow_failure:
            print("       continuing anyway (this stage is not required for the rest of the pipeline)")
            return False
        sys.exit(1)
    return output_marker.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Run the full LiDAR pipeline: read -> fit ground -> cluster -> track -> georeference results."
    )
    parser.add_argument("pcap", help="Input .pcap file from the LiDAR sensor")
    parser.add_argument("--output-dir", default="outputs", help="Directory for all generated files. Default outputs/.")
    parser.add_argument("--lat", type=float, default=None, help="Sensor latitude (required for the georeference/map stage)")
    parser.add_argument("--lon", type=float, default=None, help="Sensor longitude (required for the georeference/map stage)")
    parser.add_argument("--heading", type=float, default=None, help="Sensor heading, degrees (required for the georeference/map stage)")
    parser.add_argument("--heading-offset", type=float, default=0.0, help="Correction added to --heading. Default 0.")
    parser.add_argument("--ground-threshold", type=float, default=0.15,
                         help="Max distance (m) from the ground plane to be considered ground. Default 0.15.")
    parser.add_argument("--ransac-iterations", type=int, default=200,
                         help="Number of RANSAC iterations for the one-time ground plane fit. Default 200.")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                         help="Voxel downsampling size (m), used when fitting the ground plane and by the "
                              "cluster/track stages. Default 0.05.")
    parser.add_argument("--method", choices=["dbscan", "hdbscan", "voxel-cc", "euclidean"], default="dbscan",
                         help="Clustering algorithm for both the cluster and track stages. Default dbscan.")
    parser.add_argument("--eps", type=float, default=0.5, help="Clustering radius/voxel size (m). Default 0.5.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum points per cluster. Default 10.")
    parser.add_argument("--frame", type=int, default=None,
                         help="If set, the cluster stage only looks at this single rotation instead of the whole file.")
    parser.add_argument("--frame-start", type=int, default=None, help="First frame for the track stage. Default: 0.")
    parser.add_argument("--frame-end", type=int, default=None, help="Last frame (inclusive) for the track stage. Default: last in file.")
    parser.add_argument("--skip-map", action="store_true", help="Skip the final georeference/OSM map stage.")
    parser.add_argument("--force", action="store_true", help="Re-run every stage even if its output already exists.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    point_cloud_path = output_dir / "point_cloud.ply"
    ground_plane_json = output_dir / "ground_plane.json"
    detected_objects_csv = output_dir / "detected_objects.csv"
    tracks_summary_csv = output_dir / "tracks_summary.csv"
    clusters_map_png = output_dir / "clusters_map.png"

    # 1. read
    run_stage(
        "read lidar data (pcap -> local point cloud)",
        point_cloud_path,
        [sys.executable, str(SRC_DIR / "read_lidar_data.py"), args.pcap, str(point_cloud_path)],
        force=args.force,
    )

    # 2. fit ground plane once, shared by both the cluster and track stages below
    run_stage(
        "fit ground plane (RANSAC, once)",
        ground_plane_json,
        [
            sys.executable, str(SRC_DIR / "fit_ground_plane.py"), str(point_cloud_path),
            "--ground-threshold", str(args.ground_threshold), "--ransac-iterations", str(args.ransac_iterations),
            "--voxel-size", str(args.voxel_size), "--output", str(ground_plane_json),
        ],
        force=args.force,
    )

    # 3. cluster (one snapshot over the local cloud)
    cluster_cmd = [
        sys.executable, str(SRC_DIR / "cluster_objects.py"), str(point_cloud_path),
        "--ground-plane", str(ground_plane_json), "--ground-threshold", str(args.ground_threshold),
        "--voxel-size", str(args.voxel_size),
        "--method", args.method, "--eps", str(args.eps), "--min-samples", str(args.min_samples),
        "--output-prefix", f"{output_dir}/",
    ]
    if args.frame is not None:
        cluster_cmd += ["--frame", str(args.frame)]
    run_stage("cluster objects (local point cloud -> detected objects)", detected_objects_csv, cluster_cmd, force=args.force)

    # 4. track (across all frames / a chosen frame range)
    track_cmd = [
        sys.executable, str(SRC_DIR / "track_objects.py"), str(point_cloud_path),
        "--ground-plane", str(ground_plane_json), "--ground-threshold", str(args.ground_threshold),
        "--voxel-size", str(args.voxel_size),
        "--method", args.method, "--eps", str(args.eps), "--min-samples", str(args.min_samples),
        "--output-prefix", f"{output_dir}/",
    ]
    if args.frame_start is not None:
        track_cmd += ["--frame-start", str(args.frame_start)]
    if args.frame_end is not None:
        track_cmd += ["--frame-end", str(args.frame_end)]
    run_stage("track objects (local point cloud -> tracks across frames)", tracks_summary_csv, track_cmd, force=args.force)

    # 5. georeference results (project detected objects onto a real-world map)
    if args.skip_map:
        print("[skip] georeference results: --skip-map was passed")
    elif args.lat is None or args.lon is None or args.heading is None:
        print("[skip] georeference results: --lat/--lon/--heading not all given")
    else:
        map_cmd = [
            sys.executable, str(SRC_DIR / "plot_clusters_on_map.py"), str(detected_objects_csv),
            "--lat", str(args.lat), "--lon", str(args.lon), "--heading", str(args.heading),
            "--heading-offset", str(args.heading_offset),
            "--point-cloud", str(point_cloud_path),
            "--output", str(clusters_map_png),
        ]
        # needs internet access to fetch OSM tiles -- don't let that fail the whole pipeline
        run_stage("georeference results (detected objects -> OSM map)", clusters_map_png, map_cmd,
                  force=args.force, allow_failure=True)

    print("\nDone. Outputs in", output_dir)


if __name__ == "__main__":
    main()
