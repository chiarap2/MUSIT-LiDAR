"""
plot_point_cloud.py

Generates 2D plots from a point cloud PLY file (as produced by
read_lidar_pcap.py or georeference_point_cloud.py):
  - top-down view (bird's-eye), colored by intensity
  - side view (elevation profile), colored by elevation

Works on both local point clouds (x, y, z in meters relative to the sensor)
and georeferenced ones (x, y in UTM meters, z relative elevation) — it just
labels the axes accordingly.

Usage:
    python plot_point_cloud.py point_cloud.ply --output-prefix local_
    python plot_point_cloud.py geo_v1.ply --output-prefix geo_ --georeferenced

Dependencies:
    pip install numpy matplotlib
"""

import sys
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


def subsample(cloud, max_points):
    """Randomly subsamples the cloud to at most max_points, for fast plotting."""
    if len(cloud) <= max_points:
        return cloud
    idx = np.random.choice(len(cloud), size=max_points, replace=False)
    return cloud[idx]


def plot_top_down(cloud, output_path, georeferenced=False, point_size=0.5):
    """Bird's-eye view: X vs Y, colored by intensity."""
    x, y, intensity = cloud[:, 0], cloud[:, 1], cloud[:, 3]

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(x, y, c=intensity, s=point_size, cmap="viridis")
    ax.set_aspect("equal")

    if georeferenced:
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")
        ax.set_title("Point cloud - top-down view (georeferenced, UTM)")
    else:
        ax.set_xlabel("Local X (m, sensor-centered)")
        ax.set_ylabel("Local Y (m, sensor-centered)")
        ax.set_title("Point cloud - top-down view (sensor-local frame)")

    plt.colorbar(scatter, ax=ax, label="Intensity", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_side_view(cloud, output_path, point_size=0.5):
    """Side view: horizontal distance from sensor/origin vs Z (elevation), colored by elevation."""
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    horizontal_distance = np.sqrt(x**2 + y**2)

    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(horizontal_distance, z, c=z, s=point_size, cmap="terrain")
    ax.set_xlabel("Horizontal distance from origin (m)")
    ax.set_ylabel("Elevation Z (m)")
    ax.set_title("Point cloud - side view (elevation profile)")
    plt.colorbar(scatter, ax=ax, label="Elevation (m)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_consecutive_timestamps(cloud, timestamp_col_index, output_path,
                                 n_timestamps=3, start_index=0,
                                 georeferenced=False, point_size=3.0):
    """
    Selects `n_timestamps` consecutive unique timestamp values (starting
    at `start_index` into the sorted list of unique timestamps) and plots
    each one as its own top-down subplot, side by side, so you can visually
    compare what each individual packet captured.

    Note: since each unique timestamp corresponds to a single UDP packet
    (~384 points, a thin ~5-degree slice of the sensor's 360-degree sweep),
    these subplots will look like narrow arcs, not full scenes.
    """
    timestamps = cloud[:, timestamp_col_index]
    unique_ts = np.unique(timestamps)

    if start_index >= len(unique_ts):
        raise ValueError(f"start_index {start_index} is out of range: only {len(unique_ts)} unique timestamps found")

    selected_ts = unique_ts[start_index:start_index + n_timestamps]
    print(f"Selected {len(selected_ts)} timestamps (out of {len(unique_ts)} unique): {selected_ts}")

    # shared axis limits and color scale across subplots, for fair comparison
    x_all, y_all, intensity_all = cloud[:, 0], cloud[:, 1], cloud[:, 3]
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    vmin, vmax = intensity_all.min(), intensity_all.max()

    fig, axes = plt.subplots(1, len(selected_ts), figsize=(6 * len(selected_ts), 6), squeeze=False)
    axes = axes[0]

    for ax, ts_value in zip(axes, selected_ts):
        mask = timestamps == ts_value
        x, y, intensity = cloud[mask, 0], cloud[mask, 1], cloud[mask, 3]

        scatter = ax.scatter(x, y, c=intensity, s=point_size, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(f"t = {ts_value:.6f}s\n({mask.sum()} points)")
        ax.set_xlabel("UTM Easting (m)" if georeferenced else "Local X (m)")
        ax.set_ylabel("UTM Northing (m)" if georeferenced else "Local Y (m)")

    fig.colorbar(scatter, ax=axes, label="Intensity", shrink=0.7)
    fig.suptitle(f"{len(selected_ts)} consecutive packet timestamps, side by side")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_consecutive_frames(cloud, frame_ids, timestamp_col_index, output_path,
                             n_frames=3, start_index=0,
                             georeferenced=False, point_size=1.0):
    """
    Selects `n_frames` consecutive full-rotation frames (starting at
    `start_index` into the sorted list of unique frame indices) and plots
    each one as its own top-down subplot, side by side. Each frame is a
    full 360-degree sweep of the sensor, so for a stationary sensor these
    should all look like the same scene — good for spotting noise,
    inconsistencies, or moving objects across rotations.

    `frame_ids` is a 1D array (same length as cloud) giving each point's
    frame number — either read from a 'frame_index' column, or derived
    on the fly with derive_frame_index().
    """
    unique_frames = np.unique(frame_ids)

    if start_index >= len(unique_frames):
        raise ValueError(f"start_index {start_index} is out of range: only {len(unique_frames)} frames found")

    selected_frames = unique_frames[start_index:start_index + n_frames]
    print(f"Selected {len(selected_frames)} frames (out of {len(unique_frames)} total): {selected_frames.astype(int)}")

    # shared axis limits and color scale across subplots, for fair comparison
    x_all, y_all, intensity_all = cloud[:, 0], cloud[:, 1], cloud[:, 3]
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    vmin, vmax = intensity_all.min(), intensity_all.max()

    fig, axes = plt.subplots(1, len(selected_frames), figsize=(6 * len(selected_frames), 6), squeeze=False)
    axes = axes[0]

    for ax, frame_id in zip(axes, selected_frames):
        mask = frame_ids == frame_id
        x, y, intensity = cloud[mask, 0], cloud[mask, 1], cloud[mask, 3]

        scatter = ax.scatter(x, y, c=intensity, s=point_size, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")

        if timestamp_col_index is not None:
            t_start, t_end = cloud[mask, timestamp_col_index].min(), cloud[mask, timestamp_col_index].max()
            ax.set_title(f"Frame {int(frame_id)}\nt = {t_start:.3f}-{t_end:.3f}s  ({mask.sum()} points)")
        else:
            ax.set_title(f"Frame {int(frame_id)}\n({mask.sum()} points)")

        ax.set_xlabel("UTM Easting (m)" if georeferenced else "Local X (m)")
        ax.set_ylabel("UTM Northing (m)" if georeferenced else "Local Y (m)")

    fig.colorbar(scatter, ax=axes, label="Intensity", shrink=0.7)
    fig.suptitle(f"{len(selected_frames)} consecutive full rotations, side by side")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a point cloud PLY file (top-down and side views).")
    parser.add_argument("input", help="Input PLY point cloud file")
    parser.add_argument("--max-points", type=int, default=300_000,
                         help="Max number of points to plot (randomly subsampled). Default 300000.")
    parser.add_argument("--output-prefix", default="", help="Prefix for output PNG filenames")
    parser.add_argument("--georeferenced", action="store_true",
                         help="Label axes as UTM easting/northing instead of local sensor-frame x/y")
    parser.add_argument("--point-size", type=float, default=0.5, help="Marker size in the scatter plots")
    parser.add_argument("--timestamps", type=int, default=0,
                         help="If > 0, plot this many consecutive unique PACKET timestamps side by side "
                              "(thin ~5-degree slices — requires a 'timestamp' column). "
                              "Prefer --frames for comparing full rotations instead.")
    parser.add_argument("--start-timestamp-index", type=int, default=0,
                         help="Index into the sorted list of unique timestamps to start from (default 0).")
    parser.add_argument("--frames", type=int, default=0,
                         help="If > 0, plot this many consecutive full 360-degree rotations side by side "
                              "(requires a 'frame_index' column). This is usually more useful than "
                              "--timestamps for comparing what the sensor saw over time.")
    parser.add_argument("--start-frame-index", type=int, default=0,
                         help="Index into the sorted list of unique frames to start from (default 0).")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    full_cloud, field_names = read_ply_full(args.input)
    print(f"Loaded {len(full_cloud)} points with fields: {field_names}")

    if args.frames > 0:
        if args.georeferenced:
            print("Warning: deriving frame_index from x/y on a georeferenced file is not reliable "
                  "(x/y are UTM coordinates, not sensor-local). Run --frames on the local point "
                  "cloud instead, before georeferencing.")
        frame_ids = full_cloud[:, 5] if "frame_index" in field_names else derive_frame_index(full_cloud)
        timestamp_col_index = field_names.index("timestamp") if "timestamp" in field_names else None

        frames_path = f"{args.output_prefix}frames_{args.start_frame_index}_{args.start_frame_index + args.frames - 1}.png"
        plot_consecutive_frames(
            full_cloud, frame_ids, timestamp_col_index, frames_path,
            n_frames=args.frames, start_index=args.start_frame_index,
            georeferenced=args.georeferenced, point_size=max(args.point_size, 1.0),
        )
        print(f"Saved {frames_path}")
        return

    if args.timestamps > 0:
        if "timestamp" not in field_names:
            print("Error: --timestamps requires the input PLY to have a 'timestamp' field.")
            print(f"This file only has: {field_names}")
            sys.exit(1)
        timestamp_col_index = field_names.index("timestamp")

        multi_ts_path = f"{args.output_prefix}timestamps_{args.start_timestamp_index}_{args.start_timestamp_index + args.timestamps - 1}.png"
        plot_consecutive_timestamps(
            full_cloud, timestamp_col_index, multi_ts_path,
            n_timestamps=args.timestamps, start_index=args.start_timestamp_index,
            georeferenced=args.georeferenced, point_size=max(args.point_size, 2.0),
        )
        print(f"Saved {multi_ts_path}")
        return

    plot_cloud = subsample(full_cloud, args.max_points)
    if len(plot_cloud) < len(full_cloud):
        print(f"Subsampled to {len(plot_cloud)} points for plotting.")

    top_down_path = f"{args.output_prefix}top_down.png"
    side_view_path = f"{args.output_prefix}side_view.png"

    plot_top_down(plot_cloud, top_down_path, georeferenced=args.georeferenced, point_size=args.point_size)
    print(f"Saved {top_down_path}")

    plot_side_view(plot_cloud, side_view_path, point_size=args.point_size)
    print(f"Saved {side_view_path}")


if __name__ == "__main__":
    main()