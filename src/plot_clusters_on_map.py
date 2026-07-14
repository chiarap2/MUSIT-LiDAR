"""
plot_clusters_on_map.py

Takes the cluster summary CSV produced by cluster_objects.py (in local,
sensor-centered coordinates) and the sensor's GPS position/heading, converts
each cluster's centroid and bounding box into real-world UTM coordinates,
and plots them over an OpenStreetMap basemap.

Bounding boxes are axis-aligned in the local sensor frame, but once rotated
by the sensor heading they become rotated rectangles in real-world space --
this script draws them as such (not axis-aligned boxes).

Usage:
    python plot_clusters_on_map.py detected_objects.csv \
        --lat 37.430785059983215 --lon 24.941538092360716 \
        --heading 280 --heading-offset 0 \
        --output clusters_map.png

    # optionally show the full local point cloud too, lightly, for context
    python plot_clusters_on_map.py detected_objects.csv \
        --lat 37.430785059983215 --lon 24.941538092360716 --heading 280 \
        --point-cloud point_cloud.ply --output clusters_map.png

Dependencies:
    pip install numpy matplotlib contextily pyproj

NOTE: downloading the OSM basemap tiles needs normal internet access at
runtime. If you're running this in a sandboxed environment without access
to tile.openstreetmap.org (or your chosen provider's domain), the basemap
step will fail -- run it on your own machine.
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
from matplotlib.patches import Polygon
import contextily as ctx
from pyproj import Transformer

from lidar_io import read_ply_xyzi
from georeference_point_cloud import utm_epsg_for, rotate_to_utm


def read_clusters_csv(path):
    """Reads the cluster summary CSV from cluster_objects.py into a list of dicts (all values as float, except cluster_id/n_points as int)."""
    clusters = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clusters.append({
                "cluster_id": int(row["cluster_id"]),
                "n_points": int(row["n_points"]),
                "centroid_x": float(row["centroid_x"]),
                "centroid_y": float(row["centroid_y"]),
                "min_x": float(row["min_x"]), "max_x": float(row["max_x"]),
                "min_y": float(row["min_y"]), "max_y": float(row["max_y"]),
            })
    return clusters



def main():
    parser = argparse.ArgumentParser(description="Georeference detected clusters and plot them over OpenStreetMap.")
    parser.add_argument("input", help="Cluster summary CSV from cluster_objects.py")
    parser.add_argument("--lat", type=float, required=True, help="Sensor latitude (decimal degrees)")
    parser.add_argument("--lon", type=float, required=True, help="Sensor longitude (decimal degrees)")
    parser.add_argument("--heading", type=float, required=True, help="Sensor heading, degrees, 0=North, 90=East, clockwise")
    parser.add_argument("--heading-offset", type=float, default=0.0, help="Correction added to --heading (e.g. -90)")
    parser.add_argument("--point-cloud", default=None,
                         help="Optional: local point_cloud.ply to show lightly in the background for context")
    parser.add_argument("--output", default="clusters_map.png", help="Output PNG path")
    parser.add_argument("--basemap-style", default="OpenStreetMap.Mapnik",
                         help="Contextily provider path, e.g. 'OpenStreetMap.Mapnik' or 'CartoDB.Positron'")
    parser.add_argument("--max-context-points", type=int, default=100_000,
                         help="Max number of --point-cloud points to draw (randomly subsampled)")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    clusters = read_clusters_csv(args.input)
    print(f"Loaded {len(clusters)} clusters.")

    epsg = utm_epsg_for(args.lon, args.lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    sensor_easting, sensor_northing = transformer.transform(args.lon, args.lat)
    heading_used = (args.heading + args.heading_offset) % 360.0
    print(f"Sensor UTM position: ({sensor_easting:.2f}, {sensor_northing:.2f}), EPSG:{epsg}")
    print(f"Using heading = {heading_used} deg")

    fig, ax = plt.subplots(figsize=(12, 12))

    # optional background context: the full local point cloud, georeferenced too
    if args.point_cloud:
        print(f"Reading context point cloud {args.point_cloud} ...")
        context_cloud = read_ply_xyzi(args.point_cloud)
        if len(context_cloud) > args.max_context_points:
            idx = np.random.choice(len(context_cloud), size=args.max_context_points, replace=False)
            context_cloud = context_cloud[idx]
        ce, cn = rotate_to_utm(context_cloud[:, 0], context_cloud[:, 1], sensor_easting, sensor_northing, heading_used)
        ax.scatter(ce, cn, s=0.5, c="steelblue", alpha=0.3, zorder=2, label="scan points")

    # sensor position marker
    ax.scatter([sensor_easting], [sensor_northing], marker="*", s=300, c="red", edgecolor="black",
               zorder=4, label="sensor")

    # clusters: bounding box (rotated) + centroid + label
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(clusters):
        color = cmap(i % 20)

        # 4 corners of the local axis-aligned box, in order, so the rotated result is still a proper quadrilateral
        corners_x = np.array([c["min_x"], c["max_x"], c["max_x"], c["min_x"]])
        corners_y = np.array([c["min_y"], c["min_y"], c["max_y"], c["max_y"]])
        corners_e, corners_n = rotate_to_utm(corners_x, corners_y, sensor_easting, sensor_northing, heading_used)

        polygon = Polygon(np.column_stack([corners_e, corners_n]), fill=False,
                           edgecolor=color, linewidth=2, zorder=3)
        ax.add_patch(polygon)

        centroid_e, centroid_n = rotate_to_utm(c["centroid_x"], c["centroid_y"], sensor_easting, sensor_northing, heading_used)
        ax.scatter([centroid_e], [centroid_n], s=40, color=color, zorder=4)
        ax.annotate(f"#{c['cluster_id']} ({c['n_points']}pts)", (centroid_e, centroid_n),
                    fontsize=9, color=color, fontweight="bold", zorder=5,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_aspect("equal")

    # resolve provider path (e.g. "OpenStreetMap.Mapnik")
    provider = ctx.providers
    for part in args.basemap_style.split("."):
        provider = provider[part]

    print(f"Downloading basemap tiles (EPSG:{epsg}, provider: {args.basemap_style}) ...")
    ctx.add_basemap(ax, crs=f"EPSG:{epsg}", source=provider, zorder=1)

    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")
    ax.set_title(f"Detected objects ({len(clusters)}) over OpenStreetMap")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()