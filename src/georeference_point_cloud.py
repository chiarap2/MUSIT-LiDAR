"""
georeference_point_cloud.py

Converts a LOCAL point cloud (x, y relative to the sensor, in meters) into
real-world UTM coordinates, given the sensor's GPS position and heading at
capture time. z (relative elevation), intensity, and any extra columns
(e.g. timestamp) pass through unchanged -- only x, y are rotated by the
sensor heading and translated to the sensor's UTM position.

This is a standalone utility for producing a GIS-viewable point cloud (e.g.
to open in CloudCompare/QGIS next to a basemap). It is intentionally NOT
part of the main pipeline's clustering/tracking steps: those need the
LOCAL point cloud, since georeferenced coordinates are huge UTM numbers
(~300000+) that lose ~0.5m of precision once stored as float32 -- enough to
corrupt small object bounding boxes. Cluster/track first on the local
cloud, then georeference just the detected objects' centroids/boxes
afterwards for mapping (see plot_clusters_on_map.py), or georeference the
whole cloud here only for visualization.

`utm_epsg_for` and `rotate_to_utm` are the canonical UTM conversion used
across this repo -- plot_clusters_on_map.py imports them from here rather
than keeping its own copy.

Usage:
    python georeference_point_cloud.py point_cloud.ply \
        --lat 37.430785059983215 --lon 24.941538092360716 \
        --heading 280 --heading-offset 0 --output geo_v1.ply

(These are the confirmed real capture parameters for this dataset: transforming
them through pyproj lands within a few tens of meters of geo_v1.ply's actual
point coordinates.)

Dependencies:
    pip install numpy pyproj
"""

import argparse
import numpy as np
from pyproj import Transformer

from lidar_io import read_ply_full, save_point_cloud_ply


def utm_epsg_for(lon, lat):
    """Returns the EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def rotate_to_utm(x_local, y_local, sensor_easting, sensor_northing, heading_used_deg):
    """
    Rotates local sensor-frame (x, y) offsets by the sensor heading and
    translates them to the sensor's UTM position. Local y-axis points along
    the sensor's azimuth-zero direction, azimuth increases clockwise (like
    a bearing).
    """
    h = np.radians(heading_used_deg)
    east_offset = x_local * np.cos(h) + y_local * np.sin(h)
    north_offset = -x_local * np.sin(h) + y_local * np.cos(h)
    return sensor_easting + east_offset, sensor_northing + north_offset


def georeference_cloud(cloud, sensor_lat, sensor_lon, heading_deg, heading_offset_deg=0.0):
    """
    Georeferences a local point cloud's x, y columns in place (on a copy).
    Returns (geo_cloud, epsg, (sensor_easting, sensor_northing), heading_used_deg).
    """
    epsg = utm_epsg_for(sensor_lon, sensor_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    sensor_easting, sensor_northing = transformer.transform(sensor_lon, sensor_lat)
    heading_used = (heading_deg + heading_offset_deg) % 360.0

    geo_cloud = cloud.copy()
    geo_cloud[:, 0], geo_cloud[:, 1] = rotate_to_utm(
        cloud[:, 0], cloud[:, 1], sensor_easting, sensor_northing, heading_used
    )
    return geo_cloud, epsg, (sensor_easting, sensor_northing), heading_used


def main():
    parser = argparse.ArgumentParser(
        description="Georeference a local LiDAR point cloud into UTM coordinates."
    )
    parser.add_argument("input", help="Input PLY point cloud file (LOCAL frame)")
    parser.add_argument("--lat", type=float, required=True, help="Sensor latitude (decimal degrees)")
    parser.add_argument("--lon", type=float, required=True, help="Sensor longitude (decimal degrees)")
    parser.add_argument("--heading", type=float, required=True, help="Sensor heading, degrees, 0=North, 90=East, clockwise")
    parser.add_argument("--heading-offset", type=float, default=0.0, help="Correction added to --heading (e.g. -90)")
    parser.add_argument("--output", default="geo_v1.ply", help="Output PLY path")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    cloud, field_names = read_ply_full(args.input)
    print(f"Loaded {len(cloud)} points with fields: {field_names}")

    geo_cloud, epsg, (sensor_easting, sensor_northing), heading_used = georeference_cloud(
        cloud, args.lat, args.lon, args.heading, args.heading_offset
    )
    print(f"Sensor UTM position: ({sensor_easting:.2f}, {sensor_northing:.2f}), EPSG:{epsg}")
    print(f"Using heading = {heading_used} deg")

    save_point_cloud_ply(geo_cloud, args.output, field_names=field_names)
    print(f"Saved georeferenced point cloud to {args.output}")


if __name__ == "__main__":
    main()
