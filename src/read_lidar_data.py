"""
read_lidar_pcap.py

Reads a .pcap file containing data from a Velodyne VLP-16 LiDAR sensor,
decodes the UDP packets, and produces a 3D point cloud (x, y, z, intensity).

It also auto-detects the sensor model and return mode from the packet's
factory bytes.

Usage:
    python read_lidar_pcap.py data.pcap output.csv

Dependencies:
    pip install dpkt numpy
"""

import sys
import time
import struct
from datetime import datetime, timezone
import numpy as np
import dpkt

from lidar_io import save_point_cloud_ply, save_point_cloud_csv

# ---------------------------------------------------------------------------
# Velodyne VLP-16 protocol constants
# ---------------------------------------------------------------------------

DATA_PORT = 2368          # standard UDP port the sensor sends data on
BLOCKS_PER_PACKET = 12
CHANNELS_PER_BLOCK = 32   # 16 lasers x 2 firing sequences per block
BYTES_PER_BLOCK = 100
BLOCK_FLAG = 0xEEFF
DISTANCE_RESOLUTION_M = 0.002   # each distance unit = 2 mm

# Vertical angles (in degrees) of the 16 VLP-16 lasers, in the order they
# are actually fired within each firing sequence.
# (Standard calibration table provided by the manufacturer)
VERTICAL_ANGLES = [
    -15, 1, -13, 3, -11, 5, -9, 7,
    -7, 9, -5, 11, -3, 13, -1, 15,
]

# Product ID -> sensor name (last factory byte of the packet).
# NOTE: some values (e.g. VLS-128) vary between firmware/manual revisions.
PRODUCT_ID = {
    0x21: "HDL-32E",
    0x22: "VLP-16 / Puck LITE",
    0x24: "Puck Hi-Res",
    0x28: "VLP-32C",
    0x31: "Velarray",
    0x63: "VLS-128 / Alpha Prime (some revisions)",
    0xA1: "VLS-128 / Alpha Prime (other revisions)",
}

RETURN_MODE = {
    0x37: "Strongest Return",
    0x38: "Last Return",
    0x39: "Dual Return",
}


def read_udp_packets(pcap_path, port=DATA_PORT):
    """
    Yields the payload (bytes) of every UDP packet addressed to the LiDAR's
    data port, in the order they were captured in the pcap file.
    """
    with open(pcap_path, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for timestamp, buf in reader:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
                if not isinstance(ip, dpkt.ip.IP):
                    continue
                udp = ip.data
                if not isinstance(udp, dpkt.udp.UDP):
                    continue
                if udp.dport != port:
                    continue
                yield timestamp, udp.data
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                # truncated or malformed packet: skip it
                continue


def decode_packet(payload):
    """
    Decodes a single Velodyne data packet (1206 bytes) and returns a list
    of raw measurements: (azimuth_deg, laser_id, distance_m, intensity)
    """
    if len(payload) != 1206:
        return []

    measurements = []

    for block_index in range(BLOCKS_PER_PACKET):
        start = block_index * BYTES_PER_BLOCK
        block = payload[start:start + BYTES_PER_BLOCK]

        flag, azimuth_raw = struct.unpack_from("<HH", block, 0)
        if flag != BLOCK_FLAG:
            # invalid/unrecognized block: skip it
            continue

        azimuth_deg = azimuth_raw / 100.0  # hundredths of a degree -> degrees

        # 32 channels of 3 bytes each, starting at offset 4
        channels_offset = 4
        for channel_index in range(CHANNELS_PER_BLOCK):
            off = channels_offset + channel_index * 3
            distance_raw, intensity = struct.unpack_from("<HB", block, off)

            if distance_raw == 0:
                # no valid return (beam went out into empty space)
                continue

            distance_m = distance_raw * DISTANCE_RESOLUTION_M
            laser_id = channel_index % 16  # 16 lasers, repeated over 2 sequences

            measurements.append((azimuth_deg, laser_id, distance_m, intensity))

    return measurements


def decode_packet_vectorized(payload):
    """
    Decodes a single Velodyne data packet (1206 bytes) using NumPy vector
    operations instead of a per-channel Python loop.

    Returns four 1D numpy arrays (all points from all blocks in the packet,
    invalid returns already filtered out): azimuth_deg, laser_id, distance_m, intensity
    """
    if len(payload) != 1206:
        empty = np.empty(0)
        return empty, empty, empty, empty

    raw = np.frombuffer(payload, dtype=np.uint8)
    blocks = raw[: BLOCKS_PER_PACKET * BYTES_PER_BLOCK].reshape(BLOCKS_PER_PACKET, BYTES_PER_BLOCK)

    # --- header (first 4 bytes of each block): flag (u16) + azimuth (u16) ---
    flags = blocks[:, 0].astype(np.uint16) | (blocks[:, 1].astype(np.uint16) << 8)
    azimuth_raw = blocks[:, 2].astype(np.uint16) | (blocks[:, 3].astype(np.uint16) << 8)
    valid_blocks = flags == BLOCK_FLAG  # (12,) boolean

    # --- channel data (96 bytes = 32 channels x 3 bytes), per block ---
    channel_bytes = blocks[:, 4:100].reshape(BLOCKS_PER_PACKET, CHANNELS_PER_BLOCK, 3)
    distance_raw = channel_bytes[:, :, 0].astype(np.uint16) | (
        channel_bytes[:, :, 1].astype(np.uint16) << 8
    )  # (12, 32)
    intensity = channel_bytes[:, :, 2]  # (12, 32)

    # broadcast azimuth and block validity to the (12, 32) channel grid
    azimuth_deg_grid = np.repeat((azimuth_raw / 100.0)[:, None], CHANNELS_PER_BLOCK, axis=1)
    valid_grid = np.repeat(valid_blocks[:, None], CHANNELS_PER_BLOCK, axis=1)
    laser_id_grid = np.tile(np.arange(CHANNELS_PER_BLOCK) % 16, (BLOCKS_PER_PACKET, 1))

    # a return is valid if its block flag is correct AND distance != 0
    valid = valid_grid & (distance_raw != 0)

    azimuth_deg = azimuth_deg_grid[valid]
    laser_id = laser_id_grid[valid]
    distance_m = distance_raw[valid].astype(np.float64) * DISTANCE_RESOLUTION_M
    intensity = intensity[valid]

    return azimuth_deg, laser_id, distance_m, intensity


def spherical_to_cartesian_vectorized(azimuth_deg, laser_id, distance_m):
    """
    Vectorized version of spherical_to_cartesian: converts whole arrays of
    measurements into arrays of x, y, z coordinates (meters) at once.
    """
    vertical_angles = np.asarray(VERTICAL_ANGLES)
    vertical_angle_deg = vertical_angles[laser_id]

    az = np.radians(azimuth_deg)
    el = np.radians(vertical_angle_deg)

    horizontal_radius = distance_m * np.cos(el)
    x = horizontal_radius * np.sin(az)
    y = horizontal_radius * np.cos(az)
    z = distance_m * np.sin(el)
    return x, y, z


def detect_sensor(pcap_path):
    """
    Reads the first valid (1206-byte) data packet and extracts the
    return mode and product ID from its trailing factory bytes.
    """
    for ts, payload in read_udp_packets(pcap_path):
        if len(payload) != 1206:
            continue
        return_mode_byte, product_id_byte = struct.unpack_from("<BB", payload, 1204)
        return_mode = RETURN_MODE.get(return_mode_byte, f"unknown (0x{return_mode_byte:02X})")
        sensor = PRODUCT_ID.get(product_id_byte, f"unknown (0x{product_id_byte:02X})")
        return sensor, return_mode
    return None, None


def pcap_to_point_cloud(pcap_path, packet_limit=None, progress_every=20000):
    """
    Reads the entire pcap file and returns:
      - an Nx5 numpy array: (x, y, z, intensity, timestamp_offset_s)
      - the absolute start timestamp (Unix epoch seconds, float) that
        timestamp_offset_s is relative to

    `timestamp_offset_s` is the pcap capture timestamp of the packet each
    point came from (all ~384 points in a packet share the same value),
    expressed as seconds elapsed since the first packet in the file.
    This is precise enough for a stationary-sensor capture; it is NOT
    per-point timing and should not be used for motion compensation.

    Uses vectorized per-packet decoding for speed, and prints progress
    every `progress_every` packets so you can tell it's not stuck.
    """
    chunks = []  # list of (N,5) arrays, one per packet, concatenated at the end
    start_ts = None

    for i, (ts, payload) in enumerate(read_udp_packets(pcap_path)):
        if packet_limit is not None and i >= packet_limit:
            break

        if start_ts is None:
            start_ts = ts

        if progress_every and i % progress_every == 0 and i > 0:
            print(f"  ... {i} packets processed")

        azimuth_deg, laser_id, distance_m, intensity = decode_packet_vectorized(payload)
        if len(azimuth_deg) == 0:
            continue

        x, y, z = spherical_to_cartesian_vectorized(azimuth_deg, laser_id, distance_m)
        timestamp_offset = np.full(len(x), ts - start_ts, dtype=np.float32)
        chunk = np.column_stack([x, y, z, intensity, timestamp_offset]).astype(np.float32)
        chunks.append(chunk)

    if not chunks:
        return np.empty((0, 5), dtype=np.float32), start_ts
    return np.concatenate(chunks, axis=0), start_ts


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_lidar_pcap.py data.pcap [output_path] [--csv]")
        print("  Default output format is binary PLY (fast, opens in CloudCompare/MeshLab/Open3D).")
        print("  Pass --csv to write a text CSV instead (much slower for large files).")
        sys.exit(1)

    pcap_path = sys.argv[1]
    use_csv = "--csv" in sys.argv
    positional_args = [a for a in sys.argv[2:] if not a.startswith("--")]
    default_output = "point_cloud.csv" if use_csv else "point_cloud.ply"
    output_path = positional_args[0] if positional_args else default_output

    field_names = ("x", "y", "z", "intensity", "timestamp")

    print(f"Reading {pcap_path} ...")

    t0 = time.time()
    sensor, return_mode = detect_sensor(pcap_path)
    print(f"Detected sensor: {sensor}")
    print(f"Return mode: {return_mode}")

    t1 = time.time()
    cloud, start_ts = pcap_to_point_cloud(pcap_path)
    print(f"Extracted {len(cloud)} points in {time.time() - t1:.1f}s.")
    if start_ts is not None:
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        print(f"Capture start (UTC): {start_dt.isoformat()}  (epoch {start_ts:.6f})")
        print("The 'timestamp' column in the output is seconds elapsed since this moment.")

    t2 = time.time()
    if use_csv:
        save_point_cloud_csv(cloud, output_path, field_names=field_names)
    else:
        save_point_cloud_ply(cloud, output_path, field_names=field_names)
    print(f"Point cloud saved to {output_path} in {time.time() - t2:.1f}s.")
    print(f"Total time: {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()