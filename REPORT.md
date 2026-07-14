# Methodology report

How each pipeline stage works, why it's built that way, and what its
limitations are. See [README.md](README.md) for usage.

## 1. Reading the sensor data (`src/read_lidar_data.py`)

The sensor is a Velodyne VLP-16 (16 lasers at fixed vertical angles from
-15° to +15°), captured as raw UDP packets in a `.pcap` file. Each packet
carries 12 firing blocks; each block has an azimuth angle and 32 channel
returns (16 lasers × 2 firing sequences). The sensor/return-mode are
auto-detected from the packet's factory bytes.

Decoding is vectorized per packet (NumPy over the raw byte buffer, not a
Python loop per channel) — this matters because a single capture is often
tens of millions of points. Each valid return (distance, azimuth, laser
elevation) is converted to Cartesian `(x, y, z)` in the sensor's **local**
frame via standard spherical-to-Cartesian conversion. Output is a binary PLY
(`x, y, z, intensity, timestamp`) — far faster to read/write than text CSV at
this scale.

`timestamp` is the pcap capture time of the packet each point came from,
relative to the first packet — precise enough to derive per-rotation frames
(below) for a stationary sensor, but **not** per-point timing, so it can't be
used for motion compensation of a moving sensor.

## 2. Local vs. georeferenced coordinates, and multi-rotation frames

Everything through clustering/tracking operates in the sensor's **local**
frame: `x = r·sin(azimuth)`, `y = r·cos(azimuth)`. This lets a "frame index"
(one per full 360° rotation) be derived on the fly from just `x, y`, by
unwrapping azimuth and detecting the wraparounds (`lidar_io.derive_frame_index`)
— no need for a dedicated per-point frame column, but it only works before
georeferencing, since afterward `atan2(x, y)` no longer reflects scan angle.

Georeferencing (`src/georeference_point_cloud.py`) rotates local `(x, y)` by
the sensor's heading and translates by its UTM position; `z` is left as
relative elevation. This is a one-way, informational transform for mapping —
UTM eastings/northings are ~300000+ in magnitude, and storing that as
float32 leaves only ~0.5m of precision, enough to meaningfully distort a
1-2m object's bounding box. **Clustering/tracking must run on the local
cloud** for this reason; only the final detected-object centroids/boxes get
georeferenced, in `src/plot_clusters_on_map.py`.

## 3. Ground removal (`src/ground.py`, fit stage: `src/fit_ground_plane.py`)

A single plane `ax + by + cz + d = 0` is fit via RANSAC: repeatedly sample 3
points, form the plane through them, count inliers within a distance
threshold, keep the best. Points within `--ground-threshold` of the best
plane are "ground" and excluded from clustering. Simple and dependency-free;
assumes one dominant near-flat ground surface, which held for this capture
site.

For multi-rotation captures, the same physical ground gets resampled every
rotation at nearly identical points, which can blow up RANSAC/clustering
time and memory — `voxel_downsample` (average all columns per grid cell)
is applied first specifically to counteract this.

RANSAC is the expensive part of the whole pipeline (hundreds of iterations,
each scoring every point), and the ground plane doesn't change between
clustering method comparisons or between frames of a stationary-sensor
capture — so it's fit **exactly once** and reused everywhere else, rather
than being repeated per run:
- `src/fit_ground_plane.py` runs the fit and saves `(a, b, c, d)` plus fit
  metadata to a small `ground_plane.json`.
- `src/cluster_objects.py` and `src/track_objects.py` both accept
  `--ground-plane ground_plane.json` to load it directly (`split_by_plane`
  is the only per-run cost afterwards — one dot product + threshold per
  point, not a re-fit) instead of calling `fit_ground_plane_ransac` again.
  Without `--ground-plane`, both scripts still fall back to fitting inline,
  for standalone convenience.
- `main.py` always fits it once (stage 2) and passes the same
  `ground_plane.json` to both the cluster and track stages.

## 4. Object detection: four clustering methods (`src/cluster_objects.py`)

All four take the same `xyz` array and return the same label convention
(`-1` = noise), so they're drop-in comparable via `--method`:

| Method | Idea | Tradeoff |
|---|---|---|
| `dbscan` (default) | Core points (≥`min_samples` neighbors within `eps`) chain into clusters | Well-understood, but a single global `eps` struggles when near/far objects have very different point densities |
| `hdbscan` | Hierarchical density clustering, extracts the most stable clusters across a range of densities | No `eps` to tune; handles mixed near/far density better; costs a bit more compute |
| `voxel-cc` | Snap points to a voxel grid, flood-fill (union-find) 26-connected occupied voxels | Much faster on dense clouds (dict lookup vs. tree query); clusters are grid-aligned, so nearby distinct objects can merge more easily than with a radius search |
| `euclidean` | Classic PCL-style: KD-tree region growing by radius, filtered by min cluster size at the end | Very similar to `dbscan`, but no core-point requirement — differs slightly at sparse cluster edges/tails |

Ground-removed points get a bounding box, centroid, point count, and mean
intensity per cluster (`summarize_clusters`), saved to CSV and plotted
top-down with bounding boxes.

## 5. Tracking objects across frames (`src/track_objects.py`)

Each rotation is clustered independently (method above), then this frame's
cluster centroids are matched against the previous frame's active tracks:
all (cluster, track) pairs within `--max-match-distance` are sorted by
distance and assigned greedily, closest first, one-to-one. This is a
simplification of the optimal (Hungarian) assignment — chosen because it's
simple and fast, and works well as long as objects don't move by more than
`--max-match-distance` between rotations or pass close by one another
(a harder ambiguous case that optimal assignment or motion prediction would
handle better, see Limitations).

A track survives up to `--max-missed-frames` without a match before being
retired, so one rotation occasionally dropping an object below
`--min-samples` doesn't split its track into two. Per track, aggregate
stats — total path length, net first-to-last displacement, and max
distance from the track's own mean position — classify it `static` (the
same physical spot being re-detected every rotation, e.g. a bench or pole)
vs. `moving` (`--static-threshold` on the max-distance-from-mean value).
This directly answers "is this the same object moving over time, or the
same spot being re-detected?"

## 6. Georeferencing results onto a map (`src/plot_clusters_on_map.py`, notebook)

The sensor's lat/lon is converted to UTM via `pyproj` (auto-selecting the
UTM zone from longitude), and each detected object's local centroid/box
corners are rotated by the sensor heading and translated to that UTM
position — the same `rotate_to_utm`/`utm_epsg_for` functions live in
`src/georeference_point_cloud.py` and are imported everywhere else that
needs them, so there's exactly one place defining this convention.
Axis-aligned local boxes become rotated rectangles in real-world space once
the heading rotation is applied, and are drawn as such. The interactive
notebook exists because calibrating `heading_offset` against real map
features is inherently iterative.

**Confirmed capture parameters for the included dataset**: lat
`37.430785059983215`, lon `24.941538092360716`, heading `280°`,
heading-offset `0`. These were cross-checked by transforming them through
`pyproj` (EPSG:32635) and comparing against `outputs/geo_v1.ply`'s actual
point coordinates — they land within a few tens of meters (plausible LiDAR
range) of each other.

## Known limitations

- **No per-point motion compensation.** `timestamp` is per-packet, not
  per-point — fine for a stationary sensor, not suitable for a moving one.
- **Axis-aligned bounding boxes** in the local frame, even for objects that
  are rotated relative to the sensor — a real object's true extent may be
  overestimated.
- **Greedy, not optimal, track assignment.** Two similar-looking objects
  passing close to each other could have their tracks swapped; Hungarian
  assignment (or adding motion prediction) would handle this better at the
  cost of more code and a `scipy`/`scipy.optimize` dependency already
  available if this becomes worth it.
- **Single dominant ground plane assumption.** A site with two differently
  sloped ground regions (e.g. a ramp) would need a more general ground
  model than one RANSAC plane.
- **`outputs/output.csv`** is a stale ~3GB text-CSV export predating
  `point_cloud.ply` — looks redundant, flagged in the README for the user
  to confirm and delete.
