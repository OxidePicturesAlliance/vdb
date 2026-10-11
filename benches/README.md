# Rust vs C++ OpenVDB benchmarks

`compare.rs` and `cpp/compare.cpp` are twin programs: every benchmark has the same name,
the same inputs (the random ones come from the same SplitMix64 sequence) and the same
parameters on both sides. Each benchmark also prints a result summary (voxel counts, value
sums, mesh sizes, hit counts) which `compare.sh` compares, so both sides are known to do the
same work.

```sh
benches/compare.sh                        # all benchmarks, 5 repetitions, all cores
benches/compare.sh --threads 1 --reps 3   # single-threaded
benches/compare.sh mesh                   # only benchmarks whose name contains "mesh"
cargo bench --bench compare -- [filter] [--reps N] [--threads N]   # Rust side only
```

The C++ side needs OpenVDB 10 with TBB and Imath (`apt install libopenvdb-dev libboost-dev`
on Ubuntu 24.04). The script builds it with `g++ -O3 -DNDEBUG` (override with `CXX` /
`CXXFLAGS`), builds the Rust side with the default `bench` profile, and prints a Markdown
table. Raw outputs go to `target/bench-compare/{rust,cpp}.tsv`. Times are the median of the
repetitions; setup (copying input grids, generating inputs) is not timed.

## Benchmarks

Unless noted, the input is a narrow-band level set of a sphere of radius 1 with voxel size
0.004 (250 voxels in radius) and half-width 3: 4.7 million active voxels.

| benchmark | operation |
|---|---|
| `sphere_create` | `createLevelSetSphere` |
| `grid_copy` | deep copy of the grid |
| `active_voxel_count` | `activeVoxelCount()` |
| `iterate_active` | sum of all active values with a value iterator |
| `random_insert` | 2M `setValue` calls through an accessor at random coordinates in a 512³ box, into an empty grid |
| `random_read` | 4M accessor reads at random coordinates in the sphere's bounding box |
| `coherent_read` | accessor reads of every voxel of a 256³ block, z fastest |
| `box_sample` | 2M world-space trilinear (`BoxSampler`) samples at random positions |
| `write_*` / `read_*` | write / read the grid to / from an in-memory stream with active-mask compression, plus nothing, zip or Blosc (the grid class is set to "unknown" so that zip applies) |
| `csg_union`, `csg_difference` | with a second sphere centred at (0.5, 0, 0), pruning cancelled tiles |
| `dilate_3`, `erode_3` | `dilateActiveValues` / `erodeActiveValues`, 3 iterations, face neighbours, `IGNORE_TILES` |
| `signed_flood_fill` | `signedFloodFill` |
| `filter_gaussian`, `filter_median` | `Filter::gaussian(1, 1)`, `Filter::median(1, 1)` |
| `ls_filter_mean_curvature` | one `LevelSetFilter::meanCurvature` step (including renormalization) |
| `gradient`, `laplacian`, `mean_curvature` | the grid operators |
| `ls_measure` | `LevelSetMeasure` area and volume |
| `volume_to_mesh`, `volume_to_mesh_adaptive` | `volumeToMesh` with adaptivity 0 and 0.5 |
| `mesh_to_level_set` | `meshToLevelSet` of a 131k-quad torus at voxel size 0.004 |
| `ray_intersect` | 500k `LevelSetRayIntersector::intersectsWS` rays, one thread |
| `particles_to_sdf` | `particlesToSdf` of 50k particles of radius 2.5 voxels |
| `point_index_grid`, `point_data_grid` | `createPointIndexGrid` / `createPointDataGrid<NullCodec>` from 2M random positions, voxel size 0.02 |

## Results

Measured on 2026-10-11 on a 4-vCPU Intel Xeon @ 2.10 GHz cloud VM (shared, so expect
±10–20% noise), 15 GB RAM, Ubuntu 24.04. C++: OpenVDB 10.0.1 and oneTBB 2021.11 from Ubuntu,
g++ 13.3. Rust: rustc 1.97.0. "4T" is 4 threads (rayon / TBB defaults, 5 repetitions), "1T"
is `--threads 1` (3 repetitions). Ratios above 1 mean Rust is slower.

| benchmark | C++ 4T (ms) | Rust 4T (ms) | Rust/C++ 4T | C++ 1T (ms) | Rust 1T (ms) | Rust/C++ 1T |
|---|---:|---:|---:|---:|---:|---:|
| `sphere_create` | 103.01 | 137.74 | 1.34x | 257.24 | 336.40 | 1.31x |
| `grid_copy` | 8.96 | 24.87 | 2.78x | 21.10 | 41.63 | 1.97x |
| `active_voxel_count` | 0.45 | 1.38 | 3.07x | 1.17 | 3.17 | 2.71x |
| `iterate_active` | 38.14 | 18.24 | 0.48x | 34.51 | 22.81 | 0.66x |
| `random_insert` | 511.50 | 523.32 | 1.02x | 499.57 | 777.31 | 1.56x |
| `random_read` | 194.12 | 206.38 | 1.06x | 182.62 | 212.64 | 1.16x |
| `coherent_read` | 72.77 | 131.19 | 1.80x | 62.43 | 123.95 | 1.99x |
| `box_sample` | 272.76 | 358.54 | 1.31x | 249.47 | 322.18 | 1.29x |
| `write_none` | 91.38 | 73.53 | 0.80x | 107.42 | 73.45 | 0.68x |
| `read_none` | 27.73 | 74.69 | 2.69x | 28.04 | 64.55 | 2.30x |
| `write_zip` | 1393.15 | 863.91 | 0.62x | 2169.74 | 823.75 | 0.38x |
| `read_zip` | 30.95 | 275.95 | 8.92x | 26.16 | 298.96 | 11.43x |
| `write_blosc` | 148.49 | 155.70 | 1.05x | 216.08 | 142.90 | 0.66x |
| `read_blosc` | 27.31 | 127.95 | 4.69x | 30.31 | 98.53 | 3.25x |
| `csg_union` | 10.27 | 39.71 | 3.86x | 25.17 | 31.33 | 1.24x |
| `csg_difference` | 8.47 | 32.65 | 3.85x | 25.04 | 24.97 | 1.00x |
| `dilate_3` | 36.49 | 225.92 | 6.19x | 99.28 | 368.34 | 3.71x |
| `erode_3` | 11.23 | 95.16 | 8.47x | 30.41 | 185.47 | 6.10x |
| `signed_flood_fill` | 10.03 | 14.48 | 1.44x | 28.25 | 26.22 | 0.93x |
| `filter_gaussian` | 465.79 | 878.51 | 1.89x | 1258.45 | 2282.12 | 1.81x |
| `filter_median` | 593.98 | 634.78 | 1.07x | 1155.27 | 1483.09 | 1.28x |
| `ls_filter_mean_curvature` | 1851.10 | 3315.34 | 1.79x | 5015.09 | 7573.88 | 1.51x |
| `gradient` | 87.42 | 118.77 | 1.36x | 242.57 | 240.70 | 0.99x |
| `laplacian` | 82.47 | 97.36 | 1.18x | 236.74 | 228.80 | 0.97x |
| `mean_curvature` | 335.06 | 313.68 | 0.94x | 800.79 | 775.52 | 0.97x |
| `ls_measure` | 52.81 | 71.19 | 1.35x | 154.39 | 188.81 | 1.22x |
| `volume_to_mesh` | 164.38 | 249.21 | 1.52x | 433.06 | 573.55 | 1.32x |
| `volume_to_mesh_adaptive` | 482.87 | 612.23 | 1.27x | 1322.67 | 1470.65 | 1.11x |
| `mesh_to_level_set` | 2411.66 | 3406.34 | 1.41x | 6771.68 | 7295.72 | 1.08x |
| `ray_intersect` | 379.61 | 619.94 | 1.63x | 375.52 | 621.96 | 1.66x |
| `particles_to_sdf` | 366.11 | 1184.10 | 3.23x | 693.32 | 1235.04 | 1.78x |
| `point_index_grid` | 47.89 | 153.46 | 3.20x | 142.66 | 334.87 | 2.35x |
| `point_data_grid` | 72.88 | 177.72 | 2.44x | 213.08 | 420.97 | 1.98x |
| **geometric mean** | | | **1.86x** | | | **1.50x** |

All result summaries agree between the two libraries (same active voxel counts, the same
value sums to 7 significant digits, the same mesh sizes, ray hits and point leaf counts).
The only differences are the sizes of the written streams, which are within 0.4% of each other
(C++ / Rust bytes: 22,793,521 / 22,777,189 uncompressed, 21,872,216 / 21,782,325 zip,
21,551,656 / 21,479,252 Blosc). Rust's Blosc is its own pure-Rust implementation and its zip
uses `flate2`'s default backend (miniz_oxide), so the compressed sizes are not expected to match.

### Observations

- **Overall:** Rust is about 1.5× slower than C++ on one thread and about 1.9× slower with
  4 threads (geometric means), so part of the gap is in how the tools are parallelised.
- **On par or faster:** the grid operators (`gradient`, `laplacian`, `mean_curvature`),
  `signed_flood_fill`, `csg_*` and `mesh_to_level_set` (single-threaded), `iterate_active`,
  and every **write** benchmark (zip writing is 1.6–2.6× faster than C++).
- **Reading files** is 2.3–11× slower. `read_zip` is the worst case (≈280 ms vs ≈30 ms). It is
  not the deflate backend: switching `flate2` to `zlib-rs` or decompressing each chunk in
  one shot changed `read_zip` by less than 10%. The time goes elsewhere in the read path and
  needs profiling.
- **Morphology** is the largest algorithmic gap: `dilate_3` is 3.7× and `erode_3` 6.1×
  slower single-threaded (6–8× with threads).
- **Scaling with threads:** `csg_*` (3.9× vs 1.0–1.2× on one thread), `particles_to_sdf`
  (Rust barely speeds up from 1 to 4 threads: 1235 → 1184 ms, against 693 → 366 ms for C++),
  `grid_copy`, the point grids, and `active_voxel_count` fall behind once threads are
  enabled.
- **Tree access:** random accessor reads are within 5–15% of C++, but coherent reads are
  about 2× slower, and `box_sample` is about 1.3× slower.
