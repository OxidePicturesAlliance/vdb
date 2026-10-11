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

Measured on 2026-10-11 (after the read-path optimization below) on a 4-vCPU Intel Xeon
@ 2.10 GHz cloud VM (shared, so expect ±10–20% noise), 15 GB RAM, Ubuntu 24.04. C++: OpenVDB 10.0.1 and oneTBB 2021.11 from Ubuntu,
g++ 13.3. Rust: rustc 1.97.0. "4T" is 4 threads (rayon / TBB defaults, 5 repetitions), "1T"
is `--threads 1` (3 repetitions). Ratios above 1 mean Rust is slower.

| benchmark | C++ 4T (ms) | Rust 4T (ms) | Rust/C++ 4T | C++ 1T (ms) | Rust 1T (ms) | Rust/C++ 1T |
|---|---:|---:|---:|---:|---:|---:|
| `sphere_create` | 95.96 | 136.08 | 1.42x | 310.92 | 355.09 | 1.14x |
| `grid_copy` | 8.37 | 26.18 | 3.13x | 23.53 | 43.40 | 1.84x |
| `active_voxel_count` | 0.39 | 3.00 | 7.78x | 1.58 | 3.34 | 2.11x |
| `iterate_active` | 39.01 | 21.74 | 0.56x | 38.70 | 19.08 | 0.49x |
| `random_insert` | 464.53 | 544.08 | 1.17x | 512.28 | 754.14 | 1.47x |
| `random_read` | 178.56 | 207.43 | 1.16x | 174.25 | 226.54 | 1.30x |
| `coherent_read` | 47.00 | 126.67 | 2.70x | 48.59 | 127.14 | 2.62x |
| `box_sample` | 300.14 | 357.96 | 1.19x | 240.83 | 335.07 | 1.39x |
| `write_none` | 104.39 | 77.99 | 0.75x | 105.10 | 78.84 | 0.75x |
| `read_none` | 38.59 | 33.85 | 0.88x | 34.89 | 34.66 | 0.99x |
| `write_zip` | 1495.88 | 862.10 | 0.58x | 2281.90 | 930.72 | 0.41x |
| `read_zip` | 220.18 | 213.48 | 0.97x | 217.64 | 206.25 | 0.95x |
| `write_blosc` | 140.41 | 162.21 | 1.16x | 215.19 | 148.94 | 0.69x |
| `read_blosc` | 49.05 | 46.39 | 0.95x | 48.58 | 46.60 | 0.96x |
| `csg_union` | 11.99 | 31.90 | 2.66x | 25.16 | 32.70 | 1.30x |
| `csg_difference` | 11.52 | 29.70 | 2.58x | 24.98 | 28.27 | 1.13x |
| `dilate_3` | 40.13 | 30.36 | 0.76x | 96.76 | 61.39 | 0.63x |
| `erode_3` | 9.52 | 12.45 | 1.31x | 24.52 | 21.96 | 0.90x |
| `signed_flood_fill` | 8.11 | 10.47 | 1.29x | 26.19 | 26.53 | 1.01x |
| `filter_gaussian` | 384.85 | 676.43 | 1.76x | 1238.32 | 2276.73 | 1.84x |
| `filter_median` | 378.38 | 431.69 | 1.14x | 1241.05 | 1462.93 | 1.18x |
| `ls_filter_mean_curvature` | 1609.84 | 2386.59 | 1.48x | 5404.35 | 7552.76 | 1.40x |
| `gradient` | 81.21 | 106.64 | 1.31x | 278.26 | 275.48 | 0.99x |
| `laplacian` | 83.98 | 93.86 | 1.12x | 267.04 | 251.68 | 0.94x |
| `mean_curvature` | 268.85 | 263.27 | 0.98x | 940.02 | 765.09 | 0.81x |
| `ls_measure` | 54.07 | 69.46 | 1.28x | 161.21 | 202.26 | 1.25x |
| `volume_to_mesh` | 176.38 | 232.41 | 1.32x | 441.40 | 559.86 | 1.27x |
| `volume_to_mesh_adaptive` | 436.82 | 525.79 | 1.20x | 1337.16 | 1410.33 | 1.05x |
| `mesh_to_level_set` | 1951.61 | 2616.26 | 1.34x | 7006.84 | 7785.22 | 1.11x |
| `ray_intersect` | 402.36 | 651.90 | 1.62x | 487.46 | 650.93 | 1.34x |
| `particles_to_sdf` | 320.01 | 1307.70 | 4.09x | 727.79 | 1351.74 | 1.86x |
| `point_index_grid` | 48.95 | 124.77 | 2.55x | 184.80 | 305.13 | 1.65x |
| `point_data_grid` | 73.63 | 177.40 | 2.41x | 239.53 | 438.21 | 1.83x |
| **geometric mean** | | | **1.60x** | | | **1.29x** |

All result summaries agree between the two libraries (same active voxel counts, the same
value sums to 7 significant digits, the same mesh sizes, ray hits and point leaf counts).
The only differences are the sizes of the written streams, which are within 0.4% of each other
(C++ / Rust bytes: 22,793,521 / 22,777,189 uncompressed, 21,872,216 / 21,782,325 zip,
21,551,656 / 21,479,252 Blosc). Rust's Blosc is its own pure-Rust implementation and its zip
uses `flate2`'s default backend (miniz_oxide), so the compressed sizes are not expected to match.

### Observations

- **Overall:** Rust is about 1.3× slower than C++ on one thread and about 1.6× slower with
  4 threads (geometric means), so a good part of the gap is in how the tools are
  parallelised.
- **On par or faster:** reading and writing files (except `write_blosc` with threads), the
  grid operators (`gradient`, `laplacian`, `mean_curvature`), `signed_flood_fill`, `csg_*`
  and `mesh_to_level_set` (single-threaded), and `iterate_active`. Zip writing is 1.7–2.5×
  faster than C++.
- **Morphology:** `dilate_3` is faster than C++ and `erode_3` is on par single-threaded
  (1.3× with threads, where both take ~10 ms).
- **Scaling with threads:** `csg_*` (2.6× vs 1.1–1.3× on one thread), `particles_to_sdf`
  (Rust does not speed up from 1 to 4 threads, while C++ goes from 728 to 320 ms),
  `grid_copy`, the point grids and `active_voxel_count` fall behind once threads are enabled.
- **Tree access:** random accessor reads are within 15–30% of C++, but coherent reads are
  about 2.6× slower, and `box_sample` is about 1.2–1.4× slower.

### History

- **Morphology:** dilation and erosion now work directly on the 64-bit active-mask words of
  each leaf and its 6 / 18 / 26 neighbor leaves (shifting and masking words for the z, y
  and x neighbors), instead of copying them into a 24³ scratch volume per leaf. Erosion
  skips neighbors that don't share active border voxels with the leaf. Single-threaded,
  `dilate_3` went from 422 to 61 ms and `erode_3` from 200 to 22 ms (192 → 30 ms and
  76 → 12 ms with 4 threads). The `dilate_3` / `erode_3` rows above are from this change.
- **Read path (I/O):** leaf buffers are now decoded straight from the (decompressed) bytes
  into the leaf, using per-thread reusable buffers and zlib state, instead of going through
  three temporary vectors per node; Blosc unshuffling is specialized for 4- and 8-byte
  types. Single-threaded reads went from 65 / 299 / 99 ms (none / zip / Blosc) to
  35 / 206 / 47 ms.
- **Benchmark fix:** the first version of `compare.cpp` read streams with OpenVDB's default
  delayed loading, so its `read_*` timings skipped decoding the leaf buffers and were too
  low (the earlier "2.3–11× slower" read results were wrong). It now passes
  `delayLoad = false`, and the `read_*` checks include the sum of the decoded values.
