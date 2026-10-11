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
on Ubuntu 24.04). Where the distribution's OpenVDB is too old (e.g. 9.x on RHEL/Rocky),
use the pixi environment in `pixi.toml`, which installs OpenVDB 13.1, TBB and a matching
g++ from [Anders Langlands' ASWF channel](https://github.com/anderslanglands/aswf-pixi)
and conda-forge:

```sh
pixi run bench-compare [filter] [--reps N] [--threads N]
```

The script builds it with `g++ -O3 -DNDEBUG` (override with `CXX` /
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

Measured on 2026-10-11 (after the optimizations listed under History) on a 4-vCPU Intel
Xeon @ 2.10 GHz cloud VM (shared, so expect ±10–20% noise, more for single benchmarks run in
isolation), 15 GB RAM, Ubuntu 24.04. C++: OpenVDB 10.0.1 and oneTBB 2021.11 from Ubuntu,
g++ 13.3. Rust: rustc 1.97.0. "4T" is 4 threads (rayon / TBB defaults, 5 repetitions), "1T"
is `--threads 1` (3 repetitions). Ratios above 1 mean Rust is slower.

| benchmark | C++ 4T (ms) | Rust 4T (ms) | Rust/C++ 4T | C++ 1T (ms) | Rust 1T (ms) | Rust/C++ 1T |
|---|---:|---:|---:|---:|---:|---:|
| `sphere_create` | 92.13 | 102.63 | 1.11x | 232.37 | 342.36 | 1.47x |
| `grid_copy` | 9.33 | 6.13 | 0.66x | 23.97 | 26.83 | 1.12x |
| `active_voxel_count` | 0.50 | 0.61 | 1.22x | 1.69 | 1.86 | 1.10x |
| `iterate_active` | 40.72 | 16.61 | 0.41x | 36.65 | 20.94 | 0.57x |
| `random_insert` | 605.22 | 545.39 | 0.90x | 646.87 | 655.67 | 1.01x |
| `random_read` | 190.99 | 256.71 | 1.34x | 185.69 | 255.75 | 1.38x |
| `coherent_read` | 43.20 | 75.95 | 1.76x | 40.92 | 83.59 | 2.04x |
| `box_sample` | 233.95 | 353.12 | 1.51x | 256.15 | 374.24 | 1.46x |
| `write_none` | 70.39 | 68.45 | 0.97x | 100.29 | 63.48 | 0.63x |
| `read_none` | 34.77 | 34.77 | 1.00x | 76.45 | 31.60 | 0.41x |
| `write_zip` | 1272.97 | 752.52 | 0.59x | 1993.17 | 771.14 | 0.39x |
| `read_zip` | 215.52 | 193.02 | 0.90x | 211.54 | 158.63 | 0.75x |
| `write_blosc` | 143.53 | 125.08 | 0.87x | 217.71 | 130.79 | 0.60x |
| `read_blosc` | 48.91 | 42.89 | 0.88x | 51.05 | 47.61 | 0.93x |
| `csg_union` | 10.70 | 8.31 | 0.78x | 24.05 | 24.85 | 1.03x |
| `csg_difference` | 8.55 | 8.54 | 1.00x | 23.91 | 22.12 | 0.93x |
| `dilate_3` | 31.89 | 29.24 | 0.92x | 85.01 | 62.30 | 0.73x |
| `erode_3` | 9.07 | 11.19 | 1.23x | 31.54 | 23.10 | 0.73x |
| `signed_flood_fill` | 7.06 | 7.05 | 1.00x | 24.93 | 32.23 | 1.29x |
| `filter_gaussian` | 341.67 | 342.97 | 1.00x | 1258.80 | 1253.53 | 1.00x |
| `filter_median` | 350.47 | 302.02 | 0.86x | 1156.73 | 972.56 | 0.84x |
| `ls_filter_mean_curvature` | 1406.47 | 1408.28 | 1.00x | 4769.00 | 4774.32 | 1.00x |
| `gradient` | 75.08 | 68.09 | 0.91x | 258.29 | 146.13 | 0.57x |
| `laplacian` | 74.89 | 44.88 | 0.60x | 225.30 | 136.88 | 0.61x |
| `mean_curvature` | 226.10 | 131.38 | 0.58x | 855.67 | 447.65 | 0.52x |
| `ls_measure` | 48.02 | 29.94 | 0.62x | 150.19 | 130.20 | 0.87x |
| `volume_to_mesh` | 136.94 | 160.75 | 1.17x | 425.67 | 455.36 | 1.07x |
| `volume_to_mesh_adaptive` | 369.63 | 412.02 | 1.11x | 1249.22 | 1454.57 | 1.16x |
| `mesh_to_level_set` | 1795.14 | 2203.40 | 1.23x | 6442.82 | 6161.66 | 0.96x |
| `ray_intersect` | 374.98 | 524.76 | 1.40x | 443.21 | 528.60 | 1.19x |
| `particles_to_sdf` | 325.38 | 321.01 | 0.99x | 691.00 | 559.54 | 0.81x |
| `point_index_grid` | 41.67 | 38.05 | 0.91x | 133.52 | 95.86 | 0.72x |
| `point_data_grid` | 79.68 | 74.56 | 0.94x | 199.92 | 197.63 | 0.99x |
| **geometric mean** | | | **0.94x** | | | **0.88x** |

All result summaries agree between the two libraries (same active voxel counts, the same
value sums to 7 significant digits, the same mesh sizes, ray hits and point leaf counts).
The only differences are the sizes of the written streams, which are within 0.4% of each other
(C++ / Rust bytes: 22,793,521 / 22,777,189 uncompressed, 21,872,216 / 21,782,325 zip,
21,551,656 / 21,479,252 Blosc). Rust's Blosc is its own pure-Rust implementation and its zip
uses `flate2`'s default backend (miniz_oxide), so the compressed sizes are not expected to match.

### Observations

- **Overall:** Rust is now about 10% faster than C++ on one thread and about 5% faster with
  4 threads (geometric means), from 1.14× and 1.42× slower before the changes below.
- **Faster than C++:** file I/O (zip writing 1.7–2.6× faster), the grid operators,
  `iterate_active`, `ls_measure`, `filter_median`, morphology (single-threaded),
  `particles_to_sdf` and the point grids.
- **On par (within ~10%):** CSG, the Gaussian and level-set filters, `random_insert`,
  `grid_copy`, `volume_to_mesh` (single-threaded) and `mesh_to_level_set` (single-threaded).
- **Still slower:** random-access reads (`random_read`, `box_sample`, `ray_intersect`, 1.2–1.5×)
  and especially `coherent_read` (about 1.8–2×); meshing and `mesh_to_level_set` with 4
  threads and `sphere_create`, `signed_flood_fill` and adaptive meshing on one thread
  (1.1–1.5×).

### History

- **Performance pass (2026-10-11):** geometric means went from 1.14× / 1.42× (1T / 4T)
  slower than C++ to 0.88× / 0.94×. All outputs are unchanged.
  - Tree: leaf (and other small) node masks are stored inline instead of in a separate heap
    allocation; tree copies and voxel/leaf counts run in parallel over the top-level
    children; internal nodes test the child mask before reading the child or tile arrays.
  - Accessors: the read-only accessor finds the leaf or tile holding a voxel with one
    descent and caches the last tile; box samplers read a 2×2×2 cell with one lookup when it
    lies in one leaf; stencils and grid operators read neighbors in the center voxel's leaf
    straight from its buffer and look up axis-aligned stencil offsets with a direct table.
  - `Coord::floor`/`round`/`ceil` no longer call libm's `floor`/`ceil` (the baseline x86-64
    target has no rounding instruction).
  - `particles_to_sdf`: buckets are rasterized in parallel into per-task trees merged in
    bucket order with a rule that reproduces the sequential result exactly; the inner loop
    tests the distance before reading the tree.
  - Point partitioning uses a stable parallel counting sort instead of a comparison sort.
  - CSG merges, sphere creation, mesh voxelization (now through cached accessors) and the
    volume-to-mesh intersection mask merge their per-task results pairwise in parallel.
  - The box filter reads in-leaf neighbors from the leaf buffer; the ray intersector reuses
    its stencil across rays and looks nodes up through the accessor cache.

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
