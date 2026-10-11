# vdb

A Rust port of [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb), the sparse
volumetric data structure and toolset used throughout film VFX.

The port follows the C++ library closely: same tree layout (Root → 32³ → 16³ → 8³ leaves),
same file format, same algorithms. Most tools are checked against numbers produced by the
C++ library (OpenVDB 10) on identical inputs.

```rust
use vdb::prelude::*;
use vdb::tools::level_set_sphere::create_level_set_sphere;

// A narrow-band level set of a sphere: radius 1, centered at the origin,
// voxel size 0.05, half-width 3 voxels.
let sphere: FloatGrid = create_level_set_sphere(1.0, Vec3f::zero(), 0.05, 3.0, true)?;
println!("{} active voxels", sphere.active_voxel_count());

vdb::io::write_file("sphere.vdb", &[sphere.into()], &MetaMap::new())?;
let (_meta, grids) = vdb::io::read_file("sphere.vdb")?;
```

## What's ported

| Area | Rust module | Notes |
|------|-------------|-------|
| Core math | `math` | `Vec2/3/4`, `Coord`, `CoordBBox`, `BBox`, `Mat3/4`, `Quat`, `Ray`, `Dda`, stats/histograms |
| Transforms | `math::maps`, `math::transform` | All linear maps and the frustum map; serialization is byte-identical to C++ |
| Finite differences | `math::finite_difference`, `math::stencils`, `math::operators` | All schemes (CD/FD/BD, WENO5, HJWENO5); world-space operators via map Jacobians |
| Tree | `tree` | `LeafNode`, `InternalNode`, `RootNode`, `Tree<T>`, cached `ValueAccessor`s |
| Grids & metadata | `grid`, `metadata` | `Grid<T>`, `AnyGrid`, grid class / vector type, typed `MetaMap` |
| File I/O | `io` | `.vdb` read/write, versions 220–225, zip and pure-Rust blosc, half floats, all mask-compression schemes |
| Level sets | `tools::level_set_*`, `signed_flood_fill`, `change_background`, `prune` | Sphere, Platonic solids, box, capsules/tubes, dilated meshes; measure (area, volume, genus); utilities |
| CSG & compositing | `tools::composite`, `tools::tree_ops` | Union / intersection / difference, comp min/max/sum/mul/div/replace, tree merge |
| Topology | `tools::morphology`, `activate`, `clip`, `mask`, `topology_to_level_set` | Dilation and erosion with every neighbour pattern and tile policy |
| Sampling | `tools::interpolation`, `tools::grid_transformer` | Point / box / quadratic (+ staggered) samplers, resampling |
| Filtering & evolution | `tools::filter`, `level_set_filter`, `level_set_tracker`, `level_set_advect`, `level_set_morph`, `volume_advect`, `fast_sweeping` | |
| Grid operators | `tools::grid_operators` | Gradient, curl, divergence, Laplacian, mean curvature, CPT, magnitude, normalize |
| Meshing | `tools::mesh_to_volume`, `tools::volume_to_mesh` | Adaptive meshing gives identical points and polygons to C++ |
| Particles & points | `tools::particles_to_level_set`, `point_index_grid`, `point_partitioner`, `point_scatter`, `points_to_mask`, `volume_to_spheres` | Scattered points are bit-identical to C++ |
| Rays | `tools::ray_intersector`, `tools::ray_tracer` | Level-set and volume intersection, a small renderer |
| Solvers | `math::conj_gradient`, `tools::poisson_solver` | Preconditioned conjugate gradient |
| Point data | `points` | `PointDataGrid` with attribute arrays, codecs, groups and strings; conversion, counting, sampling, scattering, rasterization, advection; C++-compatible file I/O |
| NanoVDB | `nanovdb` | Pointer-free GPU-friendly layout, OpenVDB↔NanoVDB conversion (incl. Fp4/8/16/N, index grids), `.nvdb` I/O, samplers, HDDA; converted grids are byte-identical to C++ |
| Diagnostics | `tools::diagnostics`, `count`, `statistics`, `find_active_values` | |

**Not planned:** OpenVDB AX (needs an LLVM JIT), the Houdini/Maya plugins, Python bindings.

**Known gaps:** the frustum point rasterizer, delayed loading and memory-mapped files, interrupters, non-standard tree
configurations (tools assume the standard 5-4-3 tree).

## Examples

```sh
cargo run --release --example vdb_print -- -l tests/data/sphere.vdb   # like OpenVDB's vdb_print
cargo run --release --example level_set_sphere
cargo run --release --example mesh_roundtrip -- torus.obj 0.01 0.0
cargo run --release --example ray_trace_sphere                         # writes a PPM image
cargo run --release --example nanovdb_sphere
```

## Testing

```sh
cargo test
```

Tests include:

- **Model-based tree tests:** random edits checked against a dense reference.
- **C++-written files:** reference `.vdb` files written by C++ OpenVDB 10.0.1 in `tests/data`.
- **Embedded C++ results:** numbers from C++ runs of the same operations, stored in the tests.

The C++ programs that produced the reference data are in `tests/data/cpp/`. With them
built, `VDB_REF=<path to vdb_ref> cargo test --test io -- --ignored` also checks that C++
reads every file the Rust writer produces.

## Benchmarks

`benches/compare.sh` runs the same operations with this crate and with C++ OpenVDB 10 and
prints a comparison table; see [`benches/README.md`](benches/README.md) for the benchmark list
and current results.

## License

MIT (see `LICENSE`). OpenVDB itself is licensed under Apache-2.0 / MPL-2.0; this is an
independent port.
