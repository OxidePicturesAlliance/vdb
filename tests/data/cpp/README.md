# Reference data generators

The `.vdb` files in `tests/data` were written by C++ OpenVDB 10.0.1 using `vdb_ref.cpp`;
the `*.dump.txt` files are the output of `vdb_ref dump` on each of them, and the Blosc test
vectors in `tests/data/blosc` come from `blosc_vectors.c` (c-blosc 1.21.5).

## Building

Requirements: OpenVDB (headers and `libopenvdb`), TBB, Imath, Boost headers (OpenVDB's
headers include a few Boost headers), and for the Blosc vectors c-blosc. On Ubuntu:
`apt install libopenvdb-dev libboost-dev libblosc-dev`.

```sh
cd tests/data/cpp
g++ -std=c++17 -O2 vdb_ref.cpp -o vdb_ref -lopenvdb -ltbb -lImath-3_1
gcc blosc_vectors.c -o blosc_vectors -lblosc -lm
```

(`vdb_ref` and `blosc_vectors` binaries are ignored by git; add `-I<boost include dir>` if
Boost is not installed system-wide.)

## Regenerating the reference files

```sh
./vdb_ref gen ..                     # writes ../*.vdb
for f in ../*.vdb; do ./vdb_ref dump "$f" > "${f%.vdb}.dump.txt"; done
./blosc_vectors ../blosc
```

Values asserted in `tests/io.rs` were obtained with `./vdb_ref probe <file> <grid> <x> <y> <z>`.
If the generator changes, update those expectations too.

## Checking Rust-written files with the C++ library

`cargo test -- --ignored` runs `cpp_reads_rust_files`, which writes files with every
compression option (with and without half floats, and as a stream), runs `vdb_ref dump` on
them and compares the output with the crate's own reading. It looks for the tool in
`$VDB_REF`, or at `tests/data/cpp/vdb_ref`:

```sh
VDB_REF=$PWD/tests/data/cpp/vdb_ref cargo test --test io -- --ignored
```

## OpenVDB 13.1

The tools also build against OpenVDB 13.1.0 (the pixi environment in `pixi.toml`; its
OpenVDB is built with its own half type, so link without `-lImath-3_1`, and take the
NanoVDB headers from the `nanovdb/nanovdb` directory of the `v13.1.0` source tag):

```sh
g++ -std=c++17 -O2 -I$CONDA_PREFIX/include vdb_ref.cpp -o vdb_ref \
    -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lopenvdb -ltbb
```

With 13.1.0 tools, all three `--ignored` interop tests pass (C++ reads every file the Rust
writer produces). Reference files regenerated with 13.1.0 have the same contents as the
committed ones: the `vdb_ref` and `points_ref` dumps differ only in their first line
(13.1 writes file format 225, `OPENVDB_FILE_VERSION_HALF_GRID`, instead of 224), and the
`.nvdb` files are byte-identical (13.1.0 ships NanoVDB 32.9.2). Against regenerated files the
whole suite passes once the version expectations in `tests/io.rs` (`sphere_level_set`,
`file_versions_222_and_223`, `rewrite_reference_files`) are changed to 225 / 13.1. The
committed files stay at 10.0.1 so the tests keep covering format 224.

## NanoVDB reference files (`nanovdb_ref.cpp`)

The files in `tests/data/nanovdb` were written by `nanovdb_ref.cpp` with the header-only
NanoVDB 32.9.2 (the `nanovdb/nanovdb` directory of the OpenVDB repository, which also
compiles against an installed OpenVDB 10 for the OpenVDB → NanoVDB path), zlib and c-blosc:

| file | contents |
|------|----------|
| `types.nvdb` (ZIP) | one grid per segment: float, double, int32, int64, Vec3f, Vec3d, bool, a bool grid as `ValueMask`, the float grid as `Fp4`/`Fp8`/`Fp16`/`FpN`, and as `ValueIndex`/`ValueOnIndex` index grids (with and without a channel, statistics and tiles) |
| `modes.nvdb` (ZIP) | the float grid with every statistics mode and checksum mode, a 310-character grid name (stored as blind data) and an empty grid |
| `band.nvdb` (ZIP) | a narrow band built voxel by voxel, as float, dithered `Fp4`, `Fp16` and two `FpN` variants |
| `primitives.nvdb` (BLOSC) | `createLevelSetSphere<float>`, `createLevelSetSphere<Fp8>`, `createLevelSetTorus`, `createLevelSetBox` and `createFogVolumeSphere` |
| `multi.nvdb` (ZIP) | three grids merged with `mergeGrids` and written as a single segment |
| `single_none.nvdb` | a single-voxel grid, uncompressed |
| `raw.nvdb` | a raw grid buffer (no file header) holding two grids made of root tiles |
| `samples.txt` | values of the samplers of order 0–3 (cached and not), trilinear gradients and `zeroCrossing`/`firstActive` results on the sphere, as float bit patterns |

The source grids are mirrored exactly by `tests/nanovdb.rs`, which checks that the Rust
conversion produces byte-for-byte identical grid buffers (including statistics and
checksums), that values, statistics and samples agree, and that files round-trip.

Building and regenerating (`<nanovdb>` is the directory containing `nanovdb/NanoVDB.h`):

```sh
cd tests/data/cpp
g++ -std=c++17 -O1 -I<nanovdb> nanovdb_ref.cpp -o nanovdb_ref -lopenvdb -ltbb -lImath-3_1 -lz -lblosc
./nanovdb_ref gen ../nanovdb
```

Do not define `NANOVDB_USE_TBB`: the statistics must be reduced serially so that averages
and standard deviations are bit-for-bit reproducible.

`cargo test --test nanovdb -- --ignored` runs `cpp_reads_rust_nvdb_files`, which writes
`.nvdb` files with every codec (and a merged segment and a raw buffer), runs
`nanovdb_ref check` on them (C++ `readGrids`, `validateChecksum`, `GridValidator::isValid`
and a full CRC of each grid) and compares its output with the Rust-side expectations. The
tool is looked up in `$NANOVDB_REF` or at `tests/data/cpp/nanovdb_ref`:

```sh
NANOVDB_REF=$PWD/tests/data/cpp/nanovdb_ref cargo test --test nanovdb -- --ignored
## Point data reference files (`points_ref.cpp`)

`points_ref.cpp` generates the point data and point index grid files in `tests/data/points/`
and dumps them in the text format compared by `tests/points.rs`:

```sh
cd tests/data/cpp
g++ -std=c++17 -O1 points_ref.cpp -o points_ref -lopenvdb -ltbb -lImath-3_1
./points_ref gen ../points                     # writes ../points/*.vdb
for f in attrs none zip codecs misc scatter; do
  ./points_ref dump ../points/points_$f.vdb > ../points/points_$f.dump.txt
done
./points_ref dumpindex ../points/points_index.vdb > ../points/points_index.dump.txt
```

(`points_ref` is ignored by git.) The ignored test `cpp_reads_rust_points` writes point
data and point index grids from Rust (every compression mode, including multi-page paged
attribute streams) and checks that `points_ref dump` / `dumpindex` read them identically.
It looks for the tool in `$POINTS_REF`, or at `tests/data/cpp/points_ref`:

```sh
POINTS_REF=$PWD/tests/data/cpp/points_ref cargo test --test points -- --ignored
```
