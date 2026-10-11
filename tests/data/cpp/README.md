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
