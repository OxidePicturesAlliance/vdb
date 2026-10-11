#!/usr/bin/env bash
# Run the Rust and C++ OpenVDB comparison benchmarks and print a Markdown table.
#
#   benches/compare.sh [filter] [--reps N] [--threads N]
#
# Requires OpenVDB (headers + libopenvdb), TBB and Imath for the C++ side, e.g.
# `apt install libopenvdb-dev libboost-dev` on Ubuntu, or run it through pixi
# (`pixi run bench-compare`, see pixi.toml), which provides OpenVDB 13. Set CXX / CXXFLAGS
# to override the compiler and flags.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${VDB_BENCH_OUT:-$root/target/bench-compare}"
mkdir -p "$out"

cxx="${CXX:-g++}"
cxxflags="${CXXFLAGS:--O3 -DNDEBUG}"
cpp_bin="$out/compare_cpp"
cpp_libs=(-lopenvdb -ltbb -lImath-3_1)
if [[ -n "${PIXI_PROJECT_ROOT:-}" && -n "${CONDA_PREFIX:-}" ]]; then
    # Inside a pixi environment: link its OpenVDB (built with its own half type, so no Imath).
    cpp_bin="$out/compare_cpp_pixi"
    cpp_libs=(-I"$CONDA_PREFIX/include" -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" -lopenvdb -ltbb)
fi
if [[ ! -x "$cpp_bin" || "$root/benches/cpp/compare.cpp" -nt "$cpp_bin" ]]; then
    echo "building C++ benchmark ($cxx $cxxflags)" >&2
    $cxx -std=c++17 $cxxflags "$root/benches/cpp/compare.cpp" -o "$cpp_bin" "${cpp_libs[@]}"
fi
(cd "$root" && cargo build --release --bench compare --quiet)
rust_bin="$(cd "$root" && cargo bench --bench compare --no-run --message-format=json 2>/dev/null \
    | python3 -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m["target"]["name"] == "compare" and m.get("executable"):
        print(m["executable"])')"

"$rust_bin" "$@" > "$out/rust.tsv"
"$cpp_bin" "$@" > "$out/cpp.tsv"

python3 - "$out/rust.tsv" "$out/cpp.tsv" <<'PY'
import re, sys

def load(path):
    rows = {}
    for line in open(path):
        name, med, mn, check = line.rstrip("\n").split("\t")
        rows[name] = (float(med), float(mn), check)
    return rows

def norm(check):
    # C++ prints exponents as e+02, Rust as e2.
    return re.sub(r"e([+-]?)0*(\d)", lambda m: "e" + ("-" if m.group(1) == "-" else "") + m.group(2), check)

rust, cpp = load(sys.argv[1]), load(sys.argv[2])
print("| benchmark | C++ (ms) | Rust (ms) | Rust / C++ | results |")
print("|---|---:|---:|---:|---|")
ratios = []
for name, (rmed, _, rcheck) in rust.items():
    if name not in cpp:
        continue
    cmed, _, ccheck = cpp[name]
    ratio = rmed / cmed if cmed > 0 else float("inf")
    ratios.append(ratio)
    same = "same" if norm(rcheck) == norm(ccheck) else f"differ: C++ `{ccheck}`, Rust `{rcheck}`"
    print(f"| {name} | {cmed:.2f} | {rmed:.2f} | {ratio:.2f}x | {same} |")
if ratios:
    import math
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f"\nGeometric mean Rust / C++: {geo:.2f}x over {len(ratios)} benchmarks")
PY
