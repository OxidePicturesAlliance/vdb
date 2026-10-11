// Reference generator and checker for the NanoVDB port (see README.md).
//
//   nanovdb_ref gen <dir>     write the reference .nvdb files into <dir>
//   nanovdb_ref check <file>  read a .nvdb file and print one line per grid (used by the
//                             ignored Rust test `cpp_reads_rust_nvdb_files`)
//
// The grids built here are mirrored exactly by tests/nanovdb.rs.

#define NANOVDB_USE_OPENVDB
#define NANOVDB_USE_ZIP
#define NANOVDB_USE_BLOSC
#include <openvdb/openvdb.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/tools/GridChecksum.h>
#include <nanovdb/tools/GridValidator.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace openvdb;
using HandleT = nanovdb::GridHandle<nanovdb::HostBuffer>;

// ---------------------------------------------------------------------------------------
// Source grids
// ---------------------------------------------------------------------------------------

template<typename GridT, typename F>
typename GridT::Ptr makeGrid(const std::string& name, typename GridT::ValueType bg, F val)
{
    auto g = GridT::create(bg);
    g->setName(name);
    auto t = math::Transform::createLinearTransform(0.5);
    t->postTranslate(Vec3d(1.0, -2.0, 3.0));
    g->setTransform(t);
    auto& tree = g->tree();
    tree.setValue(Coord(0, 0, 0), val(0));
    tree.setValue(Coord(1, 2, 3), val(1));
    tree.setValue(Coord(-5, 7, 100), val(2));
    tree.setValue(Coord(1000, -2000, 3000), val(3));
    tree.setValueOff(Coord(10, 10, 10), val(4));
    tree.fill(CoordBBox(Coord(-256), Coord(-129)), val(5), true);
    tree.fill(CoordBBox(Coord(64), Coord(71)), val(6), true);
    tree.addTile(3, Coord(8192, 0, 0), val(7), true);
    tree.addTile(3, Coord(-8192, 4096, 0), val(8), false);
    for (int i = 0; i < 20; ++i) tree.setValue(Coord(20 + i, 30, 40 + (i % 3)), val(9 + i));
    return g;
}

FloatGrid::Ptr floatGrid()
{
    auto g = makeGrid<FloatGrid>("float_grid", 2.0f, [](int i) { return 0.25f * float(i) - 1.0f; });
    g->setGridClass(GRID_FOG_VOLUME);
    return g;
}
DoubleGrid::Ptr doubleGrid()
{
    return makeGrid<DoubleGrid>("double_grid", -1.5, [](int i) { return 0.125 * double(i) - 2.0; });
}
Int32Grid::Ptr int32Grid()
{
    return makeGrid<Int32Grid>("int32_grid", 7, [](int i) { return i * 3 - 10; });
}
Int64Grid::Ptr int64Grid()
{
    return makeGrid<Int64Grid>("int64_grid", -9, [](int i) { return int64_t(i) * 1000000007LL - 5; });
}
Vec3fGrid::Ptr vec3fGrid()
{
    return makeGrid<Vec3fGrid>("vec3f_grid", Vec3f(0.5f, 0.0f, -1.0f),
                               [](int i) { return Vec3f(float(i), -0.5f * float(i), 2.0f); });
}
Vec3dGrid::Ptr vec3dGrid()
{
    return makeGrid<Vec3dGrid>("vec3d_grid", Vec3d(0.0, 1.0, 0.0),
                               [](int i) { return Vec3d(0.5 * i, 3.0, -double(i)); });
}
BoolGrid::Ptr boolGrid()
{
    return makeGrid<BoolGrid>("bool_grid", false, [](int i) { return i % 3 != 1; });
}

/// A narrow band of sphere-like distances, built voxel by voxel.
FloatGrid::Ptr bandGrid()
{
    auto g = FloatGrid::create(0.3f);
    g->setName("band");
    g->setGridClass(GRID_LEVEL_SET);
    g->setTransform(math::Transform::createLinearTransform(0.1));
    auto acc = g->getAccessor();
    for (int i = -20; i <= 20; ++i)
        for (int j = -20; j <= 20; ++j)
            for (int k = -20; k <= 20; ++k) {
                const float d = std::sqrt(float(i * i + j * j + k * k)) - 15.0f;
                if (std::fabs(d) < 3.0f) acc.setValue(Coord(i, j, k), 0.1f * d);
            }
    return g;
}

FloatGrid::Ptr singleVoxelGrid()
{
    auto g = FloatGrid::create(0.0f);
    g->setName("single");
    g->tree().setValue(Coord(3, 4, 5), 42.0f);
    return g;
}

FloatGrid::Ptr rootTilesGrid(const std::string& name)
{
    auto g = FloatGrid::create(1.0f);
    g->setName(name);
    g->tree().addTile(3, Coord(0, 0, 0), 2.0f, true);
    g->tree().addTile(3, Coord(-4096, 0, 4096), 3.0f, false);
    return g;
}

// ---------------------------------------------------------------------------------------
// gen
// ---------------------------------------------------------------------------------------

using nanovdb::tools::StatsMode;
using nanovdb::CheckMode;

static uint32_t bits(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}

template<int Order, bool Cache, typename AccT>
static void writeSamples(FILE* f, const AccT& acc)
{
    auto s = nanovdb::math::SampleFromVoxels<AccT, Order, Cache>(acc);
    for (int n = 0; n < 64; ++n) {
        const nanovdb::Vec3d p(-5.2 + (n % 13) * 1.37, -4.5 + (n % 11) * 1.71, -13.0 + (n % 7) * 2.33);
        fprintf(f, "s %d %d %.17g %.17g %.17g %08x\n", Order, int(Cache), p[0], p[1], p[2], bits(s(p)));
    }
}

/// Interpolated values, gradients and ray intersections on the level-set sphere.
static void samples(const std::string& file)
{
    auto handle = nanovdb::tools::createLevelSetSphere<float>(5.0, nanovdb::Vec3d(1.5, 2.0, -3.0), 0.5, 3.0,
                                                              nanovdb::Vec3d(0.0), "sphere");
    auto* grid = handle.grid<float>();
    FILE* f = fopen(file.c_str(), "w");
    {
        auto acc = grid->getAccessor();
        writeSamples<0, true>(f, acc);
        writeSamples<0, false>(f, acc);
        writeSamples<1, true>(f, acc);
        writeSamples<1, false>(f, acc);
        writeSamples<2, true>(f, acc);
        writeSamples<2, false>(f, acc);
        writeSamples<3, true>(f, acc);
        writeSamples<3, false>(f, acc);
        auto s = nanovdb::math::SampleFromVoxels<decltype(acc), 1, false>(acc);
        for (int n = 0; n < 16; ++n) {
            const nanovdb::Vec3d p(-2.3 + n * 0.71, 1.1 + n * 0.37, -9.7 + n * 0.53);
            const auto g = s.gradient(p);
            fprintf(f, "g %.17g %.17g %.17g %08x %08x %08x\n", p[0], p[1], p[2], bits(g[0]), bits(g[1]), bits(g[2]));
        }
    }
    for (int k = 0; k < 16; ++k) {
        using RayT = nanovdb::math::Ray<float>;
        const nanovdb::Vec3f eye(-20.0f, -2.0f + 0.5f * k, -10.0f + 0.25f * k);
        const nanovdb::Vec3f dir(1.0f, 0.05f * k - 0.3f, 0.02f * k);
        {
            RayT ray(eye, dir);
            auto acc = grid->getAccessor();
            nanovdb::Coord ijk;
            float v = 0.0f, t = 0.0f;
            const bool hit = nanovdb::math::zeroCrossing(ray, acc, ijk, v, t);
            fprintf(f, "z %d %d %d %d %d %08x %08x\n", k, int(hit), ijk[0], ijk[1], ijk[2], bits(hit ? v : 0.0f),
                    bits(hit ? t : 0.0f));
        }
        {
            RayT ray(eye, dir);
            auto acc = grid->getAccessor();
            nanovdb::Coord ijk;
            float t = 0.0f;
            const bool hit = nanovdb::math::firstActive(ray, acc, ijk, t);
            fprintf(f, "a %d %d %d %d %d %08x\n", k, int(hit), ijk[0], ijk[1], ijk[2], bits(hit ? t : 0.0f));
        }
    }
    fclose(f);
}

static void gen(const std::string& dir)
{
    namespace nt = nanovdb::tools;
    {   // all value and build types, one segment per grid
        std::vector<HandleT> h;
        auto f = floatGrid();
        h.push_back(nt::createNanoGrid(*f));
        h.push_back(nt::createNanoGrid(*doubleGrid()));
        h.push_back(nt::createNanoGrid(*int32Grid()));
        h.push_back(nt::createNanoGrid(*int64Grid()));
        h.push_back(nt::createNanoGrid(*vec3fGrid()));
        h.push_back(nt::createNanoGrid(*vec3dGrid()));
        auto b = boolGrid();
        h.push_back(nt::createNanoGrid(*b));
        h.push_back(nt::createNanoGrid<BoolGrid, nanovdb::ValueMask>(*b));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::Fp4>(*f));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::Fp8>(*f));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::Fp16>(*f));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::FpN>(*f));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::ValueIndex>(*f, 1u, true, true));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::ValueOnIndex>(*f, 1u, true, true));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::ValueOnIndex>(*f, 0u, false, false));
        nanovdb::io::writeGrids(dir + "/types.nvdb", h, nanovdb::io::Codec::ZIP);
    }
    {   // statistics and checksum modes, long names, empty grids
        std::vector<HandleT> h;
        auto f = floatGrid();
        h.push_back(nt::createNanoGrid(*f, StatsMode::Disable, CheckMode::Partial));
        h.push_back(nt::createNanoGrid(*f, StatsMode::BBox, CheckMode::Full));
        h.push_back(nt::createNanoGrid(*f, StatsMode::MinMax, CheckMode::Disable));
        h.push_back(nt::createNanoGrid(*f, StatsMode::All, CheckMode::Full));
        auto s = singleVoxelGrid();
        s->setName(std::string(300, 'x') + "_long_name");
        h.push_back(nt::createNanoGrid(*s));
        auto e = FloatGrid::create(5.0f);
        e->setName("empty");
        h.push_back(nt::createNanoGrid(*e, StatsMode::All, CheckMode::Full));
        nanovdb::io::writeGrids(dir + "/modes.nvdb", h, nanovdb::io::Codec::ZIP);
    }
    {   // quantization of a narrow band
        std::vector<HandleT> h;
        auto g = bandGrid();
        h.push_back(nt::createNanoGrid(*g));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::Fp4>(*g, StatsMode::All, CheckMode::Full, true));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::Fp16>(*g, StatsMode::All, CheckMode::Full, false));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::FpN>(*g, StatsMode::All, CheckMode::Full, true));
        h.push_back(nt::createNanoGrid<FloatGrid, nanovdb::FpN>(*g, StatsMode::MinMax, CheckMode::Full, false,
                                                                0, nt::AbsDiff(0.001f)));
        nanovdb::io::writeGrids(dir + "/band.nvdb", h, nanovdb::io::Codec::ZIP);
    }
    {   // NanoVDB primitives, BLOSC compressed
        std::vector<HandleT> h;
        h.push_back(nt::createLevelSetSphere<float>(5.0, nanovdb::Vec3d(1.5, 2.0, -3.0), 0.5, 3.0,
                                                    nanovdb::Vec3d(0.0), "sphere"));
        h.push_back(nt::createLevelSetSphere<nanovdb::Fp8>(5.0, nanovdb::Vec3d(1.5, 2.0, -3.0), 0.5, 3.0,
                                                           nanovdb::Vec3d(0.0), "sphere_fp8"));
        h.push_back(nt::createLevelSetTorus<float>(6.0, 2.0, nanovdb::Vec3d(0.0), 0.5, 3.0,
                                                   nanovdb::Vec3d(0.25, 0.0, 0.0), "torus"));
        h.push_back(nt::createLevelSetBox<float>(10.0, 6.0, 4.0, nanovdb::Vec3d(1.0, 0.0, 0.0), 0.5, 3.0,
                                                 nanovdb::Vec3d(0.0), "box"));
        h.push_back(nt::createFogVolumeSphere<float>(5.0, nanovdb::Vec3d(-2.0, 0.0, 1.0), 0.25, 3.0,
                                                     nanovdb::Vec3d(0.0), "fog_sphere"));
        nanovdb::io::writeGrids(dir + "/primitives.nvdb", h, nanovdb::io::Codec::BLOSC);
    }
    {   // several grids in one segment
        std::vector<HandleT> h;
        h.push_back(nt::createNanoGrid(*floatGrid()));
        h.push_back(nt::createNanoGrid(*int32Grid()));
        h.push_back(nt::createNanoGrid(*vec3fGrid()));
        auto merged = nanovdb::mergeGrids(h);
        nanovdb::io::writeGrid(dir + "/multi.nvdb", merged, nanovdb::io::Codec::ZIP);
    }
    {   // uncompressed single grid
        nanovdb::io::writeGrid(dir + "/single_none.nvdb", nt::createNanoGrid(*singleVoxelGrid()),
                               nanovdb::io::Codec::NONE);
    }
    {   // raw grid buffer with two grids (no file header)
        std::vector<HandleT> h;
        h.push_back(nt::createNanoGrid(*rootTilesGrid("tiles_a")));
        h.push_back(nt::createNanoGrid(*rootTilesGrid("tiles_b"), StatsMode::All, CheckMode::Full));
        nanovdb::mergeGrids(h).write(dir + "/raw.nvdb");
    }
    samples(dir + "/samples.txt");
}

// ---------------------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------------------

template<typename BuildT>
static int validate(const HandleT& h, uint32_t n)
{
    auto* g = h.grid<BuildT>(n);
    return g ? int(nanovdb::tools::isValid(g, CheckMode::Full)) : -1;
}

static int validateAny(const HandleT& h, uint32_t n, nanovdb::GridType t)
{
    using GT = nanovdb::GridType;
    switch (t) {
    case GT::Float: return validate<float>(h, n);
    case GT::Double: return validate<double>(h, n);
    case GT::Int32: return validate<int32_t>(h, n);
    case GT::Int64: return validate<int64_t>(h, n);
    case GT::Vec3f: return validate<nanovdb::Vec3f>(h, n);
    case GT::Vec3d: return validate<nanovdb::Vec3d>(h, n);
    case GT::Boolean: return validate<bool>(h, n);
    case GT::Mask: return validate<nanovdb::ValueMask>(h, n);
    case GT::Fp4: return validate<nanovdb::Fp4>(h, n);
    case GT::Fp8: return validate<nanovdb::Fp8>(h, n);
    case GT::Fp16: return validate<nanovdb::Fp16>(h, n);
    case GT::FpN: return validate<nanovdb::FpN>(h, n);
    case GT::Index: return validate<nanovdb::ValueIndex>(h, n);
    case GT::OnIndex: return validate<nanovdb::ValueOnIndex>(h, n);
    default: return -1;
    }
}

static void check(const std::string& file)
{
    auto handles = nanovdb::io::readGrids(file);
    for (auto& h : handles) {
        for (uint32_t n = 0; n < h.gridCount(); ++n) {
            const nanovdb::GridData* d = h.gridData(n);
            char type[32], cls[32];
            const auto full = nanovdb::tools::evalChecksum(d, CheckMode::Full);
            const auto* tree = reinterpret_cast<const nanovdb::TreeData*>(d->treePtr());
            printf("%s|%s|%s|%llu|%llu|%u,%u,%u|%u,%u,%u|%016llx|%d|%d|%016llx\n", d->gridName(),
                   nanovdb::toStr(type, d->mGridType), nanovdb::toStr(cls, d->mGridClass),
                   (unsigned long long)d->mGridSize, (unsigned long long)tree->mVoxelCount, tree->mNodeCount[0],
                   tree->mNodeCount[1], tree->mNodeCount[2], tree->mTileCount[0], tree->mTileCount[1],
                   tree->mTileCount[2], (unsigned long long)d->mChecksum.full(),
                   int(nanovdb::tools::validateChecksum(d, CheckMode::Full)),
                   // the validator dereferences the first node of every level
                   tree->mNodeCount[0] ? validateAny(h, n, d->mGridType) : 2,
                   (unsigned long long)full.full());
        }
    }
}

int main(int argc, char** argv)
{
    openvdb::initialize();
    if (argc == 3 && std::string(argv[1]) == "gen") {
        gen(argv[2]);
    } else if (argc == 3 && std::string(argv[1]) == "check") {
        check(argv[2]);
    } else {
        fprintf(stderr, "usage: %s gen <dir> | check <file.nvdb>\n", argv[0]);
        return 1;
    }
    return 0;
}
