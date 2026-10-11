// C++ side of the Rust <-> C++ OpenVDB comparison benchmark.
//
// Every benchmark here has a twin with the same name, inputs and parameters in
// benches/compare.rs; see that file and benches/compare.sh. Each prints one line:
//
//   name <TAB> median_ms <TAB> min_ms <TAB> check
//
// Build (OpenVDB 10, TBB, Imath):
//   g++ -std=c++17 -O3 -DNDEBUG compare.cpp -o compare -lopenvdb -ltbb -lImath-3_1
// Usage:
//   ./compare [filter] [--reps N] [--threads N]

#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/math/Ray.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/PointIndexGrid.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/tools/VolumeToMesh.h>

#include <tbb/global_control.h>
#include <tbb/info.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace openvdb;

namespace {

// Shared parameters (keep in sync with compare.rs).
const float VOXEL = 0.004f; // sphere of radius 1 -> 250 voxels
const float HALF_WIDTH = 3.0f;

// SplitMix64, identical to the Rust side, so both use the same random inputs.
struct Rng {
    uint64_t s;
    uint64_t next()
    {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double unit() { return double(next() >> 11) * (1.0 / double(1ULL << 53)); }
    double rangeF(double lo, double hi) { return lo + (hi - lo) * unit(); }
    int rangeI(int lo, int hi) { return lo + int(next() % uint64_t(hi - lo)); }
    // Draw x, then y, then z (C++ leaves argument evaluation order unspecified).
    Coord coord(int lo, int hi) { int x = rangeI(lo, hi); int y = rangeI(lo, hi); int z = rangeI(lo, hi); return {x, y, z}; }
    Vec3d vec(double lo, double hi) { double x = rangeF(lo, hi); double y = rangeF(lo, hi); double z = rangeF(lo, hi); return {x, y, z}; }
};

template<typename T> inline void doNotOptimize(const T& v) { asm volatile("" : : "g"(&v) : "memory"); }

std::string fmt(const char* f, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

struct Ctx {
    std::string filter;
    int reps = 5;

    // Time `run` over `reps` repetitions, each on a fresh state from `setup` (untimed);
    // `check` summarizes the last state (untimed).
    template<typename Setup, typename Run, typename Check>
    void bench(const std::string& name, Setup setup, Run run, Check check) const
    {
        if (!filter.empty() && name.find(filter) == std::string::npos) return;
        using State = decltype(setup());
        using Result = decltype(run(std::declval<State&>()));
        std::vector<double> times;
        std::unique_ptr<State> state;
        std::unique_ptr<Result> result;
        for (int i = 0; i < reps; ++i) {
            state.reset(); result.reset();
            state = std::make_unique<State>(setup());
            auto t0 = std::chrono::steady_clock::now();
            result = std::make_unique<Result>(run(*state));
            auto t1 = std::chrono::steady_clock::now();
            doNotOptimize(*result);
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::string c = check(*state, *result);
        std::sort(times.begin(), times.end());
        std::printf("%s\t%.3f\t%.3f\t%s\n", name.c_str(), times[times.size() / 2], times[0], c.c_str());
        std::fflush(stdout);
    }
};

struct Unit {};

FloatGrid::Ptr sphere(const Vec3f& c) { return tools::createLevelSetSphere<FloatGrid>(1.0f, c, VOXEL, HALF_WIDTH); }

void torus(float R, float r, uint32_t nMajor, uint32_t nMinor, std::vector<Vec3s>& points, std::vector<Vec4I>& quads)
{
    const float tau = 6.283185307179586f;
    for (uint32_t i = 0; i < nMajor; ++i) {
        float u = float(i) / float(nMajor) * tau;
        for (uint32_t j = 0; j < nMinor; ++j) {
            float v = float(j) / float(nMinor) * tau;
            float w = R + r * std::cos(v);
            points.emplace_back(w * std::cos(u), r * std::sin(v), w * std::sin(u));
        }
    }
    for (uint32_t i = 0; i < nMajor; ++i) {
        for (uint32_t j = 0; j < nMinor; ++j) {
            uint32_t a = i * nMinor + j;
            uint32_t b = ((i + 1) % nMajor) * nMinor + j;
            uint32_t c = ((i + 1) % nMajor) * nMinor + (j + 1) % nMinor;
            uint32_t d = i * nMinor + (j + 1) % nMinor;
            quads.emplace_back(a, d, c, b);
        }
    }
}

double sumActive(const FloatGrid& g)
{
    double s = 0;
    for (auto it = g.cbeginValueOn(); it; ++it) s += double(*it);
    return s;
}

std::string count(const GridBase& g) { return std::to_string(g.activeVoxelCount()); }

struct ParticleList {
    std::vector<Vec3d> pos;
    std::vector<double> rad;
    using PosType = Vec3R;
    size_t size() const { return pos.size(); }
    void getPos(size_t n, Vec3R& xyz) const { xyz = pos[n]; }
    void getPosRad(size_t n, Vec3R& xyz, Real& r) const { xyz = pos[n]; r = rad[n]; }
};

} // namespace

int main(int argc, char** argv)
{
    Ctx ctx;
    int threads = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--reps") ctx.reps = std::atoi(argv[++i]);
        else if (a == "--threads") threads = std::atoi(argv[++i]);
        else ctx.filter = a;
    }
    std::unique_ptr<tbb::global_control> gc;
    if (threads > 0) gc = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, threads);
    openvdb::initialize();
    std::fprintf(stderr, "c++: %d threads, %d reps\n",
        int(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)), ctx.reps);

    FloatGrid::Ptr base = sphere(Vec3f(0.0f));
    auto none = [] { return Unit{}; };

    // ---- Tree construction and access ----
    ctx.bench("sphere_create", none, [](Unit&) { return sphere(Vec3f(0.0f)); },
        [](Unit&, FloatGrid::Ptr& g) { return count(*g); });
    ctx.bench("grid_copy", none, [&](Unit&) { return base->deepCopy(); },
        [](Unit&, FloatGrid::Ptr& g) { return count(*g); });
    ctx.bench("active_voxel_count", none, [&](Unit&) { return base->activeVoxelCount(); },
        [](Unit&, Index64 c) { return std::to_string(c); });
    ctx.bench("iterate_active", none, [&](Unit&) { return sumActive(*base); },
        [](Unit&, double s) { return fmt("%.6e", s); });

    Rng rng{1};
    std::vector<Coord> insertCoords;
    for (int i = 0; i < 2000000; ++i) insertCoords.push_back(rng.coord(-256, 256));
    ctx.bench("random_insert", [] { return FloatGrid::create(0.0f); },
        [&](FloatGrid::Ptr& g) {
            auto acc = g->getAccessor();
            for (size_t i = 0; i < insertCoords.size(); ++i) acc.setValue(insertCoords[i], float(i));
            return 0;
        },
        [](FloatGrid::Ptr& g, int) { return count(*g); });
    std::vector<Coord> readCoords;
    for (int i = 0; i < 4000000; ++i) readCoords.push_back(rng.coord(-260, 260));
    ctx.bench("random_read", none,
        [&](Unit&) {
            auto acc = base->getConstAccessor();
            double s = 0;
            for (const auto& c : readCoords) s += double(acc.getValue(c));
            return s;
        },
        [](Unit&, double s) { return fmt("%.6e", s); });
    ctx.bench("coherent_read", none,
        [&](Unit&) {
            auto acc = base->getConstAccessor();
            double s = 0;
            for (int i = -128; i < 128; ++i)
                for (int j = -128; j < 128; ++j)
                    for (int k = 0; k < 256; ++k) s += double(acc.getValue(Coord(i, j, k)));
            return s;
        },
        [](Unit&, double s) { return fmt("%.6e", s); });
    std::vector<Vec3d> samplePoints;
    for (int i = 0; i < 2000000; ++i) samplePoints.push_back(rng.vec(-1.05, 1.05));
    ctx.bench("box_sample", none,
        [&](Unit&) {
            tools::GridSampler<FloatGrid::ConstAccessor, tools::BoxSampler> s(base->getConstAccessor(), base->transform());
            double sum = 0;
            for (const auto& p : samplePoints) sum += double(s.wsSample(p));
            return sum;
        },
        [](Unit&, double s) { return fmt("%.6e", s); });

    // ---- I/O (in memory) ----
    FloatGrid::Ptr fog = base->deepCopy();
    fog->setGridClass(GRID_UNKNOWN); // so that zip is applied
    GridPtrVec fogVec{fog};
    const std::vector<std::pair<std::string, uint32_t>> comps = {
        {"none", io::COMPRESS_ACTIVE_MASK},
        {"zip", io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK},
        {"blosc", io::COMPRESS_BLOSC | io::COMPRESS_ACTIVE_MASK},
    };
    for (const auto& [label, comp] : comps) {
        auto write = [&, comp = comp] {
            std::ostringstream os(std::ios_base::binary);
            io::Stream s(os);
            s.setCompression(comp);
            s.write(fogVec);
            return os.str();
        };
        ctx.bench("write_" + label, none, [&](Unit&) { return write(); },
            [](Unit&, std::string& b) { return std::to_string(b.size()) + " bytes"; });
        const std::string bytes = write();
        ctx.bench("read_" + label, [&] { return std::make_shared<std::istringstream>(bytes, std::ios_base::binary); },
            [](std::shared_ptr<std::istringstream>& is) { io::Stream s(*is); return s.getGrids(); },
            [](std::shared_ptr<std::istringstream>&, GridPtrVecPtr& g) { return count(*(*g)[0]); });
    }

    // ---- Level set tools ----
    FloatGrid::Ptr other = sphere(Vec3f(0.5f, 0.0f, 0.0f));
    using Pair = std::pair<FloatGrid::Ptr, FloatGrid::Ptr>;
    ctx.bench("csg_union", [&] { return Pair(base->deepCopy(), other->deepCopy()); },
        [](Pair& p) { tools::csgUnion(*p.first, *p.second, true, true); return 0; },
        [](Pair& p, int) { return count(*p.first); });
    ctx.bench("csg_difference", [&] { return Pair(base->deepCopy(), other->deepCopy()); },
        [](Pair& p) { tools::csgDifference(*p.first, *p.second, true, true); return 0; },
        [](Pair& p, int) { return count(*p.first); });
    auto copy = [&] { return base->deepCopy(); };
    ctx.bench("dilate_3", copy,
        [](FloatGrid::Ptr& g) { tools::dilateActiveValues(g->tree(), 3, tools::NN_FACE, tools::IGNORE_TILES, true); return 0; },
        [](FloatGrid::Ptr& g, int) { return count(*g); });
    ctx.bench("erode_3", copy,
        [](FloatGrid::Ptr& g) { tools::erodeActiveValues(g->tree(), 3, tools::NN_FACE, tools::IGNORE_TILES, true); return 0; },
        [](FloatGrid::Ptr& g, int) { return count(*g); });
    ctx.bench("signed_flood_fill", copy,
        [](FloatGrid::Ptr& g) { tools::signedFloodFill(g->tree(), true); return 0; },
        [](FloatGrid::Ptr& g, int) { return count(*g); });
    ctx.bench("filter_gaussian", copy,
        [](FloatGrid::Ptr& g) { tools::Filter<FloatGrid> f(*g); f.gaussian(1, 1); return 0; },
        [](FloatGrid::Ptr& g, int) { return fmt("%.6e", sumActive(*g)); });
    ctx.bench("filter_median", copy,
        [](FloatGrid::Ptr& g) { tools::Filter<FloatGrid> f(*g); f.median(1, 1); return 0; },
        [](FloatGrid::Ptr& g, int) { return fmt("%.6e", sumActive(*g)); });
    ctx.bench("ls_filter_mean_curvature", copy,
        [](FloatGrid::Ptr& g) { tools::LevelSetFilter<FloatGrid> f(*g); f.meanCurvature(); return 0; },
        [](FloatGrid::Ptr& g, int) { return count(*g); });
    ctx.bench("gradient", none, [&](Unit&) { return tools::gradient(*base); },
        [](Unit&, Vec3SGrid::Ptr& g) { return count(*g); });
    ctx.bench("laplacian", none, [&](Unit&) { return tools::laplacian(*base); },
        [](Unit&, FloatGrid::Ptr& g) { return count(*g); });
    ctx.bench("mean_curvature", none, [&](Unit&) { return tools::meanCurvature(*base); },
        [](Unit&, FloatGrid::Ptr& g) { return count(*g); });
    ctx.bench("ls_measure", none,
        [&](Unit&) {
            tools::LevelSetMeasure<FloatGrid> m(*base);
            return std::make_pair(double(m.area(true)), double(m.volume(true)));
        },
        [](Unit&, std::pair<double, double>& av) { return fmt("%.5f %.5f", av.first, av.second); });
    using Mesh = std::tuple<std::vector<Vec3s>, std::vector<Vec3I>, std::vector<Vec4I>>;
    for (auto [label, adaptivity] : {std::make_pair("volume_to_mesh", 0.0), std::make_pair("volume_to_mesh_adaptive", 0.5)}) {
        ctx.bench(label, none,
            [&, adaptivity = adaptivity](Unit&) {
                Mesh m;
                tools::volumeToMesh(*base, std::get<0>(m), std::get<1>(m), std::get<2>(m), 0.0, adaptivity, true);
                return m;
            },
            [](Unit&, Mesh& m) {
                return fmt("%zu pts %zu tris %zu quads", std::get<0>(m).size(), std::get<1>(m).size(), std::get<2>(m).size());
            });
    }
    std::vector<Vec3s> tpoints;
    std::vector<Vec4I> tquads;
    torus(1.0f, 0.35f, 512, 256, tpoints, tquads);
    ctx.bench("mesh_to_level_set", none,
        [&](Unit&) {
            auto xform = math::Transform::createLinearTransform(VOXEL);
            return tools::meshToLevelSet<FloatGrid>(*xform, tpoints, std::vector<Vec3I>(), tquads, HALF_WIDTH);
        },
        [](Unit&, FloatGrid::Ptr& g) { return count(*g); });
    std::vector<math::Ray<double>> rays;
    for (int i = 0; i < 500000; ++i) {
        double ex = rng.rangeF(-1.0, 1.0);
        double ey = rng.rangeF(-1.0, 1.0);
        Vec3d eye(ex, ey, 3.0);
        double tx = rng.rangeF(-0.5, 0.5);
        double ty = rng.rangeF(-0.5, 0.5);
        Vec3d target(tx, ty, 0.0);
        rays.emplace_back(eye, (target - eye).unit());
    }
    ctx.bench("ray_intersect", none,
        [&](Unit&) {
            tools::LevelSetRayIntersector<FloatGrid> lsri(*base);
            size_t hits = 0;
            double sum = 0;
            for (const auto& r : rays) {
                Vec3d xyz;
                double t;
                if (lsri.intersectsWS(r, xyz, t)) { ++hits; sum += t; }
            }
            return std::make_pair(hits, sum);
        },
        [](Unit&, std::pair<size_t, double>& r) { return fmt("%zu hits %.4e", r.first, r.second); });

    // ---- Particles and points ----
    ParticleList particles;
    for (int i = 0; i < 50000; ++i) {
        particles.pos.push_back(rng.vec(-1.0, 1.0));
        particles.rad.push_back(0.01);
    }
    ctx.bench("particles_to_sdf", [] { return openvdb::createLevelSet<FloatGrid>(VOXEL, HALF_WIDTH); },
        [&](FloatGrid::Ptr& g) { tools::particlesToSdf(particles, *g); return 0; },
        [](FloatGrid::Ptr& g, int) { return count(*g); });
    std::vector<Vec3f> positions;
    for (int i = 0; i < 2000000; ++i) {
        float x = float(rng.rangeF(-1.0, 1.0));
        float y = float(rng.rangeF(-1.0, 1.0));
        float z = float(rng.rangeF(-1.0, 1.0));
        positions.emplace_back(x, y, z);
    }
    auto pxform = math::Transform::createLinearTransform(0.02);
    ctx.bench("point_index_grid", none,
        [&](Unit&) {
            points::PointAttributeVector<Vec3f> wrapper(positions);
            return tools::createPointIndexGrid<tools::PointIndexGrid>(wrapper, *pxform);
        },
        [](Unit&, tools::PointIndexGrid::Ptr& g) { return std::to_string(g->tree().leafCount()) + " leaves"; });
    ctx.bench("point_data_grid", none,
        [&](Unit&) { return points::createPointDataGrid<points::NullCodec, points::PointDataGrid>(positions, *pxform); },
        [](Unit&, points::PointDataGrid::Ptr& g) { return std::to_string(g->tree().leafCount()) + " leaves"; });
    return 0;
}
