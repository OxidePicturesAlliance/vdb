// Reference-file generator and checker for the Rust `vdb` crate's I/O tests.
//
// Usage:
//   vdb_ref gen <outdir>     write the reference .vdb files into <outdir>
//   vdb_ref dump <file.vdb>  print a canonical, order-independent description of a file
//   vdb_ref probe <file.vdb> <grid> <x> <y> <z>  print a voxel's value and active state
//
// The `dump` format is mirrored by tests/common/mod.rs (`dump_file`), so that the
// Rust test-suite can compare its own reading of a file against what the C++ library sees.
// See README.md in this directory for build instructions.

#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/tools/LevelSetSphere.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace openvdb;

namespace {

// ---------------------------------------------------------------------------
// Hashing (must match tests/common/mod.rs)

inline uint64_t mix(uint64_t h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

inline uint64_t coordHash(const Coord& c)
{
    uint64_t h = mix(uint64_t(uint32_t(c.x())));
    h = mix(h ^ uint64_t(uint32_t(c.y())));
    h = mix(h ^ uint64_t(uint32_t(c.z())));
    return h;
}

inline uint64_t f32bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
inline uint64_t f64bits(double v) { uint64_t b; std::memcpy(&b, &v, 8); return b; }

inline uint64_t valueBits(float v) { return f32bits(v); }
inline uint64_t valueBits(double v) { return f64bits(v); }
inline uint64_t valueBits(int32_t v) { return uint64_t(uint32_t(v)); }
inline uint64_t valueBits(int64_t v) { return uint64_t(v); }
inline uint64_t valueBits(bool v) { return v ? 1 : 0; }
template<typename T> inline uint64_t valueBits(const math::Vec3<T>& v)
{
    return mix(mix(mix(valueBits(v[0])) ^ valueBits(v[1])) ^ valueBits(v[2]));
}

std::string hex(uint64_t v)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
    return buf;
}

// ---------------------------------------------------------------------------
// Metadata printing

std::string metaString(const Metadata& m)
{
    const std::string t = m.typeName();
    std::ostringstream os;
    if (t == "bool") os << (static_cast<const BoolMetadata&>(m).value() ? 1 : 0);
    else if (t == "int32") os << static_cast<const Int32Metadata&>(m).value();
    else if (t == "int64") os << static_cast<const Int64Metadata&>(m).value();
    else if (t == "float") os << hex(f32bits(static_cast<const FloatMetadata&>(m).value()));
    else if (t == "double") os << hex(f64bits(static_cast<const DoubleMetadata&>(m).value()));
    else if (t == "string") os << '"' << static_cast<const StringMetadata&>(m).value() << '"';
    else if (t == "vec2i") { auto v = static_cast<const Vec2IMetadata&>(m).value(); os << v[0] << ' ' << v[1]; }
    else if (t == "vec2s") { auto v = static_cast<const Vec2SMetadata&>(m).value(); os << hex(f32bits(v[0])) << ' ' << hex(f32bits(v[1])); }
    else if (t == "vec2d") { auto v = static_cast<const Vec2DMetadata&>(m).value(); os << hex(f64bits(v[0])) << ' ' << hex(f64bits(v[1])); }
    else if (t == "vec3i") { auto v = static_cast<const Vec3IMetadata&>(m).value(); os << v[0] << ' ' << v[1] << ' ' << v[2]; }
    else if (t == "vec3s") { auto v = static_cast<const Vec3SMetadata&>(m).value(); for (int i = 0; i < 3; ++i) os << (i ? " " : "") << hex(f32bits(v[i])); }
    else if (t == "vec3d") { auto v = static_cast<const Vec3DMetadata&>(m).value(); for (int i = 0; i < 3; ++i) os << (i ? " " : "") << hex(f64bits(v[i])); }
    else if (t == "vec4i") { auto v = static_cast<const Vec4IMetadata&>(m).value(); for (int i = 0; i < 4; ++i) os << (i ? " " : "") << v[i]; }
    else if (t == "vec4s") { auto v = static_cast<const Vec4SMetadata&>(m).value(); for (int i = 0; i < 4; ++i) os << (i ? " " : "") << hex(f32bits(v[i])); }
    else if (t == "vec4d") { auto v = static_cast<const Vec4DMetadata&>(m).value(); for (int i = 0; i < 4; ++i) os << (i ? " " : "") << hex(f64bits(v[i])); }
    else if (t == "mat4s") { auto v = static_cast<const Mat4SMetadata&>(m).value(); for (int i = 0; i < 16; ++i) os << (i ? " " : "") << hex(f32bits(v.asPointer()[i])); }
    else if (t == "mat4d") { auto v = static_cast<const Mat4DMetadata&>(m).value(); for (int i = 0; i < 16; ++i) os << (i ? " " : "") << hex(f64bits(v.asPointer()[i])); }
    else os << "<" << m.size() << " bytes>";
    return os.str();
}

void dumpMeta(std::ostream& os, const MetaMap& meta, const std::string& indent)
{
    for (auto it = meta.beginMeta(); it != meta.endMeta(); ++it) {
        os << indent << "meta " << it->first << " " << it->second->typeName() << " "
           << metaString(*it->second) << "\n";
    }
}

// ---------------------------------------------------------------------------
// Grid dumping

template<typename GridT>
void dumpTree(std::ostream& os, const GridT& grid, bool isMask)
{
    const auto& tree = grid.tree();
    os << "  background " << hex(valueBits(tree.background())) << "\n";
    os << "  active_voxels " << tree.activeVoxelCount() << "\n";
    os << "  active_tiles " << tree.activeTileCount() << "\n";
    os << "  leaf_count " << tree.leafCount() << "\n";
    CoordBBox bbox;
    if (tree.evalActiveVoxelBoundingBox(bbox)) {
        os << "  bbox " << bbox.min().x() << " " << bbox.min().y() << " " << bbox.min().z() << " "
           << bbox.max().x() << " " << bbox.max().y() << " " << bbox.max().z() << "\n";
    } else {
        os << "  bbox empty\n";
    }
    uint64_t activeHash = 0, leafHash = 0, tileHash = 0;
    for (auto leaf = tree.cbeginLeaf(); leaf; ++leaf) {
        for (Index i = 0; i < GridT::TreeType::LeafNodeType::SIZE; ++i) {
            const Coord c = leaf->offsetToGlobalCoord(i);
            const bool on = leaf->isValueOn(i);
            // For mask grids the value of a voxel is its active state.
            const uint64_t v = isMask ? valueBits(on) : valueBits(leaf->getValue(i));
            const uint64_t h = mix(coordHash(c) ^ v);
            if (on) activeHash += h;
            leafHash += mix(h ^ (on ? 0x5555ULL : 0xaaaaULL));
        }
    }
    for (auto it = tree.cbeginValueAll(); it; ++it) {
        if (it.isVoxelValue()) continue;
        CoordBBox b;
        it.getBoundingBox(b);
        const uint64_t dim = uint64_t(b.dim().x());
        const uint64_t h = mix(coordHash(b.min()) ^ valueBits(*it));
        tileHash += mix(h ^ mix(dim * 2 + (it.isValueOn() ? 1 : 0)));
    }
    os << "  active_hash " << hex(activeHash) << "\n";
    os << "  leaf_hash " << hex(leafHash) << "\n";
    os << "  tile_hash " << hex(tileHash) << "\n";
}

void dumpGrid(std::ostream& os, const GridBase& grid)
{
    if (grid.isType<FloatGrid>()) dumpTree(os, static_cast<const FloatGrid&>(grid), false);
    else if (grid.isType<DoubleGrid>()) dumpTree(os, static_cast<const DoubleGrid&>(grid), false);
    else if (grid.isType<Int32Grid>()) dumpTree(os, static_cast<const Int32Grid&>(grid), false);
    else if (grid.isType<Int64Grid>()) dumpTree(os, static_cast<const Int64Grid&>(grid), false);
    else if (grid.isType<BoolGrid>()) dumpTree(os, static_cast<const BoolGrid&>(grid), false);
    else if (grid.isType<MaskGrid>()) dumpTree(os, static_cast<const MaskGrid&>(grid), true);
    else if (grid.isType<Vec3SGrid>()) dumpTree(os, static_cast<const Vec3SGrid&>(grid), false);
    else if (grid.isType<Vec3DGrid>()) dumpTree(os, static_cast<const Vec3DGrid&>(grid), false);
    else if (grid.isType<Vec3IGrid>()) dumpTree(os, static_cast<const Vec3IGrid&>(grid), false);
    else { os << "  unsupported\n"; return; }

    const math::Mat4d m = grid.transform().baseMap()->getAffineMap()->getMat4();
    os << "  transform";
    for (int i = 0; i < 16; ++i) os << " " << hex(f64bits(m.asPointer()[i]));
    os << "\n";
}

int dump(const std::string& path)
{
    io::File file(path);
    file.open();
    std::cout << "file_version " << file.fileVersion() << "\n";
    dumpMeta(std::cout, *file.getMetadata(), "");
    GridPtrVecPtr grids = file.getGrids();
    for (const auto& grid : *grids) {
        std::cout << "grid " << grid->getName() << "\n";
        std::cout << "  type " << grid->type() << "\n";
        dumpMeta(std::cout, *grid, "  ");
        dumpGrid(std::cout, *grid);
    }
    file.close();
    return 0;
}

// ---------------------------------------------------------------------------
// Reference file generation

void addFileMeta(MetaMap& meta)
{
    meta.insertMeta("creator_tool", StringMetadata("vdb_ref"));
    meta.insertMeta("answer", Int32Metadata(42));
    meta.insertMeta("big", Int64Metadata(int64_t(1) << 40));
    meta.insertMeta("pi_f", FloatMetadata(3.14159f));
    meta.insertMeta("e_d", DoubleMetadata(2.718281828459045));
    meta.insertMeta("flag", BoolMetadata(true));
    meta.insertMeta("origin", Vec3DMetadata(Vec3d(1.5, -2.0, 0.25)));
    meta.insertMeta("res", Vec3IMetadata(Vec3i(640, 480, 1)));
    meta.insertMeta("dir", Vec3SMetadata(Vec3s(0.0f, 1.0f, -0.5f)));
    math::Mat4d m = math::Mat4d::identity();
    m.setTranslation(Vec3d(1, 2, 3));
    meta.insertMeta("xform", Mat4DMetadata(m));
}

FloatGrid::Ptr makeSphere(const std::string& name)
{
    FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(
        /*radius=*/5.0f, /*center=*/Vec3f(1.5f, -2.0f, 0.25f), /*voxelSize=*/0.5f, /*halfWidth=*/3.0f);
    grid->setName(name);
    grid->setCreator("vdb_ref");
    return grid;
}

FloatGrid::Ptr makeMaskCompGrid(const std::string& name)
{
    // One leaf per mask-compression scheme; background 2.
    FloatGrid::Ptr grid = FloatGrid::create(2.0f);
    grid->setName(name);
    auto& tree = grid->tree();
    auto leafAt = [&](int i) { return tree.touchLeaf(Coord(8 * i, 0, 0)); };
    const float bg = 2.0f;
    // 0: inactive values all +background
    { auto* l = leafAt(0); for (Index n = 0; n < 512; n += 7) l->setValueOn(n, float(n) * 0.25f); }
    // 1: inactive values all -background
    { auto* l = leafAt(1); for (Index n = 0; n < 512; ++n) l->setValueOff(n, -bg);
      for (Index n = 3; n < 512; n += 11) l->setValueOn(n, float(n)); }
    // 2: one non-background inactive value
    { auto* l = leafAt(2); for (Index n = 0; n < 512; ++n) l->setValueOff(n, 5.0f);
      for (Index n = 1; n < 512; n += 13) l->setValueOn(n, -float(n)); }
    // 3: -background and +background
    { auto* l = leafAt(3); for (Index n = 0; n < 512; ++n) l->setValueOff(n, (n & 4) ? bg : -bg);
      for (Index n = 0; n < 512; n += 17) l->setValueOn(n, 0.5f); }
    // 4: background and one other value (other value first)
    { auto* l = leafAt(4); for (Index n = 0; n < 512; ++n) l->setValueOff(n, (n & 8) ? bg : 7.0f);
      for (Index n = 0; n < 512; n += 19) l->setValueOn(n, 1.5f); }
    // 5: two non-background inactive values
    { auto* l = leafAt(5); for (Index n = 0; n < 512; ++n) l->setValueOff(n, (n & 2) ? 3.0f : 4.0f);
      for (Index n = 0; n < 512; n += 23) l->setValueOn(n, -1.0f); }
    // 6: more than two inactive values
    { auto* l = leafAt(6); for (Index n = 0; n < 512; ++n) l->setValueOff(n, float(n % 5));
      for (Index n = 0; n < 512; n += 29) l->setValueOn(n, 9.0f); }
    // 7: +background first, then -background (swap path)
    { auto* l = leafAt(7); for (Index n = 0; n < 512; ++n) l->setValueOff(n, (n < 100) ? bg : -bg);
      for (Index n = 0; n < 512; n += 31) l->setValueOn(n, 0.0f); }
    // 8: background first, then another value (swap path)
    { auto* l = leafAt(8); for (Index n = 0; n < 512; ++n) l->setValueOff(n, (n < 200) ? bg : -7.5f);
      for (Index n = 0; n < 512; n += 37) l->setValueOn(n, 2.25f); }
    // 9: fully inactive leaf with background values only
    { leafAt(9); }
    // Internal-node tiles with several distinct inactive values, and an active tile.
    tree.addTile(1, Coord(128, 0, 0), 9.0f, false);
    tree.addTile(1, Coord(256, 0, 0), -2.0f, false);
    tree.addTile(1, Coord(384, 0, 0), 11.0f, true);
    tree.addTile(2, Coord(0, 0, 512), 6.0f, false);
    return grid;
}

void genSpheres(const std::string& dir)
{
    MetaMap meta;
    addFileMeta(meta);
    {
        GridPtrVec grids{makeSphere("sphere")};
        io::File f(dir + "/sphere.vdb");
        f.setCompression(io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK);
        f.write(grids, meta);
    }
    {
        FloatGrid::Ptr s = makeSphere("sphere_half");
        s->setSaveFloatAsHalf(true);
        FloatGrid::Ptr mc = makeMaskCompGrid("maskcomp_half");
        mc->setSaveFloatAsHalf(true);
        Vec3SGrid::Ptr v = Vec3SGrid::create(Vec3s(0.5f, 0.0f, -0.5f));
        v->setName("vel_half");
        v->setSaveFloatAsHalf(true);
        for (int i = 0; i < 300; ++i) {
            v->tree().setValueOn(Coord(i % 17, i / 17, -i % 5),
                Vec3s(std::sin(i * 0.1f), std::cos(i * 0.2f), i * 0.01f));
        }
        GridPtrVec grids{s, mc, v};
        io::File f(dir + "/sphere_half.vdb");
        f.setCompression(io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK);
        f.write(grids, meta);
    }
    {
        GridPtrVec grids{makeSphere("sphere_blosc")};
        io::File f(dir + "/sphere_blosc.vdb");
        f.setCompression(io::COMPRESS_BLOSC | io::COMPRESS_ACTIVE_MASK);
        f.write(grids, meta);
    }
}

FloatGrid::Ptr makeFog()
{
    FloatGrid::Ptr grid = FloatGrid::create(0.0f);
    grid->setName("fog");
    grid->setGridClass(GRID_FOG_VOLUME);
    grid->setTransform(math::Transform::createLinearTransform(0.1));
    grid->transform().postTranslate(Vec3d(10.0, -5.0, 2.5));
    auto& tree = grid->tree();
    // Root-level active tiles (4096^3 each).
    tree.fill(CoordBBox(Coord(4096, 0, 0), Coord(3 * 4096 - 1, 4095, 4095)), 1.0f, true);
    // A root-level inactive non-background tile.
    tree.fill(CoordBBox(Coord(-4096, -4096, -4096), Coord(-1, -1, -1)), 0.25f, false);
    // Internal tiles at both levels plus partial leaves.
    tree.fill(CoordBBox(Coord(-264, 8, 0), Coord(-1, 263, 135)), 0.5f, true);
    // A few partially covered leaves.
    tree.fill(CoordBBox(Coord(-300, 1, 1), Coord(-290, 5, 6)), 0.75f, true);
    // A dense region of varying values.
    for (int x = 0; x < 40; ++x) for (int y = 0; y < 30; ++y) for (int z = 0; z < 20; ++z) {
        const float d = 0.1f + 0.9f * std::fabs(std::sin(x * 0.3f) * std::cos(y * 0.2f + z * 0.1f));
        if ((x + y + z) % 3) tree.setValue(Coord(x, y, z), d);
    }
    return grid;
}

void genFog(const std::string& dir)
{
    MetaMap meta;
    meta.insertMeta("description", StringMetadata("fog volume with tiles"));
    GridPtrVec grids{makeFog()};
    io::File f(dir + "/fog_tiles.vdb");
    f.setCompression(io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK);
    f.write(grids, meta);
}

GridPtrVec makeTypedGrids()
{
    GridPtrVec grids;

    Int32Grid::Ptr ints = Int32Grid::create(-1);
    ints->setName("ints");
    for (int i = 0; i < 400; ++i) {
        const Coord c(i % 23 + 11, (i / 23) % 19 + 300, i / 437 + 50);
        if (i % 4) ints->tree().setValueOn(c, i * 7 - 3000);
        else ints->tree().setValueOff(c, i % 3 == 0 ? 5 : -6);
    }
    ints->tree().fill(CoordBBox(Coord(1000, 1000, 1000), Coord(1127, 1127, 1127)), 77, true);
    ints->insertMeta("note", StringMetadata("integers"));
    ints->insertMeta("scale2", Vec2SMetadata(Vec2s(0.5f, 2.0f)));
    ints->insertMeta("ivec2", Vec2IMetadata(Vec2i(-3, 4)));
    ints->insertMeta("dvec4", Vec4DMetadata(Vec4d(1, 2, 3, 4)));
    ints->insertMeta("smat", Mat4SMetadata(math::Mat4s::identity()));
    grids.push_back(ints);

    // An instance sharing the int tree, with a different transform.
    Int32Grid::Ptr inst = Int32Grid::create(ints->treePtr());
    inst->setName("ints_instance");
    inst->setTransform(math::Transform::createLinearTransform(2.0));
    grids.push_back(inst);

    DoubleGrid::Ptr dbl = DoubleGrid::create(0.125);
    dbl->setName("doubles");
    {
        math::Mat4d m = math::Mat4d::identity();
        m.preScale(Vec3d(0.5, 0.5, 0.5));
        m.postRotate(math::Y_AXIS, 0.3);
        m.postTranslate(Vec3d(1.0, 2.0, 3.0));
        dbl->setTransform(math::Transform::createLinearTransform(m));
    }
    for (int i = 0; i < 120; ++i) dbl->tree().setValueOn(Coord(i, 600 - i, i * 2), 1.0 / (i + 1));
    grids.push_back(dbl);

    Int64Grid::Ptr longs = Int64Grid::create(int64_t(-5));
    longs->setName("longs");
    for (int i = 0; i < 200; ++i) longs->tree().setValueOn(Coord(i * 3, 7, -i), (int64_t(1) << 40) + i);
    grids.push_back(longs);

    BoolGrid::Ptr flags = BoolGrid::create(false);
    flags->setName("flags");
    for (int i = 0; i < 400; ++i) {
        const Coord c(i % 9, i % 13, i % 31);
        if (i % 3 == 0) flags->tree().setValueOn(c, true);
        else if (i % 3 == 1) flags->tree().setValueOn(c, false);
        else flags->tree().setValueOff(c, true);
    }
    flags->tree().fill(CoordBBox(Coord(256, 256, 256), Coord(383, 383, 383)), true, true);
    grids.push_back(flags);

    MaskGrid::Ptr mask = MaskGrid::create();
    mask->setName("mask");
    for (int i = 0; i < 100; ++i) mask->tree().setValueOn(Coord(i * 5 % 97, i, 2 * i));
    mask->tree().fill(CoordBBox(Coord(512, 512, 512), Coord(512 + 255, 512 + 127, 512 + 127)), true, true);
    grids.push_back(mask);

    Vec3SGrid::Ptr vel = Vec3SGrid::create(Vec3s(0.0f));
    vel->setName("velocity");
    vel->setVectorType(VEC_CONTRAVARIANT_RELATIVE);
    vel->setTransform(math::Transform::Ptr(new math::Transform(
        math::MapBase::Ptr(new math::ScaleTranslateMap(Vec3d(0.5, 1.0, 2.0), Vec3d(-1.0, 0.0, 4.0))))));
    for (int i = 0; i < 100; ++i) {
        vel->tree().setValueOn(Coord(i, i % 7, i % 11), Vec3s(float(i), -float(i) * 0.5f, 1.0f / (i + 1)));
    }
    grids.push_back(vel);

    Vec3DGrid::Ptr veld = Vec3DGrid::create(Vec3d(1.0, 2.0, 3.0));
    veld->setName("vel_d");
    veld->transform().postTranslate(Vec3d(0.5, 0.25, -0.125));
    for (int i = 0; i < 100; ++i) veld->tree().setValueOn(Coord(100 - i, i, 3), Vec3d(i * 0.1, i * 0.2, i * 0.3));
    grids.push_back(veld);

    Vec3IGrid::Ptr vi = Vec3IGrid::create(Vec3i(0, 0, 0));
    vi->setName("indices");
    vi->setTransform(math::Transform::Ptr(new math::Transform(
        math::MapBase::Ptr(new math::ScaleMap(Vec3d(1.0, 2.0, 3.0))))));
    for (int i = 0; i < 100; ++i) vi->tree().setValueOn(Coord(i, 2 * i, 3 * i), Vec3i(i, -i, i * i));
    grids.push_back(vi);

    grids.push_back(makeMaskCompGrid("maskcomp"));

    FloatGrid::Ptr dup1 = FloatGrid::create(0.0f);
    dup1->setName("dup");
    dup1->tree().setValueOn(Coord(1, 2, 3), 1.0f);
    FloatGrid::Ptr dup2 = FloatGrid::create(0.0f);
    dup2->setName("dup");
    dup2->tree().setValueOn(Coord(4, 5, 6), 2.0f);
    grids.push_back(dup1);
    grids.push_back(dup2);

    FloatGrid::Ptr empty = FloatGrid::create(3.5f);
    empty->setName("empty");
    grids.push_back(empty);

    return grids;
}

void genTypes(const std::string& dir)
{
    MetaMap meta;
    addFileMeta(meta);
    const GridPtrVec grids = makeTypedGrids();
    const std::pair<const char*, uint32_t> variants[] = {
        {"types_zip.vdb", io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK},
        {"types_mask.vdb", io::COMPRESS_ACTIVE_MASK},
        {"types_blosc.vdb", io::COMPRESS_BLOSC | io::COMPRESS_ACTIVE_MASK},
        {"types_zip_only.vdb", io::COMPRESS_ZIP},
    };
    for (const auto& v : variants) {
        io::File f(dir + "/" + v.first);
        f.setCompression(v.second);
        f.write(grids, meta);
    }
    // Uncompressed internal nodes are large (128KB of values per 32^3 node), so the
    // COMPRESS_NONE variant only holds a few grids.
    {
        GridPtrVec small;
        for (const auto& g : grids) {
            if (g->getName() == "maskcomp" || g->getName() == "flags") {
                small.push_back(g);
            }
        }
        io::File f(dir + "/types_none.vdb");
        f.setCompression(io::COMPRESS_NONE);
        f.write(small, meta);
    }
}

void genPoints(const std::string& dir)
{
    std::vector<Vec3f> positions{{0, 1, 0}, {1.5f, 3.5f, 1}, {-1, 6, -2}, {1.1f, 1.25f, 0.06f}};
    math::Transform::Ptr xform = math::Transform::createLinearTransform(0.5);
    points::PointDataGrid::Ptr pts =
        points::createPointDataGrid<points::NullCodec, points::PointDataGrid>(positions, *xform);
    pts->setName("points");
    FloatGrid::Ptr after = FloatGrid::create(0.0f);
    after->setName("after_points");
    after->tree().setValueOn(Coord(3, 2, 1), 42.0f);
    GridPtrVec grids{pts, after};
    io::File f(dir + "/points.vdb");
    f.write(grids);
}

void genStream(const std::string& dir)
{
    GridPtrVec grids;
    FloatGrid::Ptr a = FloatGrid::create(0.0f);
    a->setName("a");
    for (int i = 0; i < 64; ++i) a->tree().setValueOn(Coord(i, i, i), float(i));
    Int32Grid::Ptr b = Int32Grid::create(0);
    b->setName("b");
    b->tree().setValueOn(Coord(-5, 5, -5), 12345);
    grids.push_back(a);
    grids.push_back(b);
    MetaMap meta;
    meta.insertMeta("streamed", BoolMetadata(true));
    std::ofstream os(dir + "/stream.vdb", std::ios_base::binary);
    io::Stream(os).write(grids, meta);
}

template<typename GridT>
bool probeTyped(const GridBase::Ptr& grid, const Coord& c)
{
    auto typed = GridBase::grid<GridT>(grid);
    if (!typed) return false;
    const auto acc = typed->getConstAccessor();
    std::ostringstream os;
    os << std::setprecision(9) << acc.getValue(c);
    std::cout << os.str() << " " << (acc.isValueOn(c) ? "on" : "off") << "\n";
    return true;
}

int probe(const std::string& path, const std::string& name, const Coord& c)
{
    io::File file(path);
    file.open();
    GridBase::Ptr grid = file.readGrid(name);
    if (probeTyped<FloatGrid>(grid, c) || probeTyped<DoubleGrid>(grid, c) ||
        probeTyped<Int32Grid>(grid, c) || probeTyped<Int64Grid>(grid, c) ||
        probeTyped<BoolGrid>(grid, c) || probeTyped<MaskGrid>(grid, c) ||
        probeTyped<Vec3SGrid>(grid, c) || probeTyped<Vec3DGrid>(grid, c) ||
        probeTyped<Vec3IGrid>(grid, c)) return 0;
    std::cerr << "unsupported grid type\n";
    return 1;
}

} // namespace

int main(int argc, char** argv)
{
    openvdb::initialize();
    if (argc == 3 && std::string(argv[1]) == "gen") {
        const std::string dir = argv[2];
        genSpheres(dir);
        genFog(dir);
        genTypes(dir);
        genPoints(dir);
        genStream(dir);
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "dump") {
        try {
            return dump(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "error: " << e.what() << "\n";
            return 1;
        }
    }
    if (argc == 7 && std::string(argv[1]) == "probe") {
        return probe(argv[2], argv[3], Coord(std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6])));
    }
    std::cerr << "usage: vdb_ref gen <outdir> | vdb_ref dump <file.vdb> | vdb_ref probe <file.vdb> <grid> <x> <y> <z>\n";
    return 2;
}
