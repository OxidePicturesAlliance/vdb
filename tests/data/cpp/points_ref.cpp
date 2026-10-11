// Reference-file generator and checker for the Rust `vdb` crate's point data tests.
//
// Usage:
//   points_ref gen <outdir>     write the reference point .vdb files into <outdir>
//   points_ref dump <file.vdb>  print a canonical description of the point data grids
//
// The `dump` format is mirrored by tests/points.rs (`dump_points_file`), so that the Rust
// test-suite can compare its own reading of a file against what the C++ library sees.
// See README.md in this directory for build instructions.

#include <openvdb/openvdb.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointGroup.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace openvdb;
using namespace openvdb::points;

namespace {

inline uint64_t f32bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
inline uint64_t f64bits(double v) { uint64_t b; std::memcpy(&b, &v, 8); return b; }

std::string hex(uint64_t v)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
    return buf;
}

// FNV-1a over raw bytes (must match tests/points.rs).
uint64_t fnv(const void* data, size_t n, uint64_t h = 0xcbf29ce484222325ULL)
{
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

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
    else if (t == "vec3i") { auto v = static_cast<const Vec3IMetadata&>(m).value(); os << v[0] << ' ' << v[1] << ' ' << v[2]; }
    else if (t == "vec3s") { auto v = static_cast<const Vec3SMetadata&>(m).value(); for (int i = 0; i < 3; ++i) os << (i ? " " : "") << hex(f32bits(v[i])); }
    else if (t == "vec3d") { auto v = static_cast<const Vec3DMetadata&>(m).value(); for (int i = 0; i < 3; ++i) os << (i ? " " : "") << hex(f64bits(v[i])); }
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

void dumpPointGrid(std::ostream& os, const PointDataGrid& grid)
{
    const auto& tree = grid.tree();
    const math::Mat4d m = grid.transform().baseMap()->getAffineMap()->getMat4();
    os << "  transform";
    for (int i = 0; i < 16; ++i) os << " " << hex(f64bits(m.asPointer()[i]));
    os << "\n";
    os << "  leaf_count " << tree.leafCount() << "\n";
    os << "  active_voxels " << tree.activeVoxelCount() << "\n";
    os << "  points " << pointCount(tree) << "\n";
    auto leafIter = tree.cbeginLeaf();
    if (!leafIter) return;
    const auto& descr = leafIter->attributeSet().descriptor();
    // attributes by position
    std::vector<std::string> names(descr.size());
    for (const auto& np : descr.map()) names[np.second] = np.first;
    for (size_t i = 0; i < descr.size(); ++i) {
        os << "  attribute " << i << " " << names[i] << " " << descr.type(i).first << " "
           << descr.type(i).second << "\n";
    }
    for (const auto& g : descr.groupMap()) os << "  group " << g.first << " " << g.second << "\n";
    dumpMeta(os, descr.getMetadata(), "  descriptor_");
    for (; leafIter; ++leafIter) {
        const auto& leaf = *leafIter;
        const Coord o = leaf.origin();
        uint64_t maskHash = 0xcbf29ce484222325ULL;
        for (Index w = 0; w < 8; ++w) {
            uint64_t word = 0;
            for (Index b = 0; b < 64; ++b) {
                if (leaf.isValueOn(w * 64 + b)) word |= (uint64_t(1) << b);
            }
            maskHash = fnv(&word, 8, maskHash);
        }
        uint64_t offsetHash = 0xcbf29ce484222325ULL;
        for (Index i = 0; i < 512; ++i) {
            const uint32_t v = leaf.getValue(i);
            offsetHash = fnv(&v, 4, offsetHash);
        }
        os << "  leaf " << o.x() << " " << o.y() << " " << o.z() << " points " << leaf.pointCount()
           << " mask " << hex(maskHash) << " offsets " << hex(offsetHash) << "\n";
        const auto& set = leaf.attributeSet();
        for (size_t i = 0; i < set.size(); ++i) {
            const AttributeArray& a = *set.getConst(i);
            a.loadData();
            const size_t bytes = (a.isUniform() ? 1 : a.dataSize()) * a.storageTypeSize();
            os << "    array " << i << " size " << a.size() << " stride " << a.stride() << " data_size "
               << a.dataSize() << " flags " << int(a.flags()) << " uniform " << (a.isUniform() ? 1 : 0)
               << " storage " << hex(fnv(a.constDataAsByteArray(), bytes)) << "\n";
        }
    }
}

int dump(const std::string& path)
{
    io::File file(path);
    file.open(/*delayLoad=*/false);
    GridPtrVecPtr grids = file.getGrids();
    for (const auto& grid : *grids) {
        if (!grid->isType<PointDataGrid>()) continue;
        std::cout << "grid " << grid->getName() << "\n";
        dumpPointGrid(std::cout, static_cast<const PointDataGrid&>(*grid));
    }
    file.close();
    return 0;
}

// ---------------------------------------------------------------------------
// Reference file generation (the formulas are mirrored in tests/points.rs)

Vec3f position(int i)
{
    return Vec3f(float((i * 7) % 13) * 0.31f + 0.5f, float((i * 5) % 11) * 0.27f + 0.5f,
        float((i * 3) % 7) * 0.45f - 1.5f);
}

const int N = 40;
const char* NAMES[4] = {"alpha", "beta", "", "gamma"};

std::vector<Vec3f> positions()
{
    std::vector<Vec3f> p;
    for (int i = 0; i < N; ++i) p.push_back(position(i));
    return p;
}

template<typename Codec>
PointDataGrid::Ptr makeGrid(const std::string& name, bool withAttributes)
{
    const std::vector<Vec3f> pos = positions();
    math::Transform::Ptr xform = math::Transform::createLinearTransform(0.25);
    PointAttributeVector<Vec3f> wrapper(pos);
    tools::PointIndexGrid::Ptr index = tools::createPointIndexGrid<tools::PointIndexGrid>(wrapper, *xform);
    PointDataGrid::Ptr grid = createPointDataGrid<Codec, PointDataGrid>(*index, wrapper, *xform);
    grid->setName(name);
    if (!withAttributes) return grid;

    auto& tree = grid->tree();
    std::vector<float> density;
    std::vector<Vec3f> v;
    std::vector<int32_t> id;
    std::vector<Name> names;
    std::vector<float> arr;
    std::vector<short> even, first;
    for (int i = 0; i < N; ++i) {
        density.push_back(float(i) * 0.5f);
        v.push_back(Vec3f(float(i), -float(i), 0.5f * float(i)));
        id.push_back(100 + i);
        names.push_back(NAMES[i % 4]);
        for (int k = 0; k < 3; ++k) arr.push_back(float(i) + 0.25f * float(k));
        even.push_back(i % 2 == 0 ? 1 : 0);
        first.push_back(i < 5 ? 1 : 0);
    }
    appendAttribute<float>(tree, "density");
    populateAttribute(tree, index->tree(), "density", PointAttributeVector<float>(density));
    appendAttribute<Vec3f>(tree, "v");
    populateAttribute(tree, index->tree(), "v", PointAttributeVector<Vec3f>(v));
    appendAttribute<int32_t>(tree, "id");
    populateAttribute(tree, index->tree(), "id", PointAttributeVector<int32_t>(id));
    appendAttribute<Name>(tree, "name");
    populateAttribute(tree, index->tree(), "name", PointAttributeVector<Name>(names));
    appendAttribute<float>(tree, "arr", 0.0f, /*stride=*/3);
    populateAttribute(tree, index->tree(), "arr", PointAttributeVector<float>(arr, 3), 3);
    FloatMetadata defaultValue(1.0f);
    appendAttribute<float>(tree, "uni", 3.5f, 1, true, &defaultValue);
    appendAttribute<int32_t>(tree, "hid", 7, 1, true, nullptr, /*hidden=*/true);
    appendAttribute<float>(tree, "tmp", 2.0f, 1, true, nullptr, false, /*transient=*/true);
    appendGroup(tree, "even");
    setGroup(tree, index->tree(), even, "even");
    appendGroup(tree, "first");
    setGroup(tree, index->tree(), first, "first");
    appendGroup(tree, "unused");
    return grid;
}

void write(const std::string& path, const GridPtrVec& grids, uint32_t compression)
{
    io::File f(path);
    f.setCompression(compression);
    f.write(grids);
    f.close();
}

int gen(const std::string& dir)
{
    // Null-codec positions and every kind of attribute, with blosc (paged) compression.
    write(dir + "/points_attrs.vdb", {makeGrid<NullCodec>("attrs", true)},
        io::COMPRESS_BLOSC | io::COMPRESS_ACTIVE_MASK);
    // The same without compression and with zip (both unpaged).
    write(dir + "/points_none.vdb", {makeGrid<NullCodec>("attrs", true)}, io::COMPRESS_ACTIVE_MASK);
    write(dir + "/points_zip.vdb", {makeGrid<FixedPointCodec<false>>("attrs", true)},
        io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK);

    // Position codecs and attribute codecs.
    {
        PointDataGrid::Ptr fx16 = makeGrid<FixedPointCodec<false>>("fxpt16", false);
        PointDataGrid::Ptr fx8 = makeGrid<FixedPointCodec<true>>("fxpt8", false);
        PointDataGrid::Ptr tr = makeGrid<TruncateCodec>("trnc", false);
        // codec attributes on the 16-bit grid
        auto& tree = fx16->tree();
        appendAttribute<Vec3f, UnitVecCodec>(tree, "N");
        appendAttribute<float, FixedPointCodec<true, UnitRange>>(tree, "alpha");
        appendAttribute<float, TruncateCodec>(tree, "h");
        appendAttribute<Vec3f, FixedPointCodec<false, UnitRange>>(tree, "col");
        for (auto leaf = tree.beginLeaf(); leaf; ++leaf) {
            AttributeWriteHandle<Vec3f> n(leaf->attributeArray("N"));
            AttributeWriteHandle<float> a(leaf->attributeArray("alpha"));
            AttributeWriteHandle<float> h(leaf->attributeArray("h"));
            AttributeWriteHandle<Vec3f> c(leaf->attributeArray("col"));
            for (auto it = leaf->beginIndexOn(); it; ++it) {
                const float f = float(*it) / 10.0f;
                n.set(*it, Vec3f(1.0f, f, -0.5f).unit());
                a.set(*it, f * 0.5f);
                h.set(*it, 1.0f + f);
                c.set(*it, Vec3f(f, 0.5f, 1.0f - f * 0.5f));
            }
        }
        write(dir + "/points_codecs.vdb", {fx16, fx8, tr}, io::COMPRESS_BLOSC | io::COMPRESS_ACTIVE_MASK);
    }

    // An empty grid and a grid with points in inactive voxels.
    {
        PointDataGrid::Ptr empty = PointDataGrid::create();
        empty->setName("empty");
        PointDataGrid::Ptr inactive = makeGrid<FixedPointCodec<true>>("inactive", false);
        auto leaf = inactive->tree().beginLeaf();
        for (auto it = leaf->beginValueOn(); it; ++it) {
            leaf->setValueOff(it.pos());
            break;
        }
        write(dir + "/points_misc.vdb", {empty, inactive}, io::COMPRESS_BLOSC | io::COMPRESS_ACTIVE_MASK);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    openvdb::initialize();
    if (argc == 3 && std::string(argv[1]) == "gen") return gen(argv[2]);
    if (argc == 3 && std::string(argv[1]) == "dump") return dump(argv[2]);
    std::cerr << "usage: points_ref gen <outdir> | dump <file.vdb>\n";
    return 2;
}
