//! Random-access `.vdb` files and sequential streams (port of `openvdb/io/File` and
//! `openvdb/io/Stream`).

#[cfg(feature = "write")]
use super::archive::write_archive;
#[cfg(feature = "read")]
use super::archive::{
    copy_tree_from, read_grid_head, read_grid_tree, read_header, string_as_unique_name, FileHeader, GridDescriptor,
    GridHead,
};
#[cfg(feature = "read")]
use super::bytes::Reader;
#[cfg(feature = "read")]
use super::compression::Compression;
#[cfg(feature = "read")]
use super::error::Error;
use super::error::Result;
#[cfg(feature = "read")]
use super::metadata_io::read_meta_map;
#[cfg(feature = "write")]
use super::WriteOptions;
use crate::grid::AnyGrid;
#[cfg(feature = "read")]
use crate::math::Transform;
use crate::metadata::MetaMap;
#[cfg(feature = "read")]
use std::collections::HashMap;
#[cfg(feature = "write")]
use std::io::Write;
#[cfg(feature = "read")]
use std::io::{BufReader, Read};
#[cfg(feature = "read")]
use std::path::{Path, PathBuf};

/// A grid's descriptor, metadata and transform, read without its tree.
#[cfg(feature = "read")]
#[derive(Clone, Debug)]
pub struct GridMetadata {
    /// The grid's descriptor (names, type, offsets).
    pub descriptor: GridDescriptor,
    /// Compression flags with which the grid's tree was written.
    pub compression: Compression,
    /// The grid's metadata (name, class, statistics written by the library, ...).
    pub metadata: MetaMap,
    /// The OpenVDB type name of the transform's map (e.g. `"UniformScaleMap"`).
    pub map_type: String,
    /// The grid's transform, or `None` if its map type is not supported
    /// (e.g. `"NonlinearFrustumMap"`).
    pub transform: Option<Transform>,
}

#[cfg(feature = "read")]
impl GridMetadata {
    fn new(descriptor: GridDescriptor, head: GridHead) -> Self {
        GridMetadata {
            descriptor,
            compression: head.compression,
            metadata: head.metadata,
            map_type: head.map_type,
            transform: head.transform.ok(),
        }
    }
}

/// Find the index of the descriptor for `name`: a unique name (also accepted in the
/// printable `"name[n]"` form) or else the first grid with that grid name.
#[cfg(feature = "read")]
fn find_descriptor(descs: &[GridDescriptor], name: &str) -> Option<usize> {
    let unique = string_as_unique_name(name);
    descs
        .iter()
        .position(|d| d.unique_name == name || d.unique_name == unique)
        .or_else(|| descs.iter().position(|d| d.grid_name == name))
}

/// Resolve instanced grids by copying their parents' trees.
#[cfg(feature = "read")]
fn connect_instances(descs: &[GridDescriptor], grids: &mut [Option<AnyGrid>]) -> Result<()> {
    let index: HashMap<&str, usize> = descs.iter().enumerate().map(|(i, d)| (d.unique_name.as_str(), i)).collect();
    for (i, d) in descs.iter().enumerate() {
        if !d.is_instance() || grids[i].is_none() {
            continue;
        }
        let p = *index.get(d.instance_parent.as_str()).ok_or_else(|| {
            Error::corrupt(format!("missing instance parent \"{}\" for grid {}", d.instance_parent, d.display_name()))
        })?;
        let parent = grids[p].clone().ok_or_else(|| Error::UnsupportedGridType(descs[p].grid_type.clone()))?;
        copy_tree_from(grids[i].as_mut().unwrap(), &parent)?;
    }
    Ok(())
}

/// Read a grid's data at the reader's position. Returns `None` for an unsupported grid type,
/// after skipping to the descriptor's end offset (which requires valid offsets).
#[cfg(feature = "read")]
fn read_grid_data<R: Read>(
    r: &mut Reader<R>,
    header: &FileHeader,
    desc: &GridDescriptor,
) -> Result<(GridMetadata, Option<AnyGrid>)> {
    let head = read_grid_head(r, header)?;
    if !desc.is_supported() {
        let meta = GridMetadata::new(desc.clone(), head);
        let end = desc.end_pos as u64;
        if desc.end_pos <= 0 || end < r.pos() {
            return Err(Error::UnsupportedGridType(desc.grid_type.clone()));
        }
        r.skip(end - r.pos())?;
        return Ok((meta, None));
    }
    let meta = GridMetadata::new(desc.clone(), head.clone());
    let grid = read_grid_tree(r, header, desc, head, !desc.is_instance())?;
    Ok((meta, Some(grid)))
}

/// Everything in a sequentially read archive: file metadata, descriptors, grid metadata
/// and grids (`None` for unsupported types).
#[cfg(feature = "read")]
type SequentialArchive = (MetaMap, Vec<GridDescriptor>, Vec<GridMetadata>, Vec<Option<AnyGrid>>);

/// Read all grids of a sequential archive (after the header).
#[cfg(feature = "read")]
fn read_sequential<R: Read>(r: &mut Reader<R>, header: &FileHeader) -> Result<SequentialArchive> {
    let metadata = read_meta_map(r)?;
    let count = r.read_i32()?;
    let mut descs = Vec::new();
    let mut metas = Vec::new();
    let mut grids = Vec::new();
    for _ in 0..count.max(0) {
        let desc = GridDescriptor::read(r)?;
        let (m, g) = read_grid_data(r, header, &desc)?;
        descs.push(desc);
        metas.push(m);
        grids.push(g);
    }
    connect_instances(&descs, &mut grids)?;
    Ok((metadata, descs, metas, grids))
}

/// A `.vdb` file opened for reading.
///
/// Opening reads the header, the file-level metadata and the grid descriptors; grids are
/// loaded on demand. Files without grid offsets (written as streams) are loaded entirely
/// when opened.
///
/// ```no_run
/// let file = vdb::io::File::open("smoke.vdb")?;
/// for name in file.grid_names() {
///     println!("{}", name);
/// }
/// let density = file.read_grid("density")?;
/// # Ok::<(), vdb::io::Error>(())
/// ```
#[cfg(feature = "read")]
#[derive(Debug)]
pub struct File {
    path: PathBuf,
    header: FileHeader,
    metadata: MetaMap,
    descriptors: Vec<GridDescriptor>,
    /// For files without grid offsets: everything, read when the file was opened.
    preloaded: Option<(Vec<GridMetadata>, Vec<Option<AnyGrid>>)>,
}

#[cfg(feature = "read")]
impl File {
    /// Open a file and read its header, metadata and grid descriptors.
    pub fn open(path: impl AsRef<Path>) -> Result<File> {
        let path = path.as_ref().to_path_buf();
        let mut r = Reader::new(BufReader::new(std::fs::File::open(&path)?));
        let header = read_header(&mut r)?;
        if !header.has_grid_offsets {
            let (metadata, descriptors, metas, grids) = read_sequential(&mut r, &header)?;
            return Ok(File { path, header, metadata, descriptors, preloaded: Some((metas, grids)) });
        }
        let metadata = read_meta_map(&mut r)?;
        let count = r.read_i32()?;
        let mut descriptors = Vec::new();
        for _ in 0..count.max(0) {
            let d = GridDescriptor::read(&mut r)?;
            if d.end_pos < r.pos() as i64 {
                return Err(Error::corrupt(format!("invalid end offset for grid {}", d.display_name())));
            }
            r.seek_to(d.end_pos as u64)?;
            descriptors.push(d);
        }
        Ok(File { path, header, metadata, descriptors, preloaded: None })
    }

    /// The file's path.
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// The file header.
    pub fn header(&self) -> &FileHeader {
        &self.header
    }
    /// OpenVDB's version string, `"<major>.<minor>/<file version>"`.
    pub fn version(&self) -> String {
        self.header.version_string()
    }
    /// File-level metadata.
    pub fn metadata(&self) -> &MetaMap {
        &self.metadata
    }
    /// Descriptors of all grids in the file, in file order, including grids of
    /// unsupported types.
    pub fn descriptors(&self) -> &[GridDescriptor] {
        &self.descriptors
    }
    /// Names of all grids in file order (possibly with duplicates).
    pub fn grid_names(&self) -> Vec<String> {
        self.descriptors.iter().map(|d| d.grid_name.clone()).collect()
    }
    /// True if the file has a grid with the given name or unique name.
    pub fn has_grid(&self, name: &str) -> bool {
        find_descriptor(&self.descriptors, name).is_some()
    }

    fn reader(&self) -> Result<Reader<BufReader<std::fs::File>>> {
        Ok(Reader::new(BufReader::new(std::fs::File::open(&self.path)?)))
    }

    fn index_of(&self, name: &str) -> Result<usize> {
        find_descriptor(&self.descriptors, name).ok_or_else(|| Error::GridNotFound(name.to_string()))
    }

    fn read_metadata_at(&self, r: &mut Reader<BufReader<std::fs::File>>, i: usize) -> Result<GridMetadata> {
        if let Some((metas, _)) = &self.preloaded {
            return Ok(metas[i].clone());
        }
        let d = &self.descriptors[i];
        r.seek_to(d.grid_pos as u64)?;
        Ok(GridMetadata::new(d.clone(), read_grid_head(r, &self.header)?))
    }

    /// Read a grid's metadata and transform without loading its tree.
    pub fn read_grid_metadata(&self, name: &str) -> Result<GridMetadata> {
        let i = self.index_of(name)?;
        self.read_metadata_at(&mut self.reader()?, i)
    }

    /// Read the metadata and transforms of all grids (in file order) without loading trees.
    pub fn read_all_grid_metadata(&self) -> Result<Vec<GridMetadata>> {
        if let Some((metas, _)) = &self.preloaded {
            return Ok(metas.clone());
        }
        let mut r = self.reader()?;
        (0..self.descriptors.len()).map(|i| self.read_metadata_at(&mut r, i)).collect()
    }

    fn read_grid_at(&self, r: &mut Reader<BufReader<std::fs::File>>, i: usize) -> Result<AnyGrid> {
        let d = &self.descriptors[i];
        if let Some((_, grids)) = &self.preloaded {
            return grids[i].clone().ok_or_else(|| Error::UnsupportedGridType(d.grid_type.clone()));
        }
        if !d.is_supported() {
            return Err(Error::UnsupportedGridType(d.grid_type.clone()));
        }
        r.seek_to(d.grid_pos as u64)?;
        let head = read_grid_head(r, &self.header)?;
        let mut grid = read_grid_tree(r, &self.header, d, head, !d.is_instance())?;
        if d.is_instance() {
            let p = self
                .descriptors
                .iter()
                .position(|p| p.unique_name == d.instance_parent)
                .ok_or_else(|| Error::corrupt(format!("missing instance parent for grid {}", d.display_name())))?;
            if p == i {
                return Err(Error::corrupt("grid is its own instance parent"));
            }
            let parent = self.read_grid_at(r, p)?;
            copy_tree_from(&mut grid, &parent)?;
        }
        Ok(grid)
    }

    /// Read a grid by name. `name` may be a grid name (the first grid with that name is
    /// returned) or a unique name, also in the printable `"name[n]"` form.
    pub fn read_grid(&self, name: &str) -> Result<AnyGrid> {
        let i = self.index_of(name)?;
        self.read_grid_at(&mut self.reader()?, i)
    }

    /// Read all grids of supported types, in file order. Grids of unsupported types
    /// (e.g. point data grids) are skipped; their descriptors remain available through
    /// [`descriptors`](Self::descriptors).
    pub fn read_grids(&self) -> Result<Vec<AnyGrid>> {
        if let Some((_, grids)) = &self.preloaded {
            return Ok(grids.iter().flatten().cloned().collect());
        }
        let mut r = self.reader()?;
        let mut grids: Vec<Option<AnyGrid>> = Vec::with_capacity(self.descriptors.len());
        for (i, d) in self.descriptors.iter().enumerate() {
            if !d.is_supported() {
                grids.push(None);
                continue;
            }
            r.seek_to(d.grid_pos as u64)?;
            let head = read_grid_head(&mut r, &self.header)?;
            grids.push(Some(read_grid_tree(&mut r, &self.header, &self.descriptors[i], head, !d.is_instance())?));
        }
        connect_instances(&self.descriptors, &mut grids)?;
        Ok(grids.into_iter().flatten().collect())
    }
}

/// A `.vdb` archive read sequentially from (or written to) a byte stream
/// (port of `io::Stream`).
///
/// Streams are written without grid offsets. When reading, grids of unsupported types
/// can only be skipped if the data carries offsets (i.e. it was written as a file).
#[cfg(feature = "read")]
#[derive(Debug, Clone)]
pub struct Stream {
    /// The archive header.
    pub header: FileHeader,
    /// File-level metadata.
    pub metadata: MetaMap,
    /// Descriptors of all grids, in stream order.
    pub descriptors: Vec<GridDescriptor>,
    /// Metadata and transforms of all grids, in stream order.
    pub grid_metadata: Vec<GridMetadata>,
    /// The loaded grids, in stream order; `None` for grids of unsupported types.
    pub grids: Vec<Option<AnyGrid>>,
}

#[cfg(feature = "read")]
impl Stream {
    /// Read a complete archive from a reader.
    pub fn read<R: Read>(reader: R) -> Result<Stream> {
        let mut r = Reader::new(reader);
        let header = read_header(&mut r)?;
        let (metadata, descriptors, grid_metadata, grids) = read_sequential(&mut r, &header)?;
        Ok(Stream { header, metadata, descriptors, grid_metadata, grids })
    }

    /// The loaded grids of supported types.
    pub fn into_grids(self) -> Vec<AnyGrid> {
        self.grids.into_iter().flatten().collect()
    }

    /// Write grids and file-level metadata to a writer, without grid offsets.
    #[cfg(feature = "write")]
    pub fn write<W: Write>(writer: W, grids: &[AnyGrid], metadata: &MetaMap, options: &WriteOptions) -> Result<()> {
        write_stream_to(writer, grids, metadata, options)
    }
}

/// Write grids and file-level metadata to a writer, without grid offsets.
#[cfg(feature = "write")]
pub(crate) fn write_stream_to<W: Write>(
    mut writer: W,
    grids: &[AnyGrid],
    metadata: &MetaMap,
    options: &WriteOptions,
) -> Result<()> {
    let refs: Vec<&AnyGrid> = grids.iter().collect();
    let bytes = write_archive(&refs, metadata, options, false)?;
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}

/// Write grids and file-level metadata to a writer in the random-access file layout
/// (with grid offsets). The archive is assembled in memory first.
#[cfg(feature = "write")]
pub fn write_to<W: Write>(mut writer: W, grids: &[AnyGrid], metadata: &MetaMap, options: &WriteOptions) -> Result<()> {
    let refs: Vec<&AnyGrid> = grids.iter().collect();
    let bytes = write_archive(&refs, metadata, options, true)?;
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}
