//! A simple ray tracer for level sets and a volume renderer (port of
//! `openvdb/tools/RayTracer.h`).
//!
//! * [`Film`] — an RGBA image that can be saved as a binary PPM.
//! * [`PerspectiveCamera`] and [`OrthographicCamera`] — generate primary rays per pixel.
//! * Shaders implementing [`Shader`]: [`MatteShader`], [`NormalShader`], [`PositionShader`]
//!   and [`DiffuseShader`], with either a constant color or colors sampled from a `Vec3f` grid.
//! * [`LevelSetRayTracer`] / [`ray_trace`] — first-hit rendering of level sets with optional
//!   jittered super-sampling.
//! * [`VolumeRender`] — single-scattering rendering of density volumes.
//!
//! ```no_run
//! use vdb::prelude::*;
//! use vdb::tools::ray_tracer::{ray_trace, DiffuseShader, Film, PerspectiveCamera};
//! # fn sphere() -> FloatGrid { unimplemented!() }
//! let grid = sphere();
//! let mut film = Film::new(320, 240);
//! let camera = PerspectiveCamera::new(&film, Vec3d::zero(), Vec3d::new(0.0, 0.0, 5.0), 50.0, 41.2136);
//! ray_trace(&grid, &DiffuseShader::new(Default::default()), &camera, &mut film, 1, 0).unwrap();
//! # #[cfg(feature = "write")]
//! film.save_ppm("sphere.ppm").unwrap();
//! ```

use super::interpolation::{BoxSampler, PointSampler, Sampler};
use super::ray_intersector::{LevelSetRayIntersector, Ray, RayIntersectorError, VolumeRayIntersector};
use crate::grid::Grid;
use crate::math::{Axis, Mat4d, Vec3d, Vec3f};
use crate::types::RealValue;
use rayon::prelude::*;
use std::io::Write;
use std::ops::{Add, AddAssign, Mul};

/// An RGBA color with `f32` components (`Film::RGBA`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgba {
    /// Red.
    pub r: f32,
    /// Green.
    pub g: f32,
    /// Blue.
    pub b: f32,
    /// Alpha.
    pub a: f32,
}

impl Default for Rgba {
    /// Opaque black.
    fn default() -> Self {
        Rgba { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
    }
}

impl Rgba {
    /// An opaque color.
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Rgba { r, g, b, a: 1.0 }
    }
    /// A color with alpha.
    pub const fn with_alpha(r: f32, g: f32, b: f32, a: f32) -> Self {
        Rgba { r, g, b, a }
    }
    /// An opaque gray level.
    pub const fn gray(v: f32) -> Self {
        Rgba { r: v, g: v, b: v, a: 1.0 }
    }
    /// Composite `rhs` under this color (`RGBA::over`).
    pub fn over(&mut self, rhs: &Rgba) {
        let s = rhs.a * (1.0 - self.a);
        self.r = self.a * self.r + s * rhs.r;
        self.g = self.a * self.g + s * rhs.g;
        self.b = self.a * self.b + s * rhs.b;
        self.a += s;
    }
}

impl Mul<f32> for Rgba {
    type Output = Rgba;
    /// Scale the color channels (alpha becomes 1, as in OpenVDB).
    fn mul(self, s: f32) -> Rgba {
        Rgba::new(self.r * s, self.g * s, self.b * s)
    }
}
impl Mul for Rgba {
    type Output = Rgba;
    /// Multiply the color channels (alpha becomes 1).
    fn mul(self, o: Rgba) -> Rgba {
        Rgba::new(self.r * o.r, self.g * o.g, self.b * o.b)
    }
}
impl Add for Rgba {
    type Output = Rgba;
    /// Add the color channels (alpha becomes 1).
    fn add(self, o: Rgba) -> Rgba {
        Rgba::new(self.r + o.r, self.g + o.g, self.b + o.b)
    }
}
impl AddAssign for Rgba {
    /// Add all four channels.
    fn add_assign(&mut self, o: Rgba) {
        self.r += o.r;
        self.g += o.g;
        self.b += o.b;
        self.a += o.a;
    }
}

/// A frame buffer of RGBA pixels (`tools::Film`).
#[derive(Clone, Debug)]
pub struct Film {
    width: usize,
    height: usize,
    pixels: Vec<Rgba>,
}

impl Film {
    /// A film of the given size filled with opaque black.
    pub fn new(width: usize, height: usize) -> Self {
        Self::with_background(width, height, Rgba::default())
    }
    /// A film filled with `bg`.
    pub fn with_background(width: usize, height: usize, bg: Rgba) -> Self {
        Film { width, height, pixels: vec![bg; width * height] }
    }
    /// Width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }
    /// Height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }
    /// Number of pixels.
    pub fn num_pixels(&self) -> usize {
        self.pixels.len()
    }
    /// The pixel in column `i` and row `j` (row 0 is the top).
    pub fn pixel(&self, i: usize, j: usize) -> &Rgba {
        &self.pixels[i + j * self.width]
    }
    /// Mutable access to a pixel.
    pub fn pixel_mut(&mut self, i: usize, j: usize) -> &mut Rgba {
        &mut self.pixels[i + j * self.width]
    }
    /// All pixels in row-major order.
    pub fn pixels(&self) -> &[Rgba] {
        &self.pixels
    }
    /// Fill with a color.
    pub fn fill(&mut self, c: Rgba) {
        self.pixels.iter_mut().for_each(|p| *p = c);
    }
    /// Fill with a checkerboard of squares of `size` pixels (a power of two).
    pub fn checkerboard(&mut self, c1: Rgba, c2: Rgba, size: usize) {
        for j in 0..self.height {
            for i in 0..self.width {
                self.pixels[i + j * self.width] = if ((i & size) ^ (j & size)) != 0 { c1 } else { c2 };
            }
        }
    }
    /// 8-bit RGB(A) bytes (values are scaled by 255 and truncated, without clamping, like
    /// OpenVDB).
    pub fn to_bytes(&self, alpha: bool) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * if alpha { 4 } else { 3 });
        let q = |v: f32| (255.0f32 * v) as u8;
        for p in &self.pixels {
            out.push(q(p.r));
            out.push(q(p.g));
            out.push(q(p.b));
            if alpha {
                out.push(q(p.a));
            }
        }
        out
    }
    /// Write a binary (P6) PPM image.
    pub fn write_ppm<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        write!(w, "P6\n{} {}\n255\n", self.width, self.height)?;
        w.write_all(&self.to_bytes(false))
    }
    /// Save a binary PPM file (".ppm" is appended if the name has no extension). Requires
    /// the `write` feature.
    #[cfg(feature = "write")]
    pub fn save_ppm(&self, path: &str) -> std::io::Result<()> {
        let mut name = path.to_string();
        if !name.contains('.') {
            name.push_str(".ppm");
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(name)?);
        self.write_ppm(&mut f)
    }
}

/// Generates primary rays for the pixels of a film (`tools::BaseCamera`).
pub trait Camera: Sync {
    /// Image width in pixels.
    fn width(&self) -> usize;
    /// Image height in pixels.
    fn height(&self) -> usize;
    /// The world-space ray through the point `(i + i_offset, j + j_offset)` of the raster.
    fn get_ray(&self, i: usize, j: usize, i_offset: f64, j_offset: f64) -> Ray;
}

/// State shared by the cameras (`tools::BaseCamera`).
#[derive(Clone, Debug)]
struct CameraBase {
    width: usize,
    height: usize,
    scale_width: f64,
    scale_height: f64,
    ray: Ray,
    screen_to_world: Mat4d,
}

impl CameraBase {
    fn new(film: &Film, rotation: Vec3d, translation: Vec3d, frame_width: f64, near: f64, far: f64) -> Self {
        assert!(near > 0.0 && far > near, "invalid camera clipping planes");
        let mut m = Mat4d::identity();
        let d2r = std::f64::consts::PI / 180.0;
        m.post_rotate(Axis::X, rotation[0] * d2r);
        m.post_rotate(Axis::Y, rotation[1] * d2r);
        m.post_rotate(Axis::Z, rotation[2] * d2r);
        m.post_translate(translation);
        let mut c = CameraBase {
            width: film.width(),
            height: film.height(),
            scale_width: frame_width,
            scale_height: frame_width * film.height() as f64 / film.width() as f64,
            ray: Ray::default(),
            screen_to_world: m,
        };
        c.init_ray(near, far);
        c
    }
    fn init_ray(&mut self, t0: f64, t1: f64) {
        self.ray.set_times(t0, t1);
        self.ray.set_eye(self.screen_to_world.transform(Vec3d::zero()));
        self.ray.set_dir(self.screen_to_world.transform3x3(Vec3d::new(0.0, 0.0, -1.0)));
    }
    fn look_at(&mut self, xyz: Vec3d, up: Vec3d) {
        let orig = self.screen_to_world.transform(Vec3d::zero());
        let forward = (orig - xyz).unit();
        let horizontal = up.unit().cross(&forward).unit();
        let up = forward.cross(&horizontal).unit();
        let mut m = Mat4d::identity();
        for c in 0..3 {
            m.0[0][c] = horizontal[c];
            m.0[1][c] = up[c];
            m.0[2][c] = forward[c];
        }
        m.post_translate(orig);
        self.screen_to_world = m;
        self.init_ray(self.ray.t0(), self.ray.t1());
    }
    fn raster_to_screen(&self, i: f64, j: f64, z: f64) -> Vec3d {
        Vec3d::new(
            (2.0 * i / self.width as f64 - 1.0) * self.scale_width,
            (1.0 - 2.0 * j / self.height as f64) * self.scale_height,
            z,
        )
    }
}

/// A pinhole camera (`tools::PerspectiveCamera`). Rotations are in degrees, applied about x,
/// then y, then z, followed by the translation; the camera looks down its local `-z` axis.
#[derive(Clone, Debug)]
pub struct PerspectiveCamera {
    base: CameraBase,
}

impl PerspectiveCamera {
    /// A camera for `film` with the given focal length and aperture (both in mm; OpenVDB's
    /// defaults are 50 and 41.2136), near plane 1e-3 and an infinite far plane.
    pub fn new(film: &Film, rotation: Vec3d, translation: Vec3d, focal_length: f64, aperture: f64) -> Self {
        Self::with_planes(film, rotation, translation, focal_length, aperture, 1e-3, f64::MAX)
    }
    /// A camera with explicit near and far planes.
    pub fn with_planes(
        film: &Film,
        rotation: Vec3d,
        translation: Vec3d,
        focal_length: f64,
        aperture: f64,
        near: f64,
        far: f64,
    ) -> Self {
        PerspectiveCamera {
            base: CameraBase::new(film, rotation, translation, 0.5 * aperture / focal_length, near, far),
        }
    }
    /// Orient the camera to look at `xyz` with the given up vector.
    pub fn look_at(&mut self, xyz: Vec3d, up: Vec3d) {
        self.base.look_at(xyz, up);
    }
    /// Field of view (degrees) for a focal length and aperture.
    pub fn focal_length_to_field_of_view(length: f64, aperture: f64) -> f64 {
        360.0 / std::f64::consts::PI * (aperture / (2.0 * length)).atan()
    }
    /// Focal length for a field of view (degrees) and aperture.
    pub fn field_of_view_to_focal_length(fov: f64, aperture: f64) -> f64 {
        aperture / (2.0 * (fov * std::f64::consts::PI / 360.0).tan())
    }
}

impl Camera for PerspectiveCamera {
    fn width(&self) -> usize {
        self.base.width
    }
    fn height(&self) -> usize {
        self.base.height
    }
    fn get_ray(&self, i: usize, j: usize, i_offset: f64, j_offset: f64) -> Ray {
        let mut ray = self.base.ray;
        let dir = self.base.raster_to_screen(i as f64 + i_offset, j as f64 + j_offset, -1.0);
        let mut dir = self.base.screen_to_world.transform3x3(dir);
        dir.normalize();
        let s = 1.0 / dir.dot(&ray.dir());
        ray.set_times(ray.t0() * s, ray.t1() * s);
        ray.set_dir(dir);
        ray
    }
}

/// An orthographic camera (`tools::OrthographicCamera`).
#[derive(Clone, Debug)]
pub struct OrthographicCamera {
    base: CameraBase,
}

impl OrthographicCamera {
    /// A camera whose image plane is `frame_width` world units wide.
    pub fn new(film: &Film, rotation: Vec3d, translation: Vec3d, frame_width: f64) -> Self {
        OrthographicCamera { base: CameraBase::new(film, rotation, translation, 0.5 * frame_width, 1e-3, f64::MAX) }
    }
    /// Orient the camera to look at `xyz` with the given up vector.
    pub fn look_at(&mut self, xyz: Vec3d, up: Vec3d) {
        self.base.look_at(xyz, up);
    }
}

impl Camera for OrthographicCamera {
    fn width(&self) -> usize {
        self.base.width
    }
    fn height(&self) -> usize {
        self.base.height
    }
    fn get_ray(&self, i: usize, j: usize, i_offset: f64, j_offset: f64) -> Ray {
        let mut ray = self.base.ray;
        let eye = self.base.raster_to_screen(i as f64 + i_offset, j as f64 + j_offset, 0.0);
        ray.set_eye(self.base.screen_to_world.transform(eye));
        ray
    }
}

/// Computes the color of a surface hit (`tools::BaseShader`).
pub trait Shader: Sync {
    /// Color at world position `xyz` with unit normal `nml`, hit by a ray with direction `dir`.
    fn shade(&self, xyz: Vec3d, nml: Vec3d, dir: Vec3d) -> Rgba;
}

/// Where a shader takes its base color from.
#[derive(Clone, Copy, Debug)]
pub enum ShaderColor<'a> {
    /// A constant color.
    Constant(Rgba),
    /// Colors sampled (nearest neighbour) from a vector grid at the hit point.
    Grid(&'a Grid<Vec3f>),
}

impl Default for ShaderColor<'_> {
    /// White.
    fn default() -> Self {
        ShaderColor::Constant(Rgba::gray(1.0))
    }
}

impl ShaderColor<'_> {
    #[inline]
    fn at(&self, xyz: Vec3d) -> Rgba {
        match self {
            ShaderColor::Constant(c) => *c,
            ShaderColor::Grid(g) => {
                let v = PointSampler::sample(g.tree(), g.transform().world_to_index(xyz));
                Rgba::new(v[0], v[1], v[2])
            }
        }
    }
}

/// Returns the base color (`tools::MatteShader`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MatteShader<'a> {
    color: ShaderColor<'a>,
}
impl<'a> MatteShader<'a> {
    /// A matte shader.
    pub fn new(color: ShaderColor<'a>) -> Self {
        MatteShader { color }
    }
}
impl Shader for MatteShader<'_> {
    fn shade(&self, xyz: Vec3d, _: Vec3d, _: Vec3d) -> Rgba {
        self.color.at(xyz)
    }
}

/// Colors by the surface normal (`tools::NormalShader`).
#[derive(Clone, Copy, Debug, Default)]
pub struct NormalShader<'a> {
    color: ShaderColor<'a>,
}
impl<'a> NormalShader<'a> {
    /// A normal shader.
    pub fn new(color: ShaderColor<'a>) -> Self {
        NormalShader { color }
    }
}
impl Shader for NormalShader<'_> {
    fn shade(&self, xyz: Vec3d, n: Vec3d, _: Vec3d) -> Rgba {
        let nn = Rgba::new((n[0] + 1.0) as f32, (n[1] + 1.0) as f32, (n[2] + 1.0) as f32);
        match self.color {
            ShaderColor::Constant(c) => (c * 0.5f32) * nn,
            ShaderColor::Grid(_) => self.color.at(xyz) * nn,
        }
    }
}

/// Colors by the position within a bounding box (`tools::PositionShader`).
#[derive(Clone, Copy, Debug)]
pub struct PositionShader<'a> {
    min: Vec3d,
    inv_dim: Vec3d,
    color: ShaderColor<'a>,
}
impl<'a> PositionShader<'a> {
    /// A position shader mapping the box `[min, max]` to colors.
    pub fn new(min: Vec3d, max: Vec3d, color: ShaderColor<'a>) -> Self {
        let e = max - min;
        PositionShader { min, inv_dim: Vec3d::new(1.0 / e[0], 1.0 / e[1], 1.0 / e[2]), color }
    }
}
impl Shader for PositionShader<'_> {
    fn shade(&self, xyz: Vec3d, _: Vec3d, _: Vec3d) -> Rgba {
        let rgb = (xyz - self.min).mul_elem(&self.inv_dim);
        self.color.at(xyz) * Rgba::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32)
    }
}

/// Lambertian-like shading by `|n · dir|` (`tools::DiffuseShader`).
#[derive(Clone, Copy, Debug, Default)]
pub struct DiffuseShader<'a> {
    color: ShaderColor<'a>,
}
impl<'a> DiffuseShader<'a> {
    /// A diffuse shader.
    pub fn new(color: ShaderColor<'a>) -> Self {
        DiffuseShader { color }
    }
}
impl Shader for DiffuseShader<'_> {
    fn shade(&self, xyz: Vec3d, n: Vec3d, dir: Vec3d) -> Rgba {
        self.color.at(xyz) * (n.dot(&dir).abs() as f32)
    }
}

/// The 32-bit Mersenne twister (`std::mt19937`), used for reproducible sub-pixel jitter.
struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }
    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.state[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.state[i] = v;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
    /// `std::uniform_real_distribution<double>(0, 1)` as implemented by libstdc++.
    fn next_f64(&mut self) -> f64 {
        let a = self.next_u32() as f64;
        let b = self.next_u32() as f64;
        let r = (a + b * 4294967296.0) / 18446744073709551616.0;
        if r >= 1.0 {
            1.0 - f64::EPSILON / 2.0
        } else {
            r
        }
    }
}

/// First-hit ray tracer for level sets (`tools::LevelSetRayTracer`).
pub struct LevelSetRayTracer<'a, T: RealValue> {
    grid: &'a Grid<T>,
    iso: T,
    rand: Vec<f64>,
    sub_pixels: usize,
}

impl<'a, T: RealValue> LevelSetRayTracer<'a, T> {
    /// A tracer for the zero level set of `grid` with `pixel_samples` samples per pixel (> 0)
    /// jittered with the given seed.
    pub fn new(grid: &'a Grid<T>, pixel_samples: usize, seed: u32) -> Result<Self, RayIntersectorError> {
        Self::with_iso_value(grid, <T as crate::types::Value>::zero(), pixel_samples, seed)
    }
    /// A tracer for the `iso` level set of `grid`.
    pub fn with_iso_value(
        grid: &'a Grid<T>,
        iso: T,
        pixel_samples: usize,
        seed: u32,
    ) -> Result<Self, RayIntersectorError> {
        // Validate the grid once; every render thread builds its own intersector.
        LevelSetRayIntersector::with_iso_value(grid, iso)?;
        let mut t = LevelSetRayTracer { grid, iso, rand: Vec::new(), sub_pixels: 0 };
        t.set_pixel_samples(pixel_samples, seed);
        Ok(t)
    }
    /// Set the number of samples per pixel (> 0) and the jitter seed.
    ///
    /// # Panics
    /// If `pixel_samples` is zero.
    pub fn set_pixel_samples(&mut self, pixel_samples: usize, seed: u32) {
        assert!(pixel_samples > 0, "pixelSamples must be larger than zero!");
        self.sub_pixels = pixel_samples - 1;
        self.rand.clear();
        if self.sub_pixels > 0 {
            let mut rng = Mt19937::new(seed);
            self.rand = (0..16).map(|_| rng.next_f64()).collect();
        }
    }
    /// Render into `film`; pixels that miss the surface keep their current color.
    pub fn render<C: Camera, S: Shader>(&self, camera: &C, shader: &S, film: &mut Film) {
        assert!(film.width() == camera.width() && film.height() == camera.height(), "film/camera size mismatch");
        let frac = 1.0f32 / (1.0 + self.sub_pixels as f32);
        let width = film.width();
        film.pixels.par_chunks_mut(width).enumerate().for_each_init(
            || LevelSetRayIntersector::with_iso_value(self.grid, self.iso).expect("validated in new"),
            |inter, (j, row)| {
                // Jitter offsets are consumed in raster order, as in OpenVDB's serial render.
                let mut n = 2 * self.sub_pixels * j * width;
                for (i, px) in row.iter_mut().enumerate() {
                    let bg = *px;
                    let mut trace = |ray: Ray| match inter.intersects_ws_normal(&ray) {
                        Some((h, nml)) => shader.shade(h.position, nml, ray.dir()),
                        None => bg,
                    };
                    let mut c = trace(camera.get_ray(i, j, 0.5, 0.5));
                    for _ in 0..self.sub_pixels {
                        let ray = camera.get_ray(i, j, self.rand[n & 15], self.rand[(n + 1) & 15]);
                        c += trace(ray);
                        n += 2;
                    }
                    *px = c * frac;
                }
            },
        );
    }
}

/// Render the zero level set of `grid` into `film` (`tools::rayTrace`).
pub fn ray_trace<T: RealValue, C: Camera, S: Shader>(
    grid: &Grid<T>,
    shader: &S,
    camera: &C,
    film: &mut Film,
    pixel_samples: usize,
    seed: u32,
) -> Result<(), RayIntersectorError> {
    LevelSetRayTracer::new(grid, pixel_samples, seed)?.render(camera, shader, film);
    Ok(())
}

/// Single-scattering volume renderer for density grids (`tools::VolumeRender`).
pub struct VolumeRender<'a, T: RealValue> {
    grid: &'a Grid<T>,
    primary_step: f64,
    shadow_step: f64,
    cutoff: f64,
    light_gain: f64,
    light_dir: Vec3d,
    light_color: Vec3d,
    absorption: Vec3d,
    scattering: Vec3d,
}

impl<'a, T: RealValue + crate::types::ArithValue> VolumeRender<'a, T> {
    /// A renderer for the density grid `grid`, with OpenVDB's default parameters.
    pub fn new(grid: &'a Grid<T>) -> Result<Self, RayIntersectorError> {
        VolumeRayIntersector::new(grid, 0)?;
        Ok(VolumeRender {
            grid,
            primary_step: 1.0,
            shadow_step: 3.0,
            cutoff: 0.005,
            light_gain: 0.2,
            light_dir: Vec3d::new(0.3, 0.3, 0.0).unit(),
            light_color: Vec3d::splat(0.7),
            absorption: Vec3d::splat(0.1),
            scattering: Vec3d::splat(1.5),
        })
    }
    /// Direction towards the light.
    pub fn set_light_dir(&mut self, x: f64, y: f64, z: f64) {
        self.light_dir = Vec3d::new(x, y, z).unit();
    }
    /// Light color.
    pub fn set_light_color(&mut self, r: f64, g: f64, b: f64) {
        self.light_color = Vec3d::new(r, g, b);
    }
    /// Integration step along primary rays, in voxels.
    pub fn set_primary_step(&mut self, s: f64) {
        self.primary_step = s;
    }
    /// Integration step along shadow rays, in voxels.
    pub fn set_shadow_step(&mut self, s: f64) {
        self.shadow_step = s;
    }
    /// Scattering coefficients.
    pub fn set_scattering(&mut self, x: f64, y: f64, z: f64) {
        self.scattering = Vec3d::new(x, y, z);
    }
    /// Absorption coefficients.
    pub fn set_absorption(&mut self, x: f64, y: f64, z: f64) {
        self.absorption = Vec3d::new(x, y, z);
    }
    /// In-scattering gain along shadow rays.
    pub fn set_light_gain(&mut self, g: f64) {
        self.light_gain = g;
    }
    /// Cut-off for density and transmittance.
    pub fn set_cut_off(&mut self, c: f64) {
        self.cutoff = c;
    }

    /// Render into `film` (every pixel is overwritten; alpha holds the opacity).
    pub fn render<C: Camera>(&self, camera: &C, film: &mut Film) {
        assert!(film.width() == camera.width() && film.height() == camera.height(), "film/camera size mismatch");
        let extinction = -self.scattering - self.absorption;
        let albedo =
            self.light_color.mul_elem(&self.scattering).zip_map(self.scattering + self.absorption, |a, b| a / b);
        let (p_step, s_step, cutoff, gain) = (self.primary_step, self.shadow_step, self.cutoff, self.light_gain);
        let exp = |v: Vec3d| v.map(f64::exp);
        let width = film.width();
        let grid = self.grid;
        let light_dir = self.light_dir;
        film.pixels.par_chunks_mut(width).enumerate().for_each_init(
            || {
                (
                    VolumeRayIntersector::new(grid, 0).expect("checked in new"),
                    VolumeRayIntersector::new(grid, 0).expect("checked in new"),
                    grid.accessor(),
                )
            },
            |(primary, shadow, acc), (j, row)| {
                let xform = grid.transform();
                let ws = |p: Vec3d| BoxSampler::sample(&*acc, xform.world_to_index(p)).to_f64();
                for (i, px) in row.iter_mut().enumerate() {
                    *px = Rgba::with_alpha(0.0, 0.0, 0.0, 0.0);
                    let pray = camera.get_ray(i, j, 0.5, 0.5);
                    if !primary.set_world_ray(&pray) {
                        continue;
                    }
                    let mut p_trans = Vec3d::splat(1.0);
                    let mut p_lumi = Vec3d::zero();
                    'pixel: for span in primary.hits() {
                        let mut pt = p_step * (span.t0 / p_step).ceil();
                        while pt <= span.t1 {
                            let ppos = primary.world_pos(pt);
                            pt += p_step;
                            let density = ws(ppos);
                            if density < cutoff {
                                continue;
                            }
                            let dt = exp(extinction * (density * p_step));
                            let mut s_trans = Vec3d::splat(1.0);
                            if !shadow.set_world_ray(&Ray::new(ppos, light_dir)) {
                                continue;
                            }
                            'shadow: for sspan in shadow.hits() {
                                let mut st = s_step * (sspan.t0 / s_step).ceil();
                                while st <= sspan.t1 {
                                    let d = ws(shadow.world_pos(st));
                                    let cur = st;
                                    st += s_step;
                                    if d < cutoff {
                                        continue;
                                    }
                                    s_trans = s_trans.mul_elem(&exp(extinction * (d * s_step / (1.0 + cur * gain))));
                                    if s_trans.length_sqr() < cutoff {
                                        break 'shadow;
                                    }
                                }
                            }
                            p_lumi += albedo.mul_elem(&s_trans).mul_elem(&p_trans).mul_elem(&(Vec3d::splat(1.0) - dt));
                            p_trans = p_trans.mul_elem(&dt);
                            if p_trans.length_sqr() < cutoff {
                                break 'pixel;
                            }
                        }
                    }
                    *px = Rgba::with_alpha(
                        p_lumi[0] as f32,
                        p_lumi[1] as f32,
                        p_lumi[2] as f32,
                        1.0 - (p_trans.sum() / 3.0) as f32,
                    );
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::test_util::{assert_close, make_sphere};

    #[test]
    fn mt19937_matches_std() {
        // First outputs of std::mt19937 with the default seed 5489.
        let mut r = Mt19937::new(5489);
        assert_eq!(r.next_u32(), 3499211612);
        assert_eq!(r.next_u32(), 581869302);
        let mut r = Mt19937::new(0);
        assert_close(r.next_f64(), 0.592844616516683, 1e-12);
    }

    /// Reference pixels from `scratchpad/cppref/raytrace.cpp` (OpenVDB 10).
    #[test]
    fn matches_openvdb_reference() {
        let grid = make_sphere::<f32>(1.0, Vec3d::new(0.1, -0.05, 0.02), 0.05, 3.0);
        let mut film = Film::with_background(40, 30, Rgba::gray(0.25));
        let camera =
            PerspectiveCamera::new(&film, Vec3d::new(-10.0, 20.0, 0.0), Vec3d::new(1.0, 1.0, 4.0), 50.0, 41.2136);
        ray_trace(&grid, &DiffuseShader::default(), &camera, &mut film, 1, 0).unwrap();
        let expect = [
            ((20, 15), [0.841_141_1_f32, 0.841_141_1, 0.841_141_1]),
            ((5, 5), [0.25, 0.25, 0.25]),
            ((25, 10), [0.700_018_2, 0.700_018_2, 0.700_018_2]),
            ((12, 18), [0.25, 0.25, 0.25]),
        ];
        for ((i, j), e) in expect {
            let p = film.pixel(i, j);
            assert_close(p.r as f64, e[0] as f64, 1e-5);
            assert_close(p.g as f64, e[1] as f64, 1e-5);
            assert_close(p.b as f64, e[2] as f64, 1e-5);
        }
        let mut film2 = Film::new(40, 30);
        let mut cam2 = OrthographicCamera::new(&film2, Vec3d::zero(), Vec3d::new(0.0, 0.0, 5.0), 3.0);
        cam2.look_at(Vec3d::new(0.1, 0.0, 0.0), Vec3d::new(0.0, 1.0, 0.0));
        let tracer = LevelSetRayTracer::new(&grid, 4, 7).unwrap();
        tracer.render(&cam2, &NormalShader::default(), &mut film2);
        let expect2 = [
            ((20, 15), [0.506_256_04_f32, 0.506_258_7, 0.999_687]),
            ((10, 10), [0.136_611_52, 0.693_369_6, 0.78349936]),
            ((30, 20), [0.877_458_7, 0.318_692_4, 0.772_674_8]),
            ((0, 0), [0.0, 0.0, 0.0]),
        ];
        for ((i, j), e) in expect2 {
            let p = film2.pixel(i, j);
            assert_close(p.r as f64, e[0] as f64, 1e-5);
            assert_close(p.g as f64, e[1] as f64, 1e-5);
            assert_close(p.b as f64, e[2] as f64, 1e-5);
        }
        let mut buf = Vec::new();
        film2.write_ppm(&mut buf).unwrap();
        assert!(buf.starts_with(b"P6\n40 30\n255\n"));
        assert_eq!(buf.len(), 13 + 40 * 30 * 3);
    }

    #[test]
    fn volume_render_matches_openvdb() {
        let mut fog = Grid::<f32>::new(0.0);
        fog.tree_mut().fill(&crate::math::CoordBBox::from_coords(-8, -8, -8, 8, 8, 8), 0.5, true);
        let mut film = Film::new(16, 16);
        let cam = PerspectiveCamera::new(&film, Vec3d::zero(), Vec3d::new(0.0, 0.0, 40.0), 50.0, 41.2136);
        let vr = VolumeRender::new(&fog).unwrap();
        vr.render(&cam, &mut film);
        // Reference values from the C++ VolumeRender.
        let expect =
            [((8, 8), [0.021_344_833_f32, 0.981_370_9]), ((0, 0), [0.0, 0.0]), ((6, 9), [0.021_267_513, 0.978_688])];
        for ((i, j), e) in expect {
            let p = film.pixel(i, j);
            assert_close(p.r as f64, e[0] as f64, 1e-6);
            assert_close(p.a as f64, e[1] as f64, 1e-6);
        }
    }
}
