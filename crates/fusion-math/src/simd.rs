//! Four lanes of `f32`, and four points at once.
//!
//! PLAN.md §24 sequences CPU SIMD ahead of GPU work below about 10⁵ particles,
//! on the grounds that early content is bound by branch-heavy scalar loops
//! rather than by bandwidth. The measured fluid profile agrees: at 3500
//! particles the neighbour build is 53% of a fluid substep and the density
//! solve another 27%, both of them the same shape of loop — sweep a list of
//! points, difference each against one centre, compare a squared distance.
//! That is a lane-parallel problem four wide.
//!
//! ## Why not `core::arch`
//!
//! Intrinsics — `vsubq_f32`, `_mm_sub_ps` — would be the obvious way to write
//! this, and they are not what this module does. Three reasons, in order of
//! weight:
//!
//! 1. **This crate denies `unsafe`.** Every `core::arch` intrinsic is unsafe,
//!    so the whole of L0's safety property would be spent on a shape LLVM
//!    already emits from safe code.
//! 2. **One implementation instead of three.** NEON, SSE and a scalar fallback
//!    for anything else is three code paths to keep in agreement, and they
//!    would disagree eventually — a lane-order mistake in the path that only
//!    runs on the platform nobody develops on is the kind of bug that survives
//!    for a year.
//! 3. **WASM.** §25 wants a browser demo tier later. `core::arch::wasm32`
//!    simd128 is a fourth path; a `[f32; 4]` is not.
//!
//! What replaces them: a `#[repr(align(16))]` wrapper around `[f32; 4]` whose
//! every operation is written out as four scalar expressions with no loop and
//! no branch. LLVM's SLP vectorizer packs that into one instruction per
//! operation — `Vec3x4::sub` followed by `length_squared` becomes 3 `fsub.4s`,
//! 3 `fmul.4s` and 2 `fadd.4s` on aarch64, and a cutoff comparison is one
//! `fcmge.4s`.
//!
//! ## Three things that stop it, all of them found the hard way
//!
//! This is an optimizer-dependent bargain, so the conditions are written down
//! rather than assumed. Each of these produced *correct, scalar* code that
//! looked four wide in the source.
//!
//! 1. **Bounds checks.** `Self([src[at], src[at + 1], ...])` is four checked
//!    loads, and LLVM will not form a vector across the branches they imply.
//!    It emits four scalar `fadd`s into four scalar registers instead — the
//!    right answer at roughly the speed of the code this was meant to replace.
//!    The fix is [`F32x4::from_chunk`]: hand it a slice `chunks_exact(4)` has
//!    already proven to be four long, and the checks disappear along with the
//!    scalarization. Every loop in the engine that uses this module is driven
//!    by `chunks_exact` for that reason and no other.
//! 2. **Gathers.** `src[index[k]]` for scattered `k` cannot be checked away at
//!    all, so a gather-fed loop stays scalar however it is written. The
//!    consequence is architectural rather than local: the fluid solver caches
//!    its per-pair kernel values into arrays *parallel to the neighbour list*
//!    so that its six consumer loops read contiguously and only the one loop
//!    that fills the cache pays for a gather.
//! 3. **Reading the wrong artifact.** `cargo rustc -- --emit asm` shows the
//!    crate before thin LTO and cross-crate inlining, and it showed *zero*
//!    vector instructions for code that vectorizes perfectly well in the
//!    shipped binary. Check `otool -tV` / `objdump -d` on the linked
//!    executable, not the intermediate.
//!
//! ## One operation has no vector form
//!
//! Turning a lane comparison into an integer mask is a single `movmskps` on
//! SSE and has no NEON equivalent; LLVM emits about a dozen `umov`/`csel`
//! instructions to assemble the four bits by hand. So loops that can avoid
//! asking which lanes passed — masking to zero and summing unconditionally
//! instead, which the SPH kernels permit because they are already zero outside
//! their support — do that, and only the neighbour build, which genuinely has
//! to write out indices, pays for [`F32x4::le_mask`].
//!
//! ## What it is worth
//!
//! Less than the arithmetic suggests, and the reason is worth knowing before
//! reaching for this type. A wide `fadd.4s` and four scalar `fadd`s retire in
//! about the same time on a core that issues four floating-point operations a
//! cycle; what the vector form actually saves is instruction count, issue
//! slots and register pressure. The measured wins in the fluid solver came
//! mostly from the *layout* this type forces — contiguous, single-precision,
//! cache-local — and only partly from the instructions it emits. Benchmark
//! rather than assume: `cargo run --release -p fusion-demos --bin bench
//! profile`.
//!
//! ## Lanes and determinism
//!
//! Summing four lanes and adding them at the end is a different order from
//! summing four values in sequence, and in floating point that is a different
//! answer. PLAN.md §34's tier (a) — same machine, same build, bit-identical —
//! is untouched by that, because the lane order here is fixed by the source
//! rather than by the hardware. Tier (b), cross-platform agreement within
//! tolerance, is where a different SIMD width would show up, and §25 already
//! names that as one of the reasons cross-platform bit-identity is declined.
//!
//! Every reduction in this module therefore has a **specified** order:
//! `(0 + 1) + (2 + 3)`, pairwise, which is what the vector instruction does
//! and what the scalar fallback is written to match.

/// Four `f32` lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C, align(16))]
pub struct F32x4(pub [f32; 4]);

impl F32x4 {
    /// All lanes zero.
    pub const ZERO: Self = Self([0.0; 4]);

    /// Every lane set to `v`.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self([v; 4])
    }

    /// Four lanes from four values.
    #[inline]
    pub const fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self([a, b, c, d])
    }

    /// A lane per element of a four-element slice.
    ///
    /// **This, and not an indexed load, is how the loops here read memory**,
    /// and the reason is the whole reason the module has a measured shape
    /// rather than an assumed one. `Self([src[at], src[at+1], ...])` produces
    /// bounds checks LLVM will not vectorize across; it emits four scalar
    /// `fadd`s into four scalar registers, which is correct and is not what
    /// the type is for. A slice that `chunks_exact` has already proven to be
    /// four long carries no check, and the same expression becomes vector
    /// arithmetic.
    ///
    /// So callers drive their loops with [`slice::chunks_exact`] and hand the
    /// chunks here. See [`Vec3x4::chunks`] for the three-array version.
    ///
    /// # Panics
    ///
    /// If `chunk` is not exactly four long. Unreachable from `chunks_exact`,
    /// and the check is what lets the loads be unchecked.
    #[inline]
    pub fn from_chunk(chunk: &[f32]) -> Self {
        Self(chunk.try_into().expect("a chunk of exactly four"))
    }

    /// Four values picked out of `src` by index — a gather.
    ///
    /// The neighbour loops need this and there is no way around it: a
    /// particle's neighbours are wherever they are. On hardware with a gather
    /// instruction LLVM may use one; on hardware without, this is four loads
    /// and three inserts, which is still fewer instructions than four separate
    /// scalar iterations of the surrounding loop.
    #[inline]
    pub fn gather(src: &[f32], index: [usize; 4]) -> Self {
        Self([src[index[0]], src[index[1]], src[index[2]], src[index[3]]])
    }

    /// Bit `k` is set where `self[k] <= other[k]`.
    ///
    /// A mask rather than a `[bool; 4]`, because the caller's next move is to
    /// test lanes one at a time and `mask & 1 << k` compiles to a shift and a
    /// test with no memory round-trip.
    #[inline]
    pub fn le_mask(self, other: Self) -> u32 {
        (self.0[0] <= other.0[0]) as u32
            | ((self.0[1] <= other.0[1]) as u32) << 1
            | ((self.0[2] <= other.0[2]) as u32) << 2
            | ((self.0[3] <= other.0[3]) as u32) << 3
    }

    /// Every lane scaled by one scalar.
    #[inline]
    pub fn scale(self, s: f32) -> Self {
        self * Self::splat(s)
    }

    /// Lane-wise `powf`. The one operation here with no vector instruction
    /// behind it — four library calls — and it is provided anyway so that a
    /// loop needing it does not have to fall back to scalar wholesale.
    #[inline]
    pub fn powf(self, e: f32) -> Self {
        Self([
            self.0[0].powf(e),
            self.0[1].powf(e),
            self.0[2].powf(e),
            self.0[3].powf(e),
        ])
    }

    /// Lane-wise minimum.
    #[inline]
    pub fn min(self, o: Self) -> Self {
        Self([
            self.0[0].min(o.0[0]),
            self.0[1].min(o.0[1]),
            self.0[2].min(o.0[2]),
            self.0[3].min(o.0[3]),
        ])
    }

    /// Lane-wise maximum.
    #[inline]
    pub fn max(self, o: Self) -> Self {
        Self([
            self.0[0].max(o.0[0]),
            self.0[1].max(o.0[1]),
            self.0[2].max(o.0[2]),
            self.0[3].max(o.0[3]),
        ])
    }

    /// Lane-wise square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        Self([
            self.0[0].sqrt(),
            self.0[1].sqrt(),
            self.0[2].sqrt(),
            self.0[3].sqrt(),
        ])
    }

    /// `self` where the lane's bit is set in `mask`, zero elsewhere.
    ///
    /// The branch-free way to drop a lane that failed a test, which matters
    /// more than it looks: the kernels these loops evaluate are zero outside
    /// their support anyway, so masking to zero and summing unconditionally
    /// gives the same answer as skipping the lane, without the branch.
    #[inline]
    pub fn select_mask(self, mask: u32) -> Self {
        Self([
            if mask & 1 != 0 { self.0[0] } else { 0.0 },
            if mask & 2 != 0 { self.0[1] } else { 0.0 },
            if mask & 4 != 0 { self.0[2] } else { 0.0 },
            if mask & 8 != 0 { self.0[3] } else { 0.0 },
        ])
    }

    /// Horizontal sum, pairwise: `(0 + 1) + (2 + 3)`. See the module note on
    /// why the order is written down rather than left to the compiler.
    #[inline]
    pub fn sum(self) -> f32 {
        (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
    }

    /// Read one lane out. Panics outside 0..4.
    #[inline]
    pub fn lane(self, k: usize) -> f32 {
        self.0[k]
    }
}

/// Four points, one axis per register — structure of arrays, four wide.
///
/// This is the layout the loops want and it is not the layout the data is in;
/// `gather` is where the two meet. Holding four points as `[Vec3; 4]` instead
/// would make the loads cheap and every arithmetic operation a horizontal one,
/// which is the wrong trade: a dot product across lanes needs shuffles, a dot
/// product down lanes is three multiplies and two adds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3x4 {
    /// The four x components.
    pub x: F32x4,
    /// The four y components.
    pub y: F32x4,
    /// The four z components.
    pub z: F32x4,
}

impl Vec3x4 {
    /// Four zero vectors.
    pub const ZERO: Self = Self {
        x: F32x4::ZERO,
        y: F32x4::ZERO,
        z: F32x4::ZERO,
    };

    /// From three lanes of components.
    #[inline]
    pub const fn new(x: F32x4, y: F32x4, z: F32x4) -> Self {
        Self { x, y, z }
    }

    /// The same point in all four lanes — the query centre of a neighbour
    /// sweep.
    #[inline]
    pub fn splat(v: crate::Vec3) -> Self {
        Self {
            x: F32x4::splat(v.x),
            y: F32x4::splat(v.y),
            z: F32x4::splat(v.z),
        }
    }

    /// Walk three parallel axis arrays four points at a time.
    ///
    /// The arrays are expected to be the same length; the iterator stops with
    /// the shortest, and [`Vec3x4Chunks::remainder`] hands back what was left
    /// over for the caller's scalar tail.
    #[inline]
    pub fn chunks<'a>(x: &'a [f32], y: &'a [f32], z: &'a [f32]) -> Vec3x4Chunks<'a> {
        Vec3x4Chunks {
            x: x.chunks_exact(4),
            y: y.chunks_exact(4),
            z: z.chunks_exact(4),
        }
    }

    /// Four points picked out by index.
    #[inline]
    pub fn gather(x: &[f32], y: &[f32], z: &[f32], index: [usize; 4]) -> Self {
        Self {
            x: F32x4::gather(x, index),
            y: F32x4::gather(y, index),
            z: F32x4::gather(z, index),
        }
    }

    /// Scale each lane's vector by that lane's scalar.
    #[inline]
    pub fn scale_lanes(self, s: F32x4) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Lane-wise squared length.
    #[inline]
    pub fn length_squared(self) -> F32x4 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Sum the four lanes into one vector, pairwise per component.
    #[inline]
    pub fn sum(self) -> crate::Vec3 {
        crate::Vec3::new(self.x.sum(), self.y.sum(), self.z.sum())
    }

    /// Keep the lanes whose bit is set in `mask`, zeroing the rest. The bit
    /// for lane `k` is `1 << k`.
    #[inline]
    pub fn select_mask(self, mask: u32) -> Self {
        Self {
            x: self.x.select_mask(mask),
            y: self.y.select_mask(mask),
            z: self.z.select_mask(mask),
        }
    }

    /// Read one lane out as an ordinary vector. Panics outside 0..4.
    #[inline]
    pub fn lane(self, k: usize) -> crate::Vec3 {
        crate::Vec3::new(self.x.lane(k), self.y.lane(k), self.z.lane(k))
    }
}

/// Three parallel axis arrays, four points at a time. See [`Vec3x4::chunks`].
pub struct Vec3x4Chunks<'a> {
    x: core::slice::ChunksExact<'a, f32>,
    y: core::slice::ChunksExact<'a, f32>,
    z: core::slice::ChunksExact<'a, f32>,
}

impl<'a> Vec3x4Chunks<'a> {
    /// The one to three points at the end that did not fill a chunk, as three
    /// equal-length slices.
    #[inline]
    pub fn remainder(&self) -> (&'a [f32], &'a [f32], &'a [f32]) {
        (self.x.remainder(), self.y.remainder(), self.z.remainder())
    }
}

impl Iterator for Vec3x4Chunks<'_> {
    type Item = Vec3x4;

    #[inline]
    fn next(&mut self) -> Option<Vec3x4> {
        Some(Vec3x4::new(
            F32x4::from_chunk(self.x.next()?),
            F32x4::from_chunk(self.y.next()?),
            F32x4::from_chunk(self.z.next()?),
        ))
    }
}

// The arithmetic is the `std::ops` traits rather than inherent `add`/`sub`
// methods, for the ordinary reason: `a * b` reads as arithmetic and
// `a.mul(b).add(c.mul(d))` does not, and a type whose whole purpose is to be
// written in tight numeric loops should not make those loops harder to read
// than the scalar code they replace. Every one of them is lane-wise.
macro_rules! lanewise {
    ($trait:ident, $method:ident, $op:tt) => {
        impl core::ops::$trait for F32x4 {
            type Output = Self;
            #[inline]
            fn $method(self, o: Self) -> Self {
                Self([
                    self.0[0] $op o.0[0],
                    self.0[1] $op o.0[1],
                    self.0[2] $op o.0[2],
                    self.0[3] $op o.0[3],
                ])
            }
        }
    };
}
lanewise!(Add, add, +);
lanewise!(Sub, sub, -);
lanewise!(Mul, mul, *);
lanewise!(Div, div, /);

impl core::ops::Add for Vec3x4 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }
}

impl core::ops::Sub for Vec3x4 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3;

    #[test]
    fn lanes_are_independent() {
        let a = F32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = F32x4::splat(2.0);
        assert_eq!((a + b).0, [3.0, 4.0, 5.0, 6.0]);
        assert_eq!((a - b).0, [-1.0, 0.0, 1.0, 2.0]);
        assert_eq!((a * b).0, [2.0, 4.0, 6.0, 8.0]);
        assert_eq!(a.sum(), 10.0);
    }

    #[test]
    fn chunks_walk_in_step_and_hand_back_the_tail() {
        let (x, y, z) = ([1.0f32; 6], [2.0f32; 6], [3.0f32; 6]);
        let mut chunks = Vec3x4::chunks(&x, &y, &z);
        assert_eq!(chunks.next().unwrap().sum(), Vec3::new(4.0, 8.0, 12.0));
        assert!(chunks.next().is_none());
        let (rx, ry, rz) = chunks.remainder();
        assert_eq!((rx.len(), ry.len(), rz.len()), (2, 2, 2));
    }

    #[test]
    fn mask_bit_order_is_lane_order() {
        let a = F32x4::new(1.0, 5.0, 2.0, 6.0);
        let b = F32x4::splat(3.0);
        // lanes 0 and 2 are <= 3.
        assert_eq!(a.le_mask(b), 0b0101);
        assert_eq!(a.select_mask(0b0101).0, [1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn gather_picks_by_index() {
        let src = [0.0f32, 10.0, 20.0, 30.0, 40.0];
        assert_eq!(F32x4::gather(&src, [4, 0, 2, 2]).0, [40.0, 0.0, 20.0, 20.0]);
    }

    /// The point of the type: four squared distances at once, matching what
    /// the scalar code they replace would have produced exactly.
    #[test]
    fn matches_scalar_squared_distance() {
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 0.5, 7.25),
            Vec3::new(0.125, -0.25, 0.5),
        ];
        let (xs, ys, zs): (Vec<f32>, Vec<f32>, Vec<f32>) = (
            points.iter().map(|p| p.x).collect(),
            points.iter().map(|p| p.y).collect(),
            points.iter().map(|p| p.z).collect(),
        );
        let centre = Vec3::new(0.75, -1.5, 2.0);
        let wide = (Vec3x4::chunks(&xs, &ys, &zs).next().unwrap() - Vec3x4::splat(centre))
            .length_squared();
        for (k, p) in points.iter().enumerate() {
            let scalar = *p - centre;
            let expected = (scalar.x * scalar.x + scalar.y * scalar.y) + scalar.z * scalar.z;
            // Not `assert_relative_eq`: the claim is bit-identical, because
            // the lane expression is the same three multiplies and two adds in
            // the same associativity as the scalar one.
            assert_eq!(wide.lane(k), expected, "lane {k}");
        }
    }

    #[test]
    fn vector_lanes_sum_and_scale() {
        let v = Vec3x4::new(
            F32x4::new(1.0, 2.0, 3.0, 4.0),
            F32x4::splat(1.0),
            F32x4::new(0.0, 0.0, 0.0, 0.0),
        );
        assert_eq!(v.sum(), Vec3::new(10.0, 4.0, 0.0));
        let scaled = v.scale_lanes(F32x4::new(0.0, 1.0, 2.0, 3.0));
        assert_eq!(scaled.sum(), Vec3::new(20.0, 6.0, 0.0));
        assert_eq!(v.select_mask(0b0011).sum(), Vec3::new(3.0, 2.0, 0.0));
    }
}
