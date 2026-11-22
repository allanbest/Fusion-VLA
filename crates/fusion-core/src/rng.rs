//! Deterministic pseudo-randomness.
//!
//! PLAN.md §22 puts the seed in the tuple that makes a run reproducible, and
//! is specific about how: "RNG seeds, explicitly logged, never implicit." The
//! failure it is guarding against is the ordinary one — a scene that scatters
//! debris from a thread-local generator seeded off the clock, and which
//! therefore cannot be reproduced from its own project file no matter how
//! carefully everything else was recorded.
//!
//! **Nothing in Fusion consumes randomness yet.** This exists now because the
//! seed belongs in the project format from v1 rather than being retrofitted
//! into a file format that scenes already exist in, and because "seeded from
//! the world, never from the environment" is a rule that has to be in place
//! *before* the first caller wants a random number, not after.
//!
//! PCG-XSH-RR 64/32 (O'Neill 2014). Chosen over the shift-register generators
//! that would also have done, for one reason that matters here: consecutive
//! seeds produce uncorrelated streams. A sweep over `seed = 0..100` — which is
//! exactly what PLAN.md §23's research workflow does — gets a hundred
//! different runs from an LCG-with-output-permutation and a hundred visibly
//! related ones from a naive xorshift.

/// A small, fast, reproducible generator.
///
/// Not cryptographic, and it should never be used as though it were. The
/// property being bought is that the same seed gives the same stream on every
/// machine and every run, which is the opposite of what a cryptographic
/// generator is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    state: u64,
    /// Stream selector. Two generators with the same seed and different
    /// streams produce independent sequences, which is what lets one scene
    /// give each system its own generator without them correlating.
    increment: u64,
}

impl Rng {
    const MULTIPLIER: u64 = 6_364_136_223_846_793_005;

    /// A generator for `seed`, on the default stream.
    pub fn new(seed: u64) -> Self {
        Self::with_stream(seed, 0xda3e_39cb_94b9_5bdb)
    }

    /// A generator for `seed` on an independent stream.
    ///
    /// Use one stream per system rather than sharing a generator. Sharing
    /// makes the numbers any one system sees depend on how many the others
    /// drew, so adding a feature that happens to want two random numbers
    /// changes the answer everywhere else — a reproducibility bug that looks
    /// exactly like a physics bug.
    pub fn with_stream(seed: u64, stream: u64) -> Self {
        // The increment must be odd for the LCG to have full period.
        let increment = (stream << 1) | 1;
        let mut rng = Self {
            state: 0,
            increment,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// The next 32 bits.
    pub fn next_u32(&mut self) -> u32 {
        let previous = self.state;
        self.state = previous
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.increment);
        // XSH-RR: xor-shift the high bits down, then rotate by an amount taken
        // from the bits above them. The rotation is what makes the low-order
        // bits as good as the high-order ones, which a bare LCG's are not.
        let xorshifted = (((previous >> 18) ^ previous) >> 27) as u32;
        let rotation = (previous >> 59) as u32;
        xorshifted.rotate_right(rotation)
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        (self.next_u32() as u64) << 32 | self.next_u32() as u64
    }

    /// Uniform in `[0, 1)`.
    ///
    /// Built from 24 bits rather than 32: `f32` has 24 bits of mantissa, so
    /// the extra 8 would be rounded away, and rounding away the low bits of a
    /// uniform makes 1.0 reachable. A generator that occasionally returns
    /// exactly 1.0 from a half-open range is a bug that surfaces once a
    /// million draws, in someone else's code.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Uniform in `[low, high)`.
    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// Uniform in `[0, n)`.
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        // Rejection sampling against the largest multiple of `n` that fits,
        // rather than a modulo. A bare `% n` is biased towards small values by
        // up to `n / 2^32`, which is invisible for small `n` and is exactly
        // the sort of thing that makes a sweep's results quietly wrong.
        let threshold = (u32::MAX - n + 1) % n;
        loop {
            let value = self.next_u32();
            if value >= threshold {
                return value % n;
            }
        }
    }

    /// A point uniform inside the unit sphere.
    pub fn in_unit_sphere(&mut self) -> fusion_math::Vec3 {
        loop {
            let p = fusion_math::Vec3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            if p.length_squared() <= 1.0 {
                return p;
            }
        }
    }
}
