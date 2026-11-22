use core_math::Vec3;

/// An axis-aligned box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Contains nothing; the identity of [`Aabb::union`].
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f64::INFINITY),
        max: Vec3::splat(f64::NEG_INFINITY),
    };
    /// Contains everything.
    pub const EVERYTHING: Self = Self {
        min: Vec3::splat(f64::NEG_INFINITY),
        max: Vec3::splat(f64::INFINITY),
    };

    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn around(centre: Vec3, half: Vec3) -> Self {
        Self::new(centre - half, centre + half)
    }

    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        points.into_iter().fold(Self::EMPTY, |b, p| b.including(p))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    #[inline]
    pub fn including(&self, p: Vec3) -> Self {
        Self::new(self.min.min(p), self.max.max(p))
    }

    #[inline]
    pub fn union(&self, o: &Self) -> Self {
        Self::new(self.min.min(o.min), self.max.max(o.max))
    }

    /// Grown by `margin` on every side.
    #[inline]
    pub fn grown(&self, margin: f64) -> Self {
        Self::new(
            self.min - Vec3::splat(margin),
            self.max + Vec3::splat(margin),
        )
    }

    #[inline]
    pub fn overlaps(&self, o: &Self) -> bool {
        self.min.x <= o.max.x
            && self.max.x >= o.min.x
            && self.min.y <= o.max.y
            && self.max.y >= o.min.y
            && self.min.z <= o.max.z
            && self.max.z >= o.min.z
    }

    #[inline]
    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    #[inline]
    pub fn centre(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn surface_area(&self) -> f64 {
        let s = self.size();
        2.0 * (s.x * s.y + s.y * s.z + s.z * s.x)
    }
}
