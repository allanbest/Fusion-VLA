use crate::{Quat, Vec3};

/// Where something is and which way it faces: a rotation, then a
/// translation. Maps a body's own frame to the world.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Pose {
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    #[inline]
    pub const fn new(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }

    #[inline]
    pub const fn at(position: Vec3) -> Self {
        Self::new(position, Quat::IDENTITY)
    }

    #[inline]
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.rotation.rotate(p) + self.position
    }

    #[inline]
    pub fn transform_vector(&self, v: Vec3) -> Vec3 {
        self.rotation.rotate(v)
    }

    #[inline]
    pub fn inverse_transform_point(&self, p: Vec3) -> Vec3 {
        self.rotation.rotate_inverse(p - self.position)
    }

    #[inline]
    pub fn inverse_transform_vector(&self, v: Vec3) -> Vec3 {
        self.rotation.rotate_inverse(v)
    }

    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.conjugate();
        Self::new(rotation.rotate(-self.position), rotation)
    }

    /// `self ∘ o`: first `o`, then `self`.
    pub fn then(&self, o: &Self) -> Self {
        Self::new(
            self.transform_point(o.position),
            (self.rotation * o.rotation).normalize(),
        )
    }
}
