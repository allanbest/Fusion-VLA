//! Orbit camera and the 4x4 matrices that go with it.
//!
//! `fusion-math` has no `Mat4` on purpose: a 4x4 matrix is a graphics object,
//! not a physics one. Rigid transforms are [`fusion_math::Isometry`], and
//! projection lives here, next to the only code that needs it.

use fusion_math::{vec3, Vec3};

/// Column-major 4x4, in the memory order WGSL expects.
pub type Mat4 = [[f32; 4]; 4];

pub fn identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn multiply(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for (c, col) in out.iter_mut().enumerate() {
        for (r, cell) in col.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

/// Right-handed orthographic onto wgpu's 0..1 depth range.
///
/// What a directional light sees. A sun is infinitely far away, so its
/// shadow map is an orthographic box around whatever needs to cast — not a
/// perspective frustum, which would make shadows converge towards a point
/// the light does not have.
pub fn orthographic(half_width: f32, half_height: f32, near: f32, far: f32) -> Mat4 {
    let mut m = [[0.0f32; 4]; 4];
    m[0][0] = 1.0 / half_width.max(1e-4);
    m[1][1] = 1.0 / half_height.max(1e-4);
    m[2][2] = 1.0 / (near - far);
    m[3][2] = near / (near - far);
    m[3][3] = 1.0;
    m
}

/// General 4x4 inverse, by cofactors.
///
/// Needed because SSAO reconstructs a view-space position from a depth
/// sample, which is the projection run backwards. Only ever called on the
/// projection and view matrices — once per frame, so the cofactor expansion
/// being the slow way to do it does not matter, and it is the way that is
/// obviously correct.
pub fn inverse(m: &Mat4) -> Mat4 {
    let a = |r: usize, c: usize| m[c][r];
    let mut inv = [[0.0f32; 4]; 4];
    let s0 = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    let s1 = a(0, 0) * a(1, 2) - a(1, 0) * a(0, 2);
    let s2 = a(0, 0) * a(1, 3) - a(1, 0) * a(0, 3);
    let s3 = a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2);
    let s4 = a(0, 1) * a(1, 3) - a(1, 1) * a(0, 3);
    let s5 = a(0, 2) * a(1, 3) - a(1, 2) * a(0, 3);
    let c5 = a(2, 2) * a(3, 3) - a(3, 2) * a(2, 3);
    let c4 = a(2, 1) * a(3, 3) - a(3, 1) * a(2, 3);
    let c3 = a(2, 1) * a(3, 2) - a(3, 1) * a(2, 2);
    let c2 = a(2, 0) * a(3, 3) - a(3, 0) * a(2, 3);
    let c1 = a(2, 0) * a(3, 2) - a(3, 0) * a(2, 2);
    let c0 = a(2, 0) * a(3, 1) - a(3, 0) * a(2, 1);
    let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if det.abs() < 1e-20 {
        return identity();
    }
    let d = 1.0 / det;
    let mut set = |r: usize, c: usize, v: f32| inv[c][r] = v * d;
    set(0, 0, a(1, 1) * c5 - a(1, 2) * c4 + a(1, 3) * c3);
    set(0, 1, -a(0, 1) * c5 + a(0, 2) * c4 - a(0, 3) * c3);
    set(0, 2, a(3, 1) * s5 - a(3, 2) * s4 + a(3, 3) * s3);
    set(0, 3, -a(2, 1) * s5 + a(2, 2) * s4 - a(2, 3) * s3);
    set(1, 0, -a(1, 0) * c5 + a(1, 2) * c2 - a(1, 3) * c1);
    set(1, 1, a(0, 0) * c5 - a(0, 2) * c2 + a(0, 3) * c1);
    set(1, 2, -a(3, 0) * s5 + a(3, 2) * s2 - a(3, 3) * s1);
    set(1, 3, a(2, 0) * s5 - a(2, 2) * s2 + a(2, 3) * s1);
    set(2, 0, a(1, 0) * c4 - a(1, 1) * c2 + a(1, 3) * c0);
    set(2, 1, -a(0, 0) * c4 + a(0, 1) * c2 - a(0, 3) * c0);
    set(2, 2, a(3, 0) * s4 - a(3, 1) * s2 + a(3, 3) * s0);
    set(2, 3, -a(2, 0) * s4 + a(2, 1) * s2 - a(2, 3) * s0);
    set(3, 0, -a(1, 0) * c3 + a(1, 1) * c1 - a(1, 2) * c0);
    set(3, 1, a(0, 0) * c3 - a(0, 1) * c1 + a(0, 2) * c0);
    set(3, 2, -a(3, 0) * s3 + a(3, 1) * s1 - a(3, 2) * s0);
    set(3, 3, a(2, 0) * s3 - a(2, 1) * s1 + a(2, 2) * s0);
    inv
}

/// Right-handed perspective onto wgpu's 0..1 depth range.
pub fn perspective(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y_radians * 0.5).tan();
    let mut m = [[0.0f32; 4]; 4];
    m[0][0] = f / aspect.max(1e-4);
    m[1][1] = f;
    m[2][2] = far / (near - far);
    m[2][3] = -1.0;
    m[3][2] = far * near / (near - far);
    m
}

pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = (target - eye).normalize_or_zero();
    let s = f.cross(up).normalize_or_zero();
    let u = s.cross(f);
    [
        [s.x, u.x, -f.x, 0.0],
        [s.y, u.y, -f.y, 0.0],
        [s.z, u.z, -f.z, 0.0],
        [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
    ]
}

/// A camera that orbits a focus point. Enough for a debug viewport and
/// nothing more — flythrough cameras are a v0.5 concern.
#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub distance: f32,
    /// Radians around +Y.
    pub yaw: f32,
    /// Radians above the horizon, clamped away from the poles.
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: vec3(0.0, 1.0, 0.0),
            distance: 18.0,
            yaw: 0.7,
            pitch: 0.45,
            fov_y: 55f32.to_radians(),
            near: 0.05,
            far: 2000.0,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        self.focus + vec3(cp * sy, sp, cp * cy) * self.distance
    }

    /// Point the camera at `focus` from `eye`, exactly.
    ///
    /// The inverse of [`Self::eye`]: it decomposes the offset back into the
    /// yaw, pitch and distance that would produce it. What it is *for* is
    /// moving the pivot without moving the picture — set the focus to the
    /// point under the cursor and the view does not shift by a pixel, but
    /// every subsequent orbit turns about that point instead of about
    /// wherever the camera was last framed. That is the difference between an
    /// orbit camera that is pleasant and one that swings the thing you are
    /// looking at out of frame.
    pub fn look_from(&mut self, eye: Vec3, focus: Vec3) {
        let offset = eye - focus;
        let distance = offset.length();
        if !distance.is_finite() || distance < 1e-4 {
            return;
        }
        self.focus = focus;
        self.distance = distance.clamp(Self::NEAREST, Self::FURTHEST);
        // Clamped away from the poles: at exactly straight up the yaw is
        // undefined, and a camera that arrives there loses its horizon and
        // rolls unpredictably on the next drag.
        self.pitch = (offset.y / distance)
            .clamp(-1.0, 1.0)
            .asin()
            .clamp(-1.4, 1.4);
        self.yaw = offset.x.atan2(offset.z);
    }

    /// How close and how far the camera may get. Shared with the editor's
    /// scroll handling so that zoom and framing agree about the limits.
    pub const NEAREST: f32 = 0.2;
    /// The far end of the same range.
    pub const FURTHEST: f32 = 400.0;

    pub fn view(&self) -> Mat4 {
        look_at(self.eye(), self.focus, Vec3::Y)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        perspective(self.fov_y, aspect, self.near, self.far)
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        multiply(&self.projection(aspect), &self.view())
    }

    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw -= delta_yaw;
        // Stop just short of the poles, where the up vector degenerates.
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = (self.pitch + delta_pitch).clamp(-limit, limit);
    }

    pub fn zoom(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.distance = (self.distance * factor).clamp(Self::NEAREST, Self::FURTHEST);
        }
    }

    /// Pan across the view plane, scaled by distance so it feels the same at
    /// any zoom.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.focus - self.eye()).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward);
        self.focus += (right * -dx + up * dy) * self.distance * 0.0015;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_matches_screen_drag_direction() {
        let project_x_y = |cam: &OrbitCamera, world: Vec3| {
            let view = cam.view();
            let rx =
                view[0][0] * world.x + view[1][0] * world.y + view[2][0] * world.z + view[3][0];
            let ry =
                view[0][1] * world.x + view[1][1] * world.y + view[2][1] * world.z + view[3][1];
            (rx, ry)
        };
        let point = vec3(0.0, 0.0, 1.0);
        let base = OrbitCamera {
            focus: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.0,
            pitch: 0.0,
            ..Default::default()
        };
        let (x0, y0) = project_x_y(&base, point);

        let mut right_drag = base;
        right_drag.orbit(0.2, 0.0);
        let (xr, _) = project_x_y(&right_drag, point);
        assert!(
            xr > x0,
            "dragging right should move front point right on screen"
        );

        let mut down_drag = base;
        down_drag.orbit(0.0, 0.2);
        let (_, yd) = project_x_y(&down_drag, point);
        assert!(
            yd < y0,
            "dragging down should move front point down in view space"
        );
    }
}
