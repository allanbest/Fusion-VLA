//! Render a material test scene to a PNG, with no window.
//!
//! `cargo run --release -p fusion-viewport --example shot -- out.png`
use fusion_core::Shape;
use fusion_math::{vec3, Isometry, Quat, Vec3};
use fusion_viewport::{
    encode_png, render_offscreen_with, Colormap, OrbitCamera, RenderSettings, ScalarField, Surface,
};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "shot.png".into());
    let camera = OrbitCamera {
        focus: vec3(0.0, 0.8, 0.0),
        distance: 7.0,
        yaw: 0.8,
        pitch: 0.35,
        ..Default::default()
    };

    let mut settings = RenderSettings::default();
    if let Ok(v) = std::env::var("AO") {
        settings.ao_strength = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("AMBIENT") {
        settings.ambient = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("EXPOSURE") {
        settings.exposure = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("AORADIUS") {
        settings.ao_radius = v.parse().unwrap();
    }

    let image = render_offscreen_with(900, 600, &camera, settings, |g| {
        g.set_surface(Surface::MATTE);
        g.solid_shape(
            &Shape::cuboid(vec3(14.0, 0.4, 14.0)),
            &Isometry::from_translation(vec3(0.0, -0.2, 0.0)),
            [0.55, 0.55, 0.58],
        );
        for i in 0..5 {
            let t = i as f32 / 4.0;
            g.set_surface(Surface {
                roughness: 0.05 + t * 0.9,
                metallic: 0.0,
            });
            g.solid_shape(
                &Shape::sphere(0.42),
                &Isometry::from_translation(vec3(-2.4 + i as f32 * 1.2, 0.42, -1.1)),
                [0.85, 0.28, 0.22],
            );
        }
        for i in 0..5 {
            let t = i as f32 / 4.0;
            g.set_surface(Surface {
                roughness: 0.08 + t * 0.6,
                metallic: 1.0,
            });
            g.solid_shape(
                &Shape::sphere(0.42),
                &Isometry::from_translation(vec3(-2.4 + i as f32 * 1.2, 0.42, 0.4)),
                [0.95, 0.78, 0.35],
            );
        }
        g.set_surface(Surface::PLASTIC);
        g.solid_shape(
            &Shape::cuboid(Vec3::splat(0.9)),
            &Isometry::new(vec3(1.1, 1.4, 1.6), Quat::from_rotation_y(0.6)),
            [0.30, 0.55, 0.85],
        );
        g.line(vec3(-3.0, 0.02, 2.2), vec3(3.0, 0.02, 2.2), [0.2, 1.0, 0.4]);

        // A visualization mode and its legend (PLAN.md §19). The range is
        // fixed at construction and the legend is always on screen; both are
        // rules rather than choices.
        let speed = ScalarField::new("speed", "m/s", 0.0, 12.5).with_map(Colormap::Viridis);
        g.set_surface(Surface::MATTE);
        for i in 0..7 {
            let v = i as f32 / 6.0 * 12.5;
            g.solid_shape(
                &Shape::cuboid(Vec3::splat(0.34)),
                &Isometry::from_translation(vec3(-2.4 + i as f32 * 0.8, 0.17, 2.6)),
                speed.color(v),
            );
        }
        speed.draw_legend(g);
    });
    println!("mean luminance {:.4}", image.mean_luminance());
    std::fs::write(&path, encode_png(&image)).unwrap();
    println!("wrote {path}");
}
