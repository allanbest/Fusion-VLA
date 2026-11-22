//! Tests for the viewport, by rendering frames and looking at the pixels.
//!
//! A renderer that can only be checked by a person looking at a window is a
//! renderer with no regression tests, and shading is exactly the kind of thing
//! that breaks silently — a wrong sign in a Fresnel term or a shadow matrix
//! transposed still produces a picture.
//!
//! Every test here skips rather than fails when the machine has no GPU. A
//! build box that was never going to run a viewport should not have a red CI
//! because of one, and the alternative — deleting the tests — is worse.

use fusion_core::Shape;
use fusion_math::{vec3, Isometry, Vec3};
use fusion_viewport::{
    try_render_offscreen, Colormap, Geometry, Image, OrbitCamera, RenderSettings, ScalarField,
    Surface,
};

const W: u32 = 320;
const H: u32 = 240;

fn camera() -> OrbitCamera {
    OrbitCamera {
        focus: vec3(0.0, 0.5, 0.0),
        distance: 6.0,
        yaw: 0.7,
        pitch: 0.4,
        ..Default::default()
    }
}

/// Render, or `None` if this machine has no GPU.
fn render(settings: RenderSettings, draw: impl FnOnce(&mut Geometry)) -> Option<Image> {
    try_render_offscreen(W, H, &camera(), settings, draw)
}

macro_rules! gpu {
    ($e:expr) => {
        match $e {
            Some(image) => image,
            None => {
                eprintln!("skipped: no GPU adapter on this machine");
                return;
            }
        }
    };
}

/// A floor and one box on it.
fn simple(g: &mut Geometry) {
    g.set_surface(Surface::MATTE);
    g.solid_shape(
        &Shape::cuboid(vec3(12.0, 0.4, 12.0)),
        &Isometry::from_translation(vec3(0.0, -0.2, 0.0)),
        [0.6, 0.6, 0.62],
    );
    g.solid_shape(
        &Shape::cuboid(Vec3::ONE),
        &Isometry::from_translation(vec3(0.0, 0.5, 0.0)),
        [0.8, 0.3, 0.25],
    );
}

#[test]
fn a_frame_is_drawn_and_is_not_uniform() {
    // The floor of the whole test file: something rendered, and it has
    // structure rather than being one flat colour.
    let image = gpu!(render(RenderSettings::default(), simple));
    assert_eq!(image.width, W);
    assert_eq!(image.height, H);

    let mean = image.mean_luminance();
    assert!(
        (0.02..0.98).contains(&mean),
        "mean luminance {mean} — the frame is blank or blown out"
    );

    let mut lowest = 255u8;
    let mut highest = 0u8;
    for p in image.pixels.chunks_exact(4) {
        lowest = lowest.min(p[0]);
        highest = highest.max(p[0]);
    }
    assert!(
        highest as i32 - lowest as i32 > 40,
        "the frame spans only {lowest}..{highest} — no shading is happening"
    );
}

#[test]
fn every_pixel_is_opaque() {
    // The alpha channel is written by every path. A hole in it shows up as a
    // transparent window on a compositor that honours it, on one platform.
    let image = gpu!(render(RenderSettings::default(), simple));
    assert!(image.pixels.chunks_exact(4).all(|p| p[3] == 255));
}

#[test]
fn the_sky_is_drawn_where_nothing_else_is() {
    // The top-left corner looks at empty space, and should be sky rather than
    // the clear colour — the two are different code paths and it is the sky
    // that a reflective surface samples.
    let image = gpu!(render(RenderSettings::default(), simple));
    let sky = image.pixel(4, 4);
    assert!(
        sky[2] > sky[0],
        "the top of frame is {sky:?} — a sky should be bluer than it is red"
    );
    assert!(sky[2] > 30, "the sky is nearly black: {sky:?}");
}

#[test]
fn a_shadow_darkens_the_floor() {
    // The point of a shadow map. Rendered twice — once with a caster, once
    // without — and the floor must be measurably darker with it.
    let floor_only = gpu!(render(RenderSettings::default(), |g| {
        g.set_surface(Surface::MATTE);
        g.solid_shape(
            &Shape::cuboid(vec3(12.0, 0.4, 12.0)),
            &Isometry::from_translation(vec3(0.0, -0.2, 0.0)),
            [0.6, 0.6, 0.62],
        );
    }));
    let with_caster = gpu!(render(RenderSettings::default(), |g| {
        g.set_surface(Surface::MATTE);
        g.solid_shape(
            &Shape::cuboid(vec3(12.0, 0.4, 12.0)),
            &Isometry::from_translation(vec3(0.0, -0.2, 0.0)),
            [0.6, 0.6, 0.62],
        );
        // High up and off to the side, so its shadow lands on visible floor
        // and the box itself does not cover the pixels being compared.
        g.solid_shape(
            &Shape::cuboid(Vec3::splat(1.6)),
            &Isometry::from_translation(vec3(1.6, 3.2, 1.6)),
            [0.8, 0.3, 0.25],
        );
    }));

    // Somewhere the shadow falls: down-left of the caster, since the light
    // comes from +x +y +z.
    let mut darker = 0;
    let mut compared = 0;
    for y in (H / 2)..(H - 8) {
        for x in 8..(W / 2) {
            let a = floor_only.pixel(x, y);
            let b = with_caster.pixel(x, y);
            compared += 1;
            if (b[0] as i32) < a[0] as i32 - 8 {
                darker += 1;
            }
        }
    }
    assert!(compared > 0);
    assert!(
        darker * 20 > compared,
        "only {darker} of {compared} floor pixels darkened — the caster is not casting"
    );
}

#[test]
fn ambient_occlusion_darkens_a_crease() {
    // Two boxes meeting at a right angle. The corner between them should be
    // darker with SSAO on than off; the open floor should barely move.
    let scene = |g: &mut Geometry| {
        g.set_surface(Surface::MATTE);
        g.solid_shape(
            &Shape::cuboid(vec3(12.0, 0.4, 12.0)),
            &Isometry::from_translation(vec3(0.0, -0.2, 0.0)),
            [0.7, 0.7, 0.7],
        );
        for i in 0..3 {
            g.solid_shape(
                &Shape::cuboid(vec3(0.6, 1.4, 3.0)),
                &Isometry::from_translation(vec3(-1.2 + i as f32 * 1.2, 0.7, 0.0)),
                [0.7, 0.7, 0.7],
            );
        }
    };
    let without = gpu!(render(
        RenderSettings {
            ao_strength: 0.0,
            ..Default::default()
        },
        scene
    ));
    let with = gpu!(render(
        RenderSettings {
            ao_strength: 1.0,
            ..Default::default()
        },
        scene
    ));

    assert!(
        with.mean_luminance() < without.mean_luminance(),
        "AO made the frame no darker: {} vs {}",
        with.mean_luminance(),
        without.mean_luminance()
    );

    // And it darkens *creases more than open surfaces*, which is the whole
    // claim. A uniform dimming would pass a "did it get darker" check and be
    // an exposure change wearing a hat.
    //
    // Measuring "how many pixels moved" does not distinguish the two: this
    // scene is mostly walls and floor meeting each other, so a correct SSAO
    // touches 62% of the frame and a global dimming would too. The
    // discriminator has to be *where*.
    let darkening = |x: u32, y: u32| without.pixel(x, y)[0] as i32 - with.pixel(x, y)[0] as i32;
    let mut crease = 0i32;
    let mut crease_n = 0;
    let mut open = 0i32;
    let mut open_n = 0;
    for y in 0..H {
        for x in 0..W {
            // The slabs stand in the middle of the frame; the outer edges are
            // open floor with nothing near them.
            if (x as f32) > W as f32 * 0.35 && (x as f32) < W as f32 * 0.65 && y > H / 2 {
                crease += darkening(x, y);
                crease_n += 1;
            } else if !(W / 10..=W * 9 / 10).contains(&x) {
                open += darkening(x, y);
                open_n += 1;
            }
        }
    }
    let crease_mean = crease as f32 / crease_n as f32;
    let open_mean = open as f32 / open_n.max(1) as f32;
    assert!(
        crease_mean > open_mean + 1.0,
        "creases darkened by {crease_mean:.2} and open floor by {open_mean:.2} — \
         that is a global dimming, not occlusion"
    );
}

#[test]
fn a_metal_is_not_black() {
    // Physically a metal has no diffuse lobe, so with nothing to reflect it
    // renders black — right, and visibly broken. The ambient specular term
    // exists to stop that, and this is what says it is still there.
    let image = gpu!(render(RenderSettings::default(), |g| {
        g.set_surface(Surface::METAL);
        g.solid_shape(
            &Shape::sphere(2.0),
            &Isometry::from_translation(vec3(0.0, 0.5, 0.0)),
            [0.9, 0.75, 0.4],
        );
    }));
    // The sphere covers the middle of the frame.
    let centre = image.pixel(W / 2, H / 2);
    let luminance =
        0.2126 * centre[0] as f32 + 0.7152 * centre[1] as f32 + 0.0722 * centre[2] as f32;
    assert!(
        luminance > 20.0,
        "a metal sphere rendered at luminance {luminance} — the environment term is gone"
    );
}

#[test]
fn the_material_changes_the_picture() {
    // If the material made no difference the whole PBR path would be dead code
    // that still compiles.
    //
    // Tested on a *metal*, not a dielectric. A rough plastic and a smooth
    // plastic differ only in a small specular highlight — 170 pixels out of
    // 76 800 in this framing — because the diffuse lobe dominates and does not
    // depend on roughness at all. That is correct behaviour and a terrible
    // test. A metal has no diffuse lobe, so roughness is the whole of its
    // appearance.
    let sphere = |surface: Surface| {
        move |g: &mut Geometry| {
            g.set_surface(surface);
            g.solid_shape(
                &Shape::sphere(1.6),
                &Isometry::from_translation(vec3(0.0, 0.6, 0.0)),
                [0.85, 0.82, 0.78],
            );
        }
    };
    let polished = gpu!(render(
        RenderSettings::default(),
        sphere(Surface {
            roughness: 0.05,
            metallic: 1.0
        })
    ));
    let brushed = gpu!(render(
        RenderSettings::default(),
        sphere(Surface {
            roughness: 0.9,
            metallic: 1.0
        })
    ));
    let differing = (0..W * H)
        .filter(|i| {
            let (x, y) = (i % W, i / W);
            polished.pixel(x, y)[0].abs_diff(brushed.pixel(x, y)[0]) > 4
        })
        .count();
    assert!(
        differing > (W * H / 50) as usize,
        "only {differing} pixels differ between polished and brushed metal"
    );

    // And metalness itself, which changes even a rough surface completely.
    let plastic = gpu!(render(
        RenderSettings::default(),
        sphere(Surface {
            roughness: 0.9,
            metallic: 0.0
        })
    ));
    let metal_diff = (0..W * H)
        .filter(|i| {
            let (x, y) = (i % W, i / W);
            plastic.pixel(x, y)[0].abs_diff(brushed.pixel(x, y)[0]) > 8
        })
        .count();
    assert!(
        metal_diff > (W * H / 50) as usize,
        "only {metal_diff} pixels differ between metal and dielectric"
    );
}

#[test]
fn exposure_changes_the_brightness() {
    let dim = gpu!(render(
        RenderSettings {
            exposure: 0.25,
            ..Default::default()
        },
        simple
    ));
    let bright = gpu!(render(
        RenderSettings {
            exposure: 3.0,
            ..Default::default()
        },
        simple
    ));
    assert!(dim.mean_luminance() < bright.mean_luminance());
}

#[test]
fn the_overlay_ignores_the_camera() {
    // "Always visible" in the sense §19 means: the legend is in clip space, so
    // moving the camera must not move it.
    let draw = |g: &mut Geometry| {
        simple(g);
        ScalarField::new("speed", "m/s", 0.0, 10.0)
            .with_map(Colormap::Viridis)
            .draw_legend(g);
    };
    let near = try_render_offscreen(
        W,
        H,
        &OrbitCamera {
            distance: 4.0,
            ..camera()
        },
        RenderSettings::default(),
        draw,
    );
    let near = gpu!(near);
    let far = gpu!(try_render_offscreen(
        W,
        H,
        &OrbitCamera {
            distance: 30.0,
            yaw: 2.4,
            ..camera()
        },
        RenderSettings::default(),
        draw,
    ));

    // Only the colour bar itself, which is the part guaranteed to be covered
    // by overlay geometry. A looser window catches the scene showing between
    // the glyphs, and the scene is *supposed* to change when the camera moves.
    //
    // The bar spans NDC x -0.94..-0.58 and y -0.90..-0.865, which at this
    // resolution is roughly x 10..67, y 224..228.
    let mut identical = 0;
    let mut total = 0;
    for y in 225..228 {
        for x in 14..62 {
            total += 1;
            if near.pixel(x, y) == far.pixel(x, y) {
                identical += 1;
            }
        }
    }
    assert!(total > 0);
    assert!(
        identical * 10 > total * 9,
        "only {identical} of {total} colour-bar pixels survived a camera move"
    );
}

#[test]
fn the_colormaps_are_monotone_in_lightness() {
    // What "perceptually uniform" buys, and the property a rainbow ramp does
    // not have: lightness rises steadily from one end to the other, so the
    // map never invents a boundary the data does not contain.
    for map in [Colormap::Viridis, Colormap::Inferno] {
        let lightness: Vec<f32> = (0..=32)
            .map(|i| {
                let c = map.sample(i as f32 / 32.0);
                0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            })
            .collect();
        for pair in lightness.windows(2) {
            assert!(
                pair[1] >= pair[0] - 0.02,
                "{} is not monotone in lightness: {pair:?}",
                map.name()
            );
        }
        assert!(
            lightness[lightness.len() - 1] - lightness[0] > 0.5,
            "{} barely changes lightness end to end",
            map.name()
        );
    }
}

#[test]
fn the_colormaps_match_their_published_endpoints() {
    // The check that would have caught the magma fit whose blue channel was
    // wrong by 0.75 at the top of the ramp. A colormap that has not been
    // compared against its reference is a colormap that invents features.
    let close = |got: [f32; 3], want: [f32; 3], name: &str| {
        let error = (0..3).map(|i| (got[i] - want[i]).abs()).fold(0.0, f32::max);
        assert!(
            error < 0.03,
            "{name}: got {got:?}, published value is {want:?}"
        );
    };
    // viridis runs #440154 to #FDE725.
    close(
        Colormap::Viridis.sample(0.0),
        [0.267, 0.005, 0.329],
        "viridis start",
    );
    close(
        Colormap::Viridis.sample(1.0),
        [0.993, 0.906, 0.144],
        "viridis end",
    );
    // inferno runs #000004 to #FCFFA4.
    close(
        Colormap::Inferno.sample(0.0),
        [0.001, 0.000, 0.014],
        "inferno start",
    );
    close(
        Colormap::Inferno.sample(1.0),
        [0.988, 1.000, 0.645],
        "inferno end",
    );
}

#[test]
fn a_scalar_field_keeps_the_range_it_was_given() {
    // §19's second rule. Auto-ranging per frame makes the same colour mean
    // different things over time, and an animation of it lies.
    let field = ScalarField::new("speed", "m/s", 0.0, 10.0);
    assert_eq!(field.normalize(-5.0), 0.0);
    assert_eq!(field.normalize(0.0), 0.0);
    assert_eq!(field.normalize(5.0), 0.5);
    assert_eq!(field.normalize(10.0), 1.0);
    assert_eq!(field.normalize(1000.0), 1.0);
    // The colour of a given value does not depend on anything else in flight.
    assert_eq!(field.color(5.0), field.color(5.0));

    // And `lock_from` copes with a constant field rather than dividing by zero.
    let flat = ScalarField::lock_from("x", "", [3.0, 3.0, 3.0]);
    assert!(flat.color(3.0)[0].is_finite());
    let empty = ScalarField::lock_from("x", "", []);
    assert!(empty.high > empty.low);
}

#[test]
fn blackbody_is_dark_when_cold_and_bright_when_hot() {
    let cold = fusion_viewport::blackbody(300.0);
    assert!(cold.iter().all(|c| *c < 0.02), "300 K is glowing: {cold:?}");
    let warm = fusion_viewport::blackbody(1200.0);
    assert!(warm[0] > warm[2], "1200 K should be red-dominant: {warm:?}");
    let hot = fusion_viewport::blackbody(6500.0);
    assert!(hot.iter().all(|c| *c > 0.5), "6500 K is dim: {hot:?}");
    assert!(hot[2] > warm[2], "hotter should be bluer");
}

#[test]
fn a_png_round_trips_its_own_header() {
    // The encoder is hand-written, so the parts a decoder checks first are
    // worth asserting: signature, dimensions, and the chunk CRCs.
    let image = Image {
        width: 3,
        height: 2,
        pixels: vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9,
            255,
        ],
    };
    let png = fusion_viewport::encode_png(&image);
    assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    // IHDR: length 13, tag, width, height.
    assert_eq!(&png[8..12], &13u32.to_be_bytes());
    assert_eq!(&png[12..16], b"IHDR");
    assert_eq!(&png[16..20], &3u32.to_be_bytes());
    assert_eq!(&png[20..24], &2u32.to_be_bytes());
    assert_eq!(png[24], 8, "bit depth");
    assert_eq!(png[25], 6, "colour type RGBA");
    assert!(png.ends_with(&[0xae, 0x42, 0x60, 0x82]), "IEND CRC");

    // Walk the chunks and verify every CRC, which is what a decoder does
    // before it trusts anything.
    let mut i = 8;
    let mut saw_idat = false;
    while i + 12 <= png.len() {
        let len = u32::from_be_bytes(png[i..i + 4].try_into().unwrap()) as usize;
        let tag = &png[i + 4..i + 8];
        if tag == b"IDAT" {
            saw_idat = true;
        }
        let stated = u32::from_be_bytes(png[i + 8 + len..i + 12 + len].try_into().unwrap());
        assert_eq!(stated, crc32(&png[i + 4..i + 8 + len]), "CRC on chunk");
        i += 12 + len;
    }
    assert_eq!(i, png.len(), "chunks do not tile the file");
    assert!(saw_idat);
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for &byte in data {
        c ^= byte as u32;
        for _ in 0..8 {
            let mask = (c & 1).wrapping_neg();
            c = (c >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !c
}

/// A solid with a temperature that varies across it, for the volume tests.
fn heated_solid() -> fusion_core::SolidSet {
    use fusion_core::{build_solid, SolidDesc, SolidMaterial, SolidParams};
    let mut solid = build_solid(
        &SolidDesc::block(vec3(1.0, 1.0, 1.0))
            .at(vec3(0.0, 0.5, 0.0))
            .with_params(SolidParams::at(0.1).of(SolidMaterial::ALLOY)),
    );
    // Hot at one end, cold at the other, and nothing in the corners of the
    // bounding box — which is the arrangement the fill channel has to get
    // right for the picture to be of a block rather than of a box.
    let thermal = SolidMaterial::ALLOY.thermal;
    for i in 0..solid.len() {
        let x = solid.particles.translation(i).x;
        solid.enthalpy[i] = thermal.enthalpy_at(300.0 + 250.0 * (x + 0.5));
    }
    solid
}

fn temperature_volume(solid: &fusion_core::SolidSet) -> std::sync::Arc<fusion_viewport::Volume> {
    let field = ScalarField::new("temperature", "K", 300.0, 550.0).with_map(Colormap::Inferno);
    std::sync::Arc::new(
        fusion_viewport::Volume::from_solid(solid, field, 12.0, |i| solid.temperature(i))
            .expect("a solid with points in it makes a volume"),
    )
}

#[test]
fn a_volume_draws_where_the_material_is_and_nowhere_else() {
    let solid = heated_solid();
    let volume = temperature_volume(&solid);

    let empty = gpu!(render(RenderSettings::default(), |g| g.set_surface(Surface::MATTE)));
    let filled = gpu!(render(RenderSettings::default(), |g| {
        g.set_surface(Surface::MATTE);
        g.volume = Some(volume.clone());
    }));

    // The middle of the frame is the block, and it changed.
    let (x, y) = (W / 2, H / 2);
    let (before, after) = (empty.pixel(x, y), filled.pixel(x, y));
    assert!(
        (0..3).any(|c| (before[c] as i32 - after[c] as i32).abs() > 20),
        "the volume changed nothing at the centre of frame: {before:?} -> {after:?}"
    );
    // The corner is empty space in both, and the volume must not have painted
    // its bounding box over it. This is the fill channel's whole job.
    let (before, after) = (empty.pixel(4, 4), filled.pixel(4, 4));
    assert_eq!(
        before, after,
        "the volume painted the sky at the corner of its own bounding box"
    );
}

/// The colours follow the field, not the geometry.
///
/// The block is hot at one end and cold at the other on a ramp that runs dark
/// to bright, so the two halves of it must not render the same. Which half of
/// the *frame* each end lands in depends on the camera, so the test finds the
/// volume rather than assuming where it is: the pixels that changed when the
/// volume was added are the volume, and their own centroid splits them.
#[test]
fn a_volume_is_coloured_by_its_field() {
    let solid = heated_solid();
    let volume = temperature_volume(&solid);
    let empty = gpu!(render(RenderSettings::default(), |_| {}));
    let filled = gpu!(render(RenderSettings::default(), |g| {
        g.volume = Some(volume.clone());
    }));

    let mut columns: Vec<(u32, f64)> = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let (a, b) = (empty.pixel(x, y), filled.pixel(x, y));
            if (0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > 8) {
                let luminance = (b[0] as f64 + b[1] as f64 + b[2] as f64) / 3.0;
                columns.push((x, luminance));
            }
        }
    }
    assert!(
        columns.len() > 500,
        "only {} pixels changed — the volume is not being drawn",
        columns.len()
    );
    let centroid = columns.iter().map(|(x, _)| *x as f64).sum::<f64>() / columns.len() as f64;
    let mean = |side: bool| {
        let picked: Vec<f64> = columns
            .iter()
            .filter(|(x, _)| ((*x as f64) < centroid) == side)
            .map(|(_, l)| *l)
            .collect();
        picked.iter().sum::<f64>() / picked.len() as f64
    };
    let (left, right) = (mean(true), mean(false));
    assert!(
        (left - right).abs() > 12.0,
        "the two halves of a block with a 250 K gradient across it rendered at \
         {left:.1} and {right:.1} — the lookup is not being indexed by the field"
    );
}

/// Geometry in front of a volume hides it.
///
/// The volume pass reads the scene's depth buffer to know where to stop. A
/// marcher that skipped that step would draw the field over everything, like a
/// decal on the lens, and every spatial relationship in the picture would be a
/// lie — a solid drawn as a surface would glow through its own skin.
#[test]
fn geometry_in_front_of_a_volume_occludes_it() {
    let solid = heated_solid();
    let volume = temperature_volume(&solid);
    // Halfway between the camera and what it is looking at, so it is in the
    // way whatever the camera angle happens to be.
    let camera = camera();
    let middle = (camera.eye() + camera.focus) * 0.5;

    let open = gpu!(render(RenderSettings::default(), |g| {
        g.volume = Some(volume.clone());
    }));
    let blocked = gpu!(render(RenderSettings::default(), |g| {
        g.set_surface(Surface::MATTE);
        g.solid_shape(
            &Shape::cuboid(Vec3::splat(2.5)),
            &Isometry::from_translation(middle),
            [0.7, 0.7, 0.72],
        );
        g.volume = Some(volume.clone());
    }));

    let (x, y) = (W / 2, H / 2);
    let (front, behind) = (open.pixel(x, y), blocked.pixel(x, y));
    assert_ne!(
        front, behind,
        "an opaque slab in front of the volume changed nothing"
    );
    // The slab is grey and the field there is hot. A volume that ignored the
    // depth buffer would tint it — grey is what "the volume stopped" looks
    // like, and a red-blue spread is what it looks like when it did not.
    let spread = |p: [u8; 4]| (p[0] as i32 - p[2] as i32).abs();
    assert!(
        spread(behind) < spread(front) / 2,
        "the volume bled through the slab: {front:?} in the open, {behind:?} behind it"
    );
}
