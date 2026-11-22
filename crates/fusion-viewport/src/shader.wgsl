// The v0.5 viewport.
//
// PLAN.md §18 asks for "PBR shading, shadow maps, SSAO, tone mapping" and is
// specific about the ceiling: this is a *good viewport*, not a renderer.
// Anything needing film quality exports to Blender. So there is one light that
// casts, one that fills, an ambient term the sky provides, and nothing that
// would take a week to get right and a month to make fast.
//
// Six passes, in order:
//
//   depth      geometry, depth only          -> depth
//   ssao       fullscreen, reads depth       -> ao
//   blur       fullscreen, reads ao          -> ao_blur
//   shadow     geometry from the light       -> shadow depth
//   main       geometry, PBR + shadow + ao   -> hdr
//   tonemap    fullscreen, reads hdr         -> swapchain
//
// The depth pass exists so SSAO has depth to read *before* lighting happens.
// Occlusion belongs on the ambient term, and ambient is computed in the main
// pass — so AO computed after it could only be a multiply over the final
// colour, which darkens direct light that nothing is occluding. Drawing the
// geometry twice is the cheaper mistake.

struct Uniforms {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    light_view_projection: mat4x4<f32>,
    eye: vec4<f32>,
    // xyz: direction *towards* the light. w: shadow map texel size.
    light: vec4<f32>,
    // x: viewport width, y: height, z: ao radius, w: ao strength.
    render: vec4<f32>,
    // x: exposure, y: ambient strength, z: shadow softness, w: unused.
    grade: vec4<f32>,
};

// Group 0 is everything every pass needs. Group 1 means two different things
// depending on the pipeline — lighting inputs for the main pass, a source
// texture for the fullscreen passes — which is legal because they are
// different pipeline layouts, and necessary because a texture cannot be bound
// while the same pass is writing it.
//
// Getting that wrong is how the blur pass failed: the AO texture lived in
// group 0 alongside the uniforms, so every pass bound it, including the one
// rendering into it. wgpu reports that as "encoder is invalid" at `finish()`,
// with no indication of which pass or which resource.
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var linear_sampler: sampler;

@group(1) @binding(0) var shadow_map: texture_depth_2d;
@group(1) @binding(1) var shadow_sampler: sampler_comparison;
@group(1) @binding(2) var ao_texture: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) material: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world: vec3<f32>,
    @location(3) material: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.view_projection * vec4<f32>(input.position, 1.0);
    out.normal = input.normal;
    out.color = input.color;
    out.world = input.position;
    out.material = input.material;
    return out;
}

// Overlay: positions arrive already in clip space, so the camera does not
// enter into it. This is what "always visible" means for a legend.
@vertex
fn vs_overlay(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(input.position.xy, 0.0, 1.0);
    out.normal = vec3<f32>(0.0);
    out.color = input.color;
    out.world = input.position;
    out.material = input.material;
    return out;
}

// Depth-only, for the prepass and the shadow pass. The shadow pass swaps the
// matrix for the light's, which is the only difference between them.
@vertex
fn vs_depth(input: VertexInput) -> @builtin(position) vec4<f32> {
    return u.view_projection * vec4<f32>(input.position, 1.0);
}

@vertex
fn vs_shadow(input: VertexInput) -> @builtin(position) vec4<f32> {
    return u.light_view_projection * vec4<f32>(input.position, 1.0);
}

// ---------------------------------------------------------------- lighting

const PI: f32 = 3.14159265359;

// GGX / Trowbridge-Reitz. The distribution of microfacet normals, and the
// term that makes a rough surface's highlight wide and a smooth one's tight.
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Smith's geometry term with the Schlick-GGX approximation: how much of the
// surface shadows and masks itself at grazing angles.
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn brdf(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);

    // 0.04 is the reflectance of a typical dielectric at normal incidence; a
    // metal reflects its own albedo instead, which is the whole of what
    // "metallic" means in this workflow.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-7);
    // Energy conservation: what is not reflected is available to scatter, and
    // a metal has no diffuse lobe at all.
    let diffuse = (vec3<f32>(1.0) - f) * (1.0 - metallic) * albedo / PI;
    return (diffuse + specular) * n_dot_l;
}

/// Percentage-closer filtering over a 5x5 kernel.
///
/// A single comparison gives a hard, aliased edge that crawls as the camera
/// moves; the kernel trades that for a soft one. The normal-scaled bias is
/// what keeps a surface from shadowing itself — the classic acne — without
/// the peter-panning a constant depth bias produces.
fn shadow_factor(world: vec3<f32>, n_dot_l: f32, normal: vec3<f32>) -> f32 {
    let offset = normal * (u.light.w * 2.5 / max(n_dot_l, 0.15));
    let light_clip = u.light_view_projection * vec4<f32>(world + offset, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    if (ndc.z > 1.0 || ndc.z < 0.0) {
        return 1.0;
    }
    var uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }
    let bias = 0.0015 / max(n_dot_l, 0.15);
    let radius = u.grade.z;
    var sum = 0.0;
    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let o = vec2<f32>(f32(x), f32(y)) * u.light.w * radius;
            sum = sum + textureSampleCompareLevel(
                shadow_map, shadow_sampler, uv + o, ndc.z - bias
            );
        }
    }
    return sum / 25.0;
}

const KEY: vec3<f32> = vec3<f32>(1.0, 0.95, 0.88);
const FILL_DIR: vec3<f32> = vec3<f32>(-0.55, 0.35, -0.75);
const SKY: vec3<f32> = vec3<f32>(0.42, 0.52, 0.68);
const GROUND: vec3<f32> = vec3<f32>(0.22, 0.19, 0.16);

/// The environment, as a function of direction.
///
/// Serves two purposes and has to, because there is only one of it: it is what
/// the camera sees where nothing was drawn, and it is what a mirror reflects.
/// A viewport whose background and whose reflections disagree looks like a
/// compositing error.
fn environment(dir: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    // The studio: grey, lighter above, and a soft light to catch.
    if (u.grade.w > 0.5) {
        let wall = mix(vec3<f32>(0.075, 0.075, 0.077), vec3<f32>(0.19, 0.19, 0.195), smoothstep(0.15, 0.95, up));
        let glow = max(dot(normalize(dir), normalize(u.light.xyz)), 0.0);
        return wall + vec3<f32>(0.5) * pow(glow, 24.0) * 0.25;
    }
    let horizon = vec3<f32>(0.24, 0.242, 0.248);
    let zenith = vec3<f32>(0.11, 0.112, 0.116);
    let ground = vec3<f32>(0.14, 0.14, 0.144);
    var sky = mix(horizon, zenith, smoothstep(0.5, 1.0, up));
    // A wide transition rather than a tight one. A camera looking down at a
    // scene sees mostly the *below*-horizon hemisphere behind it, and a sharp
    // cut there reads as a black band across the frame rather than as ground.
    sky = mix(ground, sky, smoothstep(0.24, 0.54, up));
    // A broad glow around the sun, so a smooth surface has something to catch.
    let sun = max(dot(normalize(dir), normalize(u.light.xyz)), 0.0);
    return sky + KEY * pow(sun, 32.0) * 0.6 + KEY * pow(sun, 4.0) * 0.06;
}

/// Fresnel with a roughness term, for the ambient specular below.
///
/// The plain Schlick form goes to white at grazing angles regardless of how
/// rough the surface is, which is right for a mirror and wrong for chalk.
fn fresnel_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let inv = vec3<f32>(1.0 - roughness);
    return f0 + (max(inv, f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let len = length(input.normal);
    // Lines and overlays carry a zero normal and are emitted at full colour.
    // They are the one thing in the frame that must not be shaded: an overlay
    // that dims when it faces away is an overlay you cannot read.
    if (len < 0.5) {
        return vec4<f32>(input.color, 1.0);
    }

    var n = input.normal / len;
    let v = normalize(u.eye.xyz - input.world);
    // Two-sided: the debug view shows the inside of things, and a back face
    // lit as though it faced away is a black hole in the middle of a scene.
    if (dot(n, v) < 0.0) {
        n = -n;
    }

    let albedo = input.color;
    let roughness = clamp(input.material.x, 0.02, 1.0);
    let metallic = clamp(input.material.y, 0.0, 1.0);

    let l = normalize(u.light.xyz);
    let n_dot_l = max(dot(n, l), 0.0);
    let shadow = shadow_factor(input.world, n_dot_l, n);

    var lit = brdf(n, v, l, albedo, roughness, metallic) * KEY * 3.2 * shadow;
    // The fill is a second directional light with no shadow and no specular
    // worth speaking of. It exists so the side facing away from the key is
    // readable, which a physically-correct single light does not achieve
    // indoors either — real rooms have walls.
    lit = lit + brdf(n, v, normalize(FILL_DIR), albedo, roughness, metallic) * SKY * 0.75;

    // Hemispherical ambient, occluded. Sky above, bounce below.
    let ao = textureSampleLevel(
        ao_texture, linear_sampler,
        input.clip_position.xy / u.render.xy, 0.0
    ).r;
    let hemi = mix(GROUND, SKY, n.y * 0.5 + 0.5);
    let diffuse_ambient = hemi * albedo * u.grade.y * ao * (1.0 - metallic);

    // Ambient specular — a one-sample stand-in for image-based lighting.
    //
    // Without it a metal is *black*, and correctly so: a metal has no diffuse
    // lobe at all, so with no environment to reflect there is nothing for it
    // to be. Physically right and visually broken, which is exactly the sort
    // of thing §18's "good viewport" has to solve rather than explain. One
    // sample of the environment along the reflection vector, weighted by a
    // roughness-aware Fresnel, is enough to make steel read as steel.
    let reflected = reflect(-v, n);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = fresnel_roughness(max(dot(n, v), 0.0), f0, roughness);
    // Rough surfaces reflect a blurred environment; with a single sample the
    // cheapest approximation of the blur is to fade towards the flat sky.
    let sharp = environment(reflected);
    let blurred = mix(SKY, GROUND, 0.35);
    let specular_ambient = mix(sharp, blurred, roughness) * f * u.grade.y * ao;

    return vec4<f32>(lit + diffuse_ambient + specular_ambient, 1.0);
}

// ------------------------------------------------------------- fullscreen

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle covering the screen, generated from the vertex index. No
// vertex buffer, no quad, no seam down the diagonal.
@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> FullscreenOut {
    var out: FullscreenOut;
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    out.uv = uv;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return out;
}

@group(1) @binding(3) var source: texture_2d<f32>;
@group(1) @binding(4) var source_depth: texture_depth_2d;

/// Depth at a UV, as a point sample.
///
/// `textureLoad` rather than a sampler: a depth buffer must not be filtered
/// for this. Bilinear blending across a silhouette produces a depth that no
/// surface in the scene actually has, and SSAO reconstructs a position from
/// it — so the occluder it invents sits in mid-air along every edge.
fn depth_at(uv: vec2<f32>) -> f32 {
    let size = vec2<i32>(u.render.xy);
    let c = clamp(vec2<i32>(uv * u.render.xy), vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(source_depth, c, 0);
}

/// View-space position from a depth sample.
fn view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view = u.inverse_projection * ndc;
    return view.xyz / view.w;
}

/// Screen-space ambient occlusion.
///
/// Normals are reconstructed from the depth buffer rather than carried in a
/// G-buffer. That costs some quality at silhouettes — the derivative spans a
/// depth discontinuity and the normal there is wrong — and saves an entire
/// render target and the bandwidth to fill it. For a diagnostic view whose
/// AO is a shading cue rather than a measurement, that is the right side of
/// the trade.
@fragment
fn fs_ssao(input: FullscreenOut) -> @location(0) vec4<f32> {
    let depth = depth_at(input.uv);
    if (depth >= 1.0) {
        return vec4<f32>(1.0);
    }
    let origin = view_position(input.uv, depth);

    let texel = vec2<f32>(1.0) / u.render.xy;
    let ux = input.uv + vec2<f32>(texel.x, 0.0);
    let uy = input.uv + vec2<f32>(0.0, texel.y);
    let dx = view_position(ux, depth_at(ux)) - origin;
    let dy = view_position(uy, depth_at(uy)) - origin;
    let normal = normalize(cross(dx, dy));

    // A rotation per pixel, from the pixel's own coordinates. A noise texture
    // would be better distributed; this is one line and needs no upload, and
    // the blur that follows hides the difference.
    let angle = fract(sin(dot(input.uv * u.render.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453) * 6.2831853;
    let rot = vec2<f32>(cos(angle), sin(angle));

    let radius = u.render.z;
    var occlusion = 0.0;
    let count = 16;
    for (var i = 0; i < count; i = i + 1) {
        // A spiral of samples in the hemisphere above the surface.
        let fi = f32(i);
        let a = fi * 2.39996323 + angle;          // golden angle
        let r = radius * sqrt((fi + 0.5) / f32(count));
        var dir = vec3<f32>(cos(a) * r, sin(a) * r, 0.0);
        dir = dir - normal * dot(dir, normal);     // project onto the plane
        let height = radius * 0.5 * sqrt((fi + 0.5) / f32(count));
        let sample_point = origin + dir + normal * height;

        let clip = u.projection * vec4<f32>(sample_point, 1.0);
        let sample_uv = (clip.xy / clip.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }
        let sample_depth = depth_at(sample_uv);
        let scene_z = view_position(sample_uv, sample_depth).z;

        // Occluded if the scene is in front of the sample point. The range
        // check stops a distant wall behind a foreground object from casting
        // occlusion onto it — the classic SSAO halo.
        let range = clamp(radius / max(abs(origin.z - scene_z), 1e-4), 0.0, 1.0);
        if (scene_z >= sample_point.z + 0.02) {
            occlusion = occlusion + range;
        }
        let _unused = rot;
    }
    let ao = 1.0 - (occlusion / f32(count)) * u.render.w;
    return vec4<f32>(clamp(ao, 0.0, 1.0));
}

/// A 4x4 box blur over the AO buffer, to take the sampling noise out.
@fragment
fn fs_blur(input: FullscreenOut) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(1.0) / u.render.xy;
    var sum = 0.0;
    for (var y = -2; y <= 1; y = y + 1) {
        for (var x = -2; x <= 1; x = x + 1) {
            sum = sum + textureSampleLevel(
                source, linear_sampler, input.uv + vec2<f32>(f32(x), f32(y)) * texel, 0.0
            ).r;
        }
    }
    return vec4<f32>(sum / 16.0);
}

/// ACES filmic tone mapping, in Narkowicz's fitted form.
///
/// The scene is lit in linear HDR and the display is not. Clamping instead
/// would blow every highlight to flat white and make a bright scene look
/// plastic; this rolls them off. The curve is a fit rather than the real ACES
/// transform, which is a chain of matrices and a spline and would be a
/// strange thing to carry for a debug view.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

/// The world-space ray through a pixel, for the sky.
fn view_ray(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let view = u.inverse_projection * ndc;
    let world = u.inverse_view * vec4<f32>(view.xyz / view.w, 0.0);
    return normalize(world.xyz);
}

@fragment
fn fs_tonemap(input: FullscreenOut) -> @location(0) vec4<f32> {
    // Where nothing was drawn, draw the environment. Doing it here rather than
    // as a clear colour is what lets the background be a direction-dependent
    // gradient that turns with the camera, and lets it be the same function a
    // reflective surface samples.
    if (depth_at(input.uv) >= 1.0) {
        return vec4<f32>(aces(environment(view_ray(input.uv)) * u.grade.x), 1.0);
    }
    let hdr = textureSampleLevel(source, linear_sampler, input.uv, 0.0).rgb;
    // The swapchain is sRGB, so the hardware does the transfer function on
    // write. Doing it here as well is the classic double-gamma that makes
    // everything look washed out.
    return vec4<f32>(aces(hdr * u.grade.x), 1.0);
}

// ---------------------------------------------------------------- volume

// The lattice, its placement, and how solid it is. Bound only by the volume
// pipeline; the other fullscreen passes share group 1 with different contents,
// which is legal because they are different pipeline layouts.
struct VolumeUniforms {
    // xyz: world position of voxel (0,0,0). w: voxel edge length, metres.
    origin: vec4<f32>,
    // xyz: voxel counts. w: opacity of full material, per metre.
    dims: vec4<f32>,
    // x: how many steps to take at most. y: step length in voxels.
    // z: 1 while a volume is bound, 0 otherwise. w: opacity floor.
    march: vec4<f32>,
};

@group(1) @binding(5) var<uniform> vu: VolumeUniforms;
@group(1) @binding(6) var volume_field: texture_3d<f32>;
@group(1) @binding(7) var volume_lut: texture_2d<f32>;

/// Where a ray enters and leaves an axis-aligned box.
///
/// The slab method. Returns `(enter, leave)` along the ray, and `leave < enter`
/// where the ray misses — the branchless test that avoids the six-case
/// analysis people write by hand and get wrong at the corners.
fn slab(origin: vec3<f32>, direction: vec3<f32>, low: vec3<f32>, high: vec3<f32>) -> vec2<f32> {
    // A component of zero is a ray parallel to that pair of planes. Dividing
    // gives an infinity of the right sign, and min/max carry it correctly, so
    // the degenerate case needs no branch — but only if the zero is signed
    // the way the ray is, which is why this is not `1.0 / max(d, eps)`.
    let inverse = 1.0 / direction;
    let a = (low - origin) * inverse;
    let b = (high - origin) * inverse;
    let near = min(a, b);
    let far = max(a, b);
    return vec2<f32>(
        max(max(near.x, near.y), near.z),
        min(min(far.x, far.y), far.z),
    );
}

/// Emission-absorption ray marching, front to back.
///
/// Drawn after tone mapping and composited over the image, which is
/// deliberate: the colours come from a `ScalarField` whose range is fixed
/// (§19), so they are display colours already. Tone mapping them a second time
/// would bend the scale — the same temperature would read as a different
/// colour depending on how bright the rest of the scene happened to be, which
/// is exactly what a fixed scale exists to prevent.
@fragment
fn fs_volume(input: FullscreenOut) -> @location(0) vec4<f32> {
    if (vu.march.z < 0.5) {
        discard;
    }
    let eye = u.eye.xyz;
    let ray = view_ray(input.uv);
    let cell = vu.origin.w;
    let low = vu.origin.xyz;
    let high = low + vu.dims.xyz * cell;

    var hit = slab(eye, ray, low, high);
    hit.x = max(hit.x, 0.0);
    if (hit.y <= hit.x) {
        discard;
    }

    // Stop at the geometry. The depth buffer is the scene's, so a solid drawn
    // as a surface occludes the field inside it and a body in front of the
    // volume occludes all of it — without this the volume floats over the
    // scene like a decal and every spatial relationship in the picture is a
    // lie.
    let depth = depth_at(input.uv);
    if (depth < 1.0) {
        // View space puts the eye at the origin, so the distance along the ray
        // is the length of the reconstructed position.
        hit.y = min(hit.y, length(view_position(input.uv, depth)));
        if (hit.y <= hit.x) {
            discard;
        }
    }

    let step = cell * vu.march.y;
    let steps = min(vu.march.x, ceil((hit.y - hit.x) / step));
    let extent = high - low;

    var colour = vec3<f32>(0.0);
    var alpha = 0.0;
    var t = hit.x + step * 0.5;
    for (var i = 0.0; i < steps; i = i + 1.0) {
        let position = eye + ray * t;
        let uvw = (position - low) / extent;
        let sample = textureSampleLevel(volume_field, linear_sampler, uvw, 0.0);
        let fill = sample.g;
        if (fill > 0.004) {
            // Opacity per *metre*, turned into opacity for this step. An
            // opacity per sample would make the picture depend on the step
            // size, so marching more finely to see more detail would make the
            // material denser.
            // The transfer function: opacity climbs with the value, from a
            // floor that keeps material at the bottom of the scale visible as
            // a mist. Without it the cold outside of a hot body is an opaque
            // shell and the volume shows exactly what a surface render
            // already did.
            let weight = vu.march.w + (1.0 - vu.march.w) * sample.r;
            let a = clamp(1.0 - exp(-fill * weight * vu.dims.w * step), 0.0, 1.0);
            let rgb = textureSampleLevel(volume_lut, linear_sampler, vec2<f32>(sample.r, 0.5), 0.0).rgb;
            // Premultiplied, front to back: each sample contributes only what
            // the samples before it left uncovered.
            colour = colour + (1.0 - alpha) * a * rgb;
            alpha = alpha + (1.0 - alpha) * a;
            if (alpha > 0.995) {
                break;
            }
        }
        t = t + step;
    }
    if (alpha <= 0.001) {
        discard;
    }
    return vec4<f32>(colour, alpha);
}
