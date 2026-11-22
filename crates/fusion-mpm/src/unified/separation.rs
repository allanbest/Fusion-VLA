//! Fluid points kept from packing tighter than water does.
//!
//! The incompressible projection keeps the grid's velocity divergence-free,
//! and the grid's density right to within a few percent — averaged over
//! three cells, which is the spline's reach. A heavy block dropped into a
//! pool drove the water under it onto the floor in a sheet a fifth denser
//! than water, with a layer above it a third short, and the two averaged out
//! to about right: the grid saw nothing, and the projection held the thin
//! layer open for as long as it ran. The weakly compressible model never had
//! this: each of its points pushes back on its own compression.
//!
//! So each point is looked at as a particle-based fluid would: two points
//! nearer than most of their rest spacing are moved apart, a share of the
//! overlap each substep. Positions only — the projection owns the velocity —
//! and only overlap: points at their rest spacing, a lattice or a settled
//! pool, are never touched — pushing from nearer than 0.95 of the spacing
//! set still water moving. Once a frame, which is often enough for packing
//! that builds over many. The FLIP literature calls it particle separation.

use fusion_core::FluidSet;
use fusion_math::{to_f32, Vec3};

/// Nearer than this, as a fraction of the rest spacing, two points are
/// pushed apart.
const NEAREST: f32 = 0.85;

/// Share of an overlap undone per substep.
const SHARE: f32 = 0.5;

/// How far to move each point of `fluid`, in its own order.
pub(super) fn separation(fluid: &FluidSet, out: &mut Vec<Vec3>) {
    let n = fluid.len();
    out.clear();
    out.resize(n, Vec3::ZERO);
    if n < 2 {
        return;
    }
    let spacing = fluid.params.spacing;
    let reach = NEAREST * spacing;
    let bin = |p: Vec3| {
        [
            (p.x / reach).floor() as i64,
            (p.y / reach).floor() as i64,
            (p.z / reach).floor() as i64,
        ]
    };
    let points: Vec<Vec3> = fluid
        .particles
        .position
        .iter()
        .map(|p| to_f32(*p))
        .collect();
    // Points sorted by bin, z slowest, and each bin's run of them, from a
    // dense table over the fluid's bounds — or a map, if they are spread
    // thin enough that the table would be mostly empty.
    let mut sorted: Vec<([i64; 3], u32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let b = bin(*p);
            ([b[2], b[1], b[0]], i as u32)
        })
        .collect();
    sorted.sort_unstable();
    let (mut lo, mut hi) = ([i64::MAX; 3], [i64::MIN; 3]);
    for (k, _) in &sorted {
        for a in 0..3 {
            lo[a] = lo[a].min(k[a]);
            hi[a] = hi[a].max(k[a]);
        }
    }
    let dims = [hi[0] - lo[0] + 3, hi[1] - lo[1] + 3, hi[2] - lo[2] + 3];
    let span = dims[0] * dims[1] * dims[2];
    let dense = span <= 8 * n as i64 + 4096;
    let slot = |k: [i64; 3]| -> Option<usize> {
        let (z, y, x) = (k[0] - lo[0] + 1, k[1] - lo[1] + 1, k[2] - lo[2] + 1);
        ((0..dims[0]).contains(&z) && (0..dims[1]).contains(&y) && (0..dims[2]).contains(&x))
            .then(|| ((z * dims[1] + y) * dims[2] + x) as usize)
    };
    let mut table: Vec<(u32, u32)> = Vec::new();
    let mut map: std::collections::HashMap<[i64; 3], (u32, u32)> = Default::default();
    if dense {
        table.resize(span as usize, (0, 0));
    }
    let mut start = 0;
    while start < sorted.len() {
        let key = sorted[start].0;
        let mut end = start + 1;
        while end < sorted.len() && sorted[end].0 == key {
            end += 1;
        }
        let run = (start as u32, end as u32);
        if dense {
            table[slot(key).expect("inside its own bounds")] = run;
        } else {
            map.insert(key, run);
        }
        start = end;
    }
    let find = |k: [i64; 3]| -> &[([i64; 3], u32)] {
        let (a, b) = if dense {
            slot(k).map(|s| table[s]).unwrap_or((0, 0))
        } else {
            map.get(&k).copied().unwrap_or((0, 0))
        };
        &sorted[a as usize..b as usize]
    };
    for (i, p) in points.iter().enumerate() {
        let b = bin(*p);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    for &(_, j) in find([b[2] + dz, b[1] + dy, b[0] + dx]) {
                        let j = j as usize;
                        // Each pair once, from its lower index.
                        if j <= i {
                            continue;
                        }
                        let apart = points[j] - *p;
                        let distance = apart.length();
                        if distance >= reach {
                            continue;
                        }
                        // Two points on one spot: apart along an axis chosen
                        // by their indices, so the answer is the same every
                        // time.
                        let direction = if distance > 1e-6 * spacing {
                            apart / distance
                        } else {
                            [Vec3::X, Vec3::Y, Vec3::Z][(i + j) % 3]
                        };
                        let push = direction * (0.5 * SHARE * (reach - distance));
                        out[i] -= push;
                        out[j] += push;
                    }
                }
            }
        }
    }
}
