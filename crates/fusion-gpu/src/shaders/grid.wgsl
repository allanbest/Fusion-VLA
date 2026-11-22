// The uniform spatial hash and the neighbour sweep, on the GPU.
//
// This is `fusion-collide::SpatialHash` and `fusion-solver::fluid::
// build_neighbours` reproduced exactly — not approximately. Every constant
// below is the same constant, the cell coordinate is floored the same way and
// the 27 cells are walked in the same order, because the test that matters
// asserts the neighbour lists are *identical* to the ones the CPU builds, in
// the same order. Anything weaker and the two paths are two fluids.
//
// The counting sort is the interesting part. The CPU version inserts in
// ascending particle order and gets ascending buckets for free; a GPU scatter
// with an atomic cursor gets whatever order the hardware ran in, which changes
// between runs and would take PLAN.md §34's bit-identical replay with it. So
// the scatter is followed by an insertion sort within each bucket, which is
// cheap — a bucket holds about eight particles — and puts the array back into
// exactly the order the CPU would have produced.

struct Params {
    count: u32,
    // u0: bucket mask (table size - 1)
    // u1: bucket count (table size)
    // u2: neighbour stride, the per-particle cap
    u0: u32, u1: u32, u2: u32,
    // f.x: 1 / cell size
    // f.y: squared query radius
    f: vec4<f32>,
};

// ---------------------------------------------------------------- hashing

fn cell_of(local: vec3<f32>, inv_cell: f32) -> vec3<i32> {
    return vec3<i32>(floor(local * inv_cell));
}

// Teschner et al.'s three primes, XOR rather than sum so permuting the axes
// does not collide, then a multiply-shift finish so the low bits the mask
// keeps are mixed rather than being the raw low bits of x.
fn bucket_at(c: vec3<i32>, mask: u32) -> u32 {
    var h: u32 = (bitcast<u32>(c.x) * 0x8da6b343u)
               ^ (bitcast<u32>(c.y) * 0xd8372179u)
               ^ (bitcast<u32>(c.z) * 0xcb1ab31fu);
    h = h ^ (h >> 16u);
    return (h * 0x7feb352du) & mask;
}

// ---------------------------------------------------------------- histogram

@group(0) @binding(0) var<storage, read_write> hist_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> hist_bucket_of: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist_counts: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> hist_p: Params;

@compute @workgroup_size(64)
fn histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * 64u;
    for (var i: u32 = gid.x; i < hist_p.count; i = i + stride) {
        let b = bucket_at(cell_of(hist_pos[i].xyz, hist_p.f.x), hist_p.u0);
        hist_bucket_of[i] = b;
        // Counts are order-independent, so this atomic is not a determinism
        // problem. The scatter below is, and is dealt with there.
        atomicAdd(&hist_counts[b], 1u);
    }
}

// ---------------------------------------------------------------- scatter

@group(0) @binding(0) var<storage, read_write> sc_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sc_bucket_of: array<u32>;
@group(0) @binding(2) var<storage, read_write> sc_start: array<u32>;
@group(0) @binding(3) var<storage, read_write> sc_cursor: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> sc_sorted: array<u32>;
@group(0) @binding(5) var<storage, read_write> sc_sorted_pos: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> sc_p: Params;

@compute @workgroup_size(64)
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * 64u;
    for (var i: u32 = gid.x; i < sc_p.count; i = i + stride) {
        let b = sc_bucket_of[i];
        let slot = sc_start[b] + atomicAdd(&sc_cursor[b], 1u);
        sc_sorted[slot] = i;
        sc_sorted_pos[slot] = sc_pos[i];
    }
}

// ---------------------------------------------------------------- ordering

@group(0) @binding(0) var<storage, read_write> so_start: array<u32>;
@group(0) @binding(1) var<storage, read_write> so_sorted: array<u32>;
@group(0) @binding(2) var<storage, read_write> so_sorted_pos: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> so_p: Params;

// One thread per bucket, insertion-sorting its own slice by particle index.
//
// This is what makes the GPU grid deterministic and what makes it agree with
// the CPU's. Insertion sort because the runs are tiny — a cell of side
// `2 * spacing` holds about eight particles at the fluid's default resolution
// — and because a small sort with no branches worth predicting is a different
// problem from a large one. A pathological bucket makes this quadratic; a
// pathological bucket makes the neighbour sweep that follows quadratic too, so
// it is not this kernel's failure mode to worry about.
@compute @workgroup_size(64)
fn sort_buckets(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * 64u;
    for (var b: u32 = gid.x; b < so_p.count; b = b + stride) {
        sort_one_bucket(b);
    }
}

fn sort_one_bucket(b: u32) {
    let lo = so_start[b];
    let hi = so_start[b + 1u];
    var j = lo + 1u;
    loop {
        if (j >= hi) { break; }
        let key = so_sorted[j];
        let key_pos = so_sorted_pos[j];
        var k = j;
        loop {
            if (k == lo) { break; }
            if (so_sorted[k - 1u] <= key) { break; }
            so_sorted[k] = so_sorted[k - 1u];
            so_sorted_pos[k] = so_sorted_pos[k - 1u];
            k = k - 1u;
        }
        so_sorted[k] = key;
        so_sorted_pos[k] = key_pos;
        j = j + 1u;
    }
}

// ---------------------------------------------------------------- sweep

@group(0) @binding(0) var<storage, read_write> nb_start: array<u32>;
@group(0) @binding(1) var<storage, read_write> nb_sorted: array<u32>;
@group(0) @binding(2) var<storage, read_write> nb_sorted_pos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> nb_count: array<u32>;
@group(0) @binding(4) var<storage, read_write> nb_list: array<u32>;
@group(0) @binding(5) var<uniform> nb_p: Params;

// One thread per *slot*, not per particle.
//
// Slots are in bucket order, so neighbouring threads in a workgroup are
// neighbouring particles in space and read the same buckets. On a CPU that
// argument was worth 1.4x through the cache; here it is worth more, because
// the threads share it.
//
// The 27 buckets are deduplicated for the reason the CPU's are: two of the 27
// cell coordinates can hash to the same bucket about a third of the time, and
// enumerating it twice hands the density solve the same neighbour twice. A
// distance test does not catch that — the point really is a neighbour — and
// the fluid reads 9% dense on a lattice that has not moved.
@compute @workgroup_size(64)
fn sweep(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * 64u;
    for (var k: u32 = gid.x; k < nb_p.count; k = k + stride) {
        sweep_one(k);
    }
}

fn sweep_one(k: u32) {
    let centre = nb_sorted_pos[k].xyz;
    let cell = cell_of(centre, nb_p.f.x);
    let base = k * nb_p.u2;

    var seen: array<u32, 27>;
    var seen_count: u32 = 0u;
    var found: u32 = 0u;

    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let b = bucket_at(cell + vec3<i32>(dx, dy, dz), nb_p.u0);
                var duplicate = false;
                for (var s: u32 = 0u; s < seen_count; s = s + 1u) {
                    if (seen[s] == b) {
                        duplicate = true;
                    }
                }
                if (duplicate) {
                    continue;
                }
                seen[seen_count] = b;
                seen_count = seen_count + 1u;

                let lo = nb_start[b];
                let hi = nb_start[b + 1u];
                for (var j: u32 = lo; j < hi; j = j + 1u) {
                    if (found >= nb_p.u2) {
                        break;
                    }
                    let d = nb_sorted_pos[j].xyz - centre;
                    if (dot(d, d) <= nb_p.f.y) {
                        nb_list[base + found] = nb_sorted[j];
                        found = found + 1u;
                    }
                }
            }
        }
    }
    nb_count[k] = found;
}
