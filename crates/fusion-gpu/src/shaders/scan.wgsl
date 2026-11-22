// Exclusive prefix sum over an array of u32, in blocks of 256.
//
// A counting sort needs one of these and there is no way around it: the bucket
// offsets are a running total, and a running total is the one thing a
// data-parallel machine cannot do by giving each element its own thread. The
// standard answer is a scan per block plus a scan of the block totals plus an
// add, recursively, and that is what this is.
//
// Blelloch's work-efficient scan would do half the additions. This is
// Hillis-Steele, which does n log n of them and is a third of the code, and
// the whole scan is a rounding error next to the neighbour sweep it feeds —
// 262 144 buckets against 100 000 particles with 200 candidates each.

struct Params {
    count: u32,
    u0: u32, u1: u32, u2: u32,
    f: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> sums: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> shared_scan: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_block(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    let t = lid.x;

    // Threads past the end take a zero rather than returning. The barriers
    // below have to be reached by every thread in the workgroup — an early
    // return here is the classic way to hang a GPU, and it hangs it only on
    // the array lengths that are not a multiple of 256.
    var value: u32 = 0u;
    if (i < p.count) {
        value = data[i];
    }
    shared_scan[t] = value;
    workgroupBarrier();

    for (var offset: u32 = 1u; offset < 256u; offset = offset << 1u) {
        var addend: u32 = 0u;
        if (t >= offset) {
            addend = shared_scan[t - offset];
        }
        workgroupBarrier();
        shared_scan[t] = shared_scan[t] + addend;
        workgroupBarrier();
    }

    // Inclusive minus own value is exclusive.
    if (i < p.count) {
        data[i] = shared_scan[t] - value;
    }
    if (t == 255u) {
        sums[wid.x] = shared_scan[t];
    }
}

@compute @workgroup_size(256)
fn add_offsets(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= p.count) {
        return;
    }
    data[i] = data[i] + sums[wid.x];
}
