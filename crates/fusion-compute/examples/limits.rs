//! What this adapter allows. A diagnostic, not a benchmark.
fn main() {
    let Some(device) = fusion_compute::Device::headless() else {
        println!("no GPU adapter");
        return;
    };
    println!("{} via {:?}", device.adapter(), device.backend());
    let l = device.limits();
    println!(
        "  workgroups per dimension : {}",
        l.max_compute_workgroups_per_dimension
    );
    println!(
        "  invocations per workgroup: {}",
        l.max_compute_invocations_per_workgroup
    );
    println!(
        "  storage binding size     : {} MB",
        l.max_storage_buffer_binding_size / 1_000_000
    );
    println!(
        "  buffer size              : {} MB",
        l.max_buffer_size / 1_000_000
    );
}
