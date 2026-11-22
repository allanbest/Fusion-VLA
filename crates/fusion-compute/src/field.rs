//! Buffers, and who owns what is in them.

use std::marker::PhantomData;

use bytemuck::Pod;

use crate::Device;

/// Which side of the bus holds the truth for an array.
///
/// PLAN.md §24 asks for this explicitly — "explicit residency declared per
/// array (CPU-authoritative vs GPU-authoritative), not implicit unified-memory
/// reliance — behaviour varies too much across vendors". The declaration is
/// not documentation: [`Graph`](crate::Graph) refuses to schedule a kernel
/// that writes a CPU-authoritative field, and [`Field::write`] refuses to
/// overwrite a GPU-authoritative one.
///
/// The bug this exists to prevent is the quiet one. A step that reads a
/// GPU-written array from the CPU without a readback gets whatever was last
/// uploaded — usually the initial conditions, which look plausible for the
/// first frame and diverge slowly. On a machine with unified memory it can
/// even appear to work, because the pointer happens to be the same one, and
/// then it does not work on the machine with a discrete card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// The CPU owns these values; the GPU may only read them. Scene data,
    /// parameters, anything authored on the host.
    Cpu,
    /// The GPU owns these values. The CPU sees them only through
    /// [`Field::read`], which is a sync stall and is counted as one.
    Gpu,
}

/// A typed array living in GPU memory.
pub struct Field<T> {
    buffer: wgpu::Buffer,
    len: usize,
    residency: Residency,
    label: String,
    id: u64,
    marker: PhantomData<fn() -> T>,
}

impl<T: Pod> Field<T> {
    /// An uninitialized field of `len` elements.
    pub fn new(device: &Device, label: &str, len: usize, residency: Residency) -> Self {
        // Zero-length buffers are a validation error, and "no fluid in this
        // scene" is an ordinary state rather than a caller mistake.
        let bytes = (len.max(1) * std::mem::size_of::<T>()) as u64;
        let buffer = device.raw().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            len,
            residency,
            label: label.to_string(),
            id: device.next_id(),
            marker: PhantomData,
        }
    }

    /// A field initialized from host data.
    pub fn from_slice(device: &Device, label: &str, data: &[T], residency: Residency) -> Self {
        let field = Self::new(device, label, data.len(), residency);
        field.upload(device, data);
        field
    }

    /// Replace the contents from the host.
    ///
    /// Panics on a GPU-authoritative field. That is the point of the
    /// declaration: overwriting a buffer the GPU is the author of is either a
    /// mistake or a residency change, and the second one should be spelled out
    /// rather than inferred from a write.
    pub fn write(&self, device: &Device, data: &[T]) {
        assert_eq!(
            self.residency,
            Residency::Cpu,
            "`{}` is GPU-authoritative; the CPU may not write it",
            self.label
        );
        self.upload(device, data);
    }

    fn upload(&self, device: &Device, data: &[T]) {
        assert!(
            data.len() <= self.len,
            "`{}` holds {} elements, not {}",
            self.label,
            self.len,
            data.len()
        );
        if data.is_empty() {
            return;
        }
        let bytes = bytemuck::cast_slice(data);
        device.queue().write_buffer(&self.buffer, 0, bytes);
        device.record(|s| s.bytes_uploaded += bytes.len() as u64);
    }

    /// Copy the contents back to the host.
    ///
    /// **This is a sync stall** — the CPU submits a copy and then blocks until
    /// the GPU has run everything queued ahead of it. §24's whole reason for
    /// command batching is to make this rare, so it is counted in
    /// [`Stats::stalls`](crate::Stats) and a benchmark that reads back every
    /// step will say so.
    pub fn read(&self, device: &Device) -> Vec<T> {
        if self.len == 0 {
            return Vec::new();
        }
        let bytes = (self.len * std::mem::size_of::<T>()) as u64;
        let staging = device.raw().create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device
            .raw()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, bytes);
        device.queue().submit(Some(encoder.finish()));
        device.record(|s| s.submits += 1);

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let _ = device.raw().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        device.record(|s| {
            s.stalls += 1;
            s.bytes_read_back += bytes;
        });
        receiver
            .recv()
            .expect("readback channel closed")
            .expect("buffer map failed");

        let view = slice.get_mapped_range();
        let out = bytemuck::cast_slice::<u8, T>(&view).to_vec();
        drop(view);
        staging.unmap();
        out
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn residency(&self) -> Residency {
        self.residency
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl<T> std::fmt::Debug for Field<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Field")
            .field("label", &self.label)
            .field("len", &self.len)
            .field("residency", &self.residency)
            .finish()
    }
}

/// A [`Field`] with its element type erased, so a graph can hold a list of
/// them. Only the parts scheduling needs: an identity, a label, a residency
/// and a buffer to bind.
pub(crate) trait AnyField {
    fn id(&self) -> u64;
    fn label(&self) -> &str;
    fn residency(&self) -> Residency;
    fn buffer(&self) -> &wgpu::Buffer;
}

impl<T: Pod> AnyField for Field<T> {
    fn id(&self) -> u64 {
        Field::id(self)
    }
    fn label(&self) -> &str {
        Field::label(self)
    }
    fn residency(&self) -> Residency {
        Field::residency(self)
    }
    fn buffer(&self) -> &wgpu::Buffer {
        Field::buffer(self)
    }
}
