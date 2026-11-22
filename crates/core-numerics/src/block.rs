//! Sparse matrices of 3×3 blocks — the natural shape of mechanics, where
//! every degree of freedom is a point in space.
//!
//! Two ways to fill one:
//! - **A fixed pattern** ([`BlockMatrix::with_pattern`]): for terms whose
//!   coupling does not change between steps (elements of a mesh). Each term
//!   looks up its [`BlockMatrix::slot`]s once and writes values straight
//!   into them; nothing is sorted per iteration.
//! - **Triplets** ([`Triplets`]): for couplings that come and go (contacts),
//!   gathered in any order and built into a matrix each time.
//!
//! A step's Hessian is then the [`Sum`] of the two.

use std::sync::Mutex;

use core_math::{Mat3, Vec3};
use core_runtime::Pool;

use crate::{BlockOperator, LinearOperator};

/// A square matrix of 3×3 blocks in compressed rows, columns ascending in
/// each row.
#[derive(Clone, Debug, Default)]
pub struct BlockMatrix {
    size: usize,
    row_start: Vec<usize>,
    columns: Vec<u32>,
    blocks: Vec<Mat3>,
}

impl BlockMatrix {
    /// A zero matrix of `size` blocks a side storing the given `(row,
    /// column)` blocks (duplicates allowed).
    pub fn with_pattern(size: usize, entries: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut pairs: Vec<(u32, u32)> = entries
            .into_iter()
            .map(|(r, c)| {
                debug_assert!(r < size && c < size);
                (r as u32, c as u32)
            })
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        let mut row_start = vec![0usize; size + 1];
        for &(r, _) in &pairs {
            row_start[r as usize + 1] += 1;
        }
        for i in 0..size {
            row_start[i + 1] += row_start[i];
        }
        let columns: Vec<u32> = pairs.iter().map(|&(_, c)| c).collect();
        let blocks = vec![Mat3::ZERO; columns.len()];
        Self {
            size,
            row_start,
            columns,
            blocks,
        }
    }

    /// Blocks a side.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Stored blocks.
    pub fn stored(&self) -> usize {
        self.blocks.len()
    }

    /// Where block `(row, column)` is stored, if it is.
    pub fn slot(&self, row: usize, column: usize) -> Option<usize> {
        let range = self.row_start[row]..self.row_start[row + 1];
        self.columns[range.clone()]
            .binary_search(&(column as u32))
            .ok()
            .map(|k| range.start + k)
    }

    /// All stored blocks, by slot.
    pub fn values(&self) -> &[Mat3] {
        &self.blocks
    }

    pub fn values_mut(&mut self) -> &mut [Mat3] {
        &mut self.blocks
    }

    /// Sets every stored block to zero, keeping the pattern.
    pub fn clear_values(&mut self) {
        self.blocks.fill(Mat3::ZERO);
    }

    /// The blocks of `row`, by column.
    pub fn row(&self, row: usize) -> impl Iterator<Item = (usize, &Mat3)> {
        let range = self.row_start[row]..self.row_start[row + 1];
        self.columns[range.clone()]
            .iter()
            .map(|&c| c as usize)
            .zip(&self.blocks[range])
    }

    /// The slots of `row`.
    pub fn row_slots(&self, row: usize) -> std::ops::Range<usize> {
        self.row_start[row]..self.row_start[row + 1]
    }

    /// The block at `(row, column)`, zero if not stored.
    pub fn get(&self, row: usize, column: usize) -> Mat3 {
        self.slot(row, column)
            .map_or(Mat3::ZERO, |s| self.blocks[s])
    }

    /// Runs `body(row, first_slot, blocks)` for every row in parallel, where
    /// `blocks` are the row's stored blocks (columns ascending) and
    /// `first_slot` the slot of the first — so a term can fill rows it owns
    /// without two threads ever writing the same block.
    pub fn fill_rows(
        &mut self,
        pool: &Pool,
        rows_per_chunk: usize,
        body: impl Fn(usize, usize, &mut [Mat3]) + Sync,
    ) {
        let rows_per_chunk = rows_per_chunk.max(1);
        let mut parts = Vec::with_capacity(self.size.div_ceil(rows_per_chunk));
        let mut rest: &mut [Mat3] = &mut self.blocks;
        let mut first = 0;
        while first < self.size {
            let last = (first + rows_per_chunk).min(self.size);
            let (part, tail) = rest.split_at_mut(self.row_start[last] - self.row_start[first]);
            parts.push(Mutex::new(Some((first, last, part))));
            rest = tail;
            first = last;
        }
        let row_start = &self.row_start;
        pool.run(parts.len(), |c| {
            let (first, last, part) = parts[c].lock().unwrap().take().unwrap();
            let base = row_start[first];
            for row in first..last {
                let range = row_start[row] - base..row_start[row + 1] - base;
                body(row, row_start[row], &mut part[range]);
            }
        });
    }

    /// Whether the matrix equals its transpose, to `tolerance` relative to
    /// its largest block.
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        let scale = self
            .blocks
            .iter()
            .map(|b| b.frobenius_squared())
            .fold(0.0, f64::max)
            .sqrt();
        (0..self.size).all(|i| {
            self.row(i).all(|(j, b)| {
                (*b - self.get(j, i).transpose()).frobenius_squared().sqrt() <= tolerance * scale
            })
        })
    }
}

impl LinearOperator for BlockMatrix {
    fn dimension(&self) -> usize {
        3 * self.size
    }

    fn add_part(&self, x: &[f64], offset: usize, out: &mut [f64]) {
        debug_assert!(offset.is_multiple_of(3) && out.len().is_multiple_of(3));
        for (k, out) in out.chunks_exact_mut(3).enumerate() {
            let mut sum = Vec3::ZERO;
            for (column, block) in self.row(offset / 3 + k) {
                sum += *block * Vec3::read(x, column);
            }
            out[0] += sum.x;
            out[1] += sum.y;
            out[2] += sum.z;
        }
    }
}

impl BlockOperator for BlockMatrix {
    fn block(&self, row: usize, column: usize) -> Mat3 {
        self.get(row, column)
    }

    fn add_diagonal(&self, diagonal: &mut [Mat3]) {
        for (i, d) in diagonal.iter_mut().enumerate() {
            if let Some(s) = self.slot(i, i) {
                *d += self.blocks[s];
            }
        }
    }

    fn for_each_row_block(&self, row: usize, visit: &mut dyn FnMut(usize, Mat3)) {
        for (column, &block) in self.row(row) {
            if block != Mat3::ZERO {
                visit(column, block);
            }
        }
    }
}

/// `A + B`, applied without forming it.
pub struct Sum<A, B>(pub A, pub B);

impl<A: LinearOperator, B: LinearOperator> LinearOperator for Sum<A, B> {
    fn dimension(&self) -> usize {
        debug_assert_eq!(self.0.dimension(), self.1.dimension());
        self.0.dimension()
    }

    fn add_part(&self, x: &[f64], offset: usize, out: &mut [f64]) {
        self.0.add_part(x, offset, out);
        self.1.add_part(x, offset, out);
    }
}

impl<A: BlockOperator, B: BlockOperator> BlockOperator for Sum<A, B> {
    fn block(&self, row: usize, column: usize) -> Mat3 {
        self.0.block(row, column) + self.1.block(row, column)
    }

    fn add_diagonal(&self, diagonal: &mut [Mat3]) {
        self.0.add_diagonal(diagonal);
        self.1.add_diagonal(diagonal);
    }

    fn for_each_row_block(&self, row: usize, visit: &mut dyn FnMut(usize, Mat3)) {
        self.0.for_each_row_block(row, visit);
        self.1.for_each_row_block(row, visit);
    }
}

impl<T: LinearOperator + ?Sized> LinearOperator for &T {
    fn dimension(&self) -> usize {
        (**self).dimension()
    }
    fn add_part(&self, x: &[f64], offset: usize, out: &mut [f64]) {
        (**self).add_part(x, offset, out)
    }
}

impl<T: BlockOperator + ?Sized> BlockOperator for &T {
    fn block(&self, row: usize, column: usize) -> Mat3 {
        (**self).block(row, column)
    }

    fn add_diagonal(&self, diagonal: &mut [Mat3]) {
        (**self).add_diagonal(diagonal)
    }

    fn for_each_row_block(&self, row: usize, visit: &mut dyn FnMut(usize, Mat3)) {
        (**self).for_each_row_block(row, visit)
    }
}

/// One entry of a [`Triplets`] list: `(row, column, block)`.
pub type Triplet = (u32, u32, Mat3);

/// Entries `(row, column, block)` gathered in any order, duplicates summed —
/// for couplings that change from one iteration to the next.
///
/// The list is kept in parts: entries added one at a time, and whole lists
/// handed over by [`Triplets::append`] without copying them (a contact-rich
/// step has a hundred thousand entries, written by several threads into
/// lists of their own). All memory is kept between one use and the next.
#[derive(Clone, Debug, Default)]
pub struct Triplets {
    size: usize,
    /// In insertion order.
    parts: Vec<Vec<Triplet>>,
    /// Whether the last part is the one [`Triplets::add`] is filling.
    open: bool,
    /// Emptied lists, to be filled again.
    spare: Vec<Vec<Triplet>>,
    scratch: Scratch,
}

/// Working memory of [`Triplets::assemble_into`].
#[derive(Clone, Debug, Default)]
struct Scratch {
    /// Per `(row, column)` of a small matrix: where its sum is kept, or
    /// [`NONE`]. All [`NONE`] between uses.
    place: Vec<u32>,
    /// The `(row, column)`s met, as `row · size + column`, in the order met,
    /// and their sums.
    keys: Vec<u32>,
    sums: Vec<Mat3>,
    order: Vec<u32>,
    start: Vec<u32>,
    fill: Vec<u32>,
    by_row: Vec<Triplet>,
    sorted: Vec<(u32, Mat3)>,
}

const NONE: u32 = u32::MAX;

/// Up to this many `(row, column)`s a matrix gets a table with a place for
/// each (4 MB), and its entries are summed in one pass in the order they
/// were added; a larger one is summed row by row.
const TABLE: usize = 1 << 20;

impl Triplets {
    /// Room for an `size`×`size`-block matrix.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            ..Self::default()
        }
    }

    /// Empties the list for a matrix of `size` blocks a side, keeping its
    /// memory.
    pub fn reset(&mut self, size: usize) {
        self.size = size;
        self.open = false;
        for mut part in self.parts.drain(..) {
            part.clear();
            self.spare.push(part);
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn len(&self) -> usize {
        self.parts.iter().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.iter().all(Vec::is_empty)
    }

    /// Adds `block` at block `(row, column)`.
    #[inline]
    pub fn add(&mut self, row: usize, column: usize, block: Mat3) {
        debug_assert!(row < self.size && column < self.size);
        if !self.open {
            self.parts.push(self.spare.pop().unwrap_or_default());
            self.open = true;
        }
        let part = self.parts.last_mut().expect("an open part");
        part.push((row as u32, column as u32, block));
    }

    /// Adds every entry of `entries`, in order, after those added so far,
    /// and leaves `entries` empty (with memory to fill again, once this
    /// list has been through a few uses). Nothing is copied.
    pub fn append(&mut self, entries: &mut Vec<Triplet>) {
        if entries.is_empty() {
            return;
        }
        debug_assert!(entries
            .iter()
            .all(|&(r, c, _)| (r as usize) < self.size && (c as usize) < self.size));
        let mut taken = self.spare.pop().unwrap_or_default();
        std::mem::swap(&mut taken, entries);
        self.parts.push(taken);
        self.open = false;
    }

    /// Adds `block` at `(row, column)` and its transpose at `(column, row)` —
    /// once, on the diagonal.
    #[inline]
    pub fn add_symmetric(&mut self, row: usize, column: usize, block: Mat3) {
        self.add(row, column, block);
        if row != column {
            self.add(column, row, block.transpose());
        }
    }

    /// The matrix, duplicates summed in the order they were added (so the
    /// result is deterministic).
    pub fn build(&self) -> BlockMatrix {
        let mut matrix = BlockMatrix::default();
        self.build_into(&mut matrix);
        matrix
    }

    /// As [`Triplets::build`], reusing `matrix`'s memory.
    pub fn build_into(&self, matrix: &mut BlockMatrix) {
        Self::assemble(self.size, &self.parts, &mut Scratch::default(), matrix);
    }

    /// As [`Triplets::build_into`], reusing this list's working memory too:
    /// for a list built into a matrix again and again.
    pub fn assemble_into(&mut self, matrix: &mut BlockMatrix) {
        Self::assemble(self.size, &self.parts, &mut self.scratch, matrix);
    }

    /// Every `(row, column)` gets the sum of its entries, taken in the
    /// order they were added: its first entry, then each later one added to
    /// it. (Sorting all the entries first, as this once did, cost more than
    /// everything else here on a contact-rich step, where each block of the
    /// matrix is the sum of a dozen entries.)
    fn assemble(n: usize, parts: &[Vec<Triplet>], scratch: &mut Scratch, matrix: &mut BlockMatrix) {
        let Scratch {
            place,
            keys,
            sums,
            order,
            start,
            fill,
            by_row,
            sorted,
        } = scratch;
        keys.clear();
        sums.clear();
        matrix.size = n;
        matrix.row_start.clear();
        matrix.columns.clear();
        matrix.blocks.clear();
        let entries = || parts.iter().flatten();
        if n.saturating_mul(n) <= TABLE {
            // One pass: a place for every (row, column) met.
            place.resize(n * n, NONE);
            for &(row, column, block) in entries() {
                let key = row as usize * n + column as usize;
                let at = place[key];
                if at == NONE {
                    place[key] = keys.len() as u32;
                    keys.push(key as u32);
                    sums.push(block);
                } else {
                    sums[at as usize] += block;
                }
            }
            for &key in keys.iter() {
                place[key as usize] = NONE;
            }
            // The places by row (a counting sort), each row then by column.
            start.clear();
            start.resize(n + 1, 0);
            for &key in keys.iter() {
                start[key as usize / n + 1] += 1;
            }
            for row in 0..n {
                start[row + 1] += start[row];
            }
            order.clear();
            order.resize(keys.len(), 0);
            for (k, &key) in keys.iter().enumerate() {
                let row = key as usize / n;
                order[start[row] as usize] = k as u32;
                start[row] += 1;
            }
            // `start[row]` is now the end of `row`: the start of the next.
            let mut first = 0;
            matrix.row_start.push(0);
            for &end in &start[..n] {
                let row = &mut order[first..end as usize];
                row.sort_unstable_by_key(|&k| keys[k as usize]);
                for &k in row.iter() {
                    matrix.columns.push(keys[k as usize] % n as u32);
                    matrix.blocks.push(sums[k as usize]);
                }
                matrix.row_start.push(matrix.columns.len());
                first = end as usize;
            }
            return;
        }
        // Row by row: a counting sort by row — linear, and stable, so each
        // row keeps its entries in insertion order — then within a row a
        // place for every column met.
        start.clear();
        start.resize(n + 1, 0);
        for &(row, _, _) in entries() {
            start[row as usize + 1] += 1;
        }
        for row in 0..n {
            start[row + 1] += start[row];
        }
        fill.clone_from(start);
        by_row.clear();
        by_row.resize(start[n] as usize, (0, 0, Mat3::ZERO));
        for &entry in entries() {
            let at = &mut fill[entry.0 as usize];
            by_row[*at as usize] = entry;
            *at += 1;
        }
        place.clear();
        place.resize(n, NONE);
        sorted.clear();
        matrix.row_start.push(0);
        for row in 0..n {
            let first = matrix.columns.len();
            for &(_, column, block) in &by_row[start[row] as usize..start[row + 1] as usize] {
                let at = place[column as usize];
                if at == NONE {
                    place[column as usize] = (matrix.columns.len() - first) as u32;
                    matrix.columns.push(column);
                    matrix.blocks.push(block);
                } else {
                    matrix.blocks[first + at as usize] += block;
                }
            }
            let columns = &mut matrix.columns[first..];
            for &column in columns.iter() {
                place[column as usize] = NONE;
            }
            if !columns.is_sorted() {
                let blocks = &mut matrix.blocks[first..];
                sorted.clear();
                sorted.extend(columns.iter().copied().zip(blocks.iter().copied()));
                sorted.sort_unstable_by_key(|&(column, _)| column);
                for (k, &(column, block)) in sorted.iter().enumerate() {
                    columns[k] = column;
                    blocks[k] = block;
                }
            }
            matrix.row_start.push(matrix.columns.len());
        }
        // The table is all NONE between uses, whatever its size next time.
        place.clear();
    }
}
