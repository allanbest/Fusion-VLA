//! Per-element attributes stored as columns, added only when a module needs
//! them: a scene without heat stores no temperatures.

/// The name of a column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Column(pub &'static str);

impl Column {
    /// K, per node.
    pub const TEMPERATURE: Self = Self("temperature");
    /// 0 (intact) to 1 (broken), per element.
    pub const DAMAGE: Self = Self("damage");
    /// Accumulated equivalent plastic strain, per element.
    pub const PLASTIC_STRAIN: Self = Self("plastic strain");
}

/// A set of `f64` columns of equal length.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Columns {
    len: usize,
    data: Vec<(Column, Vec<f64>)>,
}

impl Columns {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            data: Vec::new(),
        }
    }

    /// Entries per column.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn has(&self, column: Column) -> bool {
        self.data.iter().any(|(c, _)| *c == column)
    }

    pub fn get(&self, column: Column) -> Option<&[f64]> {
        self.data
            .iter()
            .find(|(c, _)| *c == column)
            .map(|(_, v)| v.as_slice())
    }

    pub fn get_mut(&mut self, column: Column) -> Option<&mut [f64]> {
        self.data
            .iter_mut()
            .find(|(c, _)| *c == column)
            .map(|(_, v)| v.as_mut_slice())
    }

    /// The column, created filled with `initial` if it does not exist.
    pub fn ensure(&mut self, column: Column, initial: f64) -> &mut [f64] {
        let i = match self.data.iter().position(|(c, _)| *c == column) {
            Some(i) => i,
            None => {
                self.data.push((column, vec![initial; self.len]));
                self.data.len() - 1
            }
        };
        &mut self.data[i].1
    }

    /// The columns present.
    pub fn names(&self) -> impl Iterator<Item = Column> + '_ {
        self.data.iter().map(|(c, _)| *c)
    }

    /// Copy column schema and values from `other`, avoiding allocations when
    /// both have the same columns (including when both are empty).
    pub fn clone_from_columns(&mut self, other: &Self) {
        if self.data.is_empty() && other.data.is_empty() && self.len == other.len {
            return;
        }
        if self.len == other.len
            && self.data.len() == other.data.len()
            && self
                .data
                .iter()
                .zip(&other.data)
                .all(|((ca, _), (cb, _))| ca == cb)
        {
            for ((_, dst), (_, src)) in self.data.iter_mut().zip(&other.data) {
                dst.clone_from(src);
            }
        } else {
            *self = other.clone();
        }
    }
}
