//! Data that can be cut into consecutive parts for threads: slices, and
//! tuples of equally long slices cut at the same places — so one pass can
//! update several arrays together.

/// Something with a length that splits into two at an index.
pub trait Parts: Sized + Send {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn split_at(self, mid: usize) -> (Self, Self);
}

impl<T: Send> Parts for &mut [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        self.split_at_mut(mid)
    }
}

impl<T: Sync> Parts for &[T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        <[T]>::split_at(self, mid)
    }
}

impl<A: Parts, B: Parts> Parts for (A, B) {
    fn len(&self) -> usize {
        debug_assert_eq!(self.0.len(), self.1.len());
        self.0.len()
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (a0, a1) = self.0.split_at(mid);
        let (b0, b1) = self.1.split_at(mid);
        ((a0, b0), (a1, b1))
    }
}

impl<A: Parts, B: Parts, C: Parts> Parts for (A, B, C) {
    fn len(&self) -> usize {
        debug_assert!(self.0.len() == self.1.len() && self.1.len() == self.2.len());
        self.0.len()
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (a0, a1) = self.0.split_at(mid);
        let (b0, b1) = self.1.split_at(mid);
        let (c0, c1) = self.2.split_at(mid);
        ((a0, b0, c0), (a1, b1, c1))
    }
}

impl<A: Parts, B: Parts, C: Parts, D: Parts> Parts for (A, B, C, D) {
    fn len(&self) -> usize {
        debug_assert!(
            self.0.len() == self.1.len()
                && self.1.len() == self.2.len()
                && self.2.len() == self.3.len()
        );
        self.0.len()
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (a0, a1) = self.0.split_at(mid);
        let (b0, b1) = self.1.split_at(mid);
        let (c0, c1) = self.2.split_at(mid);
        let (d0, d1) = self.3.split_at(mid);
        ((a0, b0, c0, d0), (a1, b1, c1, d1))
    }
}
