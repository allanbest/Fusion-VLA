//! A bounding-volume hierarchy over boxes: which items might touch.

use crate::Aabb;

/// Items per leaf.
const LEAF: usize = 4;

#[derive(Clone, Copy, Debug)]
struct Node {
    bounds: Aabb,
    /// A leaf: its first entry in `items`. Otherwise: its right child (the
    /// left child is the next node).
    index: u32,
    /// Items in a leaf; zero for an inner node.
    count: u32,
    /// A homogeneous subtree can be omitted in one test (e.g. an object's
    /// own surface edges). None means mixed or ungrouped.
    group: Option<u32>,
}

/// A binary tree of boxes, built top-down by median splits. Refit it when
/// the items move a little; rebuild when they move a lot.
#[derive(Clone, Debug, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    items: Vec<u32>,
    /// Each item's box, as last built or refit.
    boxes: Vec<Aabb>,
    groups: Vec<u32>,
}

impl Bvh {
    pub fn build(boxes: &[Aabb]) -> Self {
        Self::build_with_groups(boxes, &[])
    }

    /// Build with immutable item groups. [`Self::query_excluding`] skips
    /// homogeneous subtrees of the excluded group without visiting leaves.
    pub fn build_grouped(boxes: &[Aabb], groups: &[u32]) -> Self {
        assert_eq!(boxes.len(), groups.len());
        Self::build_with_groups(boxes, groups)
    }

    fn build_with_groups(boxes: &[Aabb], groups: &[u32]) -> Self {
        let mut bvh = Self {
            nodes: Vec::with_capacity(2 * boxes.len() / LEAF + 1),
            items: (0..boxes.len() as u32).collect(),
            boxes: boxes.to_vec(),
            groups: groups.to_vec(),
        };
        if !boxes.is_empty() {
            bvh.split(boxes, groups, 0, boxes.len());
        }
        bvh
    }

    fn split(&mut self, boxes: &[Aabb], groups: &[u32], start: usize, end: usize) -> usize {
        let node = self.nodes.len();
        let bounds = self.items[start..end]
            .iter()
            .fold(Aabb::EMPTY, |b, &i| b.union(&boxes[i as usize]));
        let group = if groups.is_empty() {
            None
        } else {
            let first = groups[self.items[start] as usize];
            self.items[start..end]
                .iter()
                .all(|&i| groups[i as usize] == first)
                .then_some(first)
        };
        self.nodes.push(Node {
            bounds,
            index: start as u32,
            count: (end - start) as u32,
            group,
        });
        if end - start <= LEAF {
            return node;
        }
        let centres = self.items[start..end]
            .iter()
            .fold(Aabb::EMPTY, |b, &i| b.including(boxes[i as usize].centre()));
        let axis = centres.size().max_axis();
        let middle = (start + end) / 2;
        self.items[start..end].select_nth_unstable_by(middle - start, |&a, &b| {
            let (ca, cb) = (
                boxes[a as usize].centre()[axis],
                boxes[b as usize].centre()[axis],
            );
            ca.total_cmp(&cb).then(a.cmp(&b))
        });
        self.split(boxes, groups, start, middle);
        let right = self.split(boxes, groups, middle, end);
        self.nodes[node].index = right as u32;
        self.nodes[node].count = 0;
        node
    }

    /// Recomputes every node's bounds for items that have moved, keeping
    /// the tree.
    pub fn refit(&mut self, boxes: &[Aabb]) {
        debug_assert_eq!(boxes.len(), self.boxes.len());
        self.boxes.copy_from_slice(boxes);
        // Children follow their parent, so going backwards visits them first.
        for n in (0..self.nodes.len()).rev() {
            let node = self.nodes[n];
            self.nodes[n].bounds = if node.count > 0 {
                let items = &self.items[node.index as usize..(node.index + node.count) as usize];
                items
                    .iter()
                    .fold(Aabb::EMPTY, |b, &i| b.union(&boxes[i as usize]))
            } else {
                self.nodes[n + 1]
                    .bounds
                    .union(&self.nodes[node.index as usize].bounds)
            };
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The bounds of everything.
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::EMPTY, |n| n.bounds)
    }

    /// Calls `found` with every item whose box overlaps `query`.
    pub fn query(&self, query: &Aabb, mut found: impl FnMut(usize)) {
        self.query_with_exclusion(query, None, &mut found);
    }

    pub fn query_excluding(&self, query: &Aabb, group: u32, mut found: impl FnMut(usize)) {
        self.query_with_exclusion(query, Some(group), &mut found);
    }

    fn query_with_exclusion(
        &self,
        query: &Aabb,
        excluded: Option<u32>,
        found: &mut impl FnMut(usize),
    ) {
        if self.nodes.is_empty() {
            return;
        }
        // Median splits keep the depth near log₂(n / LEAF), far below 64.
        let mut stack = [0u32; 64];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let n = stack[top];
            let node = self.nodes[n as usize];
            if (excluded.is_some() && node.group == excluded) || !node.bounds.overlaps(query) {
                continue;
            }
            if node.count > 0 {
                for &i in &self.items[node.index as usize..(node.index + node.count) as usize] {
                    if !excluded.is_some_and(|g| self.groups.get(i as usize) == Some(&g))
                        && self.boxes[i as usize].overlaps(query)
                    {
                        found(i as usize);
                    }
                }
            } else {
                stack[top] = node.index;
                stack[top + 1] = n + 1;
                top += 2;
            }
        }
    }

    /// Calls `found(a, b)`, `a < b`, for every pair of items whose boxes
    /// overlap.
    pub fn self_pairs(&self, mut found: impl FnMut(usize, usize)) {
        for (a, bounds) in self.boxes.iter().enumerate() {
            self.query(bounds, |b| {
                if a < b {
                    found(a, b);
                }
            });
        }
    }
}
