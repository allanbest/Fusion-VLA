//! Symmetric dual-tree Barnes–Hut octree for long-range 3D potentials.

use core_math::Vec3;

const LEAF_CAP: usize = 8;
const MAX_DEPTH: usize = 24;
const NONE: u32 = u32::MAX;

/// A particle-pair or multipole-cell interaction emitted by dual-tree traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interaction {
    Direct(usize, usize),
    Cells { a: u32, b: u32 },
}

#[derive(Clone, Copy, Debug)]
struct CellSpec {
    center: Vec3,
    half: f64,
    start: usize,
    count: usize,
    depth: usize,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    size: f64,
    mass: f64,
    com: Vec3,
    start: usize,
    count: usize,
    children: [u32; 8],
    is_leaf: bool,
}

/// Spatial octree organizing weighted points for symmetric `O(N log N)` force
/// and potential evaluation.
#[derive(Clone, Debug, Default)]
pub struct BarnesHut {
    nodes: Vec<Node>,
    order: Vec<usize>,
}

impl BarnesHut {
    /// Builds a deterministic octree over all indices with `masses[i] > 0.0`.
    pub fn build(positions: &[Vec3], masses: &[f64]) -> Self {
        assert_eq!(positions.len(), masses.len());
        let mut order: Vec<usize> = (0..positions.len()).filter(|&i| masses[i] > 0.0).collect();
        if order.is_empty() {
            return Self::default();
        }
        let mut min = Vec3::splat(f64::INFINITY);
        let mut max = Vec3::splat(f64::NEG_INFINITY);
        for &i in &order {
            min = min.min(positions[i]);
            max = max.max(positions[i]);
        }
        let center = (min + max) * 0.5;
        let span = max - min;
        let half = (0.5 * span.x.max(span.y).max(span.z)).max(1e-12);

        let mut tree = Self {
            nodes: Vec::with_capacity(order.len().max(1)),
            order: std::mem::take(&mut order),
        };
        let total = tree.order.len();
        tree.build_node(
            positions,
            masses,
            CellSpec {
                center,
                half,
                start: 0,
                count: total,
                depth: 0,
            },
        );
        tree
    }

    fn build_node(&mut self, positions: &[Vec3], masses: &[f64], cell: CellSpec) -> u32 {
        let CellSpec {
            center,
            half,
            start,
            count,
            depth,
        } = cell;
        let idx = self.nodes.len() as u32;
        if count <= LEAF_CAP || depth >= MAX_DEPTH {
            let mut mass = 0.0;
            let mut weighted = Vec3::ZERO;
            for &i in &self.order[start..start + count] {
                let m = masses[i];
                mass += m;
                weighted += positions[i] * m;
            }
            let com = if mass > 0.0 {
                weighted * (1.0 / mass)
            } else {
                center
            };
            self.nodes.push(Node {
                size: 2.0 * half,
                mass,
                com,
                start,
                count,
                children: [NONE; 8],
                is_leaf: true,
            });
            return idx;
        }

        // Partition indices into 8 deterministic octants while preserving
        // deterministic per-octant order.
        let mut buckets: [Vec<usize>; 8] = Default::default();
        for &i in &self.order[start..start + count] {
            let p = positions[i];
            let oct = usize::from(p.x >= center.x)
                | (usize::from(p.y >= center.y) << 1)
                | (usize::from(p.z >= center.z) << 2);
            buckets[oct].push(i);
        }
        let mut offset = start;
        let mut spans = [(0usize, 0usize); 8];
        for (oct, bucket) in buckets.iter().enumerate() {
            let len = bucket.len();
            self.order[offset..offset + len].copy_from_slice(bucket);
            spans[oct] = (offset, len);
            offset += len;
        }

        self.nodes.push(Node {
            size: 2.0 * half,
            mass: 0.0,
            com: center,
            start,
            count,
            children: [NONE; 8],
            is_leaf: false,
        });

        let child_half = 0.5 * half;
        let mut total_mass = 0.0;
        let mut total_weighted = Vec3::ZERO;
        let mut children = [NONE; 8];
        for (oct, &(c_start, c_count)) in spans.iter().enumerate() {
            if c_count == 0 {
                continue;
            }
            let sign = |bit: usize| if (oct >> bit) & 1 != 0 { 1.0 } else { -1.0 };
            let c_center = center + Vec3::new(sign(0), sign(1), sign(2)) * child_half;
            let child = self.build_node(
                positions,
                masses,
                CellSpec {
                    center: c_center,
                    half: child_half,
                    start: c_start,
                    count: c_count,
                    depth: depth + 1,
                },
            );
            children[oct] = child;
            let cn = &self.nodes[child as usize];
            total_mass += cn.mass;
            total_weighted += cn.com * cn.mass;
        }
        let com = if total_mass > 0.0 {
            total_weighted * (1.0 / total_mass)
        } else {
            center
        };
        let node = &mut self.nodes[idx as usize];
        node.mass = total_mass;
        node.com = com;
        node.children = children;
        idx
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn cell_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn cell_mass(&self, cell: u32) -> f64 {
        self.nodes[cell as usize].mass
    }

    pub fn cell_com(&self, cell: u32) -> Vec3 {
        self.nodes[cell as usize].com
    }

    pub fn cell_members(&self, cell: u32) -> &[usize] {
        let n = &self.nodes[cell as usize];
        &self.order[n.start..n.start + n.count]
    }

    pub fn cell_is_leaf(&self, cell: u32) -> bool {
        self.nodes[cell as usize].is_leaf
    }

    pub fn cell_children(&self, cell: u32) -> [u32; 8] {
        self.nodes[cell as usize].children
    }

    /// Visits every distinctunordered pair either directly or as a well-separated
    /// cell-cell multipole pair (`(size_a + size_b) < theta * r`).
    pub fn for_each_pair(&self, theta: f64, mut visit: impl FnMut(Interaction)) {
        if self.nodes.is_empty() {
            return;
        }
        self.traverse(0, 0, theta, &mut visit);
    }

    fn traverse(&self, a: u32, b: u32, theta: f64, visit: &mut impl FnMut(Interaction)) {
        let na = &self.nodes[a as usize];
        if a == b {
            if na.is_leaf {
                for u in 0..na.count {
                    for v in (u + 1)..na.count {
                        visit(Interaction::Direct(
                            self.order[na.start + u],
                            self.order[na.start + v],
                        ));
                    }
                }
            } else {
                for u in 0..8 {
                    let cu = na.children[u];
                    if cu == NONE {
                        continue;
                    }
                    self.traverse(cu, cu, theta, visit);
                    for v in (u + 1)..8 {
                        let cv = na.children[v];
                        if cv != NONE {
                            self.traverse(cu, cv, theta, visit);
                        }
                    }
                }
            }
            return;
        }

        let nb = &self.nodes[b as usize];
        if na.count == 1 && nb.count == 1 {
            visit(Interaction::Direct(
                self.order[na.start],
                self.order[nb.start],
            ));
            return;
        }
        let r = (na.com - nb.com).length();
        if theta > 0.0 && (na.size + nb.size) < theta * r && r > 0.0 {
            visit(Interaction::Cells { a, b });
            return;
        }
        if na.is_leaf && nb.is_leaf {
            for &i in &self.order[na.start..na.start + na.count] {
                for &j in &self.order[nb.start..nb.start + nb.count] {
                    visit(Interaction::Direct(i, j));
                }
            }
            return;
        }
        let split_a = !na.is_leaf && (nb.is_leaf || na.size >= nb.size);
        if split_a {
            for &c in &na.children {
                if c != NONE {
                    self.traverse(c, b, theta, visit);
                }
            }
        } else {
            for &c in &nb.children {
                if c != NONE {
                    self.traverse(a, c, theta, visit);
                }
            }
        }
    }
}
