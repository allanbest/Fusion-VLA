//! Tetrahedral meshes, and the surfaces they bound.
//!
//! A box is cut into cubes and each cube into six tetrahedra around its main
//! diagonal (Kuhn's subdivision), the same way in every cube, so that
//! neighbouring cubes' faces match and the mesh is conforming. A ball is the
//! cubes whose centres lie inside it — stepped at its surface, the price of
//! a mesher with no dependencies, until a check says otherwise.

use std::collections::HashMap;

use core_math::Vec3;

/// Nodes at rest and tetrahedra by node index, each positively oriented.
///
/// Elements are linear (four corner nodes) or, with [`TetMesh::quadratic`],
/// quadratic: ten nodes, the corners and the midpoints of the six edges,
/// whose strain can vary across the element — what bending needs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TetMesh {
    /// Rest positions in the object's own frame, m.
    pub rest: Vec<Vec3>,
    /// Each element's corners.
    pub tets: Vec<[u32; 4]>,
    /// For quadratic elements, each element's edge midpoints, in the order
    /// of [`EDGES`]; empty for linear elements.
    pub midpoints: Vec<[u32; 6]>,
}

/// A tetrahedron's edges, as corner pairs; the order of
/// [`TetMesh::midpoints`].
pub const EDGES: [(usize, usize); 6] = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)];

/// The six tetrahedra of a unit cube, as corner indices `x + 2y + 4z`: the
/// paths from corner 0 to corner 7 along the axes in each order.
const KUHN: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

impl TetMesh {
    /// A box of `size`, centred on the origin, cut into cubes of about
    /// `edge` on a side.
    pub fn cuboid(size: Vec3, edge: f64) -> Self {
        let cells = [
            ((size.x / edge).round() as usize).max(1),
            ((size.y / edge).round() as usize).max(1),
            ((size.z / edge).round() as usize).max(1),
        ];
        Self::lattice(size, cells, &|_| true)
    }

    /// A ball of `radius` centred on the origin, from cubes of about `edge`.
    pub fn ball(radius: f64, edge: f64) -> Self {
        let cells = ((2.0 * radius / edge).round() as usize).max(2);
        Self::lattice(Vec3::splat(2.0 * radius), [cells; 3], &|c| {
            c.length() <= radius
        })
    }

    /// `cells` cubes along each axis spanning `size`, centred on the origin,
    /// keeping the cubes whose centre `keep` accepts.
    fn lattice(size: Vec3, cells: [usize; 3], keep: &dyn Fn(Vec3) -> bool) -> Self {
        let step = Vec3::new(
            size.x / cells[0] as f64,
            size.y / cells[1] as f64,
            size.z / cells[2] as f64,
        );
        let corner = size * -0.5;
        let n = [cells[0] + 1, cells[1] + 1, cells[2] + 1];
        let id = |i: usize, j: usize, k: usize| i + n[0] * (j + n[1] * k);
        let mut used = vec![u32::MAX; n[0] * n[1] * n[2]];
        let mut mesh = Self::default();
        for k in 0..cells[2] {
            for j in 0..cells[1] {
                for i in 0..cells[0] {
                    let centre = corner
                        + Vec3::new(
                            (i as f64 + 0.5) * step.x,
                            (j as f64 + 0.5) * step.y,
                            (k as f64 + 0.5) * step.z,
                        );
                    if !keep(centre) {
                        continue;
                    }
                    let mut corners = [0u32; 8];
                    for (c, slot) in corners.iter_mut().enumerate() {
                        let (a, b, d) = (i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                        let g = id(a, b, d);
                        if used[g] == u32::MAX {
                            used[g] = mesh.rest.len() as u32;
                            mesh.rest.push(
                                corner
                                    + Vec3::new(
                                        a as f64 * step.x,
                                        b as f64 * step.y,
                                        d as f64 * step.z,
                                    ),
                            );
                        }
                        *slot = used[g];
                    }
                    for t in KUHN {
                        let mut tet = [corners[t[0]], corners[t[1]], corners[t[2]], corners[t[3]]];
                        if mesh.signed_volume_of(&tet) < 0.0 {
                            tet.swap(2, 3);
                        }
                        mesh.tets.push(tet);
                    }
                }
            }
        }
        mesh
    }

    /// The same mesh with quadratic elements: a node added at the middle of
    /// every edge (shared between the elements around it).
    pub fn quadratic(&self) -> Self {
        if self.is_quadratic() {
            return self.clone();
        }
        let mut rest = self.rest.clone();
        let mut at: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
        let midpoints = self
            .tets
            .iter()
            .map(|t| {
                EDGES.map(|(i, j)| {
                    let key = (t[i].min(t[j]), t[i].max(t[j]));
                    *at.entry(key).or_insert_with(|| {
                        rest.push((self.rest[t[i] as usize] + self.rest[t[j] as usize]) * 0.5);
                        (rest.len() - 1) as u32
                    })
                })
            })
            .collect();
        Self {
            rest,
            tets: self.tets.clone(),
            midpoints,
        }
    }

    pub fn is_quadratic(&self) -> bool {
        !self.midpoints.is_empty()
    }

    /// Element `e`'s nodes: its corners, then (quadratic) its midpoints.
    pub fn element(&self, e: usize) -> Vec<u32> {
        let mut nodes = self.tets[e].to_vec();
        if self.is_quadratic() {
            nodes.extend_from_slice(&self.midpoints[e]);
        }
        nodes
    }

    /// Each node's share of its element's mass, by local node: equal
    /// quarters for linear elements; for quadratic ones the diagonal of the
    /// consistent mass scaled to the total (HRZ lumping) — 1/36 to each
    /// corner, 4/27 to each midpoint, all positive.
    pub fn mass_shares(&self) -> Vec<f64> {
        if self.is_quadratic() {
            let mut shares = vec![1.0 / 36.0; 4];
            shares.extend([4.0 / 27.0; 6]);
            shares
        } else {
            vec![0.25; 4]
        }
    }

    fn signed_volume_of(&self, tet: &[u32; 4]) -> f64 {
        let p = |q: u32| self.rest[q as usize];
        (p(tet[1]) - p(tet[0]))
            .cross(p(tet[2]) - p(tet[0]))
            .dot(p(tet[3]) - p(tet[0]))
            / 6.0
    }

    /// Rest volume of each tetrahedron, m³.
    pub fn volumes(&self) -> Vec<f64> {
        self.tets.iter().map(|t| self.signed_volume_of(t)).collect()
    }

    pub fn volume(&self) -> f64 {
        self.volumes().iter().sum()
    }

    /// Whether every tetrahedron has positive volume and valid indices.
    pub fn is_valid(&self) -> bool {
        let n = self.rest.len() as u32;
        let midpoints_ok = self.midpoints.is_empty()
            || (self.midpoints.len() == self.tets.len()
                && self.midpoints.iter().flatten().all(|&i| i < n));
        midpoints_ok
            && self
                .tets
                .iter()
                .all(|t| t.iter().all(|&i| i < n) && self.signed_volume_of(t) > 0.0)
    }

    /// The boundary.
    pub fn surface(&self) -> Surface {
        let surface = Surface::of(&self.tets, self.rest.len());
        if !self.is_quadratic() {
            return surface;
        }
        // Each six-node face as four triangles through its edge midpoints.
        let mut midpoint: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        for (t, m) in self.tets.iter().zip(&self.midpoints) {
            for (k, (i, j)) in EDGES.into_iter().enumerate() {
                midpoint.insert((t[i].min(t[j]), t[i].max(t[j])), m[k]);
            }
        }
        let mid = |a: u32, b: u32| midpoint[&(a.min(b), a.max(b))];
        let faces: Vec<[u32; 3]> = surface
            .faces
            .iter()
            .flat_map(|&[a, b, c]| {
                let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
                [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            })
            .collect();
        Surface::from_faces(faces, self.rest.len())
    }
}

/// The boundary of a tetrahedral mesh: what contact sees.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Surface {
    /// Triangles, wound so their normal `(b − a) × (c − a)` points out.
    pub faces: Vec<[u32; 3]>,
    /// Edges of those triangles, each once, `a < b`.
    pub edges: Vec<[u32; 2]>,
    /// Nodes on the boundary, ascending.
    pub nodes: Vec<u32>,
}

impl Surface {
    /// Faces that belong to exactly one tetrahedron, in the order the
    /// tetrahedra list them (so deterministic).
    pub fn of(tets: &[[u32; 4]], nodes: usize) -> Self {
        // The outward faces of a positively oriented tetrahedron [a b c d].
        let faces_of = |[a, b, c, d]: [u32; 4]| [[a, c, b], [a, b, d], [a, d, c], [b, c, d]];
        let key = |f: [u32; 3]| {
            let mut k = f;
            k.sort_unstable();
            k
        };
        let mut count: HashMap<[u32; 3], u32> = HashMap::with_capacity(4 * tets.len());
        for &t in tets {
            for f in faces_of(t) {
                *count.entry(key(f)).or_default() += 1;
            }
        }
        let faces: Vec<[u32; 3]> = tets
            .iter()
            .flat_map(|&t| faces_of(t))
            .filter(|&f| count[&key(f)] == 1)
            .collect();
        Self::from_faces(faces, nodes)
    }

    /// The surface made of these outward-wound triangles.
    pub fn from_faces(faces: Vec<[u32; 3]>, nodes: usize) -> Self {
        let mut edges: Vec<[u32; 2]> = faces
            .iter()
            .flat_map(|&[a, b, c]| {
                [
                    [a.min(b), a.max(b)],
                    [b.min(c), b.max(c)],
                    [c.min(a), c.max(a)],
                ]
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();
        let mut on = vec![false; nodes];
        for f in &faces {
            for &i in f {
                on[i as usize] = true;
            }
        }
        let nodes = (0..nodes as u32).filter(|&i| on[i as usize]).collect();
        Self {
            faces,
            edges,
            nodes,
        }
    }
}
