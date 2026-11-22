//! Versioned integration history; decoded against a prepared world's layout.
use crate::step::MechanicsState;
use core_math::{Mat3, Pose, Quat, Vec3};

impl MechanicsState {
    /// Physical time represented by this state.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Lossless state and integration history, independent of scratch buffers.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer(Vec::new());
        let any_damage = self.plastic.iter().any(|p| !p.damage.is_empty());
        w.usize(if any_damage { 5 } else { 4 });
        w.f64(self.time);
        w.usize(self.nodes.len());
        for (p, v) in &self.nodes {
            w.usize(p.len());
            for &a in p.iter().chain(v) {
                w.vec(a);
            }
        }
        w.usize(self.rigids.len());
        for (p, v, s) in &self.rigids {
            w.vec(p.position);
            w.quat(p.rotation);
            w.vec(*v);
            w.vec(*s);
        }
        w.usize(self.restart.len());
        for &b in &self.restart {
            w.flag(b);
        }
        w.flag(self.acceleration.is_some());
        if let Some(a) = &self.acceleration {
            for &v in a {
                w.f64(v);
            }
        }
        w.flag(self.earlier.is_some());
        if let Some((x, v)) = &self.earlier {
            for &a in x.iter().chain(v) {
                w.f64(a);
            }
        }
        w.usize(self.affine.len());
        for a in &self.affine {
            w.flag(a.is_some());
            if let Some((q, m)) = a {
                w.quat(*q);
                for c in m.cols {
                    w.vec(c);
                }
            }
        }
        w.flag(self.gravity_seen.is_some());
        if let Some(g) = self.gravity_seen {
            w.vec(g);
        }
        w.flag(self.previous_h.is_some());
        if let Some(h) = self.previous_h {
            w.f64(h);
        }
        w.usize(self.unloading.len());
        for &e in &self.unloading {
            w.f64(e);
        }
        w.usize(self.plastic.len());
        for p in &self.plastic {
            let active = !p.fp_inv.is_empty();
            w.flag(active);
            if active {
                w.usize(p.fp_inv.len());
                for (m, &ep) in p.fp_inv.iter().zip(&p.eps_p) {
                    for c in m.cols {
                        w.vec(c);
                    }
                    w.f64(ep);
                }
            }
        }
        if any_damage {
            w.usize(self.plastic.len());
            for p in &self.plastic {
                w.usize(p.damage.len());
                for &d in &p.damage {
                    w.f64(d);
                }
            }
        }
        w.0
    }

    /// Length of [`Self::to_bytes`]'s encoding, without encoding it.
    pub fn encoded_len(&self) -> usize {
        const N: usize = 8; // a usize or an f64
        const V: usize = 3 * N;
        let nodes: usize = self
            .nodes
            .iter()
            .map(|(p, v)| N + V * (p.len() + v.len()))
            .sum();
        let affine: usize = self
            .affine
            .iter()
            .map(|a| 1 + a.map_or(0, |_| 4 * N + 3 * V))
            .sum();
        let plastic: usize = self
            .plastic
            .iter()
            .map(|p| {
                1 + if p.fp_inv.is_empty() {
                    0
                } else {
                    N + p.fp_inv.len() * (3 * V + N)
                }
            })
            .sum();
        let any_damage = self.plastic.iter().any(|p| !p.damage.is_empty());
        let damage = if any_damage {
            N + self
                .plastic
                .iter()
                .map(|p| N + N * p.damage.len())
                .sum::<usize>()
        } else {
            0
        };
        N + N // version, time
            + N + nodes
            + N + self.rigids.len() * (V + 4 * N + V + V)
            + N + self.restart.len()
            + 1 + self.acceleration.as_ref().map_or(0, |a| N * a.len())
            + 1 + self.earlier.as_ref().map_or(0, |(x, v)| N * (x.len() + v.len()))
            + N + affine
            + 1 + self.gravity_seen.map_or(0, |_| V)
            + 1 + self.previous_h.map_or(0, |_| N)
            + N + N * self.unloading.len()
            + N + plastic
            + damage
    }

    /// Decode a state with the same node/body layout as this prepared template.
    /// Lengths are checked before allocation; non-finite numbers are refused.
    pub fn from_bytes(bytes: &[u8], template: &Self) -> Result<Self, &'static str> {
        let mut r = Reader(bytes);
        let version = u64::from_le_bytes(r.take()?);
        if !(1..=5).contains(&version) {
            return Err("unsupported core state version");
        }
        let mut s = template.clone();
        s.time = r.f64()?;
        r.count(s.nodes.len())?;
        for (p, v) in &mut s.nodes {
            r.count(p.len())?;
            for a in p.iter_mut().chain(v) {
                *a = r.vec()?;
            }
        }
        r.count(s.rigids.len())?;
        for (p, v, s) in &mut s.rigids {
            *p = Pose::new(r.vec()?, r.quat()?);
            *v = r.vec()?;
            *s = r.vec()?;
        }
        r.count(s.restart.len())?;
        for b in &mut s.restart {
            *b = r.flag()?;
        }
        let n = s.restart.len() * 3;
        s.acceleration = if r.flag()? { Some(r.vector(n)?) } else { None };
        s.earlier = if r.flag()? {
            Some((r.vector(n)?, r.vector(n)?))
        } else {
            None
        };
        r.count(s.affine.len())?;
        for a in &mut s.affine {
            *a = if r.flag()? {
                Some((r.quat()?, Mat3::from_cols(r.vec()?, r.vec()?, r.vec()?)))
            } else {
                None
            };
        }
        s.gravity_seen = None;
        s.previous_h = None;
        if version >= 2 {
            s.gravity_seen = if r.flag()? { Some(r.vec()?) } else { None };
            s.previous_h = if r.flag()? {
                let h = r.f64()?;
                if h <= 0.0 {
                    return Err("invalid previous step length");
                }
                Some(h)
            } else {
                None
            };
        }
        s.unloading.iter_mut().for_each(|e| *e = 0.0);
        if version >= 3 {
            r.count(s.unloading.len())?;
            for e in &mut s.unloading {
                *e = r.f64()?;
                if *e < 0.0 {
                    return Err("invalid contact unloading budget");
                }
            }
        }
        for p in &mut s.plastic {
            p.fp_inv.fill(Mat3::IDENTITY);
            p.eps_p.fill(0.0);
            p.damage.fill(0.0);
        }
        if version >= 4 {
            r.count(s.plastic.len())?;
            for p in &mut s.plastic {
                let active = r.flag()?;
                if active != !p.fp_inv.is_empty() {
                    return Err("core state plasticity mismatch");
                }
                if active {
                    r.count(p.fp_inv.len())?;
                    for (m, ep) in p.fp_inv.iter_mut().zip(&mut p.eps_p) {
                        let mat = Mat3::from_cols(r.vec()?, r.vec()?, r.vec()?);
                        if mat.determinant() <= 0.0 {
                            return Err("invalid plastic deformation inverse");
                        }
                        *m = mat;
                        *ep = r.f64()?;
                        if *ep < 0.0 {
                            return Err("invalid equivalent plastic strain");
                        }
                    }
                }
            }
        }
        if version >= 5 {
            r.count(s.plastic.len())?;
            for p in &mut s.plastic {
                r.count(p.damage.len())?;
                for d in &mut p.damage {
                    let val = r.f64()?;
                    if !(0.0..=1.0).contains(&val) {
                        return Err("invalid cohesive damage variable");
                    }
                    *d = val;
                }
            }
        }
        if !r.0.is_empty() {
            return Err("trailing core state bytes");
        }
        Ok(s)
    }
}
struct Writer(Vec<u8>);
impl Writer {
    fn usize(&mut self, n: usize) {
        self.0.extend_from_slice(&(n as u64).to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn flag(&mut self, b: bool) {
        self.0.push(u8::from(b));
    }
    fn vec(&mut self, v: Vec3) {
        self.f64(v.x);
        self.f64(v.y);
        self.f64(v.z);
    }
    fn quat(&mut self, q: Quat) {
        self.f64(q.x);
        self.f64(q.y);
        self.f64(q.z);
        self.f64(q.w);
    }
}
struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], &'static str> {
        if self.0.len() < N {
            return Err("truncated core state");
        }
        let (a, b) = self.0.split_at(N);
        self.0 = b;
        Ok(a.try_into().unwrap())
    }
    fn count(&mut self, n: usize) -> Result<(), &'static str> {
        if u64::from_le_bytes(self.take()?) != n as u64 {
            return Err("core state layout mismatch");
        }
        Ok(())
    }
    fn f64(&mut self) -> Result<f64, &'static str> {
        let v = f64::from_le_bytes(self.take()?);
        if !v.is_finite() {
            return Err("non-finite core state");
        }
        Ok(v)
    }
    fn flag(&mut self) -> Result<bool, &'static str> {
        match self.take::<1>()?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err("invalid core state flag"),
        }
    }
    fn vec(&mut self) -> Result<Vec3, &'static str> {
        Ok(Vec3::new(self.f64()?, self.f64()?, self.f64()?))
    }
    fn quat(&mut self) -> Result<Quat, &'static str> {
        Ok(Quat::from_xyzw(
            self.f64()?,
            self.f64()?,
            self.f64()?,
            self.f64()?,
        ))
    }
    fn vector(&mut self, n: usize) -> Result<Vec<f64>, &'static str> {
        (0..n).map(|_| self.f64()).collect()
    }
}
