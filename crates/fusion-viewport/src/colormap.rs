//! Colour mapping for the visualization modes.
//!
//! PLAN.md §19 makes the architectural point first: every mode is the same
//! geometry with a different shading function, so this is one renderer plus a
//! colour-mapping stage rather than N renderers. It then lays down three rules
//! that separate a good implementation from a misleading one, and all three
//! are enforced here rather than left to the caller's discretion.
//!
//! **Perceptually uniform colormaps only. No rainbow, ever.** This is not
//! aesthetics. A rainbow ramp has uneven perceptual spacing: it changes hue
//! fast in some places and slowly in others, so it invents sharp boundaries
//! where the data is smooth and hides real structure where the hue happens to
//! stall. A rainbow stress plot shows a feature that is not there. There is no
//! `Colormap::Jet`, and adding one would defeat the purpose of the mode whose
//! entire job is being informative.
//!
//! **Never auto-range per frame.** If the scale rescales every frame the same
//! colour means different things at different times, and an animation of it is
//! a lie — the fluid does not get faster, the mapping gets tighter. A
//! [`ScalarField`] is constructed with a range and keeps it. [`lock_from`]
//! exists for the case where a sensible range is not known in advance, and it
//! samples *once*.
//!
//! **A legend with units, always visible.** A colour with no scale beside it
//! is decoration. [`ScalarField::draw_legend`] is not optional garnish; a mode
//! without it is not a measurement view.
//!
//! ## The polynomial fits
//!
//! Viridis, inferno and magma are defined by 256-entry tables. The fits below
//! are sixth-order polynomials in the normalized value, accurate to about one
//! part in 70 at the worst point, and they cost six multiply-adds instead of a
//! table lookup and an interpolation. Fits due to Matt Zucker.
//!
//! Both are checked against their published endpoints by a test, which is not
//! ceremony: the first version of this file also carried a magma fit, and its
//! blue channel at the top of the ramp was wrong by 0.75. A colormap module
//! whose whole argument is "the wrong ramp makes you see things that are not
//! there" does not get to ship an unverified ramp.

use crate::mesh::Geometry;

/// A perceptually uniform colour ramp.
///
/// Deliberately a short list. Every entry is perceptually uniform and
/// colour-blind-safe; there is no rainbow option and there will not be one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Colormap {
    /// Blue to green to yellow. The default: the most legible of the three in
    /// print and on a projector, and the one to reach for unless a dark
    /// background argues otherwise.
    #[default]
    Viridis,
    /// Black through red to pale yellow. Reads as heat, which makes it the
    /// natural choice for temperature — see also [`blackbody`], which is the
    /// physical version of the same idea.
    Inferno,
}

// Magma was here and is not, which is worth a note rather than a silent
// absence. Its polynomial fit was written from memory and got the blue channel
// at the top of the ramp wrong by 0.75 — pale yellow came out as saturated
// yellow — and its lightness went *down* over the last two steps, which is
// precisely the non-monotonicity this module exists to avoid. A colormap
// nobody has checked against its reference is exactly the failure §19 warns
// about: it invents a boundary the data does not have. It can come back when
// someone has the table to check it against.

impl Colormap {
    /// Colour for a value already normalized to `0..1`.
    pub fn sample(self, t: f32) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Viridis => polynomial(
                t,
                [0.277_727_3, 0.005_407_344_5, 0.334_099_8],
                [0.105_093_04, 1.404_613_5, 1.384_590_2],
                [-0.330_861_83, 0.214_847_56, 0.095_095_16],
                [-4.634_230_5, -5.799_101, -19.332_441],
                [6.228_27, 14.179_933, 56.690_55],
                [4.776_385, -13.745_145, -65.353_035],
                [-5.435_456, 4.645_852_6, 26.312_435],
            ),
            Colormap::Inferno => polynomial(
                t,
                [0.000_218_940_37, 0.001_651_004_6, -0.019_480_899],
                [0.106_513_42, 0.563_956_44, 3.932_712_4],
                [11.602_493, -3.972_854, -15.942_394],
                [-41.703_995, 17.436_398, 44.354_145],
                [77.162_94, -33.402_36, -81.807_31],
                [-71.319_43, 32.626_064, 73.209_52],
                [25.131_126, -12.242_669, -23.070_324],
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Colormap::Viridis => "viridis",
            Colormap::Inferno => "inferno",
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn polynomial(
    t: f32,
    c0: [f32; 3],
    c1: [f32; 3],
    c2: [f32; 3],
    c3: [f32; 3],
    c4: [f32; 3],
    c5: [f32; 3],
    c6: [f32; 3],
) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        let v =
            c0[k] + t * (c1[k] + t * (c2[k] + t * (c3[k] + t * (c4[k] + t * (c5[k] + t * c6[k])))));
        out[k] = v.clamp(0.0, 1.0);
    }
    out
}

/// Blackbody emission at a temperature, in Kelvin.
///
/// PLAN.md §19 singles this out: "temperature mode is where diagnostic and
/// beautiful meet." Mapping temperature to the spectrum a body actually
/// radiates makes hot metal glow *correctly* — the diagnostic view and the
/// pretty view become the same view, which is the best argument the whole
/// section makes.
///
/// A fit to the Planck locus, normalized so the brightest channel is one.
/// Below about 800 K a real body emits almost nothing visible, and this
/// returns near-black there rather than pretending otherwise.
pub fn blackbody(kelvin: f32) -> [f32; 3] {
    let t = (kelvin / 100.0).clamp(1.0, 400.0);
    let red = if t <= 66.0 {
        1.0
    } else {
        (1.292936 * (t - 60.0).powf(-0.1332047)).clamp(0.0, 1.0)
    };
    let green = if t <= 66.0 {
        (0.3900816 * t.ln() - 0.6318414).clamp(0.0, 1.0)
    } else {
        (1.129891 * (t - 60.0).powf(-0.0755148)).clamp(0.0, 1.0)
    };
    let blue = if t >= 66.0 {
        1.0
    } else if t <= 19.0 {
        0.0
    } else {
        (0.5432068 * (t - 10.0).ln() - 1.19625).clamp(0.0, 1.0)
    };
    // Fade to black at the bottom: an object at 400 K is not dark red, it is
    // dark. Without this every cold thing in a temperature view glows.
    let visible = ((kelvin - 500.0) / 400.0).clamp(0.0, 1.0);
    [red * visible, green * visible, blue * visible]
}

/// A scalar quantity, its range, and how to colour it.
///
/// The range is part of the object because §19's second rule is that it must
/// not move. Constructing one of these is the moment you decide what the
/// colours mean, and they mean that for the rest of the run.
#[derive(Clone, Copy, Debug)]
pub struct ScalarField {
    pub map: Colormap,
    pub low: f32,
    pub high: f32,
    /// What the number is, for the legend: "speed", "density", "temperature".
    pub label: &'static str,
    /// SI units, for the legend: "m/s", "kg/m^3", "K".
    pub units: &'static str,
}

impl ScalarField {
    pub fn new(label: &'static str, units: &'static str, low: f32, high: f32) -> Self {
        Self {
            map: Colormap::default(),
            low,
            high,
            label,
            units,
        }
    }

    pub fn with_map(mut self, map: Colormap) -> Self {
        self.map = map;
        self
    }

    /// Choose a range from data, **once**.
    ///
    /// The escape hatch for "I do not know the range yet", and it is a
    /// constructor rather than a per-frame call on purpose: lock it on the
    /// first frame and live with it. Calling this every frame is precisely the
    /// auto-ranging §19 forbids, and no API here will make that convenient.
    pub fn lock_from(
        label: &'static str,
        units: &'static str,
        values: impl IntoIterator<Item = f32>,
    ) -> Self {
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        for v in values {
            if v.is_finite() {
                low = low.min(v);
                high = high.max(v);
            }
        }
        if !low.is_finite() {
            low = 0.0;
            high = 1.0;
        }
        // A zero-width range would divide by zero and colour everything the
        // bottom of the ramp; widen it so a constant field reads as constant
        // rather than as an artefact.
        if (high - low).abs() < 1e-9 {
            high = low + 1.0;
        }
        Self::new(label, units, low, high)
    }

    /// Where a value sits in the range, clamped.
    pub fn normalize(&self, value: f32) -> f32 {
        if (self.high - self.low).abs() < 1e-12 {
            return 0.0;
        }
        ((value - self.low) / (self.high - self.low)).clamp(0.0, 1.0)
    }

    pub fn color(&self, value: f32) -> [f32; 3] {
        self.map.sample(self.normalize(value))
    }

    /// Draw the legend: a colour bar, its end values, and the units.
    ///
    /// Bottom-left, in the overlay layer, so it survives any camera move. §19
    /// wants this "always visible" and means it — a mode whose scale is off
    /// screen is a mode that cannot be read.
    pub fn draw_legend(&self, g: &mut Geometry) {
        self.draw_legend_at(g, [-0.94, -0.90]);
    }

    /// The legend, somewhere other than the bottom-left corner — for a view
    /// showing two quantities at once.
    pub fn draw_legend_at(&self, g: &mut Geometry, at: [f32; 2]) {
        let (x, y) = (at[0], at[1]);
        let (w, h) = (0.36, 0.035);
        // The bar, as a strip of quads so the gradient is smooth.
        let steps = 48;
        for i in 0..steps {
            let t0 = i as f32 / steps as f32;
            let t1 = (i + 1) as f32 / steps as f32;
            g.overlay_quad(
                [x + w * t0, y],
                [x + w * t1, y + h],
                self.map.sample((t0 + t1) * 0.5),
            );
        }
        let tick = [0.85, 0.86, 0.9];
        g.overlay_line([x, y + h], [x, y + h + 0.02], tick);
        g.overlay_line([x + w, y + h], [x + w, y + h + 0.02], tick);

        let text = 0.026;
        g.overlay_text(&format_value(self.low), [x, y - 0.045], text, tick);
        g.overlay_text(
            &format_value(self.high),
            [x + w - 0.06, y - 0.045],
            text,
            tick,
        );
        let caption = if self.units.is_empty() {
            self.label.to_string()
        } else {
            format!("{} {}", self.label, self.units)
        };
        g.overlay_text(&caption, [x, y + h + 0.035], text, tick);
    }
}

/// Short, readable numbers for a legend: no more precision than a 40-pixel
/// label can carry.
fn format_value(v: f32) -> String {
    let a = v.abs();
    if a >= 1000.0 || (a < 0.01 && a > 0.0) {
        format!("{v:.0e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}
