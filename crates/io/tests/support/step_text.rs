//! Test-only builder for STEP files written the way other systems write
//! them, record by record.
#![allow(dead_code)]

use std::fmt::Write as _;

/// The records of one STEP file's DATA section, numbered as they are added.
pub struct StepText {
    records: Vec<String>,
}

/// A length unit for [`StepText::with_unit`].
#[derive(Clone, Copy)]
pub enum Unit {
    Millimetre,
    Metre,
    Inch,
}

impl StepText {
    /// A file whose geometry is in `unit`; `#1` is its length unit and `#4`
    /// its geometric context.
    pub fn with_unit(unit: Unit) -> Self {
        let mut s = Self {
            records: Vec::new(),
        };
        match unit {
            Unit::Millimetre => {
                s.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )");
            }
            Unit::Metre => {
                s.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) )");
            }
            Unit::Inch => {
                s.add("( CONVERSION_BASED_UNIT('INCH',#6) LENGTH_UNIT() NAMED_UNIT(#5) )");
            }
        }
        s.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )");
        s.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )");
        s.add(
            "( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#1,#2,#3)) \
             REPRESENTATION_CONTEXT('','3D') )",
        );
        if matches!(unit, Unit::Inch) {
            s.add("DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)");
            s.add("LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#7)");
            s.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )");
        }
        s
    }

    pub fn new() -> Self {
        Self::with_unit(Unit::Millimetre)
    }

    pub const CONTEXT: u64 = 4;

    /// Add `record` and return its id.
    pub fn add(&mut self, record: impl Into<String>) -> u64 {
        self.records.push(record.into());
        self.records.len() as u64
    }

    pub fn point(&mut self, p: [f64; 3]) -> u64 {
        self.add(format!(
            "CARTESIAN_POINT('',({:?},{:?},{:?}))",
            p[0], p[1], p[2]
        ))
    }

    pub fn direction(&mut self, d: [f64; 3]) -> u64 {
        self.add(format!("DIRECTION('',({:?},{:?},{:?}))", d[0], d[1], d[2]))
    }

    pub fn placement(&mut self, origin: [f64; 3], z: [f64; 3], x: [f64; 3]) -> u64 {
        let (o, z, x) = (self.point(origin), self.direction(z), self.direction(x));
        self.add(format!("AXIS2_PLACEMENT_3D('',#{o},#{z},#{x})"))
    }

    pub fn vertex(&mut self, p: [f64; 3]) -> u64 {
        let p = self.point(p);
        self.add(format!("VERTEX_POINT('',#{p})"))
    }

    pub fn line(&mut self, p: [f64; 3], d: [f64; 3]) -> u64 {
        let (p, d) = (self.point(p), self.direction(d));
        let v = self.add(format!("VECTOR('',#{d},1.)"));
        self.add(format!("LINE('',#{p},#{v})"))
    }

    /// A circle about `normal` through `center`, starting along `x`.
    pub fn circle(&mut self, center: [f64; 3], normal: [f64; 3], x: [f64; 3], r: f64) -> u64 {
        let a = self.placement(center, normal, x);
        self.add(format!("CIRCLE('',#{a},{r:?})"))
    }

    pub fn edge(&mut self, from: u64, to: u64, curve: u64) -> u64 {
        self.add(format!("EDGE_CURVE('',#{from},#{to},#{curve},.T.)"))
    }

    /// An edge loop of `(edge, forward)` uses.
    pub fn edge_loop(&mut self, edges: &[(u64, bool)]) -> u64 {
        let uses: Vec<String> = edges
            .iter()
            .map(|&(e, forward)| {
                let flag = if forward { ".T." } else { ".F." };
                format!(
                    "#{}",
                    self.add(format!("ORIENTED_EDGE('',*,*,#{e},{flag})"))
                )
            })
            .collect();
        self.add(format!("EDGE_LOOP('',({}))", uses.join(",")))
    }

    pub fn vertex_loop(&mut self, vertex: u64) -> u64 {
        self.add(format!("VERTEX_LOOP('',#{vertex})"))
    }

    /// A face on `surface` bounded by `loops`, the first its outer bound.
    pub fn face(&mut self, loops: &[u64], surface: u64, same_sense: bool) -> u64 {
        let bounds: Vec<String> = loops
            .iter()
            .enumerate()
            .map(|(k, &l)| {
                let kind = if k == 0 {
                    "FACE_OUTER_BOUND"
                } else {
                    "FACE_BOUND"
                };
                format!("#{}", self.add(format!("{kind}('',#{l},.T.)")))
            })
            .collect();
        let flag = if same_sense { ".T." } else { ".F." };
        self.add(format!(
            "ADVANCED_FACE('',({}),#{surface},{flag})",
            bounds.join(",")
        ))
    }

    pub fn plane(&mut self, origin: [f64; 3], normal: [f64; 3], x: [f64; 3]) -> u64 {
        let a = self.placement(origin, normal, x);
        self.add(format!("PLANE('',#{a})"))
    }

    pub fn solid(&mut self, faces: &[u64]) -> u64 {
        let list: Vec<String> = faces.iter().map(|f| format!("#{f}")).collect();
        let shell = self.add(format!("CLOSED_SHELL('',({}))", list.join(",")));
        self.add(format!("MANIFOLD_SOLID_BREP('',#{shell})"))
    }

    /// A shape representation of `items` in the file's context.
    pub fn representation(&mut self, kind: &str, items: &[u64]) -> u64 {
        let list: Vec<String> = items.iter().map(|i| format!("#{i}")).collect();
        self.add(format!(
            "{kind}('',({}),#{})",
            list.join(","),
            Self::CONTEXT
        ))
    }

    pub fn finish(&self) -> String {
        let mut out = String::from(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('test'),'2;1');\n\
             FILE_NAME('test.step','',(''),(''),'','','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n",
        );
        for (k, r) in self.records.iter().enumerate() {
            let _ = writeln!(out, "#{}={r};", k + 1);
        }
        out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
        out
    }
}
