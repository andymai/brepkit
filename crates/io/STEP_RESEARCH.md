# STEP File Format Research for brepkit-io

Gathered 2026-03-02. Focused on ISO 10303 Part 21 / AP203 B-Rep solid export.

---

## 1. File Format Structure (ISO 10303-21)

### 1.1 Overall File Layout

A STEP Part 21 file is a plain ASCII text file with this top-level structure:

```
ISO-10303-21;
HEADER;
  FILE_DESCRIPTION(...);
  FILE_NAME(...);
  FILE_SCHEMA(...);
ENDSEC;
DATA;
  #1 = ENTITY_NAME(attr1, attr2, ...);
  #2 = ENTITY_NAME(...);
  ...
ENDSEC;
END-ISO-10303-21;
```

Key lexical rules:
- File delimiters: `ISO-10303-21;` and `END-ISO-10303-21;`
- Comments: `/* ... */` (non-nestable)
- Strings: single-quoted `'text'`
- Enumerations: period-delimited capitals `.ENUM_VALUE.`
- Instance references: `#N` (positive integer, local to file)
- Omitted (unset) attributes: `$`
- Derived (re-declared) attributes in supertypes: `*`
- Lists/sets: comma-separated in parentheses `(a, b, c)`
- Entity instances: `#N = ENTITY_NAME(attr1, attr2, ...);`
- Numbers can use scientific notation: `1.E-07`

### 1.2 HEADER Section

Three mandatory entities in fixed order:

```
FILE_DESCRIPTION(
  ('description string'),   -- LIST OF STRING: content description
  '2;1'                     -- STRING: implementation level (2;1 = Part 21 Ed.2 / Conf.1)
);

FILE_NAME(
  'filename.step',          -- STRING: name
  '2026-03-02T00:00:00',    -- STRING: ISO 8601 timestamp
  ('Author Name'),          -- LIST OF STRING: authors
  ('Organization'),         -- LIST OF STRING: organizations
  'brepkit 0.1',            -- STRING: preprocessor version
  'brepkit',                -- STRING: originating system
  ''                        -- STRING: authorization
);

FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));   -- AP203
-- OR:
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));       -- AP214 (used by OpenCASCADE)
```

**Schema names by AP:**
- AP203 (preferred for pure geometry): `CONFIG_CONTROL_DESIGN`
- AP214 (automotive, richer metadata): `AUTOMOTIVE_DESIGN`
- AP242 (newest, recommended for new implementations): `AP242_MANAGED_MODEL_BASED_3D_ENGINEERING`

### 1.3 DATA Section

Each entity line:
```
#<id> = <ENTITY_NAME>(<attr1>, <attr2>, ...);
```

IDs do not need to be sequential but must be positive integers and unique within the file.
IDs may appear in any order (forward references are legal).

Complex entity instances (multiple supertypes merged into one instance) use alphabetical grouping:
```
#345 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3)
         GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#349))
         GLOBAL_UNIT_ASSIGNED_CONTEXT((#346,#347,#348))
         REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') );
```
This compound form groups all supertype data into a single `#N` node.

---

## 2. Entity Reference Hierarchy

### 2.1 Top-Down Containment

```
ADVANCED_BREP_SHAPE_REPRESENTATION
  └── MANIFOLD_SOLID_BREP
        └── CLOSED_SHELL
              └── ADVANCED_FACE (×N faces)
                    ├── face_geometry: PLANE
                    │     └── AXIS2_PLACEMENT_3D
                    │           ├── CARTESIAN_POINT (origin)
                    │           ├── DIRECTION (z-axis / normal)
                    │           └── DIRECTION (x-axis / ref_direction)
                    └── bounds: FACE_OUTER_BOUND / FACE_BOUND
                          └── EDGE_LOOP
                                └── ORIENTED_EDGE (×N per loop)
                                      └── EDGE_CURVE
                                            ├── VERTEX_POINT (start)
                                            │     └── CARTESIAN_POINT
                                            ├── VERTEX_POINT (end)
                                            │     └── CARTESIAN_POINT
                                            └── SURFACE_CURVE (edge geometry)
                                                  ├── LINE (3D curve)
                                                  │     ├── CARTESIAN_POINT (point on line)
                                                  │     └── VECTOR
                                                  │           └── DIRECTION
                                                  └── PCURVE (×2, one per adj. face)
                                                        ├── PLANE (the face's surface)
                                                        └── DEFINITIONAL_REPRESENTATION
                                                              └── LINE (2D in param space)
```

### 2.2 Product Structure Chain

The geometry (`ADVANCED_BREP_SHAPE_REPRESENTATION`) must be connected to a product
definition so that CAD importers can identify what the shape represents:

```
PRODUCT
  └── PRODUCT_DEFINITION_FORMATION
        └── PRODUCT_DEFINITION
              └── PRODUCT_DEFINITION_SHAPE
                    └── SHAPE_DEFINITION_REPRESENTATION
                          └── ADVANCED_BREP_SHAPE_REPRESENTATION
```

Supporting entities required to fill out this chain:
- `APPLICATION_CONTEXT` — the engineering context string
- `PRODUCT_DEFINITION_CONTEXT` — 'part definition', 'design'
- `MECHANICAL_CONTEXT` — subtype of PRODUCT_CONTEXT, 'mechanical'
- `APPLICATION_PROTOCOL_DEFINITION` — links protocol name to application context
- (AP214 also uses) `PRODUCT_TYPE` — optional category node

---

## 3. Entity Definitions

### 3.1 Geometric Primitives

#### CARTESIAN_POINT
```
CARTESIAN_POINT('label', (x, y, z));
-- or 2D in parametric space:
CARTESIAN_POINT('label', (u, v));
```
- name: label string (often `''`)
- coordinates: LIST OF REAL, length 2 or 3

#### DIRECTION
```
DIRECTION('label', (dx, dy, dz));
```
- coordinates: unit vector components (must be unit length for most uses)

#### VECTOR
```
VECTOR('label', #direction_ref, magnitude);
```
- orientation: DIRECTION reference
- magnitude: REAL (length scale — typically `1.` for unit vectors on lines)

#### AXIS2_PLACEMENT_3D
```
AXIS2_PLACEMENT_3D('label', #origin_pt, #axis_dir, #ref_dir);
```
- location: CARTESIAN_POINT (origin)
- axis: DIRECTION (z-axis / normal of the placed coordinate system)
- ref_direction: DIRECTION (x-axis direction)
- y-axis is implied as cross(axis, ref_direction), normalized

For a PLANE: `axis` = plane normal, `ref_direction` = any vector in the plane (x-axis of local frame).

### 3.2 Topology Primitives

#### VERTEX_POINT
```
VERTEX_POINT('label', #cartesian_point);
```
Topological vertex linked to a geometric point.

#### LINE
```
LINE('label', #point_on_line, #vector);
```
- pnt: CARTESIAN_POINT (any point on the line)
- dir: VECTOR (direction + magnitude; magnitude usually 1.)
Defines an infinite line. Bounded by the edge's start/end vertices.

#### PLANE
```
PLANE('label', #axis2_placement_3d);
```
Defines an infinite plane via an axis placement.
The plane's normal = the placement's `axis` DIRECTION.
The plane's origin = the placement's `location` CARTESIAN_POINT.

#### SURFACE_CURVE
```
SURFACE_CURVE('label', #line_3d, (#pcurve1, #pcurve2), .PCURVE_S1.);
```
- curve_3d: the 3D LINE for this edge
- associated_geometry: LIST [1:2] OF PCURVE — one per adjacent face
- master_representation: `.PCURVE_S1.` — the first PCURVE is authoritative

SURFACE_CURVE replaces a bare LINE as the edge curve geometry for ADVANCED_FACE.
It bundles together the 3D curve with its parametric (UV-space) representation
on each adjacent surface.

#### PCURVE
```
PCURVE('label', #surface_ref, #definitional_representation);
```
- basis_surface: PLANE (the adjacent face's surface)
- reference_to_curve: DEFINITIONAL_REPRESENTATION containing a 2D LINE

The 2D LINE inside a PCURVE lives in the UV parameter space of the surface.
For a plane with AXIS2_PLACEMENT_3D having origin O, x-axis X, y-axis Y:
  - u = dot(3D_point - O, X)
  - v = dot(3D_point - O, Y)

#### DEFINITIONAL_REPRESENTATION
```
DEFINITIONAL_REPRESENTATION('label', (#line_2d), #param_context);
```
- items: the 2D curve (a LINE with 2D CARTESIAN_POINT and 2D VECTOR)
- context_of_items: compound GEOMETRIC_REPRESENTATION_CONTEXT(2) entity

The compound 2D context is typically written as:
```
#N = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
       PARAMETRIC_REPRESENTATION_CONTEXT()
       REPRESENTATION_CONTEXT('2D SPACE', '') );
```

#### EDGE_CURVE
```
EDGE_CURVE('label', #vertex_start, #vertex_end, #surface_curve, .T.);
```
- edge_start: VERTEX_POINT
- edge_end: VERTEX_POINT
- edge_geometry: SURFACE_CURVE (or LINE for simpler files)
- same_sense: BOOLEAN — `.T.` means curve direction agrees with start→end direction

#### ORIENTED_EDGE
```
ORIENTED_EDGE('', *, *, #edge_curve, .T.);
```
- name: `''`
- edge_element: `*` (derived, always `*` in supertype slots)
- edge_element: `*`
- edge_element: reference to EDGE_CURVE
- orientation: BOOLEAN — `.T.` if used in the same sense as the underlying EDGE_CURVE;
  `.F.` if used in reverse (traversed from end to start)

**Important:** Multiple ORIENTED_EDGEs can reference the same EDGE_CURVE.
The two faces sharing an edge each wrap it in their own ORIENTED_EDGE, with
opposite orientations relative to each other (CCW around each face normal).

#### EDGE_LOOP
```
EDGE_LOOP('label', (#oriented_edge1, #oriented_edge2, #oriented_edge3, ...));
```
- The loop must be closed: start vertex of edge[0] = end vertex of edge[N-1] (after orientation).
- Order is CCW when viewed from outside (above the face normal).

#### FACE_OUTER_BOUND
```
FACE_OUTER_BOUND('', #edge_loop, .T.);
```
Used for the single outer boundary of a simple face.
Only one FACE_OUTER_BOUND is allowed per face.

#### FACE_BOUND
```
FACE_BOUND('', #edge_loop, .T.);
```
Used for inner holes (additional loops). The orientation flag determines whether
the loop winding is taken as-defined (`.T.`) or reversed (`.F.`).
For holes: orientation is typically `.F.` to indicate the inner loop runs CW
when viewed from outside.

#### ADVANCED_FACE
```
ADVANCED_FACE('label', (#face_bound1, ...), #surface, .T.);
```
- bounds: SET OF FACE_BOUND — first is typically FACE_OUTER_BOUND, rest are holes
- face_geometry: PLANE (or other surface type)
- same_sense: BOOLEAN
  - `.T.` — surface normal (from axis placement) points outward from the solid
  - `.F.` — surface normal points inward; flip it to get the outward-facing normal

The `same_sense` parameter tells the importer: does the surface's natural normal
agree with the topological outward direction? For a closed shell, all face normals
should point outward from the solid interior.

#### CLOSED_SHELL
```
CLOSED_SHELL('label', (#face1, #face2, ..., #face6));
```
A connected, orientable, closed set of faces forming the boundary of a solid.
No gaps, no dangling edges.

#### MANIFOLD_SOLID_BREP
```
MANIFOLD_SOLID_BREP('label', #closed_shell);
```
The top-level solid geometry entity. References exactly one CLOSED_SHELL.

### 3.3 Shape Representation

#### ADVANCED_BREP_SHAPE_REPRESENTATION
```
ADVANCED_BREP_SHAPE_REPRESENTATION('', (#axis_placement, #manifold_solid_brep), #geom_context);
```
- name: `''`
- items: SET containing:
  - an AXIS2_PLACEMENT_3D (world coordinate frame, usually identity/origin)
  - one or more MANIFOLD_SOLID_BREP instances
- context_of_items: compound GEOMETRIC_REPRESENTATION_CONTEXT(3) with units and tolerance

**Constraints (WHERE rules):**
- WR1: items must be manifold_solid_brep, faceted_brep, mapped_item, or axis2_placement_3d
- WR2: at least one item must be manifold_solid_brep
- WR3: all manifold_solid_breps must contain only advanced_faces in their shells

### 3.4 Geometric Context and Units

The representation context ties the geometry to physical units and numerical tolerance.
This is a compound entity (multiple supertypes merged):

```
#N = ( GEOMETRIC_REPRESENTATION_CONTEXT(3)
       GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#uncertainty_measure))
       GLOBAL_UNIT_ASSIGNED_CONTEXT((#length_unit, #angle_unit, #solid_angle_unit))
       REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') );

#length_unit      = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
#angle_unit       = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );
#solid_angle_unit = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() );

#uncertainty = UNCERTAINTY_MEASURE_WITH_UNIT(
  LENGTH_MEASURE(1.E-07),  -- tolerance value
  #length_unit,
  'distance_accuracy_value',
  'confusion accuracy'
);
```

Common unit options for SI_UNIT:
- Length: `(.MILLI.,.METRE.)` = millimetres, `($,.METRE.)` = metres, `(.CENTI.,.METRE.)` = cm
- Angle: `($,.RADIAN.)` = radians, `($,.DEGREE.)` = degrees (but radians is strongly preferred)

---

## 4. Product Structure Entities

For a minimal single-part STEP file (AP203 / AP214):

```
#1 = APPLICATION_PROTOCOL_DEFINITION(
  'committee draft',    -- status
  'automotive_design',  -- protocol (or 'config_controlled_design' for AP203)
  1997,                 -- year
  #2                    -- application context
);
#2 = APPLICATION_CONTEXT(
  'core data for automotive mechanical design processes'
);

#3 = SHAPE_DEFINITION_REPRESENTATION(#4, #10);  -- links shape to product
#4 = PRODUCT_DEFINITION_SHAPE('', '', #5);
#5 = PRODUCT_DEFINITION('design', '', #6, #9);
#6 = PRODUCT_DEFINITION_FORMATION('', '', #7);
#7 = PRODUCT('Part Name', 'Part Name', '', (#8));
#8 = MECHANICAL_CONTEXT('', #2, 'mechanical');
#9 = PRODUCT_DEFINITION_CONTEXT('part definition', #2, 'design');
#10 = ADVANCED_BREP_SHAPE_REPRESENTATION('', (#11, #15), #geom_context);
-- #11 = AXIS2_PLACEMENT_3D (world origin)
-- #15 = MANIFOLD_SOLID_BREP
```

The linkage chain:
```
SHAPE_DEFINITION_REPRESENTATION(#4, #10)
  #4 = PRODUCT_DEFINITION_SHAPE → #5 = PRODUCT_DEFINITION → #6 = PRODUCT_DEFINITION_FORMATION → #7 = PRODUCT
  #10 = ADVANCED_BREP_SHAPE_REPRESENTATION
```

For AP203 (CONFIG_CONTROL_DESIGN schema), the `APPLICATION_PROTOCOL_DEFINITION`
`protocol` field should be `'config_controlled_design'`.
For AP214 (AUTOMOTIVE_DESIGN schema, used by OpenCASCADE), it is `'automotive_design'`.

---

## 5. Orientation Conventions

### 5.1 Edge orientation

EDGE_CURVE defines an edge from `edge_start` to `edge_end`.

ORIENTED_EDGE wraps EDGE_CURVE with an `orientation` flag:
- `.T.` — use the edge in the start→end direction as defined
- `.F.` — use it in reverse (end→start)

Each EDGE_CURVE is shared between the two faces that meet at it.
One face uses it with `.T.`, the other with `.F.`.

### 5.2 Loop winding (EDGE_LOOP)

Edges in an EDGE_LOOP must be traversed CCW when viewed from the **outside** of the face
(i.e., looking down the face normal toward the surface).

The right-hand rule: curl the fingers of the right hand in the edge traversal direction;
the thumb points toward the outward normal.

FACE_BOUND orientation flag:
- `.T.` — use loop as defined (edges run CCW when viewed from outside)
- `.F.` — reverse the loop sense

### 5.3 Face orientation (ADVANCED_FACE same_sense)

- `.T.` — the surface's natural normal (from AXIS2_PLACEMENT_3D `axis` direction) is the outward normal
- `.F.` — the surface's natural normal points inward; negate it to get outward

For a box, if you define each PLANE with its normal pointing outward, `same_sense = .T.`.
If you reuse a single downward-normal plane for both top and bottom faces, one will need `.F.`.

Practical approach: define each face's PLANE independently with the axis direction
pointing outward; set `same_sense = .T.` throughout.

---

## 6. Simplification Options

### 6.1 Faceted BREP (no SURFACE_CURVE / PCURVE)

For tessellated / polyhedral geometry, use `FACETED_BREP` instead of `MANIFOLD_SOLID_BREP`.
Edges use bare `LINE` references instead of `SURFACE_CURVE`.
FACE_BOUND uses `POLY_LOOP` instead of EDGE_LOOP.
This is simpler but less interoperable for downstream CAD.

### 6.2 EDGE_CURVE without SURFACE_CURVE

Some STEP writers omit `SURFACE_CURVE` and `PCURVE`, using a bare `LINE` as the
edge curve geometry:

```
#N = EDGE_CURVE('label', #v_start, #v_end, #line, .T.);
```

This is technically valid for `MANIFOLD_SOLID_BREP` (not for `ADVANCED_FACE` which
requires surface curves per WR9). Some importers are lenient, but for maximum
interoperability with strict importers (FreeCAD, CATIA, SolidWorks), include
SURFACE_CURVE + PCURVE.

### 6.3 Using FACE_BOUND vs FACE_OUTER_BOUND

Both are topologically valid. `FACE_OUTER_BOUND` is a subtype of `FACE_BOUND` that
semantically labels the outer boundary. For faces without holes, either works.
Using `FACE_OUTER_BOUND` is the stricter / more correct choice.

The cube example from OpenCASCADE uses plain `FACE_BOUND` throughout — this is accepted
by all major importers.

---

## 7. Complete Example: Unit Box [-1,1]^3

This is the complete, real STEP file from the OpenCASCADE jCAE test suite
(AP214 schema, AUTOMOTIVE_DESIGN):

```step
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Open CASCADE Model'),'2;1');
FILE_NAME('Open CASCADE Shape Model','2009-05-01T23:59:58',('Author'),(
    'Open CASCADE'),'Open CASCADE STEP processor 6.3','Open CASCADE 6.3'
  ,'Unknown');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2 { 1 2 10303 214 -1 1 5 4 }'));
ENDSEC;
DATA;
/* ---- Product structure ---- */
#1 = APPLICATION_PROTOCOL_DEFINITION('committee draft',
  'automotive_design',1997,#2);
#2 = APPLICATION_CONTEXT(
  'core data for automotive mechanical design processes');
#3 = SHAPE_DEFINITION_REPRESENTATION(#4,#10);
#4 = PRODUCT_DEFINITION_SHAPE('','',#5);
#5 = PRODUCT_DEFINITION('design','',#6,#9);
#6 = PRODUCT_DEFINITION_FORMATION('','',#7);
#7 = PRODUCT('Open CASCADE STEP translator 6.3 1',
  'Open CASCADE STEP translator 6.3 1','',(#8));
#8 = MECHANICAL_CONTEXT('',#2,'mechanical');
#9 = PRODUCT_DEFINITION_CONTEXT('part definition',#2,'design');

/* ---- Shape representation ---- */
#10 = ADVANCED_BREP_SHAPE_REPRESENTATION('',(#11,#15),#345);
#11 = AXIS2_PLACEMENT_3D('',#12,#13,#14);
#12 = CARTESIAN_POINT('',(0.,0.,0.));
#13 = DIRECTION('',(0.,0.,1.));
#14 = DIRECTION('',(1.,0.,-0.));

/* ---- Solid ---- */
#15 = MANIFOLD_SOLID_BREP('',#16);
#16 = CLOSED_SHELL('',(#17,#137,#237,#284,#331,#338));

/* ====== FACE: left (x=-1) ====== */
/* ADVANCED_FACE('name', (bounds), surface, same_sense) */
/* same_sense=.F. because the plane's axis points in +x but this face has outward normal -x */
#17 = ADVANCED_FACE('left',(#18),#32,.F.);
#18 = FACE_BOUND('',#19,.F.);
#19 = EDGE_LOOP('',(#20,#55,#83,#111));

/* Edge: left-back  (from left-back-bottom to left-back-top, along +z) */
#20 = ORIENTED_EDGE('',*,*,#21,.F.);
#21 = EDGE_CURVE('left-back',#22,#24,#26,.T.);
#22 = VERTEX_POINT('left-back-bottom',#23);
#23 = CARTESIAN_POINT('',(-1.,-1.,-1.));
#24 = VERTEX_POINT('left-back-top',#25);
#25 = CARTESIAN_POINT('',(-1.,-1.,1.));
/* SURFACE_CURVE bundles the 3D line + 2 PCURVEs (one for each adjacent face) */
#26 = SURFACE_CURVE('',#27,(#31,#43),.PCURVE_S1.);
#27 = LINE('',#28,#29);
#28 = CARTESIAN_POINT('',(-1.,-1.,-1.));
#29 = VECTOR('',#30,1.);
#30 = DIRECTION('',(0.,0.,1.));
/* PCURVE on the left face plane (#32) */
#31 = PCURVE('',#32,#37);
#32 = PLANE('',#33);           /* <-- this PLANE is re-used as the left face's surface */
#33 = AXIS2_PLACEMENT_3D('',#34,#35,#36);
#34 = CARTESIAN_POINT('',(-1.,-1.,-1.));
#35 = DIRECTION('',(1.,0.,-0.));  /* plane normal = +x direction */
#36 = DIRECTION('',(0.,0.,1.));   /* x-axis of plane's local frame = +z */
#37 = DEFINITIONAL_REPRESENTATION('',(#38),#42);
#38 = LINE('',#39,#40);
#39 = CARTESIAN_POINT('',(0.,0.));
#40 = VECTOR('',#41,1.);
#41 = DIRECTION('',(1.,0.));
#42 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
/* PCURVE on the back face plane (#44) */
#43 = PCURVE('',#44,#49);
#44 = PLANE('',#45);           /* <-- this PLANE re-used as the back face's surface */
#45 = AXIS2_PLACEMENT_3D('',#46,#47,#48);
#46 = CARTESIAN_POINT('',(-1.,-1.,-1.));
#47 = DIRECTION('',(-0.,1.,0.));  /* back face normal = -y (pointing back) */
#48 = DIRECTION('',(0.,0.,1.));
#49 = DEFINITIONAL_REPRESENTATION('',(#50),#54);
#50 = LINE('',#51,#52);
#51 = CARTESIAN_POINT('',(0.,0.));
#52 = VECTOR('',#53,1.);
#53 = DIRECTION('',(1.,0.));
#54 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );

/* Edge: left-bottom (from left-back-bottom to left-front-bottom, along +y) */
#55 = ORIENTED_EDGE('',*,*,#56,.T.);
#56 = EDGE_CURVE('left-bottom',#22,#57,#59,.T.);
#57 = VERTEX_POINT('left-front-bottom',#58);
#58 = CARTESIAN_POINT('',(-1.,1.,-1.));
#59 = SURFACE_CURVE('',#60,(#64,#71),.PCURVE_S1.);
#60 = LINE('',#61,#62);
#61 = CARTESIAN_POINT('',(-1.,-1.,-1.));
#62 = VECTOR('',#63,1.);
#63 = DIRECTION('',(-0.,1.,0.));
#64 = PCURVE('',#32,#65);
#65 = DEFINITIONAL_REPRESENTATION('',(#66),#70);
#66 = LINE('',#67,#68);
#67 = CARTESIAN_POINT('',(0.,0.));
#68 = VECTOR('',#69,1.);
#69 = DIRECTION('',(0.,-1.));
#70 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#71 = PCURVE('',#72,#77);
#72 = PLANE('',#73);           /* <-- bottom face plane */
#73 = AXIS2_PLACEMENT_3D('',#74,#75,#76);
#74 = CARTESIAN_POINT('',(-1.,-1.,-1.));
#75 = DIRECTION('',(0.,0.,1.));   /* bottom face normal = +z (but same_sense=.F. inverts it to -z) */
#76 = DIRECTION('',(1.,0.,-0.));
#77 = DEFINITIONAL_REPRESENTATION('',(#78),#82);
#78 = LINE('',#79,#80);
#79 = CARTESIAN_POINT('',(0.,0.));
#80 = VECTOR('',#81,1.);
#81 = DIRECTION('',(0.,1.));
#82 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );

/* Edge: left-front */
#83 = ORIENTED_EDGE('',*,*,#84,.T.);
#84 = EDGE_CURVE('left-front',#57,#85,#87,.T.);
#85 = VERTEX_POINT('left-front-top',#86);
#86 = CARTESIAN_POINT('',(-1.,1.,1.));
#87 = SURFACE_CURVE('',#88,(#92,#99),.PCURVE_S1.);
#88 = LINE('',#89,#90);
#89 = CARTESIAN_POINT('',(-1.,1.,-1.));
#90 = VECTOR('',#91,1.);
#91 = DIRECTION('',(0.,0.,1.));
#92 = PCURVE('',#32,#93);
#93 = DEFINITIONAL_REPRESENTATION('',(#94),#98);
#94 = LINE('',#95,#96);
#95 = CARTESIAN_POINT('',(0.,-2.));
#96 = VECTOR('',#97,1.);
#97 = DIRECTION('',(1.,0.));
#98 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#99 = PCURVE('',#100,#105);
#100 = PLANE('',#101);         /* <-- front face plane */
#101 = AXIS2_PLACEMENT_3D('',#102,#103,#104);
#102 = CARTESIAN_POINT('',(-1.,1.,-1.));
#103 = DIRECTION('',(-0.,1.,0.));  /* front face: note -0. = sign of -y */
#104 = DIRECTION('',(0.,0.,1.));
#105 = DEFINITIONAL_REPRESENTATION('',(#106),#110);
#106 = LINE('',#107,#108);
#107 = CARTESIAN_POINT('',(0.,0.));
#108 = VECTOR('',#109,1.);
#109 = DIRECTION('',(1.,0.));
#110 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );

/* Edge: left-top */
#111 = ORIENTED_EDGE('',*,*,#112,.F.);
#112 = EDGE_CURVE('left-top',#24,#85,#113,.T.);
#113 = SURFACE_CURVE('',#114,(#118,#125),.PCURVE_S1.);
#114 = LINE('',#115,#116);
#115 = CARTESIAN_POINT('',(-1.,-1.,1.));
#116 = VECTOR('',#117,1.);
#117 = DIRECTION('',(-0.,1.,0.));
#118 = PCURVE('',#32,#119);
#119 = DEFINITIONAL_REPRESENTATION('',(#120),#124);
#120 = LINE('',#121,#122);
#121 = CARTESIAN_POINT('',(2.,0.));
#122 = VECTOR('',#123,1.);
#123 = DIRECTION('',(0.,-1.));
#124 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#125 = PCURVE('',#126,#131);
#126 = PLANE('',#127);         /* <-- top face plane */
#127 = AXIS2_PLACEMENT_3D('',#128,#129,#130);
#128 = CARTESIAN_POINT('',(-1.,-1.,1.));
#129 = DIRECTION('',(0.,0.,1.));   /* top face normal = +z */
#130 = DIRECTION('',(1.,0.,-0.));
#131 = DEFINITIONAL_REPRESENTATION('',(#132),#136);
#132 = LINE('',#133,#134);
#133 = CARTESIAN_POINT('',(0.,0.));
#134 = VECTOR('',#135,1.);
#135 = DIRECTION('',(0.,1.));
#136 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );

/* ====== FACE: right (x=+1) ====== */
#137 = ADVANCED_FACE('right',(#138),#152,.T.);
#138 = FACE_BOUND('',#139,.T.);
#139 = EDGE_LOOP('',(#140,#170,#193,#216));
#140 = ORIENTED_EDGE('',*,*,#141,.F.);
#141 = EDGE_CURVE('right-back',#142,#144,#146,.T.);
#142 = VERTEX_POINT('right-back-bottom',#143);
#143 = CARTESIAN_POINT('',(1.,-1.,-1.));
#144 = VERTEX_POINT('right-back-top',#145);
#145 = CARTESIAN_POINT('',(1.,-1.,1.));
#146 = SURFACE_CURVE('',#147,(#151,#163),.PCURVE_S1.);
#147 = LINE('',#148,#149);
#148 = CARTESIAN_POINT('',(1.,-1.,-1.));
#149 = VECTOR('',#150,1.);
#150 = DIRECTION('',(0.,0.,1.));
#151 = PCURVE('',#152,#157);
#152 = PLANE('',#153);         /* right face plane */
#153 = AXIS2_PLACEMENT_3D('',#154,#155,#156);
#154 = CARTESIAN_POINT('',(1.,-1.,-1.));
#155 = DIRECTION('',(1.,0.,-0.));  /* right face normal = +x */
#156 = DIRECTION('',(0.,0.,1.));
#157 = DEFINITIONAL_REPRESENTATION('',(#158),#162);
#158 = LINE('',#159,#160);
#159 = CARTESIAN_POINT('',(0.,0.));
#160 = VECTOR('',#161,1.);
#161 = DIRECTION('',(1.,0.));
#162 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#163 = PCURVE('',#44,#164);
#164 = DEFINITIONAL_REPRESENTATION('',(#165),#169);
#165 = LINE('',#166,#167);
#166 = CARTESIAN_POINT('',(0.,2.));
#167 = VECTOR('',#168,1.);
#168 = DIRECTION('',(1.,0.));
#169 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#170 = ORIENTED_EDGE('',*,*,#171,.T.);
#171 = EDGE_CURVE('right-bottom',#142,#172,#174,.T.);
#172 = VERTEX_POINT('right-front-bottom',#173);
#173 = CARTESIAN_POINT('',(1.,1.,-1.));
#174 = SURFACE_CURVE('',#175,(#179,#186),.PCURVE_S1.);
#175 = LINE('',#176,#177);
#176 = CARTESIAN_POINT('',(1.,-1.,-1.));
#177 = VECTOR('',#178,1.);
#178 = DIRECTION('',(-0.,1.,0.));
#179 = PCURVE('',#152,#180);
#180 = DEFINITIONAL_REPRESENTATION('',(#181),#185);
#181 = LINE('',#182,#183);
#182 = CARTESIAN_POINT('',(0.,0.));
#183 = VECTOR('',#184,1.);
#184 = DIRECTION('',(0.,-1.));
#185 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#186 = PCURVE('',#72,#187);
#187 = DEFINITIONAL_REPRESENTATION('',(#188),#192);
#188 = LINE('',#189,#190);
#189 = CARTESIAN_POINT('',(2.,0.));
#190 = VECTOR('',#191,1.);
#191 = DIRECTION('',(0.,1.));
#192 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#193 = ORIENTED_EDGE('',*,*,#194,.T.);
#194 = EDGE_CURVE('right-front',#172,#195,#197,.T.);
#195 = VERTEX_POINT('right-front-top',#196);
#196 = CARTESIAN_POINT('',(1.,1.,1.));
#197 = SURFACE_CURVE('',#198,(#202,#209),.PCURVE_S1.);
#198 = LINE('',#199,#200);
#199 = CARTESIAN_POINT('',(1.,1.,-1.));
#200 = VECTOR('',#201,1.);
#201 = DIRECTION('',(0.,0.,1.));
#202 = PCURVE('',#152,#203);
#203 = DEFINITIONAL_REPRESENTATION('',(#204),#208);
#204 = LINE('',#205,#206);
#205 = CARTESIAN_POINT('',(0.,-2.));
#206 = VECTOR('',#207,1.);
#207 = DIRECTION('',(1.,0.));
#208 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#209 = PCURVE('',#100,#210);
#210 = DEFINITIONAL_REPRESENTATION('',(#211),#215);
#211 = LINE('',#212,#213);
#212 = CARTESIAN_POINT('',(0.,2.));
#213 = VECTOR('',#214,1.);
#214 = DIRECTION('',(1.,0.));
#215 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#216 = ORIENTED_EDGE('',*,*,#217,.F.);
#217 = EDGE_CURVE('right-top',#144,#195,#218,.T.);
#218 = SURFACE_CURVE('',#219,(#223,#230),.PCURVE_S1.);
#219 = LINE('',#220,#221);
#220 = CARTESIAN_POINT('',(1.,-1.,1.));
#221 = VECTOR('',#222,1.);
#222 = DIRECTION('',(0.,1.,0.));
#223 = PCURVE('',#152,#224);
#224 = DEFINITIONAL_REPRESENTATION('',(#225),#229);
#225 = LINE('',#226,#227);
#226 = CARTESIAN_POINT('',(2.,0.));
#227 = VECTOR('',#228,1.);
#228 = DIRECTION('',(0.,-1.));
#229 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#230 = PCURVE('',#126,#231);
#231 = DEFINITIONAL_REPRESENTATION('',(#232),#236);
#232 = LINE('',#233,#234);
#233 = CARTESIAN_POINT('',(2.,0.));
#234 = VECTOR('',#235,1.);
#235 = DIRECTION('',(0.,1.));
#236 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );

/* ====== FACE: back (y=-1) ====== */
/* Reuses #44 (back plane) as face geometry */
#237 = ADVANCED_FACE('back',(#238),#44,.F.);
#238 = FACE_BOUND('',#239,.F.);
#239 = EDGE_LOOP('',(#240,#261,#262,#283));
#240 = ORIENTED_EDGE('',*,*,#241,.F.);
#241 = EDGE_CURVE('back-bottom',#22,#142,#242,.T.);
#242 = SURFACE_CURVE('',#243,(#247,#254),.PCURVE_S1.);
#243 = LINE('',#244,#245);
#244 = CARTESIAN_POINT('',(-1.,-1.,-1.));
#245 = VECTOR('',#246,1.);
#246 = DIRECTION('',(1.,0.,-0.));
#247 = PCURVE('',#44,#248);
#248 = DEFINITIONAL_REPRESENTATION('',(#249),#253);
#249 = LINE('',#250,#251);
#250 = CARTESIAN_POINT('',(0.,0.));
#251 = VECTOR('',#252,1.);
#252 = DIRECTION('',(0.,1.));
#253 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#254 = PCURVE('',#72,#255);
#255 = DEFINITIONAL_REPRESENTATION('',(#256),#260);
#256 = LINE('',#257,#258);
#257 = CARTESIAN_POINT('',(0.,0.));
#258 = VECTOR('',#259,1.);
#259 = DIRECTION('',(1.,0.));
#260 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
/* These two ORIENTED_EDGEs reuse existing EDGE_CURVEs from the left and right faces */
#261 = ORIENTED_EDGE('',*,*,#21,.T.);   /* reuse left-back edge, reversed */
#262 = ORIENTED_EDGE('',*,*,#263,.T.);
#263 = EDGE_CURVE('back-top',#24,#144,#264,.T.);
#264 = SURFACE_CURVE('',#265,(#269,#276),.PCURVE_S1.);
#265 = LINE('',#266,#267);
#266 = CARTESIAN_POINT('',(-1.,-1.,1.));
#267 = VECTOR('',#268,1.);
#268 = DIRECTION('',(1.,0.,-0.));
#269 = PCURVE('',#44,#270);
#270 = DEFINITIONAL_REPRESENTATION('',(#271),#275);
#271 = LINE('',#272,#273);
#272 = CARTESIAN_POINT('',(2.,0.));
#273 = VECTOR('',#274,1.);
#274 = DIRECTION('',(0.,1.));
#275 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#276 = PCURVE('',#126,#277);
#277 = DEFINITIONAL_REPRESENTATION('',(#278),#282);
#278 = LINE('',#279,#280);
#279 = CARTESIAN_POINT('',(0.,0.));
#280 = VECTOR('',#281,1.);
#281 = DIRECTION('',(1.,0.));
#282 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#283 = ORIENTED_EDGE('',*,*,#141,.F.);  /* reuse right-back edge */

/* ====== FACE: front (y=+1) ====== */
/* Reuses #100 (front plane) as face geometry */
#284 = ADVANCED_FACE('front',(#285),#100,.T.);
#285 = FACE_BOUND('',#286,.T.);
#286 = EDGE_LOOP('',(#287,#308,#309,#330));
#287 = ORIENTED_EDGE('',*,*,#288,.F.);
#288 = EDGE_CURVE('front-bottom',#57,#172,#289,.T.);
#289 = SURFACE_CURVE('',#290,(#294,#301),.PCURVE_S1.);
#290 = LINE('',#291,#292);
#291 = CARTESIAN_POINT('',(-1.,1.,-1.));
#292 = VECTOR('',#293,1.);
#293 = DIRECTION('',(1.,0.,-0.));
#294 = PCURVE('',#100,#295);
#295 = DEFINITIONAL_REPRESENTATION('',(#296),#300);
#296 = LINE('',#297,#298);
#297 = CARTESIAN_POINT('',(0.,0.));
#298 = VECTOR('',#299,1.);
#299 = DIRECTION('',(0.,1.));
#300 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#301 = PCURVE('',#72,#302);
#302 = DEFINITIONAL_REPRESENTATION('',(#303),#307);
#303 = LINE('',#304,#305);
#304 = CARTESIAN_POINT('',(0.,2.));
#305 = VECTOR('',#306,1.);
#306 = DIRECTION('',(1.,0.));
#307 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#308 = ORIENTED_EDGE('',*,*,#84,.T.);   /* reuse left-front edge */
#309 = ORIENTED_EDGE('',*,*,#310,.T.);
#310 = EDGE_CURVE('front-top',#85,#195,#311,.T.);
#311 = SURFACE_CURVE('',#312,(#316,#323),.PCURVE_S1.);
#312 = LINE('',#313,#314);
#313 = CARTESIAN_POINT('',(-1.,1.,1.));
#314 = VECTOR('',#315,1.);
#315 = DIRECTION('',(1.,0.,-0.));
#316 = PCURVE('',#100,#317);
#317 = DEFINITIONAL_REPRESENTATION('',(#318),#322);
#318 = LINE('',#319,#320);
#319 = CARTESIAN_POINT('',(2.,0.));
#320 = VECTOR('',#321,1.);
#321 = DIRECTION('',(0.,1.));
#322 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#323 = PCURVE('',#126,#324);
#324 = DEFINITIONAL_REPRESENTATION('',(#325),#329);
#325 = LINE('',#326,#327);
#326 = CARTESIAN_POINT('',(0.,2.));
#327 = VECTOR('',#328,1.);
#328 = DIRECTION('',(1.,0.));
#329 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE','') );
#330 = ORIENTED_EDGE('',*,*,#194,.F.);  /* reuse right-front edge */

/* ====== FACE: bottom (z=-1) ====== */
/* Reuses #72 (bottom plane, normal +z, same_sense=.F. -> outward normal is -z) */
#331 = ADVANCED_FACE('bottom',(#332),#72,.F.);
#332 = FACE_BOUND('',#333,.F.);
#333 = EDGE_LOOP('',(#334,#335,#336,#337));
/* All 4 edges are already defined; just reuse with appropriate orientation */
#334 = ORIENTED_EDGE('',*,*,#56,.F.);    /* left-bottom reversed */
#335 = ORIENTED_EDGE('',*,*,#241,.T.);   /* back-bottom */
#336 = ORIENTED_EDGE('',*,*,#171,.T.);   /* right-bottom */
#337 = ORIENTED_EDGE('',*,*,#288,.F.);   /* front-bottom reversed */

/* ====== FACE: top (z=+1) ====== */
/* Reuses #126 (top plane, normal +z, same_sense=.T. -> outward normal is +z) */
#338 = ADVANCED_FACE('top',(#339),#126,.T.);
#339 = FACE_BOUND('',#340,.T.);
#340 = EDGE_LOOP('',(#341,#342,#343,#344));
#341 = ORIENTED_EDGE('',*,*,#112,.F.);   /* left-top reversed */
#342 = ORIENTED_EDGE('',*,*,#263,.T.);   /* back-top */
#343 = ORIENTED_EDGE('',*,*,#217,.T.);   /* right-top */
#344 = ORIENTED_EDGE('',*,*,#310,.F.);   /* front-top reversed */

/* ====== Geometric context: 3D, millimetres, 1e-7 tolerance ====== */
#345 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3)
GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#349)) GLOBAL_UNIT_ASSIGNED_CONTEXT
((#346,#347,#348)) REPRESENTATION_CONTEXT('Context #1',
  '3D Context with UNIT and UNCERTAINTY') );
#346 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
#347 = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );
#348 = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() );
#349 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#346,
  'distance_accuracy_value','confusion accuracy');

/* Optional: PRODUCT_TYPE for AP214 */
#350 = PRODUCT_TYPE('part',$,(#7));
ENDSEC;
END-ISO-10303-21;
```

---

## 8. Key Observations from the Example

### 8.1 Entity Sharing

PLANE and VERTEX entities are heavily shared across faces:
- `#32` (left face PLANE) is referenced by both the left face and by all PCURVEs for edges on the left face
- `#44` (back face PLANE) is used both as back face geometry and as PCURVE basis for edges on the back face
- `#22` (left-back-bottom VERTEX_POINT) is used by 3 different EDGE_CURVEs

This sharing is legal and expected: the same topological/geometric entity can be
referenced from many places.

### 8.2 PCURVE Coordinate Computation

For an edge on face F with AXIS2_PLACEMENT_3D(origin O, x-axis X, y-axis Y = cross(normal, X)):
- The 2D CARTESIAN_POINT in the PCURVE's DEFINITIONAL_REPRESENTATION is `(dot(P - O, X), dot(P - O, Y))`
  where P is the 3D start point of the edge's LINE
- The 2D DIRECTION is the projection of the 3D edge direction onto the face plane

This is how 3D geometry maps to UV parameter space of each face.

### 8.3 `-0.` Values

The format uses `-0.` for negative zero in direction vectors. This is valid IEEE 754
and appears throughout real STEP files from OpenCASCADE. For Rust output, write
direction components that are negative zero as `-0.` or just `0.`.

### 8.4 FACE_BOUND vs FACE_OUTER_BOUND

The OpenCASCADE example uses `FACE_BOUND` everywhere, even for the single outer boundary.
This is accepted by all importers. `FACE_OUTER_BOUND` is semantically more precise but
not required for simple solids without holes.

### 8.5 ORIENTED_EDGE `*` Placeholders

ORIENTED_EDGE has signature: `ORIENTED_EDGE(name, edge_start*, edge_end*, edge_element, orientation)`
The `*` fields are re-declared derived attributes from the supertype (`edge`) and must
be written as `*` (not omitted with `$`).

### 8.6 Same-Edge Sharing Between Faces

Every edge between two faces is represented by exactly ONE `EDGE_CURVE` entity.
The two faces reference it via different `ORIENTED_EDGE` wrappers, with opposite
`.orientation` booleans relative to the edge direction. For example:
- Left face uses `left-back` edge (#21) with `.F.` (reversed)
- Back face uses the same edge (#21) with `.T.` (forward)

---

## 9. Implementation Strategy for brepkit

### 9.1 ID Allocation

Use a simple sequential counter starting at 1. Allocate IDs in a depth-first
traversal of the product structure → shape representation → solid → faces → edges.

### 9.2 Deduplication

Track already-emitted entities by their source handle (`Id<T>` from the arena):
- Vertices: `HashMap<Id<Vertex>, usize>` (STEP ID)
- Edges: `HashMap<Id<Edge>, usize>`
- Faces: `HashMap<Id<Face>, usize>`
- Planes: `HashMap<Id<Face>, usize>` (one PLANE per face)

Edges are shared between faces, so emit each EDGE_CURVE only once.

### 9.3 PCURVE Generation

For a planar face, given:
- Face plane with `AXIS2_PLACEMENT_3D(origin O, normal N, ref_dir X)`
- Y-axis: `Y = normalize(cross(N, X))`
- Edge endpoint P in 3D

The 2D parametric coordinate is:
```
u = dot(P - O, X)
v = dot(P - O, Y)
```

The 2D edge direction is the projection of the 3D direction d onto the plane:
```
d_uv = normalize((dot(d, X), dot(d, Y)))
```

### 9.4 Output Order

Emit entities in this order for readability (not required, forward refs are fine):
1. Product structure (#1–#10)
2. World axis placement
3. MANIFOLD_SOLID_BREP
4. CLOSED_SHELL
5. For each ADVANCED_FACE:
   a. The PLANE + AXIS2_PLACEMENT_3D
   b. The FACE_BOUND + EDGE_LOOP
   c. For each ORIENTED_EDGE: the EDGE_CURVE (if not already emitted), VERTEXes, SURFACE_CURVEs, PCURVEs
6. Geometric context + units at the end

### 9.5 Minimal Valid File (AP203)

If maximum compatibility with older tools is needed, change:
```
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
```
and adjust APPLICATION_PROTOCOL_DEFINITION accordingly. The geometry entities are
identical between AP203 and AP214 — only the product structure metadata changes.

---

## 10. Sources

- [ISO 10303-21 - Wikipedia](https://en.wikipedia.org/wiki/ISO_10303-21) — file format syntax overview
- [STEP Part 21 GitHub Wiki (Bitub)](https://github.com/Bitub/step/wiki/STEP-Part-21-(ISO-10303-21)) — format tokens, entity syntax
- [STEPutils p21 documentation](https://steputils.readthedocs.io/en/latest/p21.html) — attribute encoding rules
- [STEPtools: advanced_brep_shape_representation](https://www.steptools.com/stds/stp_aim/html/t_advanced_brep_shape_representation.html) — WHERE constraints
- [STEPtools: manifold_solid_brep](https://www.steptools.com/stds/stp_aim/html/t_manifold_solid_brep.html) — entity definition
- [STEPtools: advanced_face](https://www.steptools.com/stds/stp_aim/html/t_advanced_face.html) — entity definition
- [STEPtools: surface_curve](https://www.steptools.com/stds/stp_aim/html/t_surface_curve.html) — entity definition
- [STEPtools: pcurve](https://www.steptools.com/stds/stp_aim/html/t_pcurve.html) — entity definition
- [STEPtools: face_outer_bound](https://www.steptools.com/stds/stp_aim/html/t_face_outer_bound.html) — outer vs inner bound semantics
- [AP203 Usage Notes (STEPtools)](https://steptools.com/docs/stp_aim/notes_ap203.html) — AP203 vs AP214 schema differences
- [ISO 10303-42 topology resource](https://ap238.org/SMRL_v8_final/data/resource_docs/geometric_and_topological_representation/sys/5_schema.htm) — edge_loop, oriented_edge, face_bound orientation definitions
- [jCAE cube.stp (complete real example)](https://github.com/jeromerobert/jCAE/blob/master/occjava/test/input/cube.stp) — the complete cube file this document is based on
- [Sample AP203 files (STEPtools)](http://www.steptools.com/docs/stpfiles/ap203/) — additional test files
- [OpenCASCADE STEP Translator docs](https://dev.opencascade.org/doc/overview/html/occt_user_guides__step.html) — STEP reading/writing in OCCT
- [Capvidia: Best STEP file to use](https://www.capvidia.com/blog/best-step-file-to-use-ap203-vs-ap214-vs-ap242) — AP203/AP214/AP242 comparison
