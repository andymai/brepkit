#!/usr/bin/env node
/**
 * Smoke test for the brepkit WASM package.
 * Verifies that the built package loads and basic operations work.
 *
 * Usage: node scripts/test-wasm-smoke.mjs
 */

import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");

// Use createRequire to load the CJS node entry from an ESM context.
// The node entry uses CommonJS (exports.X = ...) and is renamed to .cjs
// so Node treats it correctly even with "type": "module" in package.json.
const require = createRequire(import.meta.url);
const { BrepKernel } = require(
  resolve(projectRoot, "crates/wasm/pkg/brepkit_wasm_node.cjs"),
);

const DEFLECTION = 0.1;

// 1. Kernel creation
const kernel = new BrepKernel();
console.log("ok - BrepKernel created");

// 2. Make a box
const boxId = kernel.makeBox(10, 20, 30);
assert.equal(typeof boxId, "number", "makeBox should return a number handle");
console.log(`ok - makeBox(10, 20, 30) -> handle ${boxId}`);

// 3. Volume check
const vol = kernel.volume(boxId, DEFLECTION);
assert.ok(Math.abs(vol - 6000) < 1e-6, `volume=${vol}, expected ~6000`);
console.log(`ok - volume = ${vol}`);

// 4. Tessellation
const mesh = kernel.tessellateSolid(boxId, DEFLECTION);
assert.ok(mesh.positions.length > 0, "mesh should have positions");
assert.ok(mesh.indices.length > 0, "mesh should have indices");
assert.equal(mesh.positions.length % 3, 0, "positions should be a multiple of 3");
assert.equal(mesh.indices.length % 3, 0, "indices should be a multiple of 3");
console.log(
  `ok - tessellation: ${mesh.positions.length / 3} verts, ${mesh.indices.length / 3} tris`,
);

// 5. STL export (only if io feature is compiled in)
if (typeof kernel.exportStl === "function") {
  const stl = kernel.exportStl(boxId, DEFLECTION);
  assert.ok(stl.length > 0, "STL export should not be empty");
  console.log(`ok - STL export: ${stl.length} bytes`);
} else {
  console.log("skip - exportStl not available (io feature not enabled)");
}

console.log("\nAll smoke tests passed");
