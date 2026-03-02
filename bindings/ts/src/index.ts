/**
 * @brepkit/wasm — TypeScript bindings for the brepkit CAD kernel.
 *
 * @example
 * ```ts
 * import { initBrepkit } from '@brepkit/wasm';
 *
 * await initBrepkit();
 * // Use brepkit functions...
 * ```
 */

export { initBrepkit, isInitialized } from './init.js';
export type { Point3, Vec3, SolidHandle, FaceHandle, EdgeHandle } from './types.js';
