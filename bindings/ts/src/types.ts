/**
 * Branded types for brepkit, ensuring type safety at the TypeScript level.
 *
 * These mirror the patterns used in brepjs for consistent developer experience.
 */

/** A unique brand symbol for nominal typing. */
declare const brand: unique symbol;

/** A branded type that prevents accidental mixing of semantically different values. */
type Brand<T, B extends string> = T & { readonly [brand]: B };

/** A 3D point (position in space). */
export type Point3 = Brand<{ readonly x: number; readonly y: number; readonly z: number }, 'Point3'>;

/** A 3D vector (direction/displacement). */
export type Vec3 = Brand<{ readonly x: number; readonly y: number; readonly z: number }, 'Vec3'>;

/** An opaque handle to a solid in the kernel. */
export type SolidHandle = Brand<number, 'SolidHandle'>;

/** An opaque handle to a face in the kernel. */
export type FaceHandle = Brand<number, 'FaceHandle'>;

/** An opaque handle to an edge in the kernel. */
export type EdgeHandle = Brand<number, 'EdgeHandle'>;
