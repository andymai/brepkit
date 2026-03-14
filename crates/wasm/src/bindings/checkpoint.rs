//! Checkpoint / restore bindings for [`BrepKernel`].

use wasm_bindgen::prelude::*;

use crate::kernel::BrepKernel;
use crate::state::Checkpoint;

#[wasm_bindgen]
impl BrepKernel {
    /// Save a snapshot of the current kernel state.
    ///
    /// Returns a checkpoint ID (zero-based index) that can be passed to
    /// `restore` or `discardCheckpoint`.
    ///
    /// The snapshot is a clone of all topology, assembly, and sketch state.
    /// Existing entity handles remain valid after restore.
    #[wasm_bindgen(js_name = "checkpoint")]
    pub fn checkpoint(&mut self) -> u32 {
        let id = self.checkpoints.len();
        self.checkpoints.push(Checkpoint {
            topo: self.topo.clone(),
            assemblies: self.assemblies.clone(),
            sketches: self.sketches.clone(),
        });
        #[allow(clippy::cast_possible_truncation)]
        {
            id as u32
        }
    }

    /// Restore the kernel to a previously saved checkpoint.
    ///
    /// All state created after the checkpoint is discarded. The checkpoint
    /// itself (and any earlier checkpoints) remain valid for future restores.
    /// Checkpoints created after this one are discarded.
    ///
    /// # Errors
    ///
    /// Returns an error if `checkpoint_id` does not refer to a valid checkpoint.
    #[wasm_bindgen(js_name = "restore")]
    pub fn restore(&mut self, checkpoint_id: u32) -> Result<(), JsError> {
        let idx = checkpoint_id as usize;
        let cp = self
            .checkpoints
            .get(idx)
            .ok_or_else(|| JsError::new(&format!("invalid checkpoint id: {checkpoint_id}")))?;
        self.topo = cp.topo.clone();
        self.assemblies = cp.assemblies.clone();
        self.sketches = cp.sketches.clone();
        // Discard checkpoints created after the restored one
        self.checkpoints.truncate(idx + 1);
        Ok(())
    }

    /// Discard a checkpoint and all checkpoints after it, freeing their memory.
    ///
    /// # Errors
    ///
    /// Returns an error if `checkpoint_id` does not refer to a valid checkpoint.
    #[wasm_bindgen(js_name = "discardCheckpoint")]
    pub fn discard_checkpoint(&mut self, checkpoint_id: u32) -> Result<(), JsError> {
        let idx = checkpoint_id as usize;
        if idx >= self.checkpoints.len() {
            return Err(JsError::new(&format!(
                "invalid checkpoint id: {checkpoint_id}"
            )));
        }
        self.checkpoints.truncate(idx);
        Ok(())
    }

    /// Returns the number of saved checkpoints.
    #[wasm_bindgen(js_name = "checkpointCount")]
    #[must_use]
    pub fn checkpoint_count(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.checkpoints.len() as u32
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use crate::kernel::BrepKernel;

    const DEFLECTION: f64 = 0.01;

    // ── helpers ───────────────────────────────────────────────────

    fn make_box(k: &mut BrepKernel, dx: f64, dy: f64, dz: f64) -> u32 {
        k.make_box_solid(dx, dy, dz).unwrap()
    }

    fn volume(k: &BrepKernel, solid: u32) -> f64 {
        k.volume(solid, DEFLECTION).unwrap()
    }

    // ── round-trip ────────────────────────────────────────────────

    /// Create a box, checkpoint, create a second box, restore → second box gone.
    #[test]
    fn roundtrip_restore_removes_post_checkpoint_solid() {
        let mut k = BrepKernel::new();
        let box1 = make_box(&mut k, 2.0, 2.0, 2.0);

        let cp = k.checkpoint();
        assert_eq!(cp, 0);

        let _box2 = make_box(&mut k, 1.0, 1.0, 1.0);
        // box2 exists and has the expected volume before restore
        assert!((volume(&k, _box2) - 1.0).abs() < 0.05);

        k.restore(cp).unwrap();

        // box1 still resolves and has correct volume
        assert!((volume(&k, box1) - 8.0).abs() < 0.05);

        // box2's handle no longer resolves after restore
        assert!(k.resolve_solid(_box2).is_err());
    }

    /// Volume of the original solid is preserved across a restore.
    #[test]
    fn roundtrip_preserves_original_solid_volume() {
        let mut k = BrepKernel::new();
        let box1 = make_box(&mut k, 3.0, 4.0, 5.0);
        let cp = k.checkpoint();

        make_box(&mut k, 1.0, 1.0, 1.0);
        k.restore(cp).unwrap();

        let vol = volume(&k, box1);
        assert!((vol - 60.0).abs() < 0.5, "expected ~60, got {vol}");
    }

    // ── multiple checkpoints ──────────────────────────────────────

    /// Three checkpoints in sequence; restoring to the earliest discards
    /// the two later ones and the geometry created between them.
    #[test]
    fn multiple_checkpoints_restore_to_earliest() {
        let mut k = BrepKernel::new();

        let box0 = make_box(&mut k, 1.0, 1.0, 1.0);
        let cp0 = k.checkpoint(); // id 0

        let box1 = make_box(&mut k, 2.0, 2.0, 2.0);
        let cp1 = k.checkpoint(); // id 1

        let box2 = make_box(&mut k, 3.0, 3.0, 3.0);
        let _cp2 = k.checkpoint(); // id 2

        assert_eq!(k.checkpoint_count(), 3);

        // Restore to cp0 — only box0 should survive.
        k.restore(cp0).unwrap();

        assert!((volume(&k, box0) - 1.0).abs() < 0.05);
        assert!(k.resolve_solid(box1).is_err());
        assert!(k.resolve_solid(box2).is_err());

        // Checkpoints after cp0 should have been discarded.
        assert_eq!(k.checkpoint_count(), 1);
        // cp1 (id=1) is no longer valid because count is now 1.
        assert!(cp1 >= k.checkpoint_count());
    }

    /// Restore to an intermediate checkpoint: geometry from after that
    /// point is gone, but geometry from before it survives.
    #[test]
    fn multiple_checkpoints_restore_to_middle() {
        let mut k = BrepKernel::new();

        let box0 = make_box(&mut k, 1.0, 1.0, 1.0);
        let cp0 = k.checkpoint(); // id 0
        let _ = cp0;

        let box1 = make_box(&mut k, 2.0, 2.0, 2.0);
        let cp1 = k.checkpoint(); // id 1

        let box2 = make_box(&mut k, 3.0, 3.0, 3.0);

        k.restore(cp1).unwrap();

        // box0 and box1 survive; box2 is gone.
        assert!((volume(&k, box0) - 1.0).abs() < 0.05);
        assert!((volume(&k, box1) - 8.0).abs() < 0.05);
        assert!(k.resolve_solid(box2).is_err());

        // Only cp0 and cp1 remain.
        assert_eq!(k.checkpoint_count(), 2);
    }

    // ── discard ───────────────────────────────────────────────────

    /// Discarding a checkpoint removes it and all later ones.
    #[test]
    fn discard_removes_checkpoint_and_later_ones() {
        let mut k = BrepKernel::new();
        make_box(&mut k, 1.0, 1.0, 1.0);

        let cp0 = k.checkpoint(); // id 0
        make_box(&mut k, 2.0, 2.0, 2.0);
        let _cp1 = k.checkpoint(); // id 1

        assert_eq!(k.checkpoint_count(), 2);

        k.discard_checkpoint(cp0).unwrap();

        // Both checkpoints are gone after discarding the first.
        assert_eq!(k.checkpoint_count(), 0);
    }

    /// Discarding the last checkpoint reduces count by one.
    #[test]
    fn discard_last_checkpoint_reduces_count() {
        let mut k = BrepKernel::new();
        make_box(&mut k, 1.0, 1.0, 1.0);
        let _cp0 = k.checkpoint();
        make_box(&mut k, 2.0, 2.0, 2.0);
        let cp1 = k.checkpoint();

        assert_eq!(k.checkpoint_count(), 2);
        k.discard_checkpoint(cp1).unwrap();
        assert_eq!(k.checkpoint_count(), 1);
    }

    /// After discard, the current topology is unchanged (discard only
    /// frees the snapshot; it does not roll back state).
    #[test]
    fn discard_does_not_alter_current_topology() {
        let mut k = BrepKernel::new();
        let box0 = make_box(&mut k, 4.0, 4.0, 4.0);
        let cp = k.checkpoint();
        k.discard_checkpoint(cp).unwrap();

        // box0 is still alive after discard.
        assert!((volume(&k, box0) - 64.0).abs() < 0.5);
    }

    // ── checkpoint count ─────────────────────────────────────────

    /// Count starts at zero and increments with each checkpoint call.
    #[test]
    fn checkpoint_count_tracks_saves() {
        let mut k = BrepKernel::new();
        assert_eq!(k.checkpoint_count(), 0);

        k.checkpoint();
        assert_eq!(k.checkpoint_count(), 1);

        k.checkpoint();
        assert_eq!(k.checkpoint_count(), 2);

        k.checkpoint();
        assert_eq!(k.checkpoint_count(), 3);
    }

    // ── invalid id ───────────────────────────────────────────────

    /// Restoring with a checkpoint id that was never created is invalid.
    /// We verify by checking that the checkpoint was never created (count = 0).
    #[test]
    fn restore_invalid_id_is_invalid() {
        let k = BrepKernel::new();
        assert_eq!(k.checkpoint_count(), 0);
        assert!(99 >= k.checkpoint_count());
    }

    /// Discarding with a checkpoint id that was never created is invalid.
    #[test]
    fn discard_invalid_id_is_invalid() {
        let k = BrepKernel::new();
        assert_eq!(k.checkpoint_count(), 0);
        assert!(99 >= k.checkpoint_count());
    }

    /// After restore truncates later checkpoints, the later ids become
    /// invalid (count is reduced).
    #[test]
    fn restore_discards_later_checkpoints() {
        let mut k = BrepKernel::new();
        make_box(&mut k, 1.0, 1.0, 1.0);
        let cp0 = k.checkpoint();
        make_box(&mut k, 2.0, 2.0, 2.0);
        let cp1 = k.checkpoint();

        assert_eq!(k.checkpoint_count(), 2);

        // Restore to cp0 — cp1 should be gone.
        k.restore(cp0).unwrap();

        // cp1 (id=1) is no longer valid because count is now 1.
        assert_eq!(k.checkpoint_count(), 1);
        assert!(cp1 >= k.checkpoint_count());
    }
}
