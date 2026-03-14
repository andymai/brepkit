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
