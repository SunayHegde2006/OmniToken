//! `omnitoken` — library re-exports.
//!
//! All downstream code should depend on this crate, not on individual pipeline
//! crates, so the internal crate layout can evolve without breaking callers.

pub use vocab_ir::{self, AlgoKind, MergeRule, UnigramScore, VocabIr};
pub use trie_builder::{self, Trie};
pub use pretokenizer::{self, ByteClass};
pub use walker::{self, bpe_encode, build_merge_rank, encode};
pub use hot_cache::{self, Fingerprint, HotCache};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass]
pub struct PyTokenizer {
    ir: VocabIr,
    trie: Trie,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTokenizer {
    #[new]
    pub fn new(vocab_json_path: &str) -> PyResult<Self> {
        let content = std::fs::read_to_string(vocab_json_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let ir = vocab_ir::load_hf(&content)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let trie = trie_builder::build(&ir)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyTokenizer { ir, trie })
    }

    pub fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        walker::encode(text, &self.ir, &self.trie)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn _omnitoken(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
