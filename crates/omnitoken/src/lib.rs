//! `omnitoken` — library re-exports & universal C-ABI / Python bindings.
//!
//! All downstream code should depend on this crate, not on individual pipeline
//! crates, so the internal crate layout can evolve without breaking callers.

pub mod c_api;
pub mod dispatcher;

pub use c_api::*;
pub use dispatcher::*;

pub use omnitoken_core::{
    Capabilities, OmniError, State, TokenId, TokenSink, VecSink,
};
pub use pretokenizer::{self, ByteClass};
pub use trie_builder::{self, Trie};
pub use vocab_ir::{self, AlgoKind, MergeRule, UnigramScore, VocabIr, detect_vocab_kind, load_auto};
pub use walker::{self, BpeStreamingEncoder, StreamingEncoder, bpe_encode, build_merge_rank, encode};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass]
pub struct PyTokenizer {
    ir: VocabIr,
    trie: Trie,
    dispatcher: AdaptiveDispatcher,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTokenizer {
    #[new]
    pub fn new(vocab_path: &str) -> PyResult<Self> {
        let ir = vocab_ir::load_auto(vocab_path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let trie = trie_builder::build(&ir)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dispatcher = AdaptiveDispatcher::new();
        Ok(PyTokenizer { ir, trie, dispatcher })
    }

    pub fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        self.dispatcher
            .encode(text, &self.ir, &self.trie)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn encode_batch(&self, texts: Vec<&str>) -> PyResult<Vec<Vec<u32>>> {
        self.dispatcher
            .encode_batch(&texts, &self.ir, &self.trie)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn save_otk(&self, path: &str) -> PyResult<()> {
        self.ir
            .save_otk(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn _omnitoken(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
