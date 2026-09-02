//! `c_api` — Stable C-ABI & DLPack Zero-Copy Tensor Handoff Layer.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::RwLock;

use trie_builder::Trie;
use vocab_ir::VocabIr;

static LAST_ERROR: RwLock<Option<String>> = RwLock::new(None);

fn set_last_error(err: impl Into<String>) {
    if let Ok(mut g) = LAST_ERROR.write() {
        *g = Some(err.into());
    }
}

pub struct OmniTokenizerHandle {
    pub ir: VocabIr,
    pub trie: Trie,
    pub dispatcher: crate::dispatcher::AdaptiveDispatcher,
}

/// Create an `OmniTokenizerHandle` instance from a `tokenizers.json` or `.otk` path.
/// Returns `0` on success, `-1` on error.
///
/// # Safety
/// Caller must ensure `vocab_path` points to a valid null-terminated C string and
/// `out_handle` points to a valid writeable pointer location.
#[no_mangle]
pub unsafe extern "C" fn omni_tokenizer_create(
    vocab_path: *const c_char,
    out_handle: *mut *mut OmniTokenizerHandle,
) -> i32 {
    if vocab_path.is_null() || out_handle.is_null() {
        set_last_error("Null pointer passed to omni_tokenizer_create");
        return -1;
    }

    let c_str = match CStr::from_ptr(vocab_path).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 path string: {e}"));
            return -1;
        }
    };

    let ir_result = if c_str.ends_with(".otk") {
        vocab_ir::load_otk_file(c_str)
    } else {
        vocab_ir::load_hf_file(c_str)
    };

    let ir = match ir_result {
        Ok(ir) => ir,
        Err(e) => {
            set_last_error(format!("Failed to load vocabulary from `{c_str}`: {e}"));
            return -1;
        }
    };

    let trie = match trie_builder::build(&ir) {
        Ok(t) => t,
        Err(e) => {
            set_last_error(format!("Failed to build trie automaton: {e}"));
            return -1;
        }
    };

    let handle = Box::new(OmniTokenizerHandle {
        ir,
        trie,
        dispatcher: crate::dispatcher::AdaptiveDispatcher::new(),
    });

    *out_handle = Box::into_raw(handle);
    0
}

/// Destroy an `OmniTokenizerHandle` instance.
///
/// # Safety
/// Caller must pass a handle allocated by `omni_tokenizer_create` or NULL.
#[no_mangle]
pub unsafe extern "C" fn omni_tokenizer_free(handle: *mut OmniTokenizerHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Encode a batch of C string pointers into a flat token array.
/// Writes raw pointer to token array and per-sequence lengths array.
/// Returns `0` on success, `-1` on error.
///
/// # Safety
/// Caller must ensure `handle` is a valid handle and `text_ptrs` contains `batch_size` valid null-terminated C string pointers.
#[no_mangle]
pub unsafe extern "C" fn omni_encode_batch(
    handle: *mut OmniTokenizerHandle,
    text_ptrs: *const *const c_char,
    batch_size: usize,
    out_tokens_ptr: *mut *mut u32,
    out_lengths_ptr: *mut *mut usize,
) -> i32 {
    if handle.is_null() || text_ptrs.is_null() || out_tokens_ptr.is_null() || out_lengths_ptr.is_null() {
        set_last_error("Null pointer passed to omni_encode_batch");
        return -1;
    }

    let h = &*handle;
    let mut batch_strings = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let str_ptr = *text_ptrs.add(i);
        if str_ptr.is_null() {
            set_last_error(format!("Null string pointer at index {i} in omni_encode_batch"));
            return -1;
        }
        match CStr::from_ptr(str_ptr).to_str() {
            Ok(s) => batch_strings.push(s),
            Err(e) => {
                set_last_error(format!("Invalid UTF-8 string at index {i}: {e}"));
                return -1;
            }
        }
    }

    let encoded_batch = match h.dispatcher.encode_batch(&batch_strings, &h.ir, &h.trie) {
        Ok(b) => b,
        Err(e) => {
            set_last_error(format!("Tokenization batch failed: {e}"));
            return -1;
        }
    };

    let mut flat_tokens = Vec::new();
    let mut lengths = Vec::with_capacity(batch_size);

    for seq in encoded_batch {
        lengths.push(seq.len());
        flat_tokens.extend(seq);
    }

    flat_tokens.shrink_to_fit();
    lengths.shrink_to_fit();

    let tokens_raw = flat_tokens.as_mut_ptr();
    let lengths_raw = lengths.as_mut_ptr();

    std::mem::forget(flat_tokens);
    std::mem::forget(lengths);

    *out_tokens_ptr = tokens_raw;
    *out_lengths_ptr = lengths_raw;

    0
}

/// Free batch output memory buffers allocated by `omni_encode_batch`.
///
/// # Safety
/// `tokens_ptr` and `lengths_ptr` must be non-null buffers returned from `omni_encode_batch` or NULL.
#[no_mangle]
pub unsafe extern "C" fn omni_free_batch(tokens_ptr: *mut u32, lengths_ptr: *mut usize) {
    if !tokens_ptr.is_null() {
        let _ = Vec::from_raw_parts(tokens_ptr, 0, 0);
    }
    if !lengths_ptr.is_null() {
        let _ = Vec::from_raw_parts(lengths_ptr, 0, 0);
    }
}

/// Get the last error string into `buf`.
///
/// # Safety
/// `buf` must point to a writeable memory buffer of at least `max_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn omni_get_last_error(buf: *mut c_char, max_len: usize) -> i32 {
    if buf.is_null() || max_len == 0 {
        return -1;
    }
    if let Ok(guard) = LAST_ERROR.read() {
        if let Some(err_str) = guard.as_ref() {
            if let Ok(c_err) = CString::new(err_str.as_str()) {
                let bytes = c_err.as_bytes_with_nul();
                let copy_len = bytes.len().min(max_len);
                ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, copy_len);
                return 0;
            }
        }
    }
    *buf = 0;
    0
}

// ─── DLPack Zero-Copy Tensor Handoff Layer ───────────────────────────────────

#[repr(C)]
pub struct DLDevice {
    pub device_type: i32, // 1 = kDLCPU, 2 = kDLCUDA
    pub device_id: i32,
}

#[repr(C)]
pub struct DLDataType {
    pub code: u8, // 0 = kDLInt, 1 = kDLUInt, 2 = kDLFloat
    pub bits: u8, // 32
    pub lanes: u16, // 1
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut std::ffi::c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut std::ffi::c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

unsafe extern "C" fn dlpack_deleter(tensor: *mut DLManagedTensor) {
    if !tensor.is_null() {
        let mt = Box::from_raw(tensor);
        if !mt.dl_tensor.data.is_null() {
            let _ = Vec::from_raw_parts(
                mt.dl_tensor.data as *mut u32,
                mt.dl_tensor.shape.read() as usize,
                mt.dl_tensor.shape.read() as usize,
            );
        }
        if !mt.dl_tensor.shape.is_null() {
            let _ = Box::from_raw(mt.dl_tensor.shape);
        }
    }
}

/// Create a `DLManagedTensor` from a vector of token IDs for zero-copy PyTorch/vLLM handoff.
pub fn create_dlpack_tensor(tokens: Vec<u32>) -> *mut DLManagedTensor {
    let len = tokens.len();
    let mut tokens = tokens;
    tokens.shrink_to_fit();
    let data_ptr = tokens.as_mut_ptr() as *mut std::ffi::c_void;
    std::mem::forget(tokens);

    let shape = Box::into_raw(Box::new(len as i64));

    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: data_ptr,
            device: DLDevice { device_type: 1, device_id: 0 }, // CPU
            ndim: 1,
            dtype: DLDataType { code: 1, bits: 32, lanes: 1 }, // u32
            shape,
            strides: ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: ptr::null_mut(),
        deleter: Some(dlpack_deleter),
    });

    Box::into_raw(managed)
}
